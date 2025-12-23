"""Tests for database manager functionality.

Tests cover:
1. Database initialization
2. Table creation
3. Saving game results
4. Querying results
5. Error handling
6. Integration with GameRunner
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock
import numpy as np

from cpr_game.db_manager import DatabaseManager
from cpr_game.game_runner import GameRunner
from cpr_game.llm_agent import MockLLMAgent
from cpr_game.config import CONFIG


class TestDatabaseManagerInitialization:
    """Test database manager initialization."""

    def test_init_with_default_path(self):
        """Test initialization with default path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db.db")
            db_manager = DatabaseManager(db_path=db_path, enabled=True)
            
            assert db_manager.enabled is True
            assert db_manager.conn is not None
            assert os.path.exists(db_path)
            
            db_manager.close()

    def test_init_disabled(self):
        """Test initialization when disabled."""
        db_manager = DatabaseManager(enabled=False)
        
        assert db_manager.enabled is False
        assert db_manager.conn is None

    def test_init_creates_directory(self):
        """Test that initialization creates parent directory if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "subdir", "nested", "test_db.db")
            db_manager = DatabaseManager(db_path=db_path, enabled=True)
            
            assert os.path.exists(db_path)
            assert os.path.isdir(os.path.dirname(db_path))
            
            db_manager.close()

    def test_table_creation(self):
        """Test that table is created on initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db.db")
            db_manager = DatabaseManager(db_path=db_path, enabled=True)
            
            # Check table exists by trying to query it
            # If table doesn't exist, this will raise an error
            result = db_manager.query_results("SELECT COUNT(*) FROM game_results")
            assert len(result) == 1
            assert result[0][0] == 0  # Table exists but is empty
            
            db_manager.close()


class TestSaveGameResults:
    """Test saving game results to database."""

    def test_save_single_game_results(self):
        """Test saving results for a single game."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db.db")
            db_manager = DatabaseManager(db_path=db_path, enabled=True)
            
            # Create mock agents
            agents = []
            for i in range(2):
                agent = Mock()
                agent.player_id = i
                agent.persona = f"persona_{i}"
                agent.llm_model = "gpt-3.5-turbo"
                agents.append(agent)
            
            # Create summary
            summary = {
                "cumulative_payoffs": [100.5, 150.3]
            }
            
            # Save results
            db_manager.save_game_results(
                game_id="test_game_1",
                agents=agents,
                summary=summary,
                config=None,
                experiment_id="exp_123",
                timestamp="2024-01-01T12:00:00"
            )
            
            # Query and verify
            results = db_manager.get_all_results()
            assert len(results) == 2
            
            # Check first player
            assert results[0][0] == "test_game_1"  # game_id
            assert results[0][1] == 0  # player_id
            assert results[0][2] == "persona_0"  # persona
            assert results[0][3] == "gpt-3.5-turbo"  # model
            assert results[0][4] == 100.5  # total_reward
            assert results[0][5] == "exp_123"  # experiment_id
            # Timestamp may be stored as datetime object, check string representation
            timestamp_str = str(results[0][6])
            assert "2024-01-01" in timestamp_str or "2024-01-01T12:00:00" in timestamp_str
            
            # Check second player
            assert results[1][0] == "test_game_1"  # game_id
            assert results[1][1] == 1  # player_id
            assert results[1][2] == "persona_1"  # persona
            assert results[1][3] == "gpt-3.5-turbo"  # model
            assert results[1][4] == 150.3  # total_reward
            assert results[1][5] == "exp_123"  # experiment_id
            # Timestamp may be stored as datetime object, check string representation
            timestamp_str = str(results[1][6])
            assert "2024-01-01" in timestamp_str or "2024-01-01T12:00:00" in timestamp_str
            
            db_manager.close()

    def test_save_multiple_games(self):
        """Test saving results for multiple games."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db.db")
            db_manager = DatabaseManager(db_path=db_path, enabled=True)
            
            # Save first game
            agents1 = [Mock(player_id=0, persona="selfish", llm_model="gpt-3.5-turbo")]
            db_manager.save_game_results(
                game_id="game_1",
                agents=agents1,
                summary={"cumulative_payoffs": [200.0]},
                config=None
            )
            
            # Save second game
            agents2 = [Mock(player_id=0, persona="cooperative", llm_model="gpt-4")]
            db_manager.save_game_results(
                game_id="game_2",
                agents=agents2,
                summary={"cumulative_payoffs": [180.0]},
                config=None
            )
            
            # Verify both games are saved
            results = db_manager.get_all_results()
            assert len(results) == 2
            
            game_ids = [r[0] for r in results]
            assert "game_1" in game_ids
            assert "game_2" in game_ids
            
            db_manager.close()

    def test_save_with_config_model_fallback(self):
        """Test that model is taken from config if not in agent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db.db")
            db_manager = DatabaseManager(db_path=db_path, enabled=True)
            
            # Create agent without llm_model
            agent = Mock()
            agent.player_id = 0
            agent.persona = "test_persona"
            del agent.llm_model  # Remove llm_model attribute
            
            config = {"llm_model": "gpt-4"}
            
            db_manager.save_game_results(
                game_id="test_game",
                agents=[agent],
                summary={"cumulative_payoffs": [100.0]},
                config=config
            )
            
            # Verify model from config was used
            results = db_manager.get_all_results()
            assert results[0][3] == "gpt-4"  # model
            
            db_manager.close()

    def test_save_with_missing_cumulative_payoffs(self):
        """Test handling when cumulative_payoffs is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db.db")
            db_manager = DatabaseManager(db_path=db_path, enabled=True)
            
            agents = [Mock(player_id=0, persona="test", llm_model="gpt-3.5-turbo")]
            summary = {}  # Missing cumulative_payoffs
            
            # Should not raise error, just log warning
            db_manager.save_game_results(
                game_id="test_game",
                agents=agents,
                summary=summary,
                config=None
            )
            
            # Verify nothing was saved
            results = db_manager.get_all_results()
            assert len(results) == 0
            
            db_manager.close()

    def test_save_with_disabled_manager(self):
        """Test that save does nothing when manager is disabled."""
        db_manager = DatabaseManager(enabled=False)
        
        agents = [Mock(player_id=0, persona="test", llm_model="gpt-3.5-turbo")]
        summary = {"cumulative_payoffs": [100.0]}
        
        # Should not raise error
        db_manager.save_game_results(
            game_id="test_game",
            agents=agents,
            summary=summary,
            config=None
        )
        
        # No database connection, so nothing to verify
        assert db_manager.conn is None

    def test_save_updates_existing_record(self):
        """Test that saving same game_id+player_id updates existing record."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db.db")
            db_manager = DatabaseManager(db_path=db_path, enabled=True)
            
            # Save first time
            agents1 = [Mock(player_id=0, persona="selfish", llm_model="gpt-3.5-turbo")]
            db_manager.save_game_results(
                game_id="game_1",
                agents=agents1,
                summary={"cumulative_payoffs": [100.0]},
                config=None
            )
            
            # Save again with different values
            agents2 = [Mock(player_id=0, persona="cooperative", llm_model="gpt-4")]
            db_manager.save_game_results(
                game_id="game_1",
                agents=agents2,
                summary={"cumulative_payoffs": [200.0]},
                config=None
            )
            
            # Should only have one record (updated, not duplicated)
            results = db_manager.get_all_results()
            assert len(results) == 1
            assert results[0][2] == "cooperative"  # Updated persona
            assert results[0][3] == "gpt-4"  # Updated model
            assert results[0][4] == 200.0  # Updated reward
            
            db_manager.close()

    def test_save_with_mock_agent(self):
        """Test saving with MockLLMAgent (which doesn't have llm_model)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db.db")
            db_manager = DatabaseManager(db_path=db_path, enabled=True)
            
            # Create MockLLMAgent with full config
            config = CONFIG.copy()
            config.update({
                "llm_model": "gpt-3.5-turbo",
                "min_extraction": 0,
                "max_extraction": 100,
                "include_history_rounds": 3,
            })
            agent = MockLLMAgent(player_id=0, persona="rational_selfish", config=config)
            
            db_manager.save_game_results(
                game_id="test_game",
                agents=[agent],
                summary={"cumulative_payoffs": [150.0]},
                config=config
            )
            
            # Verify saved correctly
            results = db_manager.get_all_results()
            assert len(results) == 1
            assert results[0][1] == 0  # player_id
            assert results[0][2] == "rational_selfish"  # persona
            assert results[0][3] == "gpt-3.5-turbo"  # model from config
            assert results[0][4] == 150.0  # total_reward
            
            db_manager.close()


class TestQueryResults:
    """Test querying results from database."""

    def test_get_all_results(self):
        """Test getting all results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db.db")
            db_manager = DatabaseManager(db_path=db_path, enabled=True)
            
            # Save multiple games
            for game_id in ["game_1", "game_2", "game_3"]:
                agents = [
                    Mock(player_id=i, persona=f"persona_{i}", llm_model="gpt-3.5-turbo")
                    for i in range(2)
                ]
                db_manager.save_game_results(
                    game_id=game_id,
                    agents=agents,
                    summary={"cumulative_payoffs": [100.0 + i, 150.0 + i] for i in range(2)},
                    config=None
                )
            
            # Get all results
            results = db_manager.get_all_results()
            assert len(results) == 6  # 3 games * 2 players
            
            db_manager.close()

    def test_get_player_stats(self):
        """Test getting player statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db.db")
            db_manager = DatabaseManager(db_path=db_path, enabled=True)
            
            # Save multiple games for same player
            for game_id in ["game_1", "game_2", "game_3"]:
                agents = [Mock(player_id=0, persona="selfish", llm_model="gpt-3.5-turbo")]
                db_manager.save_game_results(
                    game_id=game_id,
                    agents=agents,
                    summary={"cumulative_payoffs": [100.0]},
                    config=None
                )
            
            # Get stats for player 0
            stats = db_manager.get_player_stats(player_id=0)
            assert len(stats) == 1
            assert stats[0][0] == 0  # player_id
            assert stats[0][3] == 3  # game_count
            assert stats[0][4] == 100.0  # avg_reward
            
            db_manager.close()

    def test_get_all_player_stats(self):
        """Test getting statistics for all players."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db.db")
            db_manager = DatabaseManager(db_path=db_path, enabled=True)
            
            # Save games with different players
            for game_id in ["game_1", "game_2"]:
                agents = [
                    Mock(player_id=i, persona=f"persona_{i}", llm_model="gpt-3.5-turbo")
                    for i in range(2)
                ]
                db_manager.save_game_results(
                    game_id=game_id,
                    agents=agents,
                    summary={"cumulative_payoffs": [100.0, 200.0]},
                    config=None
                )
            
            # Get all player stats
            stats = db_manager.get_player_stats()
            assert len(stats) == 2  # Two players
            
            db_manager.close()

    def test_custom_query(self):
        """Test executing custom queries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db.db")
            db_manager = DatabaseManager(db_path=db_path, enabled=True)
            
            # Save some data
            agents = [Mock(player_id=0, persona="selfish", llm_model="gpt-3.5-turbo")]
            db_manager.save_game_results(
                game_id="test_game",
                agents=agents,
                summary={"cumulative_payoffs": [100.0]},
                config=None
            )
            
            # Custom query
            results = db_manager.query_results(
                "SELECT COUNT(*) FROM game_results WHERE total_reward > ?",
                [50.0]
            )
            assert len(results) == 1
            assert results[0][0] == 1
            
            db_manager.close()


class TestContextManager:
    """Test context manager functionality."""

    def test_context_manager(self):
        """Test using database manager as context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db.db")
            
            with DatabaseManager(db_path=db_path, enabled=True) as db_manager:
                agents = [Mock(player_id=0, persona="test", llm_model="gpt-3.5-turbo")]
                db_manager.save_game_results(
                    game_id="test_game",
                    agents=agents,
                    summary={"cumulative_payoffs": [100.0]},
                    config=None
                )
                
                results = db_manager.get_all_results()
                assert len(results) == 1
            
            # Connection should be closed after context exit
            assert db_manager.conn is None


class TestIntegrationWithGameRunner:
    """Test integration with GameRunner."""

    def test_game_runner_saves_to_database(self):
        """Test that GameRunner automatically saves to database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db.db")
            
            # Create config with database path (use CONFIG as base)
            config = CONFIG.copy()
            config.update({
                "n_players": 2,
                "max_steps": 5,
                "initial_resource": 100,
                "regeneration_rate": 2.0,
                "min_resource": 2,
                "min_extraction": 0,
                "max_extraction": 50,
                "include_history_rounds": 3,
                "player_personas": ["rational_selfish", "cooperative"],
                "langfuse_public_key": "test_key",
                "langfuse_secret_key": "test_secret",
                "db_path": db_path,
                "db_enabled": True,
            })
            
            # Run game
            runner = GameRunner(config=config, use_mock_agents=True)
            runner.setup_game("test_game")
            summary = runner.run_episode(visualize=False, verbose=False)
            
            # Check database was created and has data
            assert os.path.exists(db_path)
            
            # Verify results were saved
            results = runner.db_manager.get_all_results()
            assert len(results) == 2  # 2 players
            
            # Verify data matches
            assert results[0][0] == "test_game"  # game_id
            assert results[0][1] == 0  # player_id
            assert results[0][4] == summary["cumulative_payoffs"][0]  # total_reward
            
            runner.db_manager.close()

    def test_game_runner_with_disabled_database(self):
        """Test that GameRunner works when database is disabled."""
        config = CONFIG.copy()
        config.update({
            "n_players": 2,
            "max_steps": 5,
            "initial_resource": 100,
            "regeneration_rate": 2.0,
            "min_resource": 2,
            "min_extraction": 0,
            "max_extraction": 50,
            "include_history_rounds": 3,
            "player_personas": ["rational_selfish", "cooperative"],
            "langfuse_public_key": "test_key",
            "langfuse_secret_key": "test_secret",
            "db_enabled": False,
        })
        
        # Should not raise error
        runner = GameRunner(config=config, use_mock_agents=True)
        runner.setup_game("test_game")
        summary = runner.run_episode(visualize=False, verbose=False)
        
        # Database should be disabled
        assert runner.db_manager is None or runner.db_manager.enabled is False
        
        # Game should still complete successfully
        assert "cumulative_payoffs" in summary

