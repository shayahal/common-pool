"""Database manager for storing game results in DuckDB.

Provides persistent storage for game run data including player results,
personas, models, and rewards.
"""

from typing import Dict, List, Optional
import duckdb
from pathlib import Path
import json
from datetime import datetime
from .logger_setup import get_logger

logger = get_logger(__name__)


class DatabaseManager:
    """Manages DuckDB database for storing game results.
    
    Creates and maintains a database table to store per-player game results
    including game_id, player_id, persona, model, and total_reward.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        enabled: bool = True
    ):
        """Initialize database manager.
        
        Args:
            db_path: Path to DuckDB database file. If None, uses default:
                     data/game_results.duckdb
            enabled: Whether database saving is enabled (default: True)
        """
        self.enabled = enabled
        
        if not self.enabled:
            logger.info("Database manager is disabled")
            self.conn = None
            return
        
        # Set default database path
        if db_path is None:
            db_path = "data/game_results.duckdb"
        
        # Ensure parent directory exists
        db_file = Path(db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Connect to DuckDB
        try:
            self.conn = duckdb.connect(db_path)
            self._create_table()
            logger.info(f"Database manager initialized: {db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}", exc_info=True)
            self.conn = None
            self.enabled = False

    def _create_table(self):
        """Create game_results table if it doesn't exist."""
        if not self.enabled or self.conn is None:
            return
        
        try:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS game_results (
                    game_id TEXT NOT NULL,
                    player_id INTEGER NOT NULL,
                    persona TEXT NOT NULL,
                    model TEXT NOT NULL,
                    total_reward DOUBLE NOT NULL,
                    experiment_id TEXT,
                    timestamp TIMESTAMP,
                    PRIMARY KEY (game_id, player_id)
                )
            """)
            
            # Add new columns if they don't exist (for migration from old schema)
            try:
                self.conn.execute("ALTER TABLE game_results ADD COLUMN IF NOT EXISTS experiment_id TEXT")
            except Exception:
                pass  # Column might already exist or error is expected
            
            try:
                self.conn.execute("ALTER TABLE game_results ADD COLUMN IF NOT EXISTS timestamp TIMESTAMP")
            except Exception:
                pass  # Column might already exist or error is expected
            
            # Create experiment tables
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    parameters TEXT NOT NULL
                )
            """)
            
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS experiment_players (
                    experiment_id TEXT NOT NULL,
                    player_index INTEGER NOT NULL,
                    persona TEXT NOT NULL,
                    model TEXT NOT NULL,
                    PRIMARY KEY (experiment_id, player_index),
                    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
                )
            """)
            
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS experiment_results (
                    experiment_id TEXT NOT NULL,
                    game_id TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    PRIMARY KEY (experiment_id, game_id),
                    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
                )
            """)
            
            logger.debug("Game results and experiment tables created or already exist")
        except Exception as e:
            logger.error(f"Failed to create table: {e}", exc_info=True)
            raise

    def save_game_results(
        self,
        game_id: str,
        agents: List,
        summary: Dict,
        config: Optional[Dict] = None,
        experiment_id: Optional[str] = None,
        timestamp: Optional[str] = None
    ):
        """Save game results to database.
        
        Args:
            game_id: Unique identifier for the game
            agents: List of agent objects with player_id, persona, llm_model
            summary: Game summary dict with 'cumulative_payoffs' list
            config: Optional config dict to get model if not in agent
            experiment_id: Optional experiment identifier
            timestamp: Optional timestamp (ISO format string or will use current time)
        """
        if not self.enabled or self.conn is None:
            return
        
        try:
            cumulative_payoffs = summary.get("cumulative_payoffs", [])
            
            if not cumulative_payoffs:
                logger.warning(f"No cumulative_payoffs in summary for game {game_id}")
                return
            
            # Prepare data for insertion
            rows = []
            for i, agent in enumerate(agents):
                if i >= len(cumulative_payoffs):
                    logger.warning(
                        f"Player {i} has no reward data in summary for game {game_id}"
                    )
                    continue
                
                player_id = getattr(agent, 'player_id', i)
                persona = getattr(agent, 'persona', 'unknown')
                
                # Get model from agent or config
                model = getattr(agent, 'llm_model', None)
                if model is None and config:
                    model = config.get('llm_model', 'unknown')
                elif model is None:
                    model = 'unknown'
                
                total_reward = float(cumulative_payoffs[i])
                
                # Use provided timestamp or current time
                if timestamp is None:
                    from datetime import datetime
                    timestamp = datetime.now().isoformat()
                
                rows.append((game_id, player_id, persona, model, total_reward, experiment_id, timestamp))
            
            if not rows:
                logger.warning(f"No valid player data to save for game {game_id}")
                return
            
            # Insert data (using ON CONFLICT for idempotency)
            self.conn.executemany("""
                INSERT INTO game_results 
                (game_id, player_id, persona, model, total_reward, experiment_id, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (game_id, player_id) 
                DO UPDATE SET 
                    persona = EXCLUDED.persona,
                    model = EXCLUDED.model,
                    total_reward = EXCLUDED.total_reward,
                    experiment_id = EXCLUDED.experiment_id,
                    timestamp = EXCLUDED.timestamp
            """, rows)
            
            self.conn.commit()
            logger.info(f"Saved {len(rows)} player results for game {game_id}")
            
            # Verify the save was successful
            verify_count = self.conn.execute(
                "SELECT COUNT(*) FROM game_results WHERE game_id = ?",
                [game_id]
            ).fetchone()[0]
            
            if verify_count != len(rows):
                logger.warning(
                    f"Save verification failed for game {game_id}: "
                    f"expected {len(rows)} rows, found {verify_count}"
                )
            
        except Exception as e:
            logger.error(
                f"Failed to save game results for {game_id}: {e}",
                exc_info=True
            )
            # Don't raise - database errors shouldn't crash the game

    def query_results(
        self,
        query: str,
        parameters: Optional[list] = None
    ) -> List[tuple]:
        """Execute a query and return results.
        
        Args:
            query: SQL query string
            parameters: Optional query parameters (list or tuple)
            
        Returns:
            List of result tuples
        """
        if not self.enabled or self.conn is None:
            logger.warning("Database is not enabled or not initialized")
            return []
        
        try:
            if parameters:
                result = self.conn.execute(query, parameters)
            else:
                result = self.conn.execute(query)
            return result.fetchall()
        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)
            return []

    def get_all_results(self) -> List[tuple]:
        """Get all game results.
        
        Returns:
            List of tuples (game_id, player_id, persona, model, total_reward)
        """
        return self.query_results("SELECT * FROM game_results ORDER BY game_id, player_id")

    def verify_game_saved(self, game_id: str) -> bool:
        """Verify that a game was saved to the database.
        
        Args:
            game_id: Game ID to check
            
        Returns:
            True if game exists in database, False otherwise
        """
        if not self.enabled or self.conn is None:
            return False
        
        try:
            result = self.conn.execute(
                "SELECT COUNT(*) FROM game_results WHERE game_id = ?",
                [game_id]
            ).fetchone()
            return result is not None and result[0] > 0
        except Exception as e:
            logger.error(f"Failed to verify game save: {e}", exc_info=True)
            return False

    def get_player_stats(self, player_id: Optional[int] = None) -> List[tuple]:
        """Get statistics for a specific player or all players.
        
        Args:
            player_id: Optional player ID to filter by
            
        Returns:
            List of tuples with player statistics
        """
        if player_id is not None:
            query = """
                SELECT 
                    player_id,
                    persona,
                    model,
                    COUNT(*) as game_count,
                    AVG(total_reward) as avg_reward,
                    MIN(total_reward) as min_reward,
                    MAX(total_reward) as max_reward,
                    SUM(total_reward) as total_reward_sum
                FROM game_results
                WHERE player_id = ?
                GROUP BY player_id, persona, model
            """
            return self.query_results(query, [player_id])
        else:
            query = """
                SELECT 
                    player_id,
                    persona,
                    model,
                    COUNT(*) as game_count,
                    AVG(total_reward) as avg_reward,
                    MIN(total_reward) as min_reward,
                    MAX(total_reward) as max_reward,
                    SUM(total_reward) as total_reward_sum
                FROM game_results
                GROUP BY player_id, persona, model
                ORDER BY player_id
            """
            return self.query_results(query)

    def close(self):
        """Close database connection."""
        if self.conn is not None:
            try:
                self.conn.close()
                logger.info("Database connection closed")
            except Exception as e:
                logger.error(f"Error closing database: {e}", exc_info=True)
            finally:
                self.conn = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    # ============================================================================
    # Experiment Management Methods
    # ============================================================================

    def save_experiment(
        self,
        experiment_id: str,
        name: str,
        players: List[Dict[str, str]],
        parameters: Dict
    ) -> bool:
        """Save an experiment definition to the database.
        
        Args:
            experiment_id: Unique identifier for the experiment
            name: Human-readable experiment name
            players: List of player dicts with 'persona' and 'model' keys
            parameters: Dictionary of experiment parameters
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or self.conn is None:
            logger.warning("Database is not enabled or not initialized")
            return False
        
        try:
            # Convert parameters to JSON string
            parameters_json = json.dumps(parameters)
            
            # Insert or update experiment
            self.conn.execute("""
                INSERT INTO experiments (experiment_id, name, status, created_at, parameters)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (experiment_id) 
                DO UPDATE SET 
                    name = EXCLUDED.name,
                    status = EXCLUDED.status,
                    parameters = EXCLUDED.parameters
            """, (experiment_id, name, "pending", datetime.now(), parameters_json))
            
            # Delete existing players for this experiment (for updates)
            self.conn.execute(
                "DELETE FROM experiment_players WHERE experiment_id = ?",
                [experiment_id]
            )
            
            # Insert players
            player_rows = [
                (experiment_id, i, player["persona"], player["model"])
                for i, player in enumerate(players)
            ]
            
            if player_rows:
                self.conn.executemany("""
                    INSERT INTO experiment_players (experiment_id, player_index, persona, model)
                    VALUES (?, ?, ?, ?)
                """, player_rows)
            
            self.conn.commit()
            logger.info(f"Saved experiment {experiment_id} with {len(players)} players")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save experiment {experiment_id}: {e}", exc_info=True)
            return False

    def load_experiment(self, experiment_id: str) -> Optional[Dict]:
        """Load an experiment definition from the database.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            Dictionary with experiment data or None if not found
        """
        if not self.enabled or self.conn is None:
            logger.warning("Database is not enabled or not initialized")
            return None
        
        try:
            # Load experiment
            exp_result = self.conn.execute(
                "SELECT name, status, created_at, parameters FROM experiments WHERE experiment_id = ?",
                [experiment_id]
            ).fetchone()
            
            if exp_result is None:
                return None
            
            name, status, created_at, parameters_json = exp_result
            parameters = json.loads(parameters_json)
            
            # Load players
            player_results = self.conn.execute(
                """SELECT player_index, persona, model 
                   FROM experiment_players 
                   WHERE experiment_id = ? 
                   ORDER BY player_index""",
                [experiment_id]
            ).fetchall()
            
            players = [
                {"persona": persona, "model": model}
                for _, persona, model in player_results
            ]
            
            return {
                "experiment_id": experiment_id,
                "name": name,
                "status": status,
                "created_at": created_at,
                "players": players,
                "parameters": parameters
            }
            
        except Exception as e:
            logger.error(f"Failed to load experiment {experiment_id}: {e}", exc_info=True)
            return None

    def list_experiments(self) -> List[Dict]:
        """List all experiments in the database.
        
        Returns:
            List of experiment dictionaries with basic info
        """
        if not self.enabled or self.conn is None:
            logger.warning("Database is not enabled or not initialized")
            return []
        
        try:
            results = self.conn.execute("""
                SELECT 
                    e.experiment_id,
                    e.name,
                    e.status,
                    e.created_at,
                    COUNT(DISTINCT ep.player_index) as player_count,
                    COUNT(DISTINCT er.game_id) as game_count
                FROM experiments e
                LEFT JOIN experiment_players ep ON e.experiment_id = ep.experiment_id
                LEFT JOIN experiment_results er ON e.experiment_id = er.experiment_id
                GROUP BY e.experiment_id, e.name, e.status, e.created_at
                ORDER BY e.created_at DESC
            """).fetchall()
            
            experiments = []
            for row in results:
                experiments.append({
                    "experiment_id": row[0],
                    "name": row[1],
                    "status": row[2],
                    "created_at": row[3],
                    "player_count": row[4] or 0,
                    "game_count": row[5] or 0
                })
            
            return experiments
            
        except Exception as e:
            logger.error(f"Failed to list experiments: {e}", exc_info=True)
            return []

    def update_experiment_status(self, experiment_id: str, status: str) -> bool:
        """Update the status of an experiment.
        
        Args:
            experiment_id: Experiment identifier
            status: New status (pending, running, completed, failed)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or self.conn is None:
            logger.warning("Database is not enabled or not initialized")
            return False
        
        try:
            self.conn.execute(
                "UPDATE experiments SET status = ? WHERE experiment_id = ?",
                [status, experiment_id]
            )
            self.conn.commit()
            logger.info(f"Updated experiment {experiment_id} status to {status}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update experiment status: {e}", exc_info=True)
            return False

    def save_experiment_result(
        self,
        experiment_id: str,
        game_id: str,
        summary: Dict
    ) -> bool:
        """Save a game result for an experiment.
        
        Args:
            experiment_id: Experiment identifier
            game_id: Game identifier
            summary: Game summary dictionary
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or self.conn is None:
            logger.warning("Database is not enabled or not initialized")
            return False
        
        try:
            summary_json = json.dumps(summary, default=str)
            timestamp = datetime.now()
            
            self.conn.execute("""
                INSERT INTO experiment_results (experiment_id, game_id, summary, timestamp)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (experiment_id, game_id) 
                DO UPDATE SET 
                    summary = EXCLUDED.summary,
                    timestamp = EXCLUDED.timestamp
            """, (experiment_id, game_id, summary_json, timestamp))
            
            self.conn.commit()
            logger.debug(f"Saved result for game {game_id} in experiment {experiment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save experiment result: {e}", exc_info=True)
            return False

    def get_experiment_results(self, experiment_id: str) -> List[Dict]:
        """Get all game results for an experiment.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            List of game result dictionaries
        """
        if not self.enabled or self.conn is None:
            logger.warning("Database is not enabled or not initialized")
            return []
        
        try:
            results = self.conn.execute(
                "SELECT game_id, summary, timestamp FROM experiment_results WHERE experiment_id = ? ORDER BY timestamp",
                [experiment_id]
            ).fetchall()
            
            game_results = []
            for game_id, summary_json, timestamp in results:
                try:
                    summary = json.loads(summary_json)
                    game_results.append({
                        "game_id": game_id,
                        "summary": summary,
                        "timestamp": timestamp
                    })
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse summary for game {game_id}: {e}")
                    continue
            
            return game_results
            
        except Exception as e:
            logger.error(f"Failed to get experiment results: {e}", exc_info=True)
            return []

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment and all associated data.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or self.conn is None:
            logger.warning("Database is not enabled or not initialized")
            return False
        
        try:
            # Manual cascade delete (DuckDB doesn't support ON DELETE CASCADE)
            # Delete in order: results -> players -> experiment
            self.conn.execute(
                "DELETE FROM experiment_results WHERE experiment_id = ?",
                [experiment_id]
            )
            self.conn.execute(
                "DELETE FROM experiment_players WHERE experiment_id = ?",
                [experiment_id]
            )
            self.conn.execute(
                "DELETE FROM experiments WHERE experiment_id = ?",
                [experiment_id]
            )
            self.conn.commit()
            logger.info(f"Deleted experiment {experiment_id} and all associated data")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete experiment: {e}", exc_info=True)
            return False

