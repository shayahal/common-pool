"""Test that FULLY replicates app.py execution to catch failures."""

import pytest
from unittest.mock import patch, MagicMock, Mock
import numpy as np
from datetime import datetime
import uuid

# Mock Streamlit completely
@pytest.fixture(autouse=True)
def mock_streamlit():
    """Mock all Streamlit functionality to match real app execution."""
    mock_st = MagicMock()
    
    # Create a proper session state mock
    mock_session_state = MagicMock()
    mock_session_state.all_runs = []
    mock_session_state.dashboard_run_history = []
    
    # Make session_state behave like a dict
    def getitem(self, key):
        if key == "all_runs":
            return mock_session_state.all_runs
        elif key == "dashboard_run_history":
            return mock_session_state.dashboard_run_history
        elif hasattr(mock_session_state, key):
            return getattr(mock_session_state, key)
        raise KeyError(f"Key {key} not found")
    
    def setitem(self, key, value):
        if key == "all_runs":
            mock_session_state.all_runs = value
        elif key == "dashboard_run_history":
            mock_session_state.dashboard_run_history = value
        else:
            setattr(mock_session_state, key, value)
    
    def contains(self, key):
        return key in ["all_runs", "dashboard_run_history"] or hasattr(mock_session_state, key)
    
    mock_session_state.__getitem__ = getitem
    mock_session_state.__setitem__ = setitem
    mock_session_state.__contains__ = contains
    
    # Mock all Streamlit functions
    mock_st.session_state = mock_session_state
    mock_st.empty.return_value = MagicMock()
    
    # columns() should return a list based on the number requested
    def mock_columns(n):
        return [MagicMock() for _ in range(n)]
    mock_st.columns = mock_columns
    
    # tabs() should return a list based on the number of tab labels
    def mock_tabs(labels):
        return [MagicMock() for _ in labels]
    mock_st.tabs = mock_tabs
    mock_st.metric.return_value = None
    mock_st.dataframe.return_value = None
    mock_st.plotly_chart.return_value = None
    mock_st.info.return_value = None
    mock_st.write.return_value = None
    mock_st.markdown.return_value = None
    mock_st.button.return_value = True  # "Run Game" button pressed
    mock_st.slider.return_value = 1.0
    mock_st.selectbox.return_value = "rational_selfish"
    mock_st.checkbox.return_value = False
    mock_st.progress.return_value = MagicMock()
    mock_st.set_page_config.return_value = None
    mock_st.title.return_value = None
    mock_st.divider.return_value = None
    mock_st.error.return_value = None
    mock_st.stop.return_value = None
    mock_st.rerun.return_value = None
    mock_st.exception.return_value = None
    
    # Patch streamlit everywhere - need to patch where it's imported
    import sys
    sys.modules['streamlit'] = mock_st
    
    # Also patch in dashboard module
    with patch('cpr_game.dashboard.st', mock_st):
        with patch('cpr_game.dashboard.st.session_state', mock_session_state):
            yield mock_st


class TestAppFullExecution:
    """Test the FULL execution flow from app.py exactly as it happens."""

    @patch('cpr_game.logging_manager.Langfuse')
    def test_app_py_full_run_game_flow(self, mock_langfuse_class):
        """Test the EXACT flow from app.py when Run Game button is pressed."""
        # Setup mock Langfuse client
        mock_client = MagicMock()
        mock_client.start_as_current_observation.return_value = MagicMock()
        mock_client.get_current_trace_id.return_value = "trace_123"
        mock_client.start_generation.return_value = MagicMock()
        mock_client.score_current_trace.return_value = MagicMock()
        mock_client.update_current_trace.return_value = MagicMock()
        mock_client.flush.return_value = None
        mock_langfuse_class.return_value = mock_client
        
        # Import after mocking
        from cpr_game.config import CONFIG
        from cpr_game.game_runner import GameRunner
        import streamlit as st
        
        # STEP 1: Initialize session state (exactly as app.py does)
        if "all_runs" not in st.session_state:
            st.session_state.all_runs = []
        
        if "dashboard_run_history" not in st.session_state:
            st.session_state.dashboard_run_history = []
        
        # STEP 2: Get configuration values (as app.py does)
        n_players = 2
        max_steps = 5
        initial_resource = 1000
        regeneration_rate = 2.0
        sustainability_threshold = 500.0
        max_fishes = 100
        personas = ["rational_selfish", "cooperative"]
        use_mock_agents = True
        auto_refresh = False
        refresh_delay = 1.0
        
        # STEP 3: Create config (exactly as app.py line 180-187)
        config = CONFIG.copy()
        config["n_players"] = n_players
        config["max_steps"] = max_steps
        config["initial_resource"] = initial_resource
        config["regeneration_rate"] = regeneration_rate
        config["sustainability_threshold"] = sustainability_threshold
        config["max_fishes"] = max_fishes
        config["player_personas"] = personas
        
        # Ensure Langfuse keys are present
        if not config.get("langfuse_public_key") or not config.get("langfuse_secret_key"):
            config["langfuse_public_key"] = "test_public_key"
            config["langfuse_secret_key"] = "test_secret_key"
        
        # STEP 4: Initialize game runner (exactly as app.py line 189-195)
        try:
            runner = GameRunner(
                config=config,
                use_mock_agents=use_mock_agents
            )
            
            game_id = runner.setup_game()
        except ValueError as e:
            pytest.fail(f"Configuration error: {e}")
        except Exception as e:
            pytest.fail(f"Error initializing game: {e}")
        
        # STEP 5: Create unique run ID (exactly as app.py line 209)
        run_id = str(uuid.uuid4())[:8]
        
        # STEP 6: Initialize dashboard (exactly as app.py line 211-221)
        try:
            dashboard = runner.dashboard
            if dashboard:
                # Initialize reasoning log
                for i in range(n_players):
                    if i not in dashboard.reasoning_log:
                        dashboard.reasoning_log[i] = []
                
                # Create new run history for this game
                dashboard.run_history = []
        except Exception as e:
            pytest.fail(f"Error initializing dashboard: {e}")
        
        # STEP 7: Start game trace (exactly as app.py line 224)
        try:
            runner.logger.start_game_trace(game_id, config)
        except Exception as e:
            pytest.fail(f"Error starting game trace: {e}")
        
        # STEP 8: Reset environment and agents (exactly as app.py line 227-229)
        try:
            observations, info = runner.env.reset()
            for agent in runner.agents:
                agent.reset()
        except Exception as e:
            pytest.fail(f"Error resetting environment/agents: {e}")
        
        # STEP 9: Initialize game loop variables (exactly as app.py line 232-238)
        done = False
        step = 0
        resource_history = [info["resource"]]
        extraction_history = []
        payoff_history = []
        cooperation_history = []
        
        # STEP 10: Progress bar (exactly as app.py line 241-242)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # STEP 11: Run at least one step of the game loop (exactly as app.py line 244-310)
        max_iterations = 3  # Limit iterations for test
        iteration = 0
        
        while not done and iteration < max_iterations:
            iteration += 1
            
            # Get actions from all agents
            actions = []
            reasonings = []
            
            for i, agent in enumerate(runner.agents):
                obs = observations[f"player_{i}"]
                action, reasoning = agent.act(obs, return_reasoning=True)
                actions.append(action)
                reasonings.append(reasoning)
                
                # Log generation (exactly as app.py line 253-262)
                prompt = agent._build_prompt(obs) if hasattr(agent, '_build_prompt') else ""
                runner.logger.log_generation(
                    player_id=i,
                    prompt=prompt,
                    response=reasoning or "",
                    action=action,
                    reasoning=reasoning
                )
                
                # Add to dashboard (exactly as app.py line 264-266)
                if dashboard and reasoning:
                    dashboard.add_reasoning(i, reasoning)
            
            # Execute step (exactly as app.py line 269)
            actions_array = np.array(actions)
            observations, rewards, terminated, truncated, info = runner.env.step(actions_array)
            done = terminated or truncated
            
            # Update agent memories (exactly as app.py line 272-274)
            for i, agent in enumerate(runner.agents):
                obs = observations[f"player_{i}"]
                agent.update_memory(obs, actions[i], rewards[i])
            
            # Collect history (exactly as app.py line 277-280)
            resource_history.append(info["resource"])
            extraction_history.append(actions)
            payoff_history.append(rewards.tolist())
            cooperation_history.append(info.get("cooperation_index", 0.0))
            
            # Log round metrics (exactly as app.py line 283-291)
            round_metrics = {
                "resource_level": info["resource"],
                "total_extraction": info["total_extraction"],
                "cooperation_index": info.get("cooperation_index", 0.0),
                "individual_extractions": actions,
                "individual_payoffs": rewards.tolist(),
            }
            runner.logger.log_round_metrics(step, round_metrics)
            
            # Update dashboard (exactly as app.py line 294-305)
            if dashboard:
                game_state = {
                    "resource": info["resource"],
                    "step": step,
                    "max_steps": max_steps,
                    "done": done,
                    "cumulative_payoffs": info.get("cumulative_payoffs", [sum(payoff_history[j][i] for j in range(len(payoff_history))) for i in range(n_players)]),
                    "resource_history": resource_history,
                    "extraction_history": extraction_history,
                    "payoff_history": payoff_history,
                    "cooperation_history": cooperation_history,
                }
                dashboard.update(game_state)
            
            # Update progress (exactly as app.py line 308-310)
            progress = (step + 1) / max_steps
            progress_bar.progress(progress)
            status_text.text(f"Round {step + 1}/{max_steps} - Resource: {info['resource']:.1f}")
            
            step += 1
        
        # STEP 12: Get summary statistics (exactly as app.py line 313)
        summary = runner.env.get_summary_stats()
        
        # STEP 13: End logging trace (exactly as app.py line 316)
        runner.logger.end_game_trace(summary)
        
        # STEP 14: Clear progress indicators (exactly as app.py line 319-320)
        progress_bar.empty()
        status_text.empty()
        
        # STEP 15: Prepare final game state (exactly as app.py line 323-332)
        final_game_state = {
            "resource": info["resource"],
            "step": step,
            "max_steps": max_steps,
            "done": True,
            "cumulative_payoffs": info.get("cumulative_payoffs", []),
            "resource_history": resource_history,
            "extraction_history": extraction_history,
            "payoff_history": payoff_history,
            "cooperation_history": cooperation_history,
        }
        
        # STEP 16: Update dashboard with final state (exactly as app.py line 335-336)
        if dashboard:
            dashboard.update(final_game_state)
        
        # STEP 17: Store complete run data (exactly as app.py line 339-365)
        run_data = {
            "run_id": run_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config": config.copy(),
            "game_id": game_id,
            "summary": summary,
            "resource_history": resource_history,
            "extraction_history": extraction_history,
            "payoff_history": payoff_history,
            "cooperation_history": cooperation_history,
            "reasoning_log": dashboard.reasoning_log.copy() if dashboard else {},
            "run_history": dashboard.run_history.copy() if dashboard else [],
            "generation_data": runner.logger.get_generation_data() if hasattr(runner.logger, 'get_generation_data') else [],
            "round_metrics": runner.logger.get_round_metrics() if hasattr(runner.logger, 'get_round_metrics') else [],
        }
        
        # STEP 18: Add to session state (exactly as app.py line 368)
        st.session_state.all_runs.append(run_data)
        
        # If we get here without exceptions, the test passes
        assert game_id is not None
        assert runner.logger is not None
        assert len(st.session_state.all_runs) > 0

