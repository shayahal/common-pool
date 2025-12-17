"""Streamlit GUI application for CPR Game.

Run with: streamlit run app.py
"""

import streamlit as st
import sys
from pathlib import Path
import numpy as np
from datetime import datetime
import uuid
import pandas as pd
import hashlib
import json
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Setup logs directory
LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)
DEBUG_LOG_PATH = LOGS_DIR / "debug.log"

from cpr_game.game_runner import GameRunner
from cpr_game.config import CONFIG
import time


def _run_game_in_tab(config, use_mock_agents):
    """Execute a game run within the current Streamlit context (for live tab rendering)."""
    try:
        # Initialize game runner
        runner = GameRunner(
            config=config,
            use_mock_agents=use_mock_agents
        )

        game_id = runner.setup_game()
    except ValueError as e:
        # Handle configuration errors
        error_msg = str(e)
        st.error(f"❌ Configuration Error: {error_msg}")
        st.info("💡 **Tip**: Make sure you have set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY environment variables.")
        return None
    except Exception as e:
        st.error(f"❌ Error initializing game: {str(e)}")
        st.exception(e)
        return None

    # Create unique run ID
    run_id = str(uuid.uuid4())[:8]
    n_players = config["n_players"]
    max_steps = config["max_steps"]

    try:
        # Initialize dashboard
        dashboard = runner.dashboard
        if dashboard:
            for i in range(n_players):
                if i not in dashboard.reasoning_log:
                    dashboard.reasoning_log[i] = []
            dashboard.run_history = []

        # Start game trace
        runner.logger.start_game_trace(game_id, config)

        # Reset environment and agents
        observations, info = runner.env.reset()
        for agent in runner.agents:
            agent.reset()

        # Main game loop
        done = False
        step = 0
        resource_history = [info["resource"]]
        extraction_history = []
        payoff_history = []
        cooperation_history = []

        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Container for reasoning table
        reasoning_container = st.container()

        while not done:
            # Set current round for logging
            runner.logger.set_current_round(step)

            # Get actions from all agents
            actions = []
            reasonings = []

            for i, agent in enumerate(runner.agents):
                obs = observations[f"player_{i}"]
                action, reasoning = agent.act(obs, return_reasoning=True)
                actions.append(action)
                reasonings.append(reasoning)

                # Get API metrics if available
                api_metrics = None
                if hasattr(agent, 'get_last_api_metrics'):
                    api_metrics = agent.get_last_api_metrics()

                # Log generation
                prompt = agent._build_prompt(obs) if hasattr(agent, '_build_prompt') else ""
                runner.logger.log_generation(
                    player_id=i,
                    prompt=prompt,
                    response=reasoning or "",
                    action=action,
                    reasoning=reasoning,
                    api_metrics=api_metrics
                )

                # Add to dashboard
                if dashboard:
                    game_state_context = {
                        "resource_level": obs.get("resource_level", [0])[0] if isinstance(obs.get("resource_level"), np.ndarray) else obs.get("resource_level", 0),
                        "step": obs.get("step", [0])[0] if isinstance(obs.get("step"), np.ndarray) else obs.get("step", step),
                    }
                    if "my_cumulative_payoff" in obs:
                        my_payoff = obs["my_cumulative_payoff"]
                        if isinstance(my_payoff, np.ndarray):
                            game_state_context["my_cumulative_payoff"] = my_payoff[0] if len(my_payoff) > 0 else 0
                        else:
                            game_state_context["my_cumulative_payoff"] = my_payoff

                    dashboard.add_reasoning(
                        player_id=i,
                        reasoning=reasoning or "",
                        prompt=prompt if prompt else None,
                        action=float(action) if action is not None else None,
                        game_state=game_state_context,
                        round_num=step + 1
                    )

            # Execute step
            actions_array = np.array(actions)
            observations, rewards, terminated, truncated, info = runner.env.step(actions_array)
            done = terminated or truncated

            # Update agent memories
            for i, agent in enumerate(runner.agents):
                obs = observations[f"player_{i}"]
                agent.update_memory(obs, actions[i], rewards[i])

            # Collect history
            resource_history.append(info["resource"])
            extraction_history.append(actions)
            payoff_history.append(rewards.tolist())
            cooperation_history.append(info.get("cooperation_index", 0.0))

            # Log round metrics
            round_metrics = {
                "resource_level": info["resource"],
                "total_extraction": info["total_extraction"],
                "cooperation_index": info.get("cooperation_index", 0.0),
                "individual_extractions": actions,
                "individual_payoffs": rewards.tolist(),
            }
            runner.logger.log_round_metrics(step, round_metrics)

            # Update progress
            progress = (step + 1) / max_steps
            progress_bar.progress(progress)
            status_text.text(f"Round {step + 1}/{max_steps} - Resource: {int(info['resource'])}")

            # Render reasoning table inline
            with reasoning_container:
                if dashboard:
                    dashboard._render_reasoning_table()

            step += 1

        # Get summary
        summary = runner.env.get_summary_stats()

        # End logging trace
        runner.logger.end_game_trace(summary)

        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()

        # Get API metrics
        api_metrics_data = runner.logger.get_api_metrics_data() if hasattr(runner.logger, 'get_api_metrics_data') else []

        # Add API metrics to dashboard
        if dashboard and api_metrics_data:
            dashboard.add_api_metrics(api_metrics_data)

        # Prepare run data
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
            "api_metrics_data": api_metrics_data,
        }

        return run_data

    except Exception as e:
        st.error(f"❌ Error during game execution: {str(e)}")
        st.exception(e)
        return None


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="CPR Game Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🌳 Common Pool Resource Game")
    st.markdown("Interactive dashboard for running and visualizing CPR games")

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Game Configuration")
        
        # Basic settings - dynamically extract defaults from CONFIG
        # #region agent log
        with open(DEBUG_LOG_PATH, "a") as f:
            n_players_val = CONFIG["n_players"]
            max_steps_val = CONFIG["max_steps"]
            f.write(json.dumps({"sessionId": "debug-session", "runId": "type-check", "hypothesisId": "B", "location": "app.py:38", "message": "slider type checks", "data": {"n_players": {"value": n_players_val, "type": type(n_players_val).__name__}, "max_steps": {"value": max_steps_val, "type": type(max_steps_val).__name__}}, "timestamp": int(time.time() * 1000)}) + "\n")
        # #endregion
        try:
            n_players = st.slider(
                "Number of Players", 
                min_value=2, 
                max_value=10, 
                value=CONFIG["n_players"]
            )
        except (TypeError, ValueError) as e:
            with open(DEBUG_LOG_PATH, "a") as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "error-catch", "hypothesisId": "D", "location": "app.py:46", "message": "ERROR in n_players slider", "data": {"error": str(e), "value": CONFIG["n_players"], "value_type": type(CONFIG["n_players"]).__name__}, "timestamp": int(time.time() * 1000)}) + "\n")
            raise
        try:
            max_steps = st.slider(
                "Max Steps", 
                min_value=10, 
                max_value=500, 
                value=CONFIG["max_steps"]
            )
        except (TypeError, ValueError) as e:
            with open(DEBUG_LOG_PATH, "a") as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "error-catch", "hypothesisId": "D", "location": "app.py:52", "message": "ERROR in max_steps slider", "data": {"error": str(e), "value": CONFIG["max_steps"], "value_type": type(CONFIG["max_steps"]).__name__}, "timestamp": int(time.time() * 1000)}) + "\n")
            raise
        initial_resource = st.number_input(
            "Initial Resource", 
            min_value=1, 
            max_value=10000, 
            value=CONFIG["initial_resource"], 
            step=10
        )
        # #region agent log
        try:
            regen_rate_val = CONFIG["regeneration_rate"]
            regen_rate_float = float(regen_rate_val)
            with open(DEBUG_LOG_PATH, "a") as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "post-fix", "hypothesisId": "A", "location": "app.py:81", "message": "regeneration_rate conversion", "data": {"original": regen_rate_val, "original_type": type(regen_rate_val).__name__, "converted": regen_rate_float, "converted_type": type(regen_rate_float).__name__}, "timestamp": int(time.time() * 1000)}) + "\n")
        except (TypeError, ValueError) as e:
            with open(DEBUG_LOG_PATH, "a") as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "error", "hypothesisId": "A", "location": "app.py:81", "message": "ERROR converting regeneration_rate", "data": {"error": str(e)}, "timestamp": int(time.time() * 1000)}) + "\n")
        # #endregion
        try:
            regeneration_rate = st.slider(
                "Regeneration Rate", 
                min_value=1.0, 
                max_value=5.0, 
                value=float(CONFIG["regeneration_rate"]),  # Convert to float to match min/max types
                step=0.1
            )
        except (TypeError, ValueError) as e:
            with open(DEBUG_LOG_PATH, "a") as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "error-catch", "hypothesisId": "A", "location": "app.py:95", "message": "ERROR in regeneration_rate slider", "data": {"error": str(e), "value": CONFIG["regeneration_rate"], "value_type": type(CONFIG["regeneration_rate"]).__name__, "converted_value": float(CONFIG["regeneration_rate"]), "converted_type": type(float(CONFIG["regeneration_rate"])).__name__}, "timestamp": int(time.time() * 1000)}) + "\n")
            raise
        sustainability_threshold = st.number_input(
            "Sustainability Threshold", 
            min_value=1, 
            max_value=10000, 
            value=CONFIG.get("sustainability_threshold", CONFIG["n_players"]), 
            step=10
        )
        max_fishes = st.number_input(
            "Max Fishes (Resource Capacity)", 
            min_value=1, 
            max_value=100000, 
            value=CONFIG["max_fishes"], 
            step=100
        )
        
        # Agent settings
        st.subheader("Agent Settings")
        use_mock_agents = st.checkbox("Use Mock Agents (No API calls)", value=True)
        
        # Persona selection - dynamically extract from CONFIG
        st.subheader("Player Personas")
        personas = []
        # Get available personas from config
        persona_options = list(CONFIG["persona_prompts"].keys())
        # Get default personas from config
        default_personas = CONFIG.get("player_personas", [])
        
        for i in range(n_players):
            # Find default persona for this player, or use first available
            default_persona = default_personas[i] if i < len(default_personas) else persona_options[0]
            default_index = persona_options.index(default_persona) if default_persona in persona_options else 0
            
            persona = st.selectbox(
                f"Player {i} Persona",
                persona_options,
                index=default_index,
                key=f"persona_{i}"
            )
            personas.append(persona)
        
        # Run button
        run_game = st.button("🚀 Run Game", type="primary", use_container_width=True)
        

    # Initialize runs storage in session state
    if "all_runs" not in st.session_state:
        st.session_state.all_runs = []

    # Initialize dashboard run history in session state (needed by Dashboard.__init__)
    if "dashboard_run_history" not in st.session_state:
        st.session_state.dashboard_run_history = []

    # Initialize current game state
    if "current_game_running" not in st.session_state:
        st.session_state.current_game_running = False
    if "current_game_dashboard" not in st.session_state:
        st.session_state.current_game_dashboard = None
    if "game_config_pending" not in st.session_state:
        st.session_state.game_config_pending = None

    # Main content area - show tabs for all runs + current running game
    tab_names = []
    tab_data = []

    # Add current running game tab if a game is in progress
    if st.session_state.current_game_running:
        tab_names.append("🔴 Live Game")
        tab_data.append({"type": "live"})

    # Add completed runs
    for i, run in enumerate(st.session_state.all_runs):
        tab_names.append(f"Run {i+1} ({run['timestamp']})")
        tab_data.append({"type": "completed", "run_idx": i})

    # Show tabs if there are any
    if len(tab_names) > 0:
        tabs = st.tabs(tab_names)

        # Display each tab
        for tab_idx, tab in enumerate(tabs):
            with tab:
                data = tab_data[tab_idx]
                if data["type"] == "live":
                    # Execute the pending game in this tab
                    if st.session_state.game_config_pending:
                        config = st.session_state.game_config_pending["config"]
                        use_mock = st.session_state.game_config_pending["use_mock_agents"]

                        st.markdown("### 🔴 Game in Progress")
                        st.info("The reasoning table below updates live as the game progresses.")

                        # Run the game in this tab context
                        run_data = _run_game_in_tab(config, use_mock)

                        if run_data:
                            # Game completed successfully
                            st.session_state.all_runs.append(run_data)
                            st.session_state.current_game_running = False
                            st.session_state.game_config_pending = None
                            st.success("✅ Game completed! View the results in the new tab.")
                            st.rerun()
                        else:
                            # Game failed
                            st.session_state.current_game_running = False
                            st.session_state.game_config_pending = None

                elif data["type"] == "completed":
                    # Show completed run
                    run_data = st.session_state.all_runs[data["run_idx"]]
                    _display_run_data(run_data)
    
    # Run new game
    if run_game:
        # Create config and store it for execution
        config = CONFIG.copy()
        config["n_players"] = n_players
        config["max_steps"] = max_steps
        config["initial_resource"] = initial_resource
        config["regeneration_rate"] = regeneration_rate
        config["sustainability_threshold"] = sustainability_threshold
        config["max_fishes"] = max_fishes
        config["player_personas"] = personas

        # Store config and flags
        st.session_state.game_config_pending = {
            "config": config,
            "use_mock_agents": use_mock_agents
        }
        st.session_state.current_game_running = True

        # Rerun to create the live tab and execute the game
        st.rerun()
        
    else:
        # Show instructions when not running
        st.info("👈 Configure your game settings in the sidebar and click 'Run Game' to start!")
        
        st.markdown("""
        ### How to Use
        
        1. **Configure Settings**: Use the sidebar to adjust:
           - Number of players
           - Game parameters (steps, resource, regeneration rate)
           - Player personas (selfish vs cooperative)
           - Agent type (mock or real LLM)
        
        2. **Run Game**: Click the "Run Game" button
        
        3. **Watch Live**: The dashboard will update in real-time showing:
           - Resource level over time
           - Player extractions
           - Cumulative payoffs
           - Cooperation metrics
           - LLM reasoning (if using real agents)
        
        4. **View Results**: After completion, see the final summary statistics
        """)


def _display_run_data(run_data: dict):
    """Display all logs and data for a specific run.
    
    Args:
        run_data: Dictionary containing all run data
    """
    from cpr_game.dashboard import Dashboard
    
    # Create a temporary dashboard instance for rendering
    temp_dashboard = Dashboard(run_data.get("config", CONFIG))
    # Set unique dashboard_id based on run_id to avoid duplicate Streamlit keys
    run_id = run_data.get('run_id', 'unknown')
    timestamp = run_data.get('timestamp', '')
    # Combine run_id and timestamp hash to ensure uniqueness across multiple displays
    unique_id = hashlib.md5(f"{run_id}_{timestamp}".encode()).hexdigest()[:8]
    temp_dashboard.dashboard_id = f"run_{run_id}_{unique_id}"
    temp_dashboard.resource_history = run_data.get("resource_history", [])
    temp_dashboard.extraction_history = [np.array(e) for e in run_data.get("extraction_history", [])]
    temp_dashboard.payoff_history = [np.array(p) for p in run_data.get("payoff_history", [])]
    temp_dashboard.cooperation_history = run_data.get("cooperation_history", [])
    temp_dashboard.reasoning_log = run_data.get("reasoning_log", {})
    temp_dashboard.run_history = run_data.get("run_history", [])
    
    # Display run header
    st.markdown(f"### Run {run_data.get('run_id', 'Unknown')} - {run_data.get('timestamp', 'Unknown time')}")
    
    # Show summary metrics
    summary = run_data.get("summary", {})
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rounds", summary.get("total_rounds", 0))
    with col2:
        tragedy = summary.get("tragedy_occurred", False)
        st.metric("Tragedy", "Yes" if tragedy else "No")
    with col3:
        st.metric("Final Resource", f"{int(summary.get('final_resource_level', 0))}")
    with col4:
        st.metric("Avg Cooperation", f"{summary.get('avg_cooperation_index', 0):.3f}")
    
    st.divider()
    
    # Create tabs for different views
    api_metrics = run_data.get("api_metrics_data", [])
    tab_names = ["📊 Charts & Metrics", "📝 Reasoning Table", "💭 Reasoning Log", "📋 All Logs"]
    if api_metrics:
        tab_names.insert(3, "📡 API Logs")  # Insert before "📋 All Logs"
    tab1, tab2, tab3, *rest_tabs = st.tabs(tab_names)

    with tab1:
        # Prepare game state for dashboard rendering
        run_config = run_data.get("config", CONFIG)
        game_state = {
            "resource": run_data.get("resource_history", [0])[-1] if run_data.get("resource_history") else 0,
            "step": len(run_data.get("extraction_history", [])),
            "max_steps": run_config.get("max_steps", CONFIG["max_steps"]),
            "done": True,
            "cumulative_payoffs": summary.get("cumulative_payoffs", []),
            "resource_history": run_data.get("resource_history", []),
            "extraction_history": run_data.get("extraction_history", []),
            "payoff_history": run_data.get("payoff_history", []),
            "cooperation_history": run_data.get("cooperation_history", []),
        }

        # Bar chart race - ALWAYS FIRST AND ON TOP
        # Use step 0 for static display (game is already done)
        temp_dashboard._render_bar_chart_race(step=0)

        # Render charts
        col1, col2 = st.columns(2)

        with col1:
            temp_dashboard._render_resource_chart()
            temp_dashboard._render_cooperation_chart()

        with col2:
            temp_dashboard._render_extraction_chart()

        # Show summary
        temp_dashboard.show_summary(summary)

    with tab2:
        temp_dashboard._render_reasoning_table()

    with tab3:
        temp_dashboard._render_reasoning_log()

    # API Logs tab (if available)
    if api_metrics and len(rest_tabs) > 0:
        with rest_tabs[0]:
            # Add API metrics to dashboard for rendering
            temp_dashboard.add_api_metrics(api_metrics)
            temp_dashboard._render_api_logs()

    # Complete Logs tab
    with rest_tabs[-1] if rest_tabs else tab3:
        st.markdown("### 📋 Complete Logs")
        
        # Generation data (prompts and responses)
        st.markdown("#### LLM Generations")
        generation_data = run_data.get("generation_data", [])
        if generation_data:
            run_config = run_data.get("config", CONFIG)
            n_players = run_config.get("n_players", CONFIG["n_players"])
            for idx, gen in enumerate(generation_data):
                round_num = idx // n_players + 1
                player_id = gen.get('player_id', idx % n_players)
                with st.expander(f"Player {player_id} - Round {round_num}", expanded=False):
                    # Display prompt in code block for better readability
                    prompt = gen.get("prompt", "")
                    if prompt:
                        st.markdown("**📝 Prompt:**")
                        st.code(prompt, language="text")
                        st.divider()
                    
                    # Display reasoning with markdown
                    reasoning = gen.get("reasoning", "")
                    if reasoning:
                        st.markdown("**💭 Reasoning:**")
                        st.markdown(reasoning)
                        st.divider()
                    
                    # Display full response if different from reasoning
                    response = gen.get("response", "")
                    if response and response != reasoning:
                        st.markdown("**📄 Full Response:**")
                        st.markdown(response)
                        st.divider()
                    
                    # Display action prominently
                    action = gen.get("action")
                    if action is not None:
                        st.metric("🎯 Action Taken", f"{int(action)}")
                    
                    # Display API metrics if available
                    api_metrics = gen.get("api_metrics")
                    if api_metrics:
                        with st.expander("📊 API Metrics", expanded=False):
                            if api_metrics.get("total_tokens"):
                                st.metric("Total Tokens", f"{api_metrics.get('total_tokens'):,}")
                            if api_metrics.get("latency"):
                                st.metric("Latency", f"{api_metrics.get('latency'):.2f}s")
                            if api_metrics.get("success") is False:
                                st.error(f"API Call Failed: {api_metrics.get('error', 'Unknown error')}")
        else:
            st.info("No generation data available.")
        
        # Round metrics
        st.markdown("#### Round Metrics")
        round_metrics = run_data.get("round_metrics", [])
        if round_metrics:
            metrics_df = pd.DataFrame(round_metrics)
            st.dataframe(metrics_df, width='stretch')
        else:
            st.info("No round metrics available.")
        
        # Full summary JSON
        st.markdown("#### Full Summary (JSON)")
        st.json(summary)
        
        # Config
        st.markdown("#### Configuration")
        st.json(run_data.get("config", {}))


if __name__ == "__main__":
    main()

