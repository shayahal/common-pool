"""Streamlit GUI application for CPR Game.

Run with: streamlit run app.py
"""

import streamlit as st
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from cpr_game.game_runner import GameRunner
from cpr_game.config import CONFIG
import time


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
        
        # Basic settings
        n_players = st.slider("Number of Players", 2, 5, 2)
        max_steps = st.slider("Max Steps", 10, 200, 50)
        initial_resource = st.number_input("Initial Resource", 100, 5000, 1000, step=100)
        regeneration_rate = st.slider("Regeneration Rate", 1.0, 3.0, 2.0, 0.1)
        sustainability_threshold = st.number_input("Sustainability Threshold", 100, 2000, 500, step=50)
        
        # Agent settings
        st.subheader("Agent Settings")
        use_mock_agents = st.checkbox("Use Mock Agents (No API calls)", value=True)
        use_mock_logging = st.checkbox("Use Mock Logging", value=True)
        
        # Persona selection
        st.subheader("Player Personas")
        personas = []
        persona_options = ["rational_selfish", "cooperative", ""]
        
        for i in range(n_players):
            persona = st.selectbox(
                f"Player {i} Persona",
                persona_options,
                index=1 if i < 2 else 2,
                key=f"persona_{i}"
            )
            personas.append(persona)
        
        # Run button
        run_game = st.button("🚀 Run Game", type="primary", use_container_width=True)
        
        # Auto-refresh option
        auto_refresh = st.checkbox("Auto-refresh during game", value=True)
        refresh_delay = st.slider("Refresh delay (seconds)", 0.1, 2.0, 0.5, 0.1) if auto_refresh else 0.0

    # Main content area
    if run_game:
        # Create config
        config = CONFIG.copy()
        config["n_players"] = n_players
        config["max_steps"] = max_steps
        config["initial_resource"] = initial_resource
        config["regeneration_rate"] = regeneration_rate
        config["sustainability_threshold"] = sustainability_threshold
        config["player_personas"] = personas

        # Initialize game runner
        runner = GameRunner(
            config=config,
            use_mock_agents=use_mock_agents,
            use_mock_logging=use_mock_logging
        )
        
        game_id = runner.setup_game()
        
        # Initialize dashboard
        dashboard = runner.dashboard
        if dashboard:
            dashboard.initialize(n_players)
        
        # Create placeholder for game output
        status_placeholder = st.empty()
        chart_placeholder = st.empty()
        
        # Start game trace
        runner.logger.start_game_trace(game_id, config)
        
        # Reset environment and agents
        observations, info = runner.env.reset()
        for agent in runner.agents:
            agent.reset()
        
        # Main game loop
        done = False
        step = 0
        resource_history = []
        extraction_history = []
        payoff_history = []
        cooperation_history = []
        
        while not done:
            # Get actions from all agents
            actions = []
            reasonings = []
            
            for i, agent in enumerate(runner.agents):
                obs = observations[f"player_{i}"]
                action, reasoning = agent.act(obs, return_reasoning=True)
                actions.append(action)
                reasonings.append(reasoning)
                
                # Log generation
                prompt = agent._build_prompt(obs) if hasattr(agent, '_build_prompt') else ""
                runner.logger.log_generation(
                    player_id=i,
                    prompt=prompt,
                    response=reasoning or "",
                    action=action,
                    reasoning=reasoning
                )
                
                # Add to dashboard
                if dashboard and reasoning:
                    dashboard.add_reasoning(i, reasoning)
            
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
            
            # Update dashboard
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
            
            # Display status
            with status_placeholder.container():
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Round", f"{step + 1} / {max_steps}")
                with col2:
                    st.metric("Resource Level", f"{info['resource']:.1f}")
                with col3:
                    coop = info.get("cooperation_index", 0.0)
                    st.metric("Cooperation Index", f"{coop:.3f}")
            
            step += 1
            
            # Auto-refresh
            if auto_refresh and not done:
                time.sleep(refresh_delay)
                st.rerun()
        
        # Get summary statistics
        summary = runner.env.get_summary_stats()
        
        # End logging trace
        runner.logger.end_game_trace(summary)
        
        # Show dashboard summary
        if dashboard:
            dashboard.show_summary(summary)
        
        # Display final summary
        st.success("✅ Game Complete!")
        st.json(summary)
        
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


if __name__ == "__main__":
    main()

