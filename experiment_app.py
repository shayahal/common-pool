"""Streamlit GUI application for managing CPR Game experiments.

Run with: streamlit run experiment_app.py
"""

import streamlit as st
import sys
from pathlib import Path
import uuid
import pandas as pd
from datetime import datetime
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from cpr_game.db_manager import DatabaseManager
from cpr_game.config import CONFIG
from cpr_game.persona_prompts import PERSONA_PROMPTS
from main import run_experiment

# Available models (common OpenAI models)
AVAILABLE_MODELS = [
    "gpt-3.5-turbo",
    "gpt-4",
    "gpt-4-turbo-preview",
    "gpt-4o",
    "gpt-4o-mini",
]

# Initialize database manager
@st.cache_resource
def get_db_manager():
    """Get cached database manager instance."""
    db_path = CONFIG.get("db_path", "data/game_results.duckdb")
    db_enabled = CONFIG.get("db_enabled", True)
    return DatabaseManager(db_path=db_path, enabled=db_enabled)


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="CPR Experiment Manager",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🧪 Common Pool Resource Game - Experiment Manager")
    st.markdown("Define and manage experiments with player pools and game parameters")

    db_manager = get_db_manager()

    if not db_manager.enabled:
        st.error("❌ Database is not enabled. Please check your configuration.")
        return

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📝 Define Experiment", "📋 Experiments List", "📊 Results"])

    with tab1:
        _show_experiment_definition(db_manager)

    with tab2:
        _show_experiments_list(db_manager)

    with tab3:
        _show_results_viewer(db_manager)


def _show_experiment_definition(db_manager: DatabaseManager):
    """Show experiment definition form."""
    st.header("Define New Experiment")

    # Experiment name
    experiment_name = st.text_input(
        "Experiment Name",
        value="",
        placeholder="e.g., 6 Players GPT-3.5 Experiment",
        help="A descriptive name for this experiment"
    )

    # Player pool builder
    st.subheader("Player Pool")
    st.markdown("Define players with persona + model combinations. These players will be randomly assigned to games.")

    # Initialize session state for players
    if "experiment_players" not in st.session_state:
        st.session_state.experiment_players = []

    # Add player button
    col1, col2 = st.columns([3, 1])
    with col1:
        new_persona = st.selectbox(
            "Persona",
            options=list(PERSONA_PROMPTS.keys()),
            key="new_player_persona"
        )
    with col2:
        new_model = st.selectbox(
            "Model",
            options=AVAILABLE_MODELS,
            index=0,
            key="new_player_model"
        )

    if st.button("➕ Add Player", type="primary"):
        st.session_state.experiment_players.append({
            "persona": new_persona,
            "model": new_model
        })
        st.rerun()

    # Display current players
    if st.session_state.experiment_players:
        st.markdown("**Current Players:**")
        player_df = pd.DataFrame(st.session_state.experiment_players)
        player_df.index = [f"Player {i+1}" for i in range(len(player_df))]
        st.dataframe(player_df, use_container_width=True)

        # Remove player buttons
        cols = st.columns(len(st.session_state.experiment_players))
        for i, col in enumerate(cols):
            with col:
                if st.button("🗑️ Remove", key=f"remove_{i}"):
                    st.session_state.experiment_players.pop(i)
                    st.rerun()
    else:
        st.info("👈 Add players to the pool using the controls above")

    # Experiment parameters
    st.subheader("Experiment Parameters")
    st.markdown("Configure game parameters for this experiment")

    param_col1, param_col2 = st.columns(2)

    with param_col1:
        # Auto-update n_players when players are added/removed
        n_players = st.number_input(
            "N_PLAYERS (Pool Size)",
            min_value=1,
            max_value=20,
            value=len(st.session_state.experiment_players) if st.session_state.experiment_players else 6,
            help="Number of players in the experiment pool (automatically matches number of players added above)"
        )
        
        # Show warning if mismatch
        if st.session_state.experiment_players and n_players != len(st.session_state.experiment_players):
            st.warning(
                f"⚠️ N_PLAYERS ({n_players}) doesn't match the number of players in pool "
                f"({len(st.session_state.experiment_players)}). "
                f"The actual pool size will be {len(st.session_state.experiment_players)}."
            )

        max_steps = st.number_input(
            "MAX_STEPS",
            min_value=10,
            max_value=500,
            value=CONFIG.get("max_steps", 50),
            help="Maximum number of rounds per game"
        )

        initial_resource = st.number_input(
            "INITIAL_RESOURCE",
            min_value=1,
            max_value=10000,
            value=CONFIG.get("initial_resource", 100),
            step=10,
            help="Initial resource level at game start"
        )

        regeneration_rate = st.number_input(
            "REGENERATION_RATE",
            min_value=1.0,
            max_value=5.0,
            value=float(CONFIG.get("regeneration_rate", 2.0)),
            step=0.1,
            help="Resource regeneration multiplier per round"
        )

    with param_col2:
        max_extraction = st.number_input(
            "MAX_EXTRACTION",
            min_value=1,
            max_value=100,
            value=CONFIG.get("max_extraction", 35),
            help="Maximum extraction per player per round"
        )

        max_fishes = st.number_input(
            "MAX_FISHES",
            min_value=1,
            max_value=100000,
            value=CONFIG.get("max_fishes", 100),
            step=100,
            help="Maximum resource capacity"
        )

        number_of_games = st.number_input(
            "Number of Games",
            min_value=1,
            max_value=1000,
            value=10,
            help="Total number of games to run in this experiment"
        )

        number_of_players_per_game = st.number_input(
            "Number of Players per Game",
            min_value=2,
            max_value=20,
            value=4,
            help="Number of players randomly selected from pool for each game"
        )

    # Validation
    if st.session_state.experiment_players:
        if number_of_players_per_game > len(st.session_state.experiment_players):
            st.warning(
                f"⚠️ Number of players per game ({number_of_players_per_game}) "
                f"exceeds pool size ({len(st.session_state.experiment_players)}). "
                f"All players will participate in every game."
            )

    # Save experiment button
    st.divider()
    
    if st.button("💾 Save Experiment", type="primary", use_container_width=True):
        if not experiment_name:
            st.error("❌ Please provide an experiment name")
        elif not st.session_state.experiment_players:
            st.error("❌ Please add at least one player to the pool")
        elif number_of_players_per_game < 2:
            st.error("❌ Number of players per game must be at least 2")
        else:
            # Generate experiment ID
            experiment_id = f"exp_{uuid.uuid4().hex[:8]}"

            # Prepare parameters (use actual pool size, not user input)
            actual_pool_size = len(st.session_state.experiment_players)
            parameters = {
                "n_players": actual_pool_size,  # Use actual pool size
                "max_steps": max_steps,
                "initial_resource": initial_resource,
                "regeneration_rate": regeneration_rate,
                "max_extraction": max_extraction,
                "max_fishes": max_fishes,
                "number_of_games": number_of_games,
                "number_of_players_per_game": number_of_players_per_game,
            }

            # Save to database
            success = db_manager.save_experiment(
                experiment_id=experiment_id,
                name=experiment_name,
                players=st.session_state.experiment_players,
                parameters=parameters
            )

            if success:
                st.success(f"✅ Experiment '{experiment_name}' saved with ID: {experiment_id}")
                # Clear form
                st.session_state.experiment_players = []
                st.rerun()
            else:
                st.error("❌ Failed to save experiment. Check logs for details.")


def _show_experiments_list(db_manager: DatabaseManager):
    """Show list of all experiments."""
    st.header("Experiments")

    # Refresh button
    if st.button("🔄 Refresh", key="refresh_experiments"):
        st.rerun()

    # Load experiments
    experiments = db_manager.list_experiments()

    if not experiments:
        st.info("No experiments found. Create one in the 'Define Experiment' tab.")
        return

    # Display as dataframe
    df = pd.DataFrame(experiments)
    df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    df = df.rename(columns={
        "experiment_id": "ID",
        "name": "Name",
        "status": "Status",
        "created_at": "Created",
        "player_count": "Players",
        "game_count": "Games"
    })

    # Reorder columns
    df = df[["Name", "ID", "Status", "Players", "Games", "Created"]]

    st.dataframe(df, use_container_width=True, hide_index=True)

    # Experiment actions
    st.subheader("Experiment Actions")

    selected_exp_id = st.selectbox(
        "Select Experiment",
        options=[exp["experiment_id"] for exp in experiments],
        format_func=lambda x: next(
            (exp["name"] for exp in experiments if exp["experiment_id"] == x),
            x
        )
    )

    if selected_exp_id:
        # Load full experiment details
        experiment = db_manager.load_experiment(selected_exp_id)

        if experiment:
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("📄 View Details", use_container_width=True):
                    st.session_state[f"view_exp_{selected_exp_id}"] = True

            with col2:
                # Check if experiment is already running
                experiment_status = experiment.get("status", "pending")
                is_running = experiment_status == "running"
                is_completed = experiment_status == "completed"
                
                if is_running:
                    st.warning("⏳ Experiment is currently running...")
                elif is_completed:
                    st.success("✅ Experiment completed")
                    if st.button("🔄 Run Again", use_container_width=True, key="run_again"):
                        st.session_state[f"run_exp_{selected_exp_id}"] = True
                else:
                    if st.button("▶️ Run Experiment", use_container_width=True, key="run_exp"):
                        st.session_state[f"run_exp_{selected_exp_id}"] = True

            with col3:
                if st.button("🗑️ Delete", use_container_width=True, type="secondary"):
                    if db_manager.delete_experiment(selected_exp_id):
                        st.success(f"✅ Experiment '{experiment['name']}' deleted")
                        st.rerun()
                    else:
                        st.error("❌ Failed to delete experiment")

            # Handle experiment run
            if st.session_state.get(f"run_exp_{selected_exp_id}", False):
                st.session_state[f"run_exp_{selected_exp_id}"] = False
                
                # Run configuration
                st.divider()
                st.markdown("### Run Experiment Configuration")
                
                run_col1, run_col2 = st.columns(2)
                with run_col1:
                    use_mock = st.checkbox(
                        "Use Mock Agents (No API Calls)",
                        value=False,
                        help="Use mock agents for testing without making API calls"
                    )
                with run_col2:
                    max_workers = st.number_input(
                        "Max Workers",
                        min_value=1,
                        max_value=20,
                        value=min(10, experiment['parameters']['number_of_games']),
                        help="Number of parallel workers for running games"
                    )
                
                if st.button("🚀 Start Experiment", type="primary", use_container_width=True):
                    # Store run parameters
                    st.session_state[f"run_exp_config_{selected_exp_id}"] = {
                        "use_mock": use_mock,
                        "max_workers": max_workers
                    }
                    st.session_state[f"should_run_{selected_exp_id}"] = True
                    st.rerun()
            
            # Execute experiment run
            if st.session_state.get(f"should_run_{selected_exp_id}", False):
                st.session_state[f"should_run_{selected_exp_id}"] = False
                config = st.session_state.get(f"run_exp_config_{selected_exp_id}", {})
                
                st.info("🚀 Starting experiment... This may take a while. Please wait.")
                
                # Run experiment (this will block, but Streamlit will show it's working)
                with st.spinner("Running experiment... This may take several minutes."):
                    try:
                        success = run_experiment(
                            experiment_id=selected_exp_id,
                            use_mock_agents=config.get("use_mock", False),
                            max_workers=config.get("max_workers", 10)
                        )
                        
                        if success:
                            st.success("✅ Experiment completed successfully!")
                            # Refresh experiment data
                            experiment = db_manager.load_experiment(selected_exp_id)
                        else:
                            st.error("❌ Experiment failed. Check logs for details.")
                    except Exception as e:
                        st.error(f"❌ Error running experiment: {str(e)}")
                        st.exception(e)
                
                # Clear config
                if f"run_exp_config_{selected_exp_id}" in st.session_state:
                    del st.session_state[f"run_exp_config_{selected_exp_id}"]

            # Show details if requested
            if st.session_state.get(f"view_exp_{selected_exp_id}", False):
                st.divider()
                st.markdown("### Experiment Details")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"**Name:** {experiment['name']}")
                    st.markdown(f"**ID:** {experiment['experiment_id']}")
                    st.markdown(f"**Status:** {experiment['status']}")
                    st.markdown(f"**Created:** {experiment['created_at']}")

                with col2:
                    st.markdown("**Parameters:**")
                    params_df = pd.DataFrame([experiment['parameters']]).T
                    params_df.columns = ["Value"]
                    st.dataframe(params_df, use_container_width=True)

                st.markdown("**Players:**")
                players_df = pd.DataFrame(experiment['players'])
                st.dataframe(players_df, use_container_width=True, hide_index=True)


def _show_results_viewer(db_manager: DatabaseManager):
    """Show experiment results viewer."""
    st.header("Experiment Results")

    # Load experiments
    experiments = db_manager.list_experiments()

    if not experiments:
        st.info("No experiments found. Create one in the 'Define Experiment' tab.")
        return

    # Select experiment
    selected_exp_id = st.selectbox(
        "Select Experiment",
        options=[exp["experiment_id"] for exp in experiments],
        format_func=lambda x: next(
            (f"{exp['name']} ({exp['game_count']} games)" for exp in experiments if exp["experiment_id"] == x),
            x
        )
    )

    if not selected_exp_id:
        return

    # Load results
    results = db_manager.get_experiment_results(selected_exp_id)

    if not results:
        st.info("No results found for this experiment. Run the experiment first.")
        return

    st.success(f"Found {len(results)} game results")

    # Aggregate statistics
    st.subheader("Aggregate Statistics")

    # Calculate aggregates
    tragedy_count = sum(1 for r in results if r["summary"].get("tragedy_occurred", False))
    avg_rounds = sum(r["summary"].get("total_rounds", 0) for r in results) / len(results)
    avg_final_resource = sum(r["summary"].get("final_resource_level", 0) for r in results) / len(results)
    avg_cooperation = sum(r["summary"].get("avg_cooperation_index", 0) for r in results) / len(results)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Games", len(results))
    with col2:
        st.metric("Tragedy Rate", f"{tragedy_count / len(results):.1%}")
    with col3:
        st.metric("Avg Rounds", f"{avg_rounds:.1f}")
    with col4:
        st.metric("Avg Cooperation", f"{avg_cooperation:.3f}")

    # Individual game results
    st.subheader("Individual Game Results")

    # Create summary dataframe
    game_summaries = []
    for result in results:
        summary = result["summary"]
        game_summaries.append({
            "Game ID": result["game_id"],
            "Rounds": summary.get("total_rounds", 0),
            "Tragedy": "Yes" if summary.get("tragedy_occurred", False) else "No",
            "Final Resource": f"{summary.get('final_resource_level', 0):.1f}",
            "Avg Cooperation": f"{summary.get('avg_cooperation_index', 0):.3f}",
            "Timestamp": result["timestamp"]
        })

    summary_df = pd.DataFrame(game_summaries)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Download results
    st.subheader("Download Results")
    results_json = json.dumps(results, default=str, indent=2)
    st.download_button(
        label="📥 Download Results (JSON)",
        data=results_json,
        file_name=f"experiment_{selected_exp_id}_results.json",
        mime="application/json"
    )

    # CSV export
    if game_summaries:
        csv = summary_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Summary (CSV)",
            data=csv,
            file_name=f"experiment_{selected_exp_id}_summary.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()

