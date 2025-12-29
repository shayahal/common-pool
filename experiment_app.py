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
from cpr_game.logger_setup import get_logger

logger = get_logger(__name__)

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
    """Get cached read-only database manager for queries.
    
    Uses READ_ONLY mode to allow concurrent access with external database tools.
    """
    db_path = CONFIG.get("db_path", "data/game_results.db")
    db_enabled = CONFIG.get("db_enabled", True)
    return DatabaseManager(db_path=db_path, enabled=db_enabled, access_mode='READ_ONLY')


def get_write_db_manager():
    """Get write database manager (not cached, created when needed).
    
    Creates a new READ_WRITE connection for write operations.
    Should be used temporarily and closed after use.
    """
    db_path = CONFIG.get("db_path", "data/game_results.db")
    db_enabled = CONFIG.get("db_enabled", True)
    return DatabaseManager(db_path=db_path, enabled=db_enabled, access_mode='READ_WRITE')


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
    
    if db_manager.conn is None:
        st.error("❌ Database connection failed. This may be due to:")
        st.error("   • Another process has the database locked (close database UI tools or other connections)")
        st.error("   • Database file permissions issue")
        st.error("   • Database file corruption")
        st.info("💡 **Tip**: Try refreshing the page or closing any database UI tools that might be accessing the database.")
        
        # Try to recreate the connection
        if st.button("🔄 Retry Database Connection"):
            get_db_manager.clear()
            db_manager = get_db_manager()
            if db_manager.conn is not None:
                st.success("✅ Database connection restored!")
                st.rerun()
            else:
                st.error("❌ Still unable to connect. Check logs for details.")
        return

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Define Experiment", 
        "📋 Experiments List", 
        "🔬 Experiment View",
        "🎮 Game View"
    ])

    with tab1:
        _show_experiment_definition(db_manager)

    with tab2:
        _show_experiments_list(db_manager)

    with tab3:
        _show_experiment_view(db_manager)

    with tab4:
        _show_game_view(db_manager)


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

    # Auto-update n_players when players are added/removed
    actual_pool_size = len(st.session_state.experiment_players) if st.session_state.experiment_players else 6
    n_players = st.number_input(
        "N_PLAYERS (Pool Size)",
        min_value=1,
        max_value=20,
        value=actual_pool_size,
        help="Number of players in the experiment pool (automatically matches number of players added above)"
    )
    
    # Show warning if mismatch
    if st.session_state.experiment_players and n_players != len(st.session_state.experiment_players):
        st.warning(
            f"⚠️ N_PLAYERS ({n_players}) doesn't match the number of players in pool "
            f"({len(st.session_state.experiment_players)}). "
            f"The actual pool size will be {len(st.session_state.experiment_players)}."
        )

    # Use actual pool size for calculations
    actual_n_players = len(st.session_state.experiment_players) if st.session_state.experiment_players else n_players

    param_col1, param_col2 = st.columns(2)

    with param_col1:
        max_steps = st.number_input(
            "MAX_STEPS",
            min_value=10,
            max_value=500,
            value=20,
            help="Maximum number of rounds per game"
        )

        number_of_games = st.number_input(
            "Number of Games",
            min_value=1,
            max_value=1000,
            value=10,
            help="Total number of games to run in this experiment"
        )

    with param_col2:
        # Default: N_players - 2, but must be at least 2
        default_players_per_game = max(2, actual_n_players - 2)
        number_of_players_per_game = st.number_input(
            "Number of Players per Game",
            min_value=2,
            max_value=20,
            value=default_players_per_game,
            help="Number of players randomly selected from pool for each game (default: N_players - 2)"
        )

    # Fixed values (not in GUI per constraints)
    INITIAL_RESOURCE = 1000
    Max_fish = 1000
    REGENERATION_RATE = 2.0  # Default regeneration rate
    
    # Calculate MAX_EXTRACTION: int(4 * INITIAL_RESOURCE / 7 * Number of Players per Game)
    max_extraction = int(4 * INITIAL_RESOURCE / (7 * number_of_players_per_game))
    
    # Show calculated values as info
    st.info(f"**Calculated values:** INITIAL_RESOURCE = {INITIAL_RESOURCE}, Max_fish = {Max_fish}, MAX_EXTRACTION = {max_extraction} (calculated: 4×{INITIAL_RESOURCE} ÷ (7×{number_of_players_per_game}))")
    
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
                "initial_resource": INITIAL_RESOURCE,  # Fixed: 1000
                "regeneration_rate": REGENERATION_RATE,  # Fixed: 2.0
                "max_extraction": max_extraction,  # Calculated
                "Max_fish": Max_fish,  # Fixed: 1000 (renamed from max_fishes)
                "number_of_games": number_of_games,
                "number_of_players_per_game": number_of_players_per_game,
            }

            # Save to database (use write manager)
            # Close read-only connection first to avoid lock conflicts
            import time
            
            if db_manager.conn is not None:
                db_manager.close()
                # Give database a moment to release the lock
                time.sleep(0.1)
            
            get_db_manager.clear()  # Clear cache so it can be recreated
            
            write_db_manager = None
            max_retries = 3
            retry_delay = 0.2
            
            for attempt in range(max_retries):
                try:
                    write_db_manager = get_write_db_manager()
                    
                    # Check if write connection was successful
                    if write_db_manager.conn is None:
                        if attempt < max_retries - 1:
                            logger.warning(f"Write connection failed, retrying ({attempt + 1}/{max_retries})...")
                            time.sleep(retry_delay)
                            continue
                        else:
                            st.error("❌ Failed to create write connection after retries. Database may be locked.")
                            logger.error("Write database manager connection is None after retries")
                            success = False
                            break
                    else:
                        # Connection successful, proceed with save
                        success = write_db_manager.save_experiment(
                            experiment_id=experiment_id,
                            name=experiment_name,
                            players=st.session_state.experiment_players,
                            parameters=parameters
                        )
                        break  # Success, exit retry loop
                        
                except Exception as e:
                    if "lock" in str(e).lower() or "locked" in str(e).lower():
                        if attempt < max_retries - 1:
                            logger.warning(f"Database lock detected, retrying ({attempt + 1}/{max_retries})...")
                            time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                            continue
                        else:
                            logger.error(f"Error saving experiment after retries: {e}", exc_info=True)
                            st.error(f"❌ Database is locked. Please close any database UI tools or other connections and try again.")
                            success = False
                            break
                    else:
                        logger.error(f"Error saving experiment: {e}", exc_info=True)
                        st.error(f"❌ Error saving experiment: {str(e)}")
                        success = False
                        break
            else:
                # If we exhausted retries without success
                success = False
            
            # Cleanup
            if write_db_manager is not None:
                try:
                    write_db_manager.close()
                except Exception as e:
                    logger.error(f"Error closing write connection: {e}", exc_info=True)
                    raise RuntimeError(f"Failed to close write connection: {e}") from e
                time.sleep(0.1)  # Give database time to release lock
            
            # Recreate read-only connection for queries
            try:
                db_manager = get_db_manager()
            except Exception as e:
                logger.error(f"Failed to recreate read-only connection: {e}", exc_info=True)
                st.warning("⚠️ Could not recreate read-only connection. Refresh the page if needed.")

            if success:
                st.success(f"✅ Experiment '{experiment_name}' saved with ID: {experiment_id}")
                st.info("💡 To run this experiment, use the experiment_worker.py script: `python experiment_worker.py --workers 4`")
                
                # Clear form
                st.session_state.experiment_players = []
                st.rerun()
            else:
                st.error("❌ Failed to save experiment. Check logs for details.")


def _show_experiments_list(db_manager: DatabaseManager):
    """Show list of all experiments."""
    st.header("Experiments")

    # Action buttons
    if st.button("🔄 Refresh", key="refresh_experiments", use_container_width=True):
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

    # Individual reset buttons for each experiment
    st.subheader("Reset Experiments to Pending")
    write_db_manager = get_write_db_manager()
    try:
        for exp in experiments:
            exp_id = exp["experiment_id"]
            exp_name = exp["name"]
            exp_status = exp.get("status", "pending")
            
            # Only show reset button if status is not already pending
            if exp_status != "pending":
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{exp_name}** (ID: {exp_id}, Status: {exp_status})")
                with col2:
                    if st.button("↩️ Reset to Pending", key=f"reset_{exp_id}", use_container_width=True, type="secondary"):
                        try:
                            if write_db_manager.update_experiment_status(exp_id, "pending"):
                                st.success(f"✅ Reset '{exp_name}' to 'pending' status")
                                st.rerun()
                            else:
                                st.error(f"❌ Failed to reset '{exp_name}'")
                        except Exception as e:
                            st.error(f"❌ Error resetting '{exp_name}': {e}")
    finally:
        write_db_manager.close()

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
                # Show experiment status
                experiment_status = experiment.get("status", "pending")
                is_running = experiment_status == "running"
                is_completed = experiment_status == "completed"
                
                if is_running:
                    st.warning("⏳ Experiment is currently running...")
                elif is_completed:
                    st.success("✅ Experiment completed")
                else:
                    st.info("⏸️ Experiment is pending. Use `experiment_worker.py` to run it.")

            with col3:
                if st.button("🗑️ Delete", use_container_width=True, type="secondary", key=f"delete_{selected_exp_id}"):
                    write_db_manager = get_write_db_manager()
                    try:
                        if write_db_manager.delete_experiment(selected_exp_id):
                            st.success(f"✅ Experiment '{experiment['name']}' deleted")
                            st.rerun()
                        else:
                            st.error("❌ Failed to delete experiment")
                    finally:
                        write_db_manager.close()

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


def _show_experiment_view(db_manager: DatabaseManager):
    """Show experiment view with summary of all games."""
    st.header("🔬 Experiment View")
    st.markdown("View detailed summary of all games in an experiment")

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
        ),
        key="exp_view_select"
    )

    if not selected_exp_id:
        return

    # Load experiment details
    experiment = db_manager.load_experiment(selected_exp_id)
    if not experiment:
        st.error("Failed to load experiment details")
        return

    # Display experiment info
    st.subheader("Experiment Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Name:** {experiment['name']}")
        st.markdown(f"**ID:** {experiment['experiment_id']}")
    with col2:
        st.markdown(f"**Status:** {experiment['status']}")
        st.markdown(f"**Created:** {experiment['created_at']}")
    with col3:
        st.markdown(f"**Players in Pool:** {len(experiment['players'])}")
        st.markdown(f"**Players per Game:** {experiment['parameters']['number_of_players_per_game']}")

    # Load all game results
    results = db_manager.get_experiment_results(selected_exp_id)

    if not results:
        st.info("No game results found for this experiment. Run the experiment first.")
        return

    st.divider()
    st.subheader(f"Games Summary ({len(results)} games)")

    # Aggregate statistics
    tragedy_count = sum(1 for r in results if r["summary"].get("tragedy_occurred", False))
    avg_rounds = sum(r["summary"].get("total_rounds", 0) for r in results) / len(results) if results else 0
    avg_final_resource = sum(r["summary"].get("final_resource_level", 0) for r in results) / len(results) if results else 0
    avg_cooperation = sum(r["summary"].get("avg_cooperation_index", 0) for r in results) / len(results) if results else 0
    
    # Calculate total payoffs
    total_payoffs = []
    for r in results:
        payoffs = r["summary"].get("cumulative_payoffs", [])
        if payoffs:
            total_payoffs.extend(payoffs)
    avg_payoff = sum(total_payoffs) / len(total_payoffs) if total_payoffs else 0

    # Calculate total cost for experiment
    total_cost = 0.0
    for result in results:
        summary = result["summary"]
        # Check if total_cost is in summary
        if "total_cost" in summary:
            total_cost += summary["total_cost"]
        # Otherwise, calculate from api_metrics if available
        elif "api_metrics" in summary or "round_metrics" in summary:
            # Try to extract costs from round metrics or api metrics
            round_metrics = summary.get("round_metrics", [])
            for round_metric in round_metrics:
                if "api_cost" in round_metric:
                    total_cost += round_metric["api_cost"]
                elif "cost" in round_metric:
                    total_cost += round_metric["cost"]
    
    # Display metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Total Games", len(results))
    with col2:
        st.metric("Tragedy Rate", f"{tragedy_count / len(results):.1%}" if results else "0%")
    with col3:
        st.metric("Avg Rounds", f"{avg_rounds:.1f}")
    with col4:
        st.metric("Avg Final Resource", f"{avg_final_resource:.1f}")
    with col5:
        st.metric("Avg Cooperation", f"{avg_cooperation:.3f}")
    with col6:
        st.metric("Total Cost", f"${total_cost:.4f}")

    # Games table
    st.subheader("All Games")
    
    game_data = []
    for result in results:
        summary = result["summary"]
        payoffs = summary.get("cumulative_payoffs", [])
        max_payoff = max(payoffs) if payoffs else 0
        min_payoff = min(payoffs) if payoffs else 0
        
        # Convert to int where appropriate (except 0-1 range numbers)
        total_rounds = int(summary.get("total_rounds", 0))
        final_resource = int(summary.get("final_resource_level", 0))
        max_payoff_int = int(max_payoff)
        min_payoff_int = int(min_payoff)
        
        # Calculate cost for this game
        game_cost = 0.0
        if "total_cost" in summary:
            game_cost = summary["total_cost"]
        elif "api_metrics" in summary or "round_metrics" in summary:
            round_metrics = summary.get("round_metrics", [])
            for round_metric in round_metrics:
                if "api_cost" in round_metric:
                    game_cost += round_metric["api_cost"]
                elif "cost" in round_metric:
                    game_cost += round_metric["cost"]
        
        game_data.append({
            "Game ID": result["game_id"],
            "Rounds": total_rounds,
            "Tragedy": "✅ Yes" if summary.get("tragedy_occurred", False) else "❌ No",
            "Final Resource": final_resource,
            "Avg Cooperation": f"{summary.get('avg_cooperation_index', 0):.3f}",
            "Max Payoff": max_payoff_int,
            "Min Payoff": min_payoff_int,
            "Gini Coefficient": f"{summary.get('gini_coefficient', 0):.3f}",
            "Cost": f"${game_cost:.4f}",
            "Timestamp": result["timestamp"]
        })

    games_df = pd.DataFrame(game_data)
    
    # Display games table
    if not games_df.empty:
        st.dataframe(games_df, use_container_width=True, hide_index=True)
        
        # Add navigation buttons for each game
        st.markdown("**Navigate to Game View:**")
        cols = st.columns(min(len(games_df), 5))  # Limit to 5 columns for better layout
        for idx, (_, row) in enumerate(games_df.iterrows()):
            col_idx = idx % len(cols)
            with cols[col_idx]:
                if st.button(f"View {row['Game ID']}", key=f"nav_game_{row['Game ID']}"):
                    st.session_state["navigate_to_game"] = {
                        "experiment_id": selected_exp_id,
                        "game_id": row["Game ID"]
                    }
                    st.rerun()

    # Player performance across all games
    st.subheader("Player Performance Across All Games")
    
    # Get all player results for this experiment
    player_stats = {}
    for result in results:
        game_id = result["game_id"]
        players = db_manager.get_game_players(selected_exp_id, game_id)
        summary = result["summary"]
        payoffs = summary.get("cumulative_payoffs", [])
        
        # Get player costs for this game
        player_costs = summary.get("player_costs", {})
        api_metrics = summary.get("api_metrics_data", [])
        
        # If player_costs not available, calculate from api_metrics
        if not player_costs and api_metrics:
            player_costs = {}
            for metric in api_metrics:
                player_id = metric.get("player_id")
                if player_id is not None:
                    cost = metric.get("cost", 0) or 0
                    if player_id not in player_costs:
                        player_costs[player_id] = 0.0
                    player_costs[player_id] += cost
        
        for i, player in enumerate(players):
            player_key = f"{player['persona']} ({player['model']})"
            if player_key not in player_stats:
                player_stats[player_key] = {
                    "persona": player["persona"],
                    "model": player["model"],
                    "games_played": 0,
                    "total_payoff": 0,
                    "total_cost": 0.0,
                    "wins": 0,
                    "payoffs": []
                }
            
            player_stats[player_key]["games_played"] += 1
            
            # Add cost for this player in this game (player_index maps to player_id)
            if i in player_costs:
                player_stats[player_key]["total_cost"] += player_costs[i]
            elif str(i) in player_costs:  # Handle string keys
                player_stats[player_key]["total_cost"] += player_costs[str(i)]
            
            if i < len(payoffs):
                payoff = payoffs[i]
                player_stats[player_key]["total_payoff"] += payoff
                player_stats[player_key]["payoffs"].append(payoff)
                
                # Check if this player won (highest payoff in this game)
                if payoff == max(payoffs):
                    player_stats[player_key]["wins"] += 1

    # Create player performance dataframe
    if player_stats:
        import statistics
        
        player_perf_data = []
        for player_key, stats in player_stats.items():
            avg_payoff = stats["total_payoff"] / stats["games_played"] if stats["games_played"] > 0 else 0
            win_rate = stats["wins"] / stats["games_played"] if stats["games_played"] > 0 else 0
            
            # Calculate median payoff
            median_payoff = statistics.median(stats["payoffs"]) if stats["payoffs"] else 0
            
            # Calculate average cost per game
            avg_cost = stats["total_cost"] / stats["games_played"] if stats["games_played"] > 0 else 0.0
            
            player_perf_data.append({
                "Player": player_key,
                "Games Played": stats["games_played"],
                "Avg Payoff": f"{avg_payoff:.2f}",
                "Median Payoff": f"{median_payoff:.2f}",
                "Total Cost": f"${stats['total_cost']:.4f}",
                "Avg Cost/Game": f"${avg_cost:.4f}",
                "Wins": stats["wins"],
                "Win Rate": f"{win_rate:.1%}"
            })
        
        player_perf_df = pd.DataFrame(player_perf_data)
        # Sort by median payoff descending
        player_perf_df["Median Payoff Float"] = player_perf_df["Median Payoff"].astype(float)
        player_perf_df = player_perf_df.sort_values("Median Payoff Float", ascending=False)
        player_perf_df = player_perf_df.drop(columns=["Median Payoff Float"])
        st.dataframe(player_perf_df, use_container_width=True, hide_index=True)
    else:
        st.info("No player performance data available")


def _show_game_view(db_manager: DatabaseManager):
    """Show detailed view of a specific game."""
    st.header("🎮 Game View")
    st.markdown("View detailed information for a specific game")

    # Check for navigation from experiment view
    if "navigate_to_game" in st.session_state:
        nav_info = st.session_state["navigate_to_game"]
        st.session_state["game_view_exp_select"] = nav_info["experiment_id"]
        st.session_state["game_view_game_select"] = nav_info["game_id"]
        del st.session_state["navigate_to_game"]

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
        ),
        key="game_view_exp_select"
    )

    if not selected_exp_id:
        return

    # Load game results for this experiment
    results = db_manager.get_experiment_results(selected_exp_id)

    if not results:
        st.info("No game results found for this experiment. Run the experiment first.")
        return

    # Select game
    game_options = {r["game_id"]: f"{r['game_id']} - {r['timestamp']}" for r in results}
    game_ids = list(game_options.keys())
    
    # Determine default index
    default_idx = 0
    if "game_view_game_select" in st.session_state:
        if st.session_state["game_view_game_select"] in game_ids:
            default_idx = game_ids.index(st.session_state["game_view_game_select"])
    
    selected_game_id = st.selectbox(
        "Select Game",
        options=game_ids,
        format_func=lambda x: game_options[x],
        key="game_view_game_select",
        index=default_idx
    )

    if not selected_game_id:
        return

    # Load game details
    try:
        game_result = db_manager.get_game_result(selected_exp_id, selected_game_id)
        game_players = db_manager.get_game_players(selected_exp_id, selected_game_id)

        if not game_result:
            st.error(f"Failed to load game details for game '{selected_game_id}' in experiment '{selected_exp_id}'. The game may not exist in the database.")
            st.info("💡 Try selecting a different game or check if the experiment has completed running.")
            return
    except Exception as e:
        st.error(f"Error loading game details: {str(e)}")
        st.exception(e)
        return

    summary = game_result["summary"]

    # Calculate total cost for this game
    game_total_cost = 0.0
    if "total_cost" in summary:
        game_total_cost = summary["total_cost"]
    elif "api_metrics" in summary or "round_metrics" in summary:
        # Try to extract costs from round metrics or api metrics
        round_metrics = summary.get("round_metrics", [])
        for round_metric in round_metrics:
            if "api_cost" in round_metric:
                game_total_cost += round_metric["api_cost"]
            elif "cost" in round_metric:
                game_total_cost += round_metric["cost"]
        # Also check if there's a direct api_metrics list
        api_metrics_list = summary.get("api_metrics", [])
        for api_metric in api_metrics_list:
            if "cost" in api_metric:
                game_total_cost += api_metric["cost"]

    # Game overview
    st.subheader("Game Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Game ID", game_result["game_id"])
    with col2:
        st.metric("Total Rounds", game_result["total_rounds"] or summary.get("total_rounds", 0))
    with col3:
        tragedy_status = "✅ Yes" if game_result["tragedy_occurred"] or summary.get("tragedy_occurred", False) else "❌ No"
        st.metric("Tragedy Occurred", tragedy_status)
    with col4:
        st.metric("Final Resource", f"{game_result['final_resource_level'] or summary.get('final_resource_level', 0):.1f}")
    with col5:
        st.metric("Total Cost", f"${game_total_cost:.4f}")

    st.markdown(f"**Timestamp:** {game_result['timestamp']}")

    # Game statistics
    st.subheader("Game Statistics")
    
    stat_col1, stat_col2 = st.columns(2)
    
    with stat_col1:
        st.markdown("**Resource & Sustainability**")
        st.json({
            "Final Resource Level": summary.get("final_resource_level", 0),
            "Total Extracted": summary.get("total_extracted", 0),
            "Sustainability Score": f"{summary.get('sustainability_score', 0):.3f}"
        })
    
    with stat_col2:
        st.markdown("**Cooperation & Fairness**")
        st.json({
            "Avg Cooperation Index": f"{summary.get('avg_cooperation_index', 0):.3f}",
            "Gini Coefficient": f"{summary.get('gini_coefficient', 0):.3f}"
        })

    # Players in this game
    st.subheader("Players in This Game")
    
    if game_players:
        # Match players with their payoffs
        payoffs = summary.get("cumulative_payoffs", [])
        max_payoff = max(payoffs) if payoffs else 0
        
        player_data = []
        for i, player in enumerate(game_players):
            player_payoff = payoffs[i] if i < len(payoffs) else 0
            is_winner = player_payoff == max_payoff and payoffs
            
            player_data.append({
                "Index": player["player_index"],
                "Persona": player["persona"],
                "Model": player["model"],
                "Total Reward": f"{player['total_reward']:.2f}",
                "Winner": "🏆 Yes" if is_winner else "No"
            })
        
        players_df = pd.DataFrame(player_data)
        st.dataframe(players_df, use_container_width=True, hide_index=True)
    else:
        st.info("No player data available for this game")

    # Winning information
    if game_result.get("winning_player_uuid") and game_players:
        st.subheader("Winner Information")
        winner_uuid = game_result["winning_player_uuid"]
        winner = next((p for p in game_players if p["player_uuid"] == winner_uuid), None)
        
        if winner:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Winner:** {winner['persona']}")
            with col2:
                st.markdown(f"**Model:** {winner['model']}")
            with col3:
                st.markdown(f"**Winning Payoff:** {game_result.get('winning_payoff', 0):.2f}")
        else:
            st.info("Winner information not available")

    # Round-by-round logs
    st.subheader("Round-by-Round Logs")
    
    # Check if round_metrics are available in summary
    round_metrics = summary.get("round_metrics", [])
    if round_metrics:
        # Create a dataframe for round-by-round data
        rounds_data = []
        for round_num, metrics in enumerate(round_metrics, 1):
            rounds_data.append({
                "Round": round_num,
                "Resource Level": int(metrics.get("resource_level", 0)),
                "Total Extraction": int(metrics.get("total_extraction", 0)),
                "Cooperation Index": f"{metrics.get('cooperation_index', 0):.3f}",
                "Individual Extractions": ", ".join([str(int(x)) for x in metrics.get("individual_extractions", [])]),
                "Individual Payoffs": ", ".join([f"{x:.2f}" for x in metrics.get("individual_payoffs", [])])
            })
        
        rounds_df = pd.DataFrame(rounds_data)
        st.dataframe(rounds_df, use_container_width=True, hide_index=True)
    else:
        # Try to reconstruct from cumulative payoffs if available
        cumulative_payoffs = summary.get("cumulative_payoffs", [])
        total_rounds = game_result.get("total_rounds") or summary.get("total_rounds", 0)
        
        if total_rounds > 0:
            st.info(f"Round-by-round metrics not available. Game had {total_rounds} rounds.")
            st.markdown("**Final cumulative payoffs:**")
            if cumulative_payoffs:
                payoff_df = pd.DataFrame({
                    "Player": [f"Player {i+1}" for i in range(len(cumulative_payoffs))],
                    "Final Cumulative Payoff": [f"{p:.2f}" for p in cumulative_payoffs]
                })
                st.dataframe(payoff_df, use_container_width=True, hide_index=True)
        else:
            st.info("No round-by-round logs available for this game.")

    # Full summary JSON
    with st.expander("📄 Full Game Summary (JSON)"):
        st.json(summary)

    # Download game data
    st.subheader("Download Game Data")
    game_json = json.dumps({
        "game_id": game_result["game_id"],
        "experiment_id": selected_exp_id,
        "timestamp": str(game_result["timestamp"]),
        "summary": summary,
        "players": game_players
    }, default=str, indent=2)
    
    st.download_button(
        label="📥 Download Game Data (JSON)",
        data=game_json,
        file_name=f"game_{selected_game_id}_data.json",
        mime="application/json"
    )


if __name__ == "__main__":
    main()

