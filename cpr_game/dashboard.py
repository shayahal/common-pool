"""Streamlit dashboard for real-time CPR game visualization.

Displays resource dynamics, player actions, payoffs, cooperation metrics,
and LLM reasoning in an interactive dashboard.
"""

from typing import Dict, List, Optional
import uuid
import streamlit as st
try:
    from streamlit.errors import StreamlitDuplicateElementKey
except ImportError:
    # Fallback for older streamlit versions
    StreamlitDuplicateElementKey = Exception
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from .config import CONFIG


class Dashboard:
    """Interactive dashboard for CPR game visualization.

    Displays multiple charts tracking game dynamics and player behavior.
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize dashboard.

        Args:
            config: Configuration dictionary
        """
        self.config = config if config is not None else CONFIG
        
        # Unique identifier for this dashboard instance
        # Use a more unique ID that includes timestamp to avoid conflicts
        if not hasattr(self, 'dashboard_id'):
            import time
            self.dashboard_id = f"{str(uuid.uuid4())[:8]}_{int(time.time() * 1000) % 100000}"

        # Display settings
        self.chart_height = self.config["chart_height"]
        self.chart_width = self.config["chart_width"]
        self.player_colors = self.config["player_colors"]
        self.resource_color = self.config["resource_color"]
        self.threshold_color = self.config["threshold_color"]

        # Game state tracking
        self.resource_history: List[int] = []
        self.extraction_history: List[np.ndarray] = []
        self.payoff_history: List[np.ndarray] = []
        self.cooperation_history: List[float] = []
        self.reasoning_log: Dict[int, List[str]] = {}
        
        # Detailed run history for each step - use session state to persist across reruns
        # Initialize session state if it doesn't exist
        # Use try/except to handle case where key doesn't exist yet
        try:
            self.run_history = st.session_state.dashboard_run_history
        except (KeyError, AttributeError):
            # Try to set it, but if that also fails, use a local list
            try:
                st.session_state.dashboard_run_history = []
                self.run_history = st.session_state.dashboard_run_history
            except (KeyError, AttributeError):
                # Fallback: use a local list if session state is not available
                self.run_history = []
        
        # Chart containers for updating without duplicate keys
        # Will be initialized in update() method using session state
        self._chart_containers = None

    def initialize(self, n_players: int):
        """Initialize dashboard layout.

        Args:
            n_players: Number of players in game
        """
        st.set_page_config(
            page_title="CPR Game Dashboard",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        st.title("🌳 Common Pool Resource Game - Live Dashboard")

        # Initialize reasoning log
        for i in range(n_players):
            self.reasoning_log[i] = []
        
        # Reset run history for new game
        if "dashboard_run_history" in st.session_state:
            st.session_state.dashboard_run_history = []
        self.run_history = st.session_state.dashboard_run_history

    def _get_chart_containers(self):
        """Get or create chart containers for this dashboard instance.
        
        Returns:
            Dict[str, streamlit.container]: Dictionary of chart containers
        """
        # Use session state to store containers per dashboard instance
        containers_key = f"dashboard_containers_{self.dashboard_id}"
        
        # Check if containers already exist in session state
        if containers_key not in st.session_state:
            # Create containers for each chart type
            # IMPORTANT: Order matters! Containers render in creation order.
            # bar_race must be FIRST to appear at the top of the GUI
            st.session_state[containers_key] = {
                "bar_race": st.empty(),  # FIRST - appears at top
                "header": st.empty(),
                "resource": st.empty(),
                "extraction": st.empty(),
                "payoff": st.empty(),
                "cooperation": st.empty(),
            }
        
        return st.session_state[containers_key]
    
    def update(self, game_state: Dict):
        """Update dashboard with new game state.

        Args:
            game_state: Current game state dictionary
        """
        # Get or create chart containers
        self._chart_containers = self._get_chart_containers()
        
        # Reset radio button creation flag at start of each update
        # This ensures it can be created in new Streamlit executions
        # but only once per execution even if update() is called multiple times
        if hasattr(self, '_radio_created'):
            # Check if this is a new Streamlit execution by checking session state
            # If the key doesn't exist in session state, it's a new execution
            view_mode_key = f"cpr_history_view_mode_{self.dashboard_id}"
            if view_mode_key not in st.session_state:
                self._radio_created = False
        
        # Extract data
        self.resource_history = game_state.get("resource_history", [])
        self.extraction_history = game_state.get("extraction_history", [])
        self.payoff_history = game_state.get("payoff_history", [])
        self.cooperation_history = game_state.get("cooperation_history", [])
        
        # Update run history with new round data
        # Note: resource_history[0] is initial resource, resource_history[1] is after round 1, etc.
        # extraction_history[0] is round 1 extractions, extraction_history[1] is round 2, etc.
        
        # Ensure we're working with the latest session state
        # Handle case where session state might not be available
        try:
            self.run_history = st.session_state.dashboard_run_history
        except (KeyError, AttributeError):
            # If session state not available, ensure we have a local list
            if not hasattr(self, 'run_history') or self.run_history is None:
                self.run_history = []
        
        if len(self.extraction_history) > len(self.run_history):
            # New round data available - process all missing rounds
            while len(self.extraction_history) > len(self.run_history):
                round_num = len(self.run_history)  # 0-indexed round number
                
                # Get extractions and payoffs for this round
                if round_num < len(self.extraction_history):
                    extractions = self.extraction_history[round_num]
                    # Handle both list and numpy array
                    if isinstance(extractions, np.ndarray):
                        extractions = extractions.tolist()
                    elif isinstance(extractions, list):
                        # Already a list, ensure it's a proper list
                        extractions = list(extractions)
                    else:
                        # Try to convert if iterable
                        extractions = list(extractions) if hasattr(extractions, '__iter__') else []
                else:
                    extractions = []
                
                if round_num < len(self.payoff_history):
                    payoffs = self.payoff_history[round_num]
                    if isinstance(payoffs, np.ndarray):
                        payoffs = payoffs.tolist()
                    elif isinstance(payoffs, list):
                        payoffs = list(payoffs)
                    else:
                        payoffs = list(payoffs) if hasattr(payoffs, '__iter__') else []
                else:
                    payoffs = []
                
                # Resource: history[0] is initial, history[1] is after round 1
                # So for round 0 (first round), before=history[0], after=history[1]
                resource_before = self.resource_history[round_num] if round_num < len(self.resource_history) else 0.0
                resource_after = self.resource_history[round_num + 1] if (round_num + 1) < len(self.resource_history) else resource_before
                
                # Cooperation index
                cooperation = self.cooperation_history[round_num] if round_num < len(self.cooperation_history) else 0.0
                
                # Get reasoning for this round if available
                round_reasoning = {}
                for player_id in self.reasoning_log:
                    if len(self.reasoning_log[player_id]) > round_num:
                        round_reasoning[player_id] = self.reasoning_log[player_id][round_num]
                
                round_data = {
                    "round": round_num + 1,  # 1-indexed for display
                    "resource_before": resource_before,
                    "resource_after": resource_after,
                    "extractions": extractions,
                    "payoffs": payoffs,
                    "cooperation_index": cooperation,
                    "reasoning": round_reasoning,
                }
                # Append directly to session state list (they reference the same object)
                # Handle case where session state might not be available
                try:
                    st.session_state.dashboard_run_history.append(round_data)
                except (KeyError, AttributeError):
                    # If session state not available, append to local list
                    self.run_history.append(round_data)
            
            # Re-sync reference to ensure we're using the latest
            # Handle case where session state might not be available
            try:
                self.run_history = st.session_state.dashboard_run_history
            except (KeyError, AttributeError):
                # If session state not available, keep using local list
                pass

        # Bar chart race for cumulative payoffs (full width) - ALWAYS FIRST IN GUI
        # Render this BEFORE header to ensure it's the first visual element
        if len(self.payoff_history) > 0:
            self._render_bar_chart_race(container=self._chart_containers.get("bar_race") if self._chart_containers else None)

        # Header metrics - render in container to replace instead of append
        header_container = self._chart_containers.get("header") if self._chart_containers else None
        if header_container:
            # Clear and render header in container to replace instead of append
            with header_container:
                self._render_header(game_state)
        else:
            self._render_header(game_state)
        
        if len(self.resource_history) > 0:
            # Organize charts in columns for better layout
            # Columns structure is stable, chart content updates via containers
            col1, col2 = st.columns(2)
            
            with col1:
                if self.config["plots"]["resource_over_time"]:
                    self._render_resource_chart(container=self._chart_containers.get("resource") if self._chart_containers else None)

                if self.config["plots"]["cooperation_index"] and len(self.cooperation_history) > 0:
                    self._render_cooperation_chart(container=self._chart_containers.get("cooperation") if self._chart_containers else None)

            with col2:
                if self.config["plots"]["individual_extractions"] and len(self.extraction_history) > 0:
                    self._render_extraction_chart(container=self._chart_containers.get("extraction") if self._chart_containers else None)

        # Only show summary and tabs when game is done
        if game_state.get("done", False):
            # Create tabs for different views
            tab1, tab2 = st.tabs(["📊 Charts & Metrics", "💭 Reasoning Log"])

            with tab1:
                # Summary
                summary = self._calculate_summary(game_state)
                self.show_summary(summary)

            with tab2:
                # Reasoning log
                if self.config["plots"]["reasoning_log"]:
                    self._render_reasoning_log()
                else:
                    st.info("Reasoning log is disabled in configuration.")

    def _render_header(self, game_state: Dict):
        """Render header with key metrics.

        Args:
            game_state: Current game state
        """
        current_resource = game_state.get("resource", 0)
        current_step = game_state.get("step", 0)
        max_steps = game_state.get("max_steps", self.config["max_steps"])
        done = game_state.get("done", False)
        cumulative_payoffs = game_state.get("cumulative_payoffs", [])

        # Status indicator
        if done:
            if current_resource <= 0:
                status = "🔴 **Resource Depleted**"
                status_color = "red"
            else:
                status = "✅ **Game Complete**"
                status_color = "green"
        else:
            status = "🟢 **Running**"
            status_color = "green"

        st.markdown(f"### {status}")

        # Metrics row
        cols = st.columns(4)

        with cols[0]:
            st.metric(
                "Resource Level",
                f"{int(current_resource)}",
                delta=None
            )

        with cols[1]:
            st.metric(
                "Round",
                f"{current_step} / {max_steps}",
                delta=None
            )

        with cols[2]:
            threshold = self.config.get("sustainability_threshold", 0)
            if threshold > 0:
                pct = (current_resource / threshold * 100)
                st.metric(
                    "Sustainability",
                    f"{pct:.1f}%",
                    delta=None
                )
            else:
                st.metric(
                    "Sustainability",
                    "N/A",
                    delta=None
                )

        with cols[3]:
            if len(self.cooperation_history) > 0:
                avg_coop = np.mean(self.cooperation_history)
                st.metric(
                    "Avg Cooperation",
                    f"{avg_coop:.3f}",
                    delta=None
                )

        # Player payoffs
        if len(cumulative_payoffs) > 0:
            st.markdown("#### Player Standings")
            payoff_cols = st.columns(len(cumulative_payoffs))
            for i, payoff in enumerate(cumulative_payoffs):
                with payoff_cols[i]:
                    st.metric(f"Player {i}", f"{payoff:.1f}")

        st.divider()

    def _render_resource_chart(self, container=None):
        """Render resource over time chart.
        
        Args:
            container: Optional Streamlit container to render into. If None, renders directly.
        """
        if len(self.resource_history) == 0:
            if container:
                container.info("No data yet...")
            else:
                st.info("No data yet...")
            return

        fig = go.Figure()

        # Resource line
        fig.add_trace(go.Scatter(
            x=list(range(len(self.resource_history))),
            y=self.resource_history,
            mode='lines+markers',
            name='Resource Level',
            line=dict(color=self.resource_color, width=3),
            marker=dict(size=6)
        ))

        # Sustainability threshold (if defined)
        threshold = self.config.get("sustainability_threshold", 0)
        if threshold > 0:
            fig.add_trace(go.Scatter(
                x=[0, len(self.resource_history) - 1],
                y=[threshold, threshold],
                mode='lines',
                name='Sustainability Threshold',
                line=dict(color=self.threshold_color, width=2, dash='dash')
            ))

        fig.update_layout(
            title="Resource Level Over Time",
            xaxis_title="Round",
            yaxis_title="Resource Amount",
            height=self.chart_height,
            showlegend=True,
            hovermode='x unified'
        )

        # Use container if provided, otherwise render directly (for backward compatibility)
        if container:
            container.plotly_chart(fig, width='stretch')
        else:
            st.plotly_chart(fig, width='stretch', key=f"{self.dashboard_id}_resource_chart")

    def _render_extraction_chart(self, container=None):
        """Render player extractions over time.
        
        Args:
            container: Optional Streamlit container to render into. If None, renders directly.
        """
        if len(self.extraction_history) == 0:
            if container:
                container.info("No data yet...")
            else:
                st.info("No data yet...")
            return

        fig = go.Figure()

        # Convert to array
        extractions = np.array(self.extraction_history)
        n_players = extractions.shape[1]

        # One line per player
        for i in range(n_players):
            fig.add_trace(go.Scatter(
                x=list(range(len(extractions))),
                y=extractions[:, i],
                mode='lines+markers',
                name=f'Player {i}',
                line=dict(color=self.player_colors[i % len(self.player_colors)], width=2),
                marker=dict(size=5)
            ))

        fig.update_layout(
            title="Player Extractions Over Time",
            xaxis_title="Round",
            yaxis_title="Extraction Amount",
            height=self.chart_height,
            showlegend=True,
            hovermode='x unified'
        )

        # Use container if provided, otherwise render directly (for backward compatibility)
        if container:
            container.plotly_chart(fig, width='stretch')
        else:
            st.plotly_chart(fig, width='stretch', key=f"{self.dashboard_id}_extraction_chart")

    def _render_payoff_chart(self, container=None):
        """Render cumulative payoffs as a bar chart showing final summary.
        
        Args:
            container: Optional Streamlit container to render into. If None, renders directly.
        """
        if len(self.payoff_history) == 0:
            if container:
                container.info("No data yet...")
            else:
                st.info("No data yet...")
            return

        fig = go.Figure()

        # Calculate cumulative payoffs
        payoffs = np.array(self.payoff_history)
        cumulative = np.cumsum(payoffs, axis=0)
        n_players = cumulative.shape[1]

        # Get final cumulative payoffs (summary of all rounds)
        final_cumulative = cumulative[-1, :] if len(cumulative) > 0 else np.zeros(n_players)
        
        # Create bar chart with one bar per player
        player_names = [f'Player {i}' for i in range(n_players)]
        colors = [self.player_colors[i % len(self.player_colors)] for i in range(n_players)]
        
        fig.add_trace(go.Bar(
            x=player_names,
            y=final_cumulative,
            name='Cumulative Payoff',
            marker_color=colors,
            text=[f'{val:.1f}' for val in final_cumulative],
            textposition='outside'
        ))

        fig.update_layout(
            title="Cumulative Payoffs Summary (All Rounds)",
            xaxis_title="Player",
            yaxis_title="Total Earnings",
            height=self.chart_height,
            showlegend=False,
            hovermode='x'
        )

        # Use container if provided, otherwise render directly (for backward compatibility)
        if container:
            container.plotly_chart(fig, width='stretch')
        else:
            st.plotly_chart(fig, width='stretch', key=f"{self.dashboard_id}_payoff_chart")

    def _render_cooperation_chart(self, container=None):
        """Render cooperation index over time.
        
        Args:
            container: Optional Streamlit container to render into. If None, renders directly.
        """
        if len(self.cooperation_history) == 0:
            if container:
                container.info("No data yet...")
            else:
                st.info("No data yet...")
            return

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=list(range(len(self.cooperation_history))),
            y=self.cooperation_history,
            mode='lines+markers',
            name='Cooperation Index',
            line=dict(color='#9B59B6', width=3),
            marker=dict(size=6),
            fill='tozeroy',
            fillcolor='rgba(155, 89, 182, 0.2)'
        ))

        fig.update_layout(
            title="Cooperation Index Over Time",
            xaxis_title="Round",
            yaxis_title="Cooperation (0-1)",
            height=self.chart_height,
            yaxis_range=[0, 1],
            showlegend=False,
            hovermode='x unified'
        )

        # Use container if provided, otherwise render directly (for backward compatibility)
        if container:
            container.plotly_chart(fig, width='stretch')
        else:
            st.plotly_chart(fig, width='stretch', key=f"{self.dashboard_id}_cooperation_chart")

    def _render_reasoning_log(self):
        """Render LLM reasoning log."""
        st.markdown("### 💭 LLM Reasoning Log")

        if not self.reasoning_log or all(len(reasons) == 0 for reasons in self.reasoning_log.values()):
            st.info("No reasoning data available yet...")
            return

        # Create tabs for each player
        n_players = len(self.reasoning_log)
        tabs = st.tabs([f"Player {i}" for i in range(n_players)])

        for i, tab in enumerate(tabs):
            with tab:
                if len(self.reasoning_log[i]) == 0:
                    st.info(f"No reasoning from Player {i} yet...")
                else:
                    # Show last few reasoning entries
                    for round_num, reasoning in enumerate(self.reasoning_log[i][-10:]):
                        with st.expander(f"Round {len(self.reasoning_log[i]) - 10 + round_num}", expanded=(round_num == len(self.reasoning_log[i][-10:]) - 1)):
                            st.write(reasoning)

    def add_reasoning(self, player_id: int, reasoning: str):
        """Add reasoning text for a player.

        Args:
            player_id: Player identifier
            reasoning: Reasoning text from LLM
        """
        if player_id not in self.reasoning_log:
            self.reasoning_log[player_id] = []
        self.reasoning_log[player_id].append(reasoning)
    
    def _render_run_history(self):
        """Render detailed history of all rounds showing what each player did."""
        st.markdown("## 📜 Run History")
        
        # Ensure we're using the latest session state
        # Handle case where session state might not be available
        try:
            self.run_history = st.session_state.dashboard_run_history
        except (KeyError, AttributeError):
            # If session state not available, ensure we have a local list
            if not hasattr(self, 'run_history') or self.run_history is None:
                self.run_history = []
        
        if len(self.run_history) == 0:
            st.info("No history available yet. The game will populate this as it progresses.")
            return
        
        # Show summary stats
        n_rounds = len(self.run_history)
        st.markdown(f"**Total Rounds:** {n_rounds}")
        
        # Option to view as table or detailed view
        # Use a unique key that includes the dashboard_id to prevent duplicates
        # when update() is called multiple times in the same execution
        view_mode_key = f"cpr_history_view_mode_{self.dashboard_id}"
        
        # Initialize default value if not set
        if view_mode_key not in st.session_state:
            st.session_state[view_mode_key] = "Table View"
        
        # Get current value from session state
        current_mode = st.session_state.get(view_mode_key, "Table View")
        default_index = 0 if current_mode == "Table View" else 1
        
        # Only create the radio button once per Streamlit script execution
        # Use an instance variable to track if we've created it in this execution
        if not hasattr(self, '_radio_created'):
            self._radio_created = False
        
        # Only create the radio button once per Streamlit script execution
        # Check if we've already created it in this execution
        if not self._radio_created:
            try:
                view_mode = st.radio(
                    "View Mode",
                    ["Table View", "Detailed View"],
                    horizontal=True,
                    key=view_mode_key,
                    index=default_index
                )
                self._radio_created = True
            except StreamlitDuplicateElementKey:
                # Widget already exists (created in a previous call in same execution)
                # Just get the value from session state
                view_mode = st.session_state.get(view_mode_key, "Table View")
                self._radio_created = True
        else:
            # Radio button already created, just get the value from session state
            view_mode = st.session_state.get(view_mode_key, "Table View")
        
        if view_mode == "Table View":
            self._render_history_table()
        else:
            self._render_history_detailed()
    
    def _render_history_table(self):
        """Render history as a compact table."""
        # Ensure we're using the latest session state
        # Handle case where session state might not be available
        try:
            self.run_history = st.session_state.dashboard_run_history
        except (KeyError, AttributeError):
            # If session state not available, ensure we have a local list
            if not hasattr(self, 'run_history') or self.run_history is None:
                self.run_history = []
        
        if len(self.run_history) == 0:
            return
        
        # Build table data
        table_data = []
        first_round = self.run_history[0]
        n_players = len(first_round.get("extractions", [])) if first_round.get("extractions") else 0
        
        if n_players == 0:
            st.warning("No player data available in history.")
            return
        
        for round_data in self.run_history:
            extractions = round_data.get("extractions", [])
            payoffs = round_data.get("payoffs", [])
            
            row = {
                "Round": round_data["round"],
                "Resource Before": f"{int(round_data['resource_before'])}",
                "Resource After": f"{int(round_data['resource_after'])}",
                "Cooperation": f"{round_data['cooperation_index']:.3f}",
            }
            
            # Add player extractions
            for i in range(n_players):
                extraction = extractions[i] if i < len(extractions) else 0.0
                payoff = payoffs[i] if i < len(payoffs) else 0.0
                row[f"P{i} Extract"] = f"{extraction:.2f}"
                row[f"P{i} Payoff"] = f"{payoff:.2f}"
            
            table_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(table_data)
        
        # Display with scrolling
        st.dataframe(
            df,
            width='stretch',
            height=min(400, 50 + len(df) * 35),
            hide_index=True
        )
    
    def _render_history_detailed(self):
        """Render history with expandable sections for each round."""
        # Ensure we're using the latest session state
        # Handle case where session state might not be available
        try:
            self.run_history = st.session_state.dashboard_run_history
        except (KeyError, AttributeError):
            # If session state not available, ensure we have a local list
            if not hasattr(self, 'run_history') or self.run_history is None:
                self.run_history = []
        
        if len(self.run_history) == 0:
            return
        
        # Show most recent rounds first
        reversed_history = list(reversed(self.run_history))
        
        # Limit display to last 50 rounds to avoid performance issues
        display_history = reversed_history[:50]
        
        if len(self.run_history) > 50:
            st.info(f"Showing last 50 rounds (out of {len(self.run_history)} total). Use Table View to see all rounds.")
        
        for round_data in display_history:
            round_num = round_data["round"]
            extractions = round_data.get("extractions", [])
            payoffs = round_data.get("payoffs", [])
            n_players = len(extractions) if extractions else 0
            
            if n_players == 0:
                continue
            
            # Create expandable section
            with st.expander(
                f"Round {round_num} - Resource: {int(round_data['resource_before'])} → {int(round_data['resource_after'])} | "
                f"Cooperation: {round_data['cooperation_index']:.3f}",
                expanded=(round_data == display_history[0])  # Expand most recent
            ):
                # Resource info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Resource Before", f"{int(round_data['resource_before'])}")
                with col2:
                    st.metric("Resource After", f"{int(round_data['resource_after'])}")
                with col3:
                    resource_change = round_data['resource_after'] - round_data['resource_before']
                    st.metric("Resource Change", f"{resource_change:+d}")
                
                st.divider()
                
                # Player actions
                st.markdown("#### Player Actions")
                player_cols = st.columns(n_players)
                
                for i in range(n_players):
                    with player_cols[i]:
                        st.markdown(f"**Player {i}**")
                        extraction = extractions[i] if i < len(extractions) else 0.0
                        payoff = payoffs[i] if i < len(payoffs) else 0.0
                        st.metric("Extraction", f"{extraction:.2f}")
                        st.metric("Payoff", f"{payoff:.2f}")
                        
                        # Show reasoning if available
                        if i in round_data.get("reasoning", {}):
                            with st.expander("Reasoning"):
                                st.write(round_data["reasoning"][i])
                
                # Summary stats
                st.divider()
                total_extraction = sum(extractions) if extractions else 0.0
                total_payoff = sum(payoffs) if payoffs else 0.0
                st.markdown(f"**Total Extraction:** {total_extraction:.2f} | **Total Payoff:** {total_payoff:.2f}")

    def _calculate_summary(self, game_state: Dict) -> Dict:
        """Calculate summary statistics from game state.
        
        Args:
            game_state: Current game state dictionary
            
        Returns:
            Dictionary with summary statistics
        """
        cumulative_payoffs = game_state.get("cumulative_payoffs", [])
        # Check if cumulative_payoffs is empty (works for both list and numpy array)
        is_empty = (cumulative_payoffs is None or 
                   len(cumulative_payoffs) == 0 or 
                   (isinstance(cumulative_payoffs, np.ndarray) and cumulative_payoffs.size == 0))
        if is_empty and len(self.payoff_history) > 0:
            # Calculate cumulative payoffs from history
            payoffs = np.array(self.payoff_history)
            cumulative_payoffs = np.cumsum(payoffs, axis=0)
            if len(cumulative_payoffs) > 0:
                cumulative_payoffs = cumulative_payoffs[-1, :].tolist()
            else:
                cumulative_payoffs = []
        
        final_resource = game_state.get("resource", 0)
        total_rounds = len(self.extraction_history)
        avg_cooperation = np.mean(self.cooperation_history) if len(self.cooperation_history) > 0 else 0.0
        
        # Calculate Gini coefficient for payoff inequality
        gini = 0.0
        if len(cumulative_payoffs) > 1 and sum(cumulative_payoffs) > 0:
            sorted_payoffs = sorted(cumulative_payoffs)
            n = len(sorted_payoffs)
            gini = (n + 1 - 2 * sum((n + 1 - i) * y for i, y in enumerate(sorted_payoffs, 1))) / (n * sum(sorted_payoffs))
        
        threshold = self.config.get("sustainability_threshold", 0)
        sustainability_score = (final_resource / threshold) if threshold > 0 else 0.0
        
        return {
            "total_rounds": total_rounds,
            "final_resource_level": final_resource,
            "tragedy_occurred": final_resource <= 0,
            "avg_cooperation_index": avg_cooperation,
            "gini_coefficient": gini,
            "sustainability_score": sustainability_score,
            "cumulative_payoffs": cumulative_payoffs,
        }
    
    def _render_bar_chart_race(self, container=None):
        """Render animated bar chart race showing cumulative payoffs over time.
        
        Args:
            container: Optional Streamlit container to render into. If None, renders directly.
        """
        if len(self.payoff_history) == 0:
            if container:
                container.info("No data yet...")
            return
        
        # Calculate cumulative payoffs over time
        payoffs = np.array(self.payoff_history)
        cumulative = np.cumsum(payoffs, axis=0)
        n_players = cumulative.shape[1]
        n_rounds = cumulative.shape[0]
        
        # Create data for animation - each frame is a round
        frames = []
        for round_num in range(n_rounds):
            # Sort players by cumulative payoff for this round
            round_payoffs = cumulative[round_num, :]
            sorted_indices = np.argsort(round_payoffs)
            
            frame_data = go.Frame(
                data=[
                    go.Bar(
                        x=[round_payoffs[i] for i in sorted_indices],
                        y=[f"Player {i}" for i in sorted_indices],
                        orientation='h',
                        marker_color=[self.player_colors[i % len(self.player_colors)] for i in sorted_indices],
                        text=[f"{round_payoffs[i]:.1f}" for i in sorted_indices],
                        textposition='outside',
                        name=f"Round {round_num + 1}"
                    )
                ],
                name=f"round_{round_num}"
            )
            frames.append(frame_data)
        
        # Initial data (latest round to show current state)
        if n_rounds > 0:
            # Show latest round by default
            latest_payoffs = cumulative[-1, :]
            sorted_indices = np.argsort(latest_payoffs)
            initial_data = [
                go.Bar(
                    x=[latest_payoffs[i] for i in sorted_indices],
                    y=[f"Player {i}" for i in sorted_indices],
                    orientation='h',
                    marker_color=[self.player_colors[i % len(self.player_colors)] for i in sorted_indices],
                    text=[f"{latest_payoffs[i]:.1f}" for i in sorted_indices],
                    textposition='outside',
                )
            ]
        else:
            initial_data = []
        
        # Create figure with animation
        fig = go.Figure(
            data=initial_data,
            frames=frames
        )
        
        # Add play button and slider
        fig.update_layout(
            title="Cumulative Payoffs Race Over Time",
            xaxis_title="Cumulative Payoff",
            yaxis_title="Player",
            height=self.chart_height,
            showlegend=False,
            updatemenus=[{
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {
                            "frame": {"duration": 500, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 300}
                        }]
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0}
                        }]
                    }
                ]
            }],
            sliders=[{
                "active": n_rounds - 1 if n_rounds > 0 else 0,  # Always show latest round
                "steps": [
                    {
                        "args": [[f"round_{i}"], {
                            "frame": {"duration": 300, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 300}
                        }],
                        "label": f"Round {i+1}",
                        "method": "animate"
                    }
                    for i in range(n_rounds)
                ],
                "transition": {"duration": 300},
                "x": 0.1,
                "len": 0.9,
                "xanchor": "left",
                "y": 0,
                "yanchor": "top",
                "pad": {"b": 10, "t": 50}
            }]
        )
        
        # Use container if provided, otherwise render directly
        # ALWAYS use 100% width
        if container:
            container.plotly_chart(fig, width='stretch')
        else:
            st.plotly_chart(fig, width='stretch', key=f"{self.dashboard_id}_bar_race")
    
    def show_summary(self, summary: Dict):
        """Display game summary statistics.

        Args:
            summary: Summary statistics dictionary
        """
        st.markdown("---")
        st.markdown("## 📊 Game Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Rounds", summary.get("total_rounds", 0))
            st.metric(
                "Final Resource",
                f"{int(summary.get('final_resource_level', 0))}"
            )

        with col2:
            tragedy = summary.get("tragedy_occurred", False)
            st.metric(
                "Tragedy Occurred",
                "Yes" if tragedy else "No"
            )
            st.metric(
                "Sustainability Score",
                f"{summary.get('sustainability_score', 0):.1%}"
            )

        with col3:
            st.metric(
                "Avg Cooperation",
                f"{summary.get('avg_cooperation_index', 0):.3f}"
            )
            st.metric(
                "Payoff Inequality (Gini)",
                f"{summary.get('gini_coefficient', 0):.3f}"
            )

        # Player payoffs table
        if "cumulative_payoffs" in summary:
            st.markdown("### Final Payoffs")
            payoff_data = {
                f"Player {i}": [payoff]
                for i, payoff in enumerate(summary["cumulative_payoffs"])
            }
            df = pd.DataFrame(payoff_data)
            st.dataframe(df, width='stretch')


def create_static_report(game_state: Dict, summary: Dict, output_path: str = "cpr_report.html"):
    """Create a static HTML report of the game.

    Args:
        game_state: Game state dictionary
        summary: Summary statistics
        output_path: Path to save HTML report
    """
    import plotly.io as pio

    # Create figures
    resource_history = game_state.get("resource_history", [])
    extraction_history = game_state.get("extraction_history", [])
    cooperation_history = game_state.get("cooperation_history", [])

    figs = []

    # Resource chart
    if len(resource_history) > 0:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(resource_history))),
            y=resource_history,
            mode='lines+markers',
            name='Resource Level'
        ))
        fig.update_layout(title="Resource Over Time", xaxis_title="Round", yaxis_title="Resource")
        figs.append(fig)

    # Extractions chart
    if len(extraction_history) > 0:
        fig = go.Figure()
        extractions = np.array(extraction_history)
        for i in range(extractions.shape[1]):
            fig.add_trace(go.Scatter(
                x=list(range(len(extractions))),
                y=extractions[:, i],
                mode='lines+markers',
                name=f'Player {i}'
            ))
        fig.update_layout(title="Extractions Over Time", xaxis_title="Round", yaxis_title="Extraction")
        figs.append(fig)

    # Save to HTML
    html_content = "<html><head><title>CPR Game Report</title></head><body>"
    html_content += "<h1>Common Pool Resource Game Report</h1>"

    # Summary
    html_content += "<h2>Summary</h2>"
    html_content += f"<p>Total Rounds: {summary.get('total_rounds', 0)}</p>"
    html_content += f"<p>Final Resource: {summary.get('final_resource_level', 0):.2f}</p>"
    html_content += f"<p>Tragedy Occurred: {'Yes' if summary.get('tragedy_occurred') else 'No'}</p>"

    # Charts
    for fig in figs:
        html_content += pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

    html_content += "</body></html>"

    with open(output_path, 'w') as f:
        f.write(html_content)

    print(f"Report saved to {output_path}")
