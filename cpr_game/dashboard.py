"""Streamlit dashboard for real-time CPR game visualization.

Displays resource dynamics, player actions, payoffs, cooperation metrics,
and LLM reasoning in an interactive dashboard.
"""

from typing import Dict, List, Optional
import streamlit as st
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

        # Display settings
        self.chart_height = self.config["chart_height"]
        self.chart_width = self.config["chart_width"]
        self.player_colors = self.config["player_colors"]
        self.resource_color = self.config["resource_color"]
        self.threshold_color = self.config["threshold_color"]

        # Game state tracking
        self.resource_history: List[float] = []
        self.extraction_history: List[np.ndarray] = []
        self.payoff_history: List[np.ndarray] = []
        self.cooperation_history: List[float] = []
        self.reasoning_log: Dict[int, List[str]] = {}

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

    def update(self, game_state: Dict):
        """Update dashboard with new game state.

        Args:
            game_state: Current game state dictionary
        """
        # Extract data
        self.resource_history = game_state.get("resource_history", [])
        self.extraction_history = game_state.get("extraction_history", [])
        self.payoff_history = game_state.get("payoff_history", [])
        self.cooperation_history = game_state.get("cooperation_history", [])

        # Header metrics
        self._render_header(game_state)

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            if self.config["plots"]["resource_over_time"]:
                self._render_resource_chart()

            if self.config["plots"]["cooperation_index"]:
                self._render_cooperation_chart()

        with col2:
            if self.config["plots"]["individual_extractions"]:
                self._render_extraction_chart()

            if self.config["plots"]["cumulative_payoffs"]:
                self._render_payoff_chart()

        # Reasoning log
        if self.config["plots"]["reasoning_log"]:
            self._render_reasoning_log()

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
                f"{current_resource:.1f}",
                delta=None
            )

        with cols[1]:
            st.metric(
                "Round",
                f"{current_step} / {max_steps}",
                delta=None
            )

        with cols[2]:
            threshold = self.config["sustainability_threshold"]
            pct = (current_resource / threshold * 100) if threshold > 0 else 0
            st.metric(
                "Sustainability",
                f"{pct:.1f}%",
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

    def _render_resource_chart(self):
        """Render resource over time chart."""
        if len(self.resource_history) == 0:
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

        # Sustainability threshold
        threshold = self.config["sustainability_threshold"]
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

        st.plotly_chart(fig, use_container_width=True)

    def _render_extraction_chart(self):
        """Render player extractions over time."""
        if len(self.extraction_history) == 0:
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

        st.plotly_chart(fig, use_container_width=True)

    def _render_payoff_chart(self):
        """Render cumulative payoffs over time."""
        if len(self.payoff_history) == 0:
            st.info("No data yet...")
            return

        fig = go.Figure()

        # Calculate cumulative payoffs
        payoffs = np.array(self.payoff_history)
        cumulative = np.cumsum(payoffs, axis=0)
        n_players = cumulative.shape[1]

        # One line per player
        for i in range(n_players):
            fig.add_trace(go.Scatter(
                x=list(range(len(cumulative))),
                y=cumulative[:, i],
                mode='lines+markers',
                name=f'Player {i}',
                line=dict(color=self.player_colors[i % len(self.player_colors)], width=2),
                marker=dict(size=5)
            ))

        fig.update_layout(
            title="Cumulative Payoffs Over Time",
            xaxis_title="Round",
            yaxis_title="Total Earnings",
            height=self.chart_height,
            showlegend=True,
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_cooperation_chart(self):
        """Render cooperation index over time."""
        if len(self.cooperation_history) == 0:
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

        st.plotly_chart(fig, use_container_width=True)

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
                f"{summary.get('final_resource_level', 0):.1f}"
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
            st.dataframe(df, use_container_width=True)


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
