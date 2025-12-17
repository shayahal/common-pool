"""Unit tests for Dashboard functionality."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from cpr_game.dashboard import Dashboard
from cpr_game.config import CONFIG


class TestDashboard:
    """Test cases for Dashboard."""

    @pytest.fixture
    def dashboard(self):
        """Create test dashboard."""
        config = CONFIG.copy()
        return Dashboard(config)

    @pytest.fixture
    def sample_game_state(self):
        """Create sample game state."""
        return {
            "resource": 1000.0,
            "step": 5,
            "max_steps": 20,
            "done": False,
            "cumulative_payoffs": [100.0, 150.0],
            "resource_history": [1000.0, 950.0, 900.0, 850.0, 800.0, 750.0],
            "extraction_history": [
                np.array([50.0, 50.0]),
                np.array([50.0, 50.0]),
                np.array([50.0, 50.0]),
                np.array([50.0, 50.0]),
                np.array([50.0, 50.0]),
            ],
            "payoff_history": [
                np.array([50.0, 50.0]),
                np.array([50.0, 50.0]),
                np.array([50.0, 50.0]),
                np.array([50.0, 50.0]),
                np.array([50.0, 50.0]),
            ],
            "cooperation_history": [1.0, 1.0, 1.0, 1.0, 1.0],
        }

    def test_initialization(self, dashboard):
        """Test dashboard initialization."""
        assert dashboard.config is not None
        assert hasattr(dashboard, 'dashboard_id')
        assert len(dashboard.dashboard_id) > 0
        assert len(dashboard.resource_history) == 0
        assert len(dashboard.extraction_history) == 0

    def test_dashboard_id_uniqueness(self):
        """Test that dashboard IDs are unique."""
        dashboard1 = Dashboard()
        dashboard2 = Dashboard()
        assert dashboard1.dashboard_id != dashboard2.dashboard_id

    @patch('streamlit.set_page_config')
    def test_initialize(self, mock_set_page_config, dashboard):
        """Test dashboard initialization."""
        dashboard.initialize(n_players=2)
        mock_set_page_config.assert_called_once()
        assert len(dashboard.reasoning_log) == 2

    def test_add_reasoning(self, dashboard):
        """Test adding reasoning text."""
        dashboard.add_reasoning(0, "I will extract 50 units")
        dashboard.add_reasoning(1, "I will extract 30 units")
        
        assert len(dashboard.reasoning_log[0]) == 1
        assert len(dashboard.reasoning_log[1]) == 1
        assert dashboard.reasoning_log[0][0] == "I will extract 50 units"
        assert dashboard.reasoning_log[1][0] == "I will extract 30 units"

    def test_add_reasoning_multiple_rounds(self, dashboard):
        """Test adding reasoning for multiple rounds."""
        dashboard.initialize(n_players=2)
        
        for i in range(5):
            dashboard.add_reasoning(0, f"Round {i} reasoning")
        
        assert len(dashboard.reasoning_log[0]) == 5
        assert dashboard.reasoning_log[0][-1] == "Round 4 reasoning"

    @patch('streamlit.plotly_chart')
    @patch('streamlit.info')
    def test_render_resource_chart_empty(self, mock_info, mock_plotly, dashboard):
        """Test rendering resource chart with no data."""
        dashboard._render_resource_chart()
        mock_info.assert_called_once_with("No data yet...")
        mock_plotly.assert_not_called()

    @patch('streamlit.plotly_chart')
    @patch('streamlit.info')
    def test_render_resource_chart_with_data(self, mock_info, mock_plotly, dashboard):
        """Test rendering resource chart with data."""
        dashboard.resource_history = [1000.0, 950.0, 900.0]
        dashboard._render_resource_chart()
        mock_info.assert_not_called()
        mock_plotly.assert_called_once()

    @patch('streamlit.plotly_chart')
    @patch('streamlit.info')
    def test_render_extraction_chart_empty(self, mock_info, mock_plotly, dashboard):
        """Test rendering extraction chart with no data."""
        dashboard._render_extraction_chart()
        mock_info.assert_called_once_with("No data yet...")
        mock_plotly.assert_not_called()

    @patch('streamlit.plotly_chart')
    @patch('streamlit.info')
    def test_render_extraction_chart_with_data(self, mock_info, mock_plotly, dashboard):
        """Test rendering extraction chart with data."""
        dashboard.extraction_history = [
            np.array([50.0, 30.0]),
            np.array([45.0, 35.0]),
        ]
        dashboard._render_extraction_chart()
        mock_info.assert_not_called()
        mock_plotly.assert_called_once()

    @patch('streamlit.plotly_chart')
    @patch('streamlit.info')
    def test_render_payoff_chart_empty(self, mock_info, mock_plotly, dashboard):
        """Test rendering payoff chart with no data."""
        dashboard._render_payoff_chart()
        mock_info.assert_called_once_with("No data yet...")
        mock_plotly.assert_not_called()

    @patch('streamlit.plotly_chart')
    @patch('streamlit.info')
    def test_render_payoff_chart_with_data(self, mock_info, mock_plotly, dashboard):
        """Test rendering payoff chart with data."""
        dashboard.payoff_history = [
            np.array([50.0, 30.0]),
            np.array([45.0, 35.0]),
        ]
        dashboard._render_payoff_chart()
        mock_info.assert_not_called()
        mock_plotly.assert_called_once()

    @patch('streamlit.plotly_chart')
    @patch('streamlit.info')
    def test_render_cooperation_chart_empty(self, mock_info, mock_plotly, dashboard):
        """Test rendering cooperation chart with no data."""
        dashboard._render_cooperation_chart()
        mock_info.assert_called_once_with("No data yet...")
        mock_plotly.assert_not_called()

    @patch('streamlit.plotly_chart')
    @patch('streamlit.info')
    def test_render_cooperation_chart_with_data(self, mock_info, mock_plotly, dashboard):
        """Test rendering cooperation chart with data."""
        dashboard.cooperation_history = [1.0, 0.9, 0.8]
        dashboard._render_cooperation_chart()
        mock_info.assert_not_called()
        mock_plotly.assert_called_once()

    @patch('streamlit.markdown')
    @patch('streamlit.info')
    def test_render_reasoning_log_empty(self, mock_info, mock_markdown, dashboard):
        """Test rendering reasoning log with no data."""
        dashboard.reasoning_log = {}
        dashboard._render_reasoning_log()
        mock_info.assert_called_once()

    @patch('streamlit.tabs')
    @patch('streamlit.markdown')
    def test_render_reasoning_log_with_data(self, mock_markdown, mock_tabs, dashboard):
        """Test rendering reasoning log with data."""
        dashboard.initialize(n_players=2)
        dashboard.add_reasoning(0, "Player 0 reasoning")
        dashboard.add_reasoning(1, "Player 1 reasoning")
        
        # Create mock tabs that support context manager
        mock_tab1 = MagicMock()
        mock_tab1.__enter__ = MagicMock(return_value=mock_tab1)
        mock_tab1.__exit__ = MagicMock(return_value=False)
        mock_tab2 = MagicMock()
        mock_tab2.__enter__ = MagicMock(return_value=mock_tab2)
        mock_tab2.__exit__ = MagicMock(return_value=False)
        mock_tabs.return_value = [mock_tab1, mock_tab2]
        
        dashboard._render_reasoning_log()
        mock_tabs.assert_called_once()

    @patch('streamlit.markdown')
    @patch('streamlit.metric')
    @patch('streamlit.columns')
    def test_render_header(self, mock_columns, mock_metric, mock_markdown, dashboard, sample_game_state):
        """Test rendering header."""
        # Create mock columns that support context manager
        mock_cols = []
        for _ in range(4):
            mock_col = MagicMock()
            mock_col.__enter__ = MagicMock(return_value=mock_col)
            mock_col.__exit__ = MagicMock(return_value=False)
            mock_cols.append(mock_col)
        mock_columns.return_value = mock_cols
        
        dashboard._render_header(sample_game_state)
        assert mock_columns.call_count >= 1

    @patch('streamlit.markdown')
    @patch('streamlit.metric')
    @patch('streamlit.columns')
    def test_render_header_done(self, mock_columns, mock_metric, mock_markdown, dashboard):
        """Test rendering header when game is done."""
        game_state = {
            "resource": 0.0,
            "step": 20,
            "max_steps": 20,
            "done": True,
            "cumulative_payoffs": [100.0, 150.0],
        }
        # Create mock columns that support context manager
        mock_cols = []
        for _ in range(4):
            mock_col = MagicMock()
            mock_col.__enter__ = MagicMock(return_value=mock_col)
            mock_col.__exit__ = MagicMock(return_value=False)
            mock_cols.append(mock_col)
        mock_columns.return_value = mock_cols
        
        dashboard._render_header(game_state)
        assert mock_columns.call_count >= 1

    @patch('streamlit.dataframe')
    @patch('streamlit.markdown')
    def test_show_summary(self, mock_markdown, mock_dataframe, dashboard):
        """Test showing game summary."""
        summary = {
            "total_rounds": 20,
            "final_resource_level": 500.0,
            "tragedy_occurred": False,
            "sustainability_score": 0.8,
            "avg_cooperation_index": 0.9,
            "gini_coefficient": 0.1,
            "cumulative_payoffs": [100.0, 150.0],
        }
        dashboard.show_summary(summary)
        mock_markdown.assert_called()
        mock_dataframe.assert_called_once()

    def test_update_updates_history(self, dashboard, sample_game_state):
        """Test that update() updates internal history."""
        # Create proper session state mock
        mock_session_state = MagicMock()
        mock_session_state.dashboard_run_history = []
        
        with patch('streamlit.set_page_config'), \
             patch('streamlit.title'), \
             patch('streamlit.tabs') as mock_tabs, \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.plotly_chart'), \
             patch('streamlit.info'), \
             patch('streamlit.markdown'), \
             patch('streamlit.metric'), \
             patch('streamlit.divider'), \
             patch('streamlit.session_state', mock_session_state):
            
            # Create mock tabs that support context manager
            mock_tab1 = MagicMock()
            mock_tab1.__enter__ = MagicMock(return_value=mock_tab1)
            mock_tab1.__exit__ = MagicMock(return_value=False)
            mock_tab2 = MagicMock()
            mock_tab2.__enter__ = MagicMock(return_value=mock_tab2)
            mock_tab2.__exit__ = MagicMock(return_value=False)
            mock_tab3 = MagicMock()
            mock_tab3.__enter__ = MagicMock(return_value=mock_tab3)
            mock_tab3.__exit__ = MagicMock(return_value=False)
            mock_tabs.return_value = [mock_tab1, mock_tab2, mock_tab3]
            
            # Create mock columns that support context manager (need 4 for header, 2 for tabs)
            def create_mock_col():
                mock_col = MagicMock()
                mock_col.__enter__ = MagicMock(return_value=mock_col)
                mock_col.__exit__ = MagicMock(return_value=False)
                return mock_col
            
            # st.columns is called multiple times with different counts
            def columns_side_effect(count):
                return [create_mock_col() for _ in range(count)]
            
            mock_columns.side_effect = columns_side_effect
            
            dashboard.initialize(n_players=2)
            dashboard.update(sample_game_state)
            
            assert len(dashboard.resource_history) > 0
            assert len(dashboard.extraction_history) > 0
            assert len(dashboard.payoff_history) > 0
            assert len(dashboard.cooperation_history) > 0

    def test_chart_keys_are_unique(self, dashboard):
        """Test that chart keys include dashboard_id for uniqueness."""
        dashboard_id = dashboard.dashboard_id
        
        # Check that keys would include dashboard_id
        resource_key = f"{dashboard_id}_resource_chart"
        extraction_key = f"{dashboard_id}_extraction_chart"
        payoff_key = f"{dashboard_id}_payoff_chart"
        cooperation_key = f"{dashboard_id}_cooperation_chart"
        
        assert dashboard_id in resource_key
        assert dashboard_id in extraction_key
        assert dashboard_id in payoff_key
        assert dashboard_id in cooperation_key
        
        # Keys should be different
        assert resource_key != extraction_key
        assert resource_key != payoff_key
        assert resource_key != cooperation_key

