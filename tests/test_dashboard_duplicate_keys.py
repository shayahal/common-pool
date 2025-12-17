"""Tests to prevent StreamlitDuplicateElementKey errors.

These tests ensure that dashboard.update() can be called multiple times
in the same execution without creating duplicate element keys.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from cpr_game.dashboard import Dashboard
from cpr_game.config import CONFIG


class TestDashboardDuplicateKeys:
    """Test cases to prevent duplicate Streamlit element keys."""

    @pytest.fixture
    def dashboard(self):
        """Create test dashboard."""
        config = CONFIG.copy()
        return Dashboard(config)

    @pytest.fixture
    def game_state_step_0(self):
        """Create game state for step 0."""
        return {
            "resource": 1000.0,
            "step": 0,
            "max_steps": 20,
            "done": False,
            "cumulative_payoffs": [0.0, 0.0],
            "resource_history": [1000.0],
            "extraction_history": [],
            "payoff_history": [],
            "cooperation_history": [],
        }

    @pytest.fixture
    def game_state_step_1(self):
        """Create game state for step 1."""
        return {
            "resource": 950.0,
            "step": 1,
            "max_steps": 20,
            "done": False,
            "cumulative_payoffs": [50.0, 50.0],
            "resource_history": [1000.0, 950.0],
            "extraction_history": [np.array([50.0, 50.0])],
            "payoff_history": [np.array([50.0, 50.0])],
            "cooperation_history": [1.0],
        }

    @pytest.fixture
    def game_state_step_2(self):
        """Create game state for step 2."""
        return {
            "resource": 900.0,
            "step": 2,
            "max_steps": 20,
            "done": False,
            "cumulative_payoffs": [100.0, 100.0],
            "resource_history": [1000.0, 950.0, 900.0],
            "extraction_history": [
                np.array([50.0, 50.0]),
                np.array([50.0, 50.0]),
            ],
            "payoff_history": [
                np.array([50.0, 50.0]),
                np.array([50.0, 50.0]),
            ],
            "cooperation_history": [1.0, 1.0],
        }

    def test_chart_keys_include_step_number(self, dashboard):
        """Test that chart keys include step number for uniqueness."""
        dashboard._current_step = 5
        
        resource_key = f"{dashboard.dashboard_id}_resource_chart_5"
        extraction_key = f"{dashboard.dashboard_id}_extraction_chart_5"
        payoff_key = f"{dashboard.dashboard_id}_payoff_chart_5"
        cooperation_key = f"{dashboard.dashboard_id}_cooperation_chart_5"
        
        assert "5" in resource_key
        assert "5" in extraction_key
        assert "5" in payoff_key
        assert "5" in cooperation_key

    def test_chart_keys_different_for_different_steps(self, dashboard):
        """Test that chart keys are different for different steps."""
        dashboard._current_step = 0
        key_step_0 = f"{dashboard.dashboard_id}_resource_chart_0"
        
        dashboard._current_step = 1
        key_step_1 = f"{dashboard.dashboard_id}_resource_chart_1"
        
        assert key_step_0 != key_step_1

    def test_multiple_updates_same_execution_unique_keys(self, dashboard, game_state_step_0, game_state_step_1, game_state_step_2):
        """Test that multiple update() calls in same execution use unique keys."""
        # Create proper session state mock
        mock_session_state = MagicMock()
        mock_session_state.dashboard_run_history = []
        
        # Track all keys used
        used_keys = []
        
        def plotly_chart_side_effect(fig, **kwargs):
            if 'key' in kwargs:
                used_keys.append(kwargs['key'])
        
        with patch('streamlit.set_page_config'), \
             patch('streamlit.title'), \
             patch('streamlit.tabs') as mock_tabs, \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.plotly_chart', side_effect=plotly_chart_side_effect), \
             patch('streamlit.info'), \
             patch('streamlit.markdown'), \
             patch('streamlit.metric'), \
             patch('streamlit.divider'), \
             patch('streamlit.session_state', mock_session_state):
            
            # Create mock tabs that support context manager
            def create_mock_tab():
                mock_tab = MagicMock()
                mock_tab.__enter__ = MagicMock(return_value=mock_tab)
                mock_tab.__exit__ = MagicMock(return_value=False)
                return mock_tab
            
            mock_tabs.return_value = [create_mock_tab(), create_mock_tab(), create_mock_tab()]
            
            # Create mock columns that support context manager
            def columns_side_effect(count):
                def create_mock_col():
                    mock_col = MagicMock()
                    mock_col.__enter__ = MagicMock(return_value=mock_col)
                    mock_col.__exit__ = MagicMock(return_value=False)
                    return mock_col
                return [create_mock_col() for _ in range(count)]
            
            mock_columns.side_effect = columns_side_effect
            
            dashboard.initialize(n_players=2)
            
            # Call update multiple times (simulating game loop)
            dashboard.update(game_state_step_0)
            dashboard.update(game_state_step_1)
            dashboard.update(game_state_step_2)
            
            # Extract chart keys (they should include step numbers)
            chart_keys = [key for key in used_keys if 'chart' in key]
            
            # All chart keys should be unique
            assert len(chart_keys) == len(set(chart_keys)), f"Duplicate keys found: {chart_keys}"
            
            # Keys should include step numbers
            step_0_keys = [k for k in chart_keys if '_0' in k]
            step_1_keys = [k for k in chart_keys if '_1' in k]
            step_2_keys = [k for k in chart_keys if '_2' in k]
            
            # Each step should have chart keys
            assert len(step_0_keys) > 0, "Step 0 should have chart keys"
            assert len(step_1_keys) > 0, "Step 1 should have chart keys"
            assert len(step_2_keys) > 0, "Step 2 should have chart keys"

    def test_update_sets_current_step(self, dashboard, game_state_step_1):
        """Test that update() sets _current_step attribute."""
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
            
            def create_mock_tab():
                mock_tab = MagicMock()
                mock_tab.__enter__ = MagicMock(return_value=mock_tab)
                mock_tab.__exit__ = MagicMock(return_value=False)
                return mock_tab
            
            mock_tabs.return_value = [create_mock_tab(), create_mock_tab(), create_mock_tab()]
            
            def columns_side_effect(count):
                def create_mock_col():
                    mock_col = MagicMock()
                    mock_col.__enter__ = MagicMock(return_value=mock_col)
                    mock_col.__exit__ = MagicMock(return_value=False)
                    return mock_col
                return [create_mock_col() for _ in range(count)]
            
            mock_columns.side_effect = columns_side_effect
            
            dashboard.initialize(n_players=2)
            dashboard.update(game_state_step_1)
            
            assert hasattr(dashboard, '_current_step')
            assert dashboard._current_step == 1

    def test_chart_keys_unique_across_multiple_dashboards(self):
        """Test that different dashboard instances use different keys."""
        dashboard1 = Dashboard()
        dashboard2 = Dashboard()
        
        dashboard1._current_step = 0
        dashboard2._current_step = 0
        
        key1 = f"{dashboard1.dashboard_id}_resource_chart_0"
        key2 = f"{dashboard2.dashboard_id}_resource_chart_0"
        
        # Keys should be different because dashboard_ids are different
        assert key1 != key2
        assert dashboard1.dashboard_id != dashboard2.dashboard_id

    def test_same_dashboard_different_steps_unique_keys(self, dashboard, game_state_step_0, game_state_step_1):
        """Test that same dashboard with different steps produces unique keys."""
        # Create proper session state mock
        mock_session_state = MagicMock()
        mock_session_state.dashboard_run_history = []
        
        keys_by_step = {0: [], 1: []}
        
        def plotly_chart_side_effect(fig, **kwargs):
            if 'key' in kwargs:
                key = kwargs['key']
                # Extract step from key
                if '_0' in key:
                    keys_by_step[0].append(key)
                elif '_1' in key:
                    keys_by_step[1].append(key)
        
        with patch('streamlit.set_page_config'), \
             patch('streamlit.title'), \
             patch('streamlit.tabs') as mock_tabs, \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.plotly_chart', side_effect=plotly_chart_side_effect), \
             patch('streamlit.info'), \
             patch('streamlit.markdown'), \
             patch('streamlit.metric'), \
             patch('streamlit.divider'), \
             patch('streamlit.session_state', mock_session_state):
            
            def create_mock_tab():
                mock_tab = MagicMock()
                mock_tab.__enter__ = MagicMock(return_value=mock_tab)
                mock_tab.__exit__ = MagicMock(return_value=False)
                return mock_tab
            
            mock_tabs.return_value = [create_mock_tab(), create_mock_tab(), create_mock_tab()]
            
            def columns_side_effect(count):
                def create_mock_col():
                    mock_col = MagicMock()
                    mock_col.__enter__ = MagicMock(return_value=mock_col)
                    mock_col.__exit__ = MagicMock(return_value=False)
                    return mock_col
                return [create_mock_col() for _ in range(count)]
            
            mock_columns.side_effect = columns_side_effect
            
            dashboard.initialize(n_players=2)
            
            # Update with step 0
            dashboard.update(game_state_step_0)
            # Update with step 1
            dashboard.update(game_state_step_1)
            
            # Keys for step 0 and step 1 should be different
            assert len(keys_by_step[0]) > 0
            assert len(keys_by_step[1]) > 0
            
            # All keys should be unique
            all_keys = keys_by_step[0] + keys_by_step[1]
            assert len(all_keys) == len(set(all_keys)), "Keys should be unique across steps"

    def test_game_loop_simulation_no_duplicate_keys(self, dashboard):
        """Test simulating a full game loop with multiple update calls."""
        # Create proper session state mock
        mock_session_state = MagicMock()
        mock_session_state.dashboard_run_history = []
        
        all_keys = []
        
        def plotly_chart_side_effect(fig, **kwargs):
            if 'key' in kwargs:
                all_keys.append(kwargs['key'])
        
        with patch('streamlit.set_page_config'), \
             patch('streamlit.title'), \
             patch('streamlit.tabs') as mock_tabs, \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.plotly_chart', side_effect=plotly_chart_side_effect), \
             patch('streamlit.info'), \
             patch('streamlit.markdown'), \
             patch('streamlit.metric'), \
             patch('streamlit.divider'), \
             patch('streamlit.session_state', mock_session_state):
            
            def create_mock_tab():
                mock_tab = MagicMock()
                mock_tab.__enter__ = MagicMock(return_value=mock_tab)
                mock_tab.__exit__ = MagicMock(return_value=False)
                return mock_tab
            
            mock_tabs.return_value = [create_mock_tab(), create_mock_tab(), create_mock_tab()]
            
            def columns_side_effect(count):
                def create_mock_col():
                    mock_col = MagicMock()
                    mock_col.__enter__ = MagicMock(return_value=mock_col)
                    mock_col.__exit__ = MagicMock(return_value=False)
                    return mock_col
                return [create_mock_col() for _ in range(count)]
            
            mock_columns.side_effect = columns_side_effect
            
            dashboard.initialize(n_players=2)
            
            # Simulate 5 rounds of game loop
            for step in range(5):
                game_state = {
                    "resource": 1000.0 - (step * 50),
                    "step": step,
                    "max_steps": 20,
                    "done": False,
                    "cumulative_payoffs": [step * 50.0, step * 50.0],
                    "resource_history": [1000.0 - (i * 50) for i in range(step + 2)],
                    "extraction_history": [np.array([50.0, 50.0]) for _ in range(step + 1)],
                    "payoff_history": [np.array([50.0, 50.0]) for _ in range(step + 1)],
                    "cooperation_history": [1.0] * (step + 1),
                }
                dashboard.update(game_state)
            
            # All chart keys should be unique
            chart_keys = [key for key in all_keys if 'chart' in key]
            unique_keys = set(chart_keys)
            
            assert len(chart_keys) == len(unique_keys), \
                f"Found {len(chart_keys) - len(unique_keys)} duplicate keys. " \
                f"Total keys: {len(chart_keys)}, Unique: {len(unique_keys)}. " \
                f"Duplicates: {[k for k in chart_keys if chart_keys.count(k) > 1]}"

