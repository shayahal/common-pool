"""Helper script to query experiment tables directly.

Examples of how to select from the experiment tables.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from cpr_game.db_manager import DatabaseManager
from cpr_game.config import CONFIG
import pandas as pd

def query_experiments():
    """Query all experiments."""
    db_path = CONFIG.get("db_path", "data/game_results.db")
    db_manager = DatabaseManager(db_path=db_path, enabled=True)
    
    # Try to query with new schema (individual columns)
    try:
        query = """
            SELECT 
                e.experiment_id,
                e.name,
                e.status,
                e.created_at,
                e.n_players,
                e.max_steps,
                e.initial_resource,
                e.regeneration_rate,
                e.max_extraction,
                e.Max_fish,
                e.number_of_games,
                e.number_of_players_per_game,
                COUNT(DISTINCT ep.player_index) as player_count,
                COUNT(DISTINCT er.game_id) as game_count
            FROM experiments e
            LEFT JOIN experiment_players ep ON e.experiment_id = ep.experiment_id
            LEFT JOIN experiment_results er ON e.experiment_id = er.experiment_id
            GROUP BY e.experiment_id, e.name, e.status, e.created_at,
                     e.n_players, e.max_steps, e.initial_resource, e.regeneration_rate,
                     e.max_extraction, e.Max_fish, e.number_of_games, e.number_of_players_per_game
            ORDER BY e.created_at DESC
        """
        
        results = db_manager.query_results(query)
        
        print("=" * 80)
        print("EXPERIMENTS")
        print("=" * 80)
        
        if results:
            for idx, row in enumerate(results, 1):
                (exp_id, name, status, created_at, n_players, max_steps, 
                 initial_resource, regen_rate, max_extraction, Max_fish,
                 number_of_games, players_per_game, player_count, game_count) = row
                
                # Format created_at
                if created_at:
                    created_str = pd.to_datetime(created_at).strftime("%Y-%m-%d %H:%M:%S")
                else:
                    created_str = "N/A"
                
                # Status emoji
                status_emoji = {
                    "pending": "⏳",
                    "running": "🔄",
                    "completed": "✅",
                    "failed": "❌"
                }.get(status, "❓")
                
                print(f"\n{idx}. {status_emoji} {name}")
                print(f"   ID: {exp_id}")
                print(f"   Status: {status.upper():10s} | Created: {created_str}")
                print(f"   ──────────────────────────────────────────────────────────────")
                print(f"   Players: {player_count or 0:2d} in pool | {players_per_game:2d} per game | {game_count or 0:3d} games completed")
                print(f"   Game Params: {max_steps:3d} max steps | {initial_resource:4d} init resource | {regen_rate:.1f}x regen")
                print(f"              : {max_extraction:2d} max extraction | {Max_fish:4d} max capacity")
        else:
            print("No experiments found")
    except Exception as e:
        # Fallback to old schema with parameters JSON column
        print(f"Warning: Could not query with new schema, trying old schema: {e}")
        query = """
            SELECT 
                experiment_id,
                name,
                status,
                created_at,
                parameters
            FROM experiments
            ORDER BY created_at DESC
        """
        
        results = db_manager.query_results(query)
        
        print("=" * 80)
        print("EXPERIMENTS (old schema)")
        print("=" * 80)
        
        if results:
            for idx, row in enumerate(results, 1):
                exp_id, name, status, created_at, params_json = row
                
                if created_at:
                    created_str = pd.to_datetime(created_at).strftime("%Y-%m-%d %H:%M:%S")
                else:
                    created_str = "N/A"
                
                status_emoji = {
                    "pending": "⏳",
                    "running": "🔄",
                    "completed": "✅",
                    "failed": "❌"
                }.get(status, "❓")
                
                print(f"\n{idx}. {status_emoji} {name}")
                print(f"   ID: {exp_id} | Status: {status.upper()} | Created: {created_str}")
                if params_json:
                    print(f"   Parameters: {params_json[:100]}...")
        else:
            print("No experiments found")
    
    print("\n" + "=" * 80)


def query_experiment_players(experiment_id: str = None):
    """Query experiment players."""
    db_path = CONFIG.get("db_path", "data/game_results.db")
    db_manager = DatabaseManager(db_path=db_path, enabled=True)
    
    if experiment_id:
        query = """
            SELECT 
                experiment_id,
                player_index,
                persona,
                model
            FROM experiment_players
            WHERE experiment_id = ?
            ORDER BY player_index
        """
        results = db_manager.query_results(query, [experiment_id])
        print(f"=" * 60)
        print(f"EXPERIMENT PLAYERS: {experiment_id}")
        print("=" * 60)
    else:
        query = """
            SELECT 
                experiment_id,
                player_index,
                persona,
                model
            FROM experiment_players
            ORDER BY experiment_id, player_index
        """
        results = db_manager.query_results(query)
        print("=" * 60)
        print("ALL EXPERIMENT PLAYERS")
        print("=" * 60)
    
    if results:
        df = pd.DataFrame(results, columns=['experiment_id', 'player_index', 'persona', 'model'])
        print(df.to_string(index=False))
    else:
        print("No players found")
    
    print()


def cleanup_nan_results():
    """Delete rows from experiment_results that have NaN values in the new columns."""
    db_path = CONFIG.get("db_path", "data/game_results.db")
    db_manager = DatabaseManager(db_path=db_path, enabled=True)
    
    if not db_manager.enabled or db_manager.conn is None:
        print("Database is not enabled or not initialized")
        return
    
    try:
        # Find rows with NULL values in any of the new columns
        query = """
            SELECT COUNT(*) 
            FROM experiment_results
            WHERE winning_player_uuid IS NULL 
               OR winning_payoff IS NULL 
               OR cumulative_payoff_sum IS NULL 
               OR total_rounds IS NULL 
               OR final_resource_level IS NULL 
               OR tragedy_occurred IS NULL
        """
        count_result = db_manager.query_results(query)
        count = count_result[0][0] if count_result else 0
        
        if count == 0:
            print("No rows with NaN values found in experiment_results")
            return
        
        print(f"Found {count} row(s) with NaN values. Deleting...")
        
        # Delete rows with NULL values
        delete_query = """
            DELETE FROM experiment_results
            WHERE winning_player_uuid IS NULL 
               OR winning_payoff IS NULL 
               OR cumulative_payoff_sum IS NULL 
               OR total_rounds IS NULL 
               OR final_resource_level IS NULL 
               OR tragedy_occurred IS NULL
        """
        db_manager.conn.execute(delete_query)
        db_manager.conn.commit()
        
        print(f"✅ Successfully deleted {count} row(s) with NaN values")
        
    except Exception as e:
        print(f"❌ Error cleaning up NaN values: {e}")
        import traceback
        traceback.print_exc()


def query_experiment_results(experiment_id: str = None):
    """Query experiment results with improved formatting."""
    db_path = CONFIG.get("db_path", "data/game_results.db")
    db_manager = DatabaseManager(db_path=db_path, enabled=True)
    
    if experiment_id:
        query = """
            SELECT 
                experiment_id,
                game_id,
                winning_player_uuid,
                winning_payoff,
                cumulative_payoff_sum,
                total_rounds,
                final_resource_level,
                tragedy_occurred,
                timestamp
            FROM experiment_results
            WHERE experiment_id = ?
            ORDER BY timestamp
        """
        results = db_manager.query_results(query, [experiment_id])
        print(f"=" * 80)
        print(f"EXPERIMENT RESULTS: {experiment_id}")
        print("=" * 80)
        columns = ['experiment_id', 'game_id', 'winning_player_uuid', 'winning_payoff', 
                  'cumulative_payoff_sum', 'total_rounds', 'final_resource_level', 
                  'tragedy_occurred', 'timestamp']
    else:
        query = """
            SELECT 
                experiment_id,
                game_id,
                winning_player_uuid,
                winning_payoff,
                cumulative_payoff_sum,
                total_rounds,
                final_resource_level,
                tragedy_occurred,
                timestamp
            FROM experiment_results
            ORDER BY experiment_id, timestamp
        """
        results = db_manager.query_results(query)
        print("=" * 80)
        print("ALL EXPERIMENT RESULTS")
        print("=" * 80)
        columns = ['experiment_id', 'game_id', 'winning_player_uuid', 'winning_payoff', 
                  'cumulative_payoff_sum', 'total_rounds', 'final_resource_level', 
                  'tragedy_occurred', 'timestamp']
    
    if results:
        df = pd.DataFrame(results, columns=columns)
        
        # Format timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime("%Y-%m-%d %H:%M:%S")
        
        # Format numeric columns
        numeric_cols = ['winning_payoff', 'cumulative_payoff_sum', 'final_resource_level']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) and x is not None else "N/A")
        
        # Format text columns (winning_player_uuid)
        if 'winning_player_uuid' in df.columns:
            df['winning_player_uuid'] = df['winning_player_uuid'].apply(
                lambda x: str(x)[:8] + "..." if pd.notna(x) and x is not None and len(str(x)) > 8 else ("N/A" if pd.isna(x) or x is None else str(x))
            )
        
        # Format integer columns
        int_cols = ['total_rounds']
        for col in int_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"{int(x)}" if pd.notna(x) and x is not None else "N/A")
        
        # Format boolean column
        if 'tragedy_occurred' in df.columns:
            df['tragedy_occurred'] = df['tragedy_occurred'].apply(
                lambda x: "Yes" if x is True else ("No" if x is False else "N/A")
            )
        
        # Print the formatted dataframe
        print(df.to_string(index=False))
        
        # Print summary statistics
        print("\n" + "-" * 80)
        print("SUMMARY STATISTICS")
        print("-" * 80)
        
        # Calculate stats from original numeric data (before formatting)
        df_stats = pd.DataFrame(results, columns=columns)
        numeric_df = df_stats.select_dtypes(include=[float, int])
        
        if len(df_stats) > 0:
            print(f"Total Games: {len(df_stats)}")
            
            if 'tragedy_occurred' in df_stats.columns:
                tragedies = df_stats['tragedy_occurred'].sum()
                tragedy_rate = (tragedies / len(df_stats)) * 100 if len(df_stats) > 0 else 0
                print(f"Tragedies: {int(tragedies)} ({tragedy_rate:.1f}%)")
            
            if 'winning_payoff' in numeric_df.columns:
                print(f"Avg Winning Payoff: {numeric_df['winning_payoff'].mean():.2f}")
                print(f"Max Winning Payoff: {numeric_df['winning_payoff'].max():.2f}")
            
            if 'total_rounds' in numeric_df.columns:
                print(f"Avg Rounds: {numeric_df['total_rounds'].mean():.1f}")
                print(f"Min Rounds: {int(numeric_df['total_rounds'].min())}")
                print(f"Max Rounds: {int(numeric_df['total_rounds'].max())}")
            
            if 'final_resource_level' in numeric_df.columns:
                print(f"Avg Final Resource: {numeric_df['final_resource_level'].mean():.2f}")
                print(f"Min Final Resource: {numeric_df['final_resource_level'].min():.2f}")
    else:
        print("No results found")
    
    print()


def query_custom(sql: str, params: list = None):
    """Run a custom SQL query."""
    db_path = CONFIG.get("db_path", "data/game_results.db")
    db_manager = DatabaseManager(db_path=db_path, enabled=True)
    
    print("=" * 60)
    print("CUSTOM QUERY")
    print("=" * 60)
    print(f"SQL: {sql}")
    if params:
        print(f"Parameters: {params}")
    print()
    
    results = db_manager.query_results(sql, params)
    
    if results:
        # Try to determine column names from query or use generic names
        # For now, just print raw results
        for i, row in enumerate(results):
            print(f"Row {i+1}: {row}")
    else:
        print("No results found")
    
    print()


def show_examples():
    """Show example queries."""
    print("=" * 60)
    print("EXAMPLE QUERIES")
    print("=" * 60)
    print("""
# Query all experiments
python query_experiments.py --experiments

# Query players for a specific experiment
python query_experiments.py --players <experiment_id>

# Query all players
python query_experiments.py --players

# Query results for a specific experiment
python query_experiments.py --results <experiment_id>

# Query all results
python query_experiments.py --results

# Run custom SQL query
python query_experiments.py --sql "SELECT * FROM experiments WHERE status = 'completed'"

# Get experiment with player count
python query_experiments.py --sql "SELECT e.*, COUNT(ep.player_index) as player_count FROM experiments e LEFT JOIN experiment_players ep ON e.experiment_id = ep.experiment_id GROUP BY e.experiment_id"

# Get experiment statistics
python query_experiments.py --sql "SELECT experiment_id, COUNT(game_id) as num_games, MIN(timestamp) as first_game, MAX(timestamp) as last_game FROM experiment_results GROUP BY experiment_id"

# Query results with new columns (winning_player_uuid, winning_payoff, etc.)
python query_experiments.py --results <experiment_id>

# Query tragedy rate by experiment
python query_experiments.py --sql "SELECT experiment_id, COUNT(*) as total_games, SUM(CASE WHEN tragedy_occurred THEN 1 ELSE 0 END) as tragedies, ROUND(100.0 * SUM(CASE WHEN tragedy_occurred THEN 1 ELSE 0 END) / COUNT(*), 2) as tragedy_rate FROM experiment_results GROUP BY experiment_id"

# Clean up rows with NaN values from experiment_results
python query_experiments.py --cleanup-nan
""")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Query experiment tables",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--experiments", action="store_true", help="Query all experiments")
    parser.add_argument("--players", nargs="?", const="", help="Query experiment players (optionally filter by experiment_id)")
    parser.add_argument("--results", nargs="?", const="", help="Query experiment results (optionally filter by experiment_id)")
    parser.add_argument("--sql", type=str, help="Run custom SQL query")
    parser.add_argument("--examples", action="store_true", help="Show example queries")
    parser.add_argument("--cleanup-nan", action="store_true", help="Delete rows with NaN values from experiment_results")
    
    args = parser.parse_args()
    
    if args.examples:
        show_examples()
        return
    
    if args.experiments:
        query_experiments()
    
    if args.players is not None:
        experiment_id = args.players if args.players else None
        query_experiment_players(experiment_id)
    
    if args.results is not None:
        experiment_id = args.results if args.results else None
        query_experiment_results(experiment_id)
    
    if args.sql:
        query_custom(args.sql)
    
    if args.cleanup_nan:
        cleanup_nan_results()
    
    if not any([args.experiments, args.players is not None, args.results is not None, args.sql, args.cleanup_nan]):
        # Show examples if no arguments provided
        show_examples()


if __name__ == "__main__":
    main()

