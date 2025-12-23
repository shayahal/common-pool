#!/usr/bin/env python3
"""Migrate data from DuckDB to SQLite database.

This script copies all data from the old DuckDB database files to the new SQLite database.
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import duckdb
except ImportError:
    print("❌ Error: duckdb module not found. Install it with: pip install duckdb")
    sys.exit(1)

import sqlite3
from cpr_game.db_manager import DatabaseManager
from cpr_game.config import CONFIG
from cpr_game.logger_setup import get_logger

logger = get_logger(__name__)


def migrate_duckdb_to_sqlite(duckdb_path: str, sqlite_path: str, dry_run: bool = False):
    """Migrate data from DuckDB to SQLite.
    
    Args:
        duckdb_path: Path to source DuckDB file
        sqlite_path: Path to destination SQLite file
        dry_run: If True, only show what would be migrated without actually migrating
    """
    if not Path(duckdb_path).exists():
        print(f"⚠️  DuckDB file not found: {duckdb_path}")
        return False
    
    print(f"\n{'='*60}")
    print(f"Migrating data from DuckDB to SQLite")
    print(f"{'='*60}")
    print(f"Source: {duckdb_path}")
    print(f"Destination: {sqlite_path}")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE MIGRATION'}")
    print(f"{'='*60}\n")
    
    # Connect to DuckDB (read-only)
    try:
        duckdb_conn = duckdb.connect(duckdb_path, read_only=True)
        print("✅ Connected to DuckDB (read-only)")
    except Exception as e:
        print(f"❌ Failed to connect to DuckDB: {e}")
        return False
    
    # Initialize SQLite database manager
    try:
        sqlite_db = DatabaseManager(db_path=sqlite_path, enabled=True, access_mode='READ_WRITE')
        if not sqlite_db.conn:
            print(f"❌ Failed to initialize SQLite database")
            return False
        print("✅ Connected to SQLite database")
    except Exception as e:
        print(f"❌ Failed to initialize SQLite database: {e}")
        duckdb_conn.close()
        return False
    
    migrated_count = {
        'experiments': 0,
        'experiment_players': 0,
        'experiment_results': 0,
        'experiment_game_players': 0,
        'game_results': 0
    }
    
    try:
        # Migrate experiments table
        print("\n📋 Migrating experiments...")
        try:
            experiments = duckdb_conn.execute("SELECT * FROM experiments").fetchall()
            columns = [desc[0] for desc in duckdb_conn.execute("DESCRIBE experiments").fetchall()]
            
            print(f"   Found {len(experiments)} experiments")
            
            for exp_row in experiments:
                exp_dict = dict(zip(columns, exp_row))
                experiment_id = exp_dict['experiment_id']
                
                if not dry_run:
                    # Load players for this experiment
                    try:
                        players_data = duckdb_conn.execute(
                            "SELECT player_index, player_uuid, persona, model FROM experiment_players WHERE experiment_id = ? ORDER BY player_index",
                            [experiment_id]
                        ).fetchall()
                        players = [
                            {"persona": p[2], "model": p[3], "player_uuid": p[1] if len(p) > 1 and p[1] else None}
                            for p in players_data
                        ]
                    except Exception as e:
                        logger.warning(f"Could not load players for {experiment_id}: {e}")
                        players = []
                    
                    # Extract parameters
                    parameters = {
                        "n_players": exp_dict.get('n_players'),
                        "max_steps": exp_dict.get('max_steps'),
                        "initial_resource": exp_dict.get('initial_resource'),
                        "regeneration_rate": exp_dict.get('regeneration_rate'),
                        "max_extraction": exp_dict.get('max_extraction'),
                        "Max_fish": exp_dict.get('Max_fish') or exp_dict.get('max_fishes', 1000),
                        "number_of_games": exp_dict.get('number_of_games'),
                        "number_of_players_per_game": exp_dict.get('number_of_players_per_game'),
                    }
                    
                    success = sqlite_db.save_experiment(
                        experiment_id=experiment_id,
                        name=exp_dict['name'],
                        players=players,
                        parameters=parameters
                    )
                    
                    if success:
                        migrated_count['experiments'] += 1
                        print(f"   ✓ Migrated experiment: {exp_dict['name']} ({experiment_id})")
                    else:
                        print(f"   ✗ Failed to migrate experiment: {experiment_id}")
                else:
                    migrated_count['experiments'] += 1
                    print(f"   [DRY RUN] Would migrate: {exp_dict['name']} ({experiment_id})")
        except Exception as e:
            print(f"   ⚠️  Error migrating experiments: {e}")
            logger.error(f"Error migrating experiments: {e}", exc_info=True)
        
        # Migrate experiment_results
        print("\n📋 Migrating experiment results...")
        try:
            results = duckdb_conn.execute("SELECT * FROM experiment_results").fetchall()
            columns = [desc[0] for desc in duckdb_conn.execute("DESCRIBE experiment_results").fetchall()]
            
            print(f"   Found {len(results)} experiment results")
            
            for result_row in results:
                result_dict = dict(zip(columns, result_row))
                experiment_id = result_dict['experiment_id']
                game_id = result_dict['game_id']
                
                if not dry_run:
                    # Parse summary JSON
                    try:
                        summary = json.loads(result_dict['summary'])
                    except:
                        summary = {}
                    
                    # Load selected players for this game
                    selected_players = None
                    try:
                        players_data = duckdb_conn.execute(
                            """SELECT player_uuid, player_index, persona, model 
                               FROM experiment_game_players 
                               WHERE experiment_id = ? AND game_id = ? 
                               ORDER BY player_index""",
                            [experiment_id, game_id]
                        ).fetchall()
                        if players_data:
                            selected_players = [
                                {"player_uuid": p[0], "persona": p[2], "model": p[3]}
                                for p in players_data
                            ]
                    except Exception as e:
                        logger.debug(f"Could not load players for game {game_id}: {e}")
                    
                    success = sqlite_db.save_experiment_result(
                        experiment_id=experiment_id,
                        game_id=game_id,
                        summary=summary,
                        selected_players=selected_players
                    )
                    
                    if success:
                        migrated_count['experiment_results'] += 1
                    else:
                        print(f"   ✗ Failed to migrate result: {experiment_id}/{game_id}")
                else:
                    migrated_count['experiment_results'] += 1
        except Exception as e:
            print(f"   ⚠️  Error migrating experiment results: {e}")
            logger.error(f"Error migrating experiment results: {e}", exc_info=True)
        
        # Migrate game_results (legacy table)
        print("\n📋 Migrating game_results (legacy)...")
        try:
            game_results = duckdb_conn.execute("SELECT * FROM game_results").fetchall()
            columns = [desc[0] for desc in duckdb_conn.execute("DESCRIBE game_results").fetchall()]
            
            print(f"   Found {len(game_results)} game results")
            
            if not dry_run and game_results:
                conn = sqlite_db.get_write_connection()
                if conn:
                    for row in game_results:
                        row_dict = dict(zip(columns, row))
                        try:
                            conn.execute("""
                                INSERT INTO game_results 
                                (game_id, player_id, player_uuid, persona, model, total_reward, experiment_id, timestamp)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                ON CONFLICT (game_id, player_id) DO NOTHING
                            """, (
                                row_dict.get('game_id'),
                                row_dict.get('player_id'),
                                row_dict.get('player_uuid'),
                                row_dict.get('persona'),
                                row_dict.get('model'),
                                row_dict.get('total_reward'),
                                row_dict.get('experiment_id'),
                                row_dict.get('timestamp')
                            ))
                            migrated_count['game_results'] += 1
                        except Exception as e:
                            logger.warning(f"Error migrating game_result: {e}")
                    conn.commit()
        except Exception as e:
            print(f"   ⚠️  Error migrating game_results: {e}")
            logger.error(f"Error migrating game_results: {e}", exc_info=True)
        
        # Summary
        print(f"\n{'='*60}")
        print("Migration Summary:")
        print(f"{'='*60}")
        print(f"Experiments: {migrated_count['experiments']}")
        print(f"Experiment Results: {migrated_count['experiment_results']}")
        print(f"Game Results (legacy): {migrated_count['game_results']}")
        print(f"{'='*60}\n")
        
        if dry_run:
            print("⚠️  This was a DRY RUN. No data was actually migrated.")
            print("   Run without --dry-run to perform the actual migration.")
        else:
            print("✅ Migration completed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        logger.error(f"Migration failed: {e}", exc_info=True)
        return False
    finally:
        duckdb_conn.close()
        sqlite_db.close()


def main():
    """Main migration function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Migrate data from DuckDB to SQLite')
    parser.add_argument('--source', default='data/game_results.duckdb',
                       help='Path to source DuckDB file (default: data/game_results.duckdb)')
    parser.add_argument('--destination', default=None,
                       help='Path to destination SQLite file (default: from config)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be migrated without actually migrating')
    
    args = parser.parse_args()
    
    # Get destination path
    if args.destination:
        sqlite_path = args.destination
    else:
        sqlite_path = CONFIG.get("db_path", "data/game_results.db")
    
    # Perform migration
    success = migrate_duckdb_to_sqlite(
        duckdb_path=args.source,
        sqlite_path=sqlite_path,
        dry_run=args.dry_run
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

