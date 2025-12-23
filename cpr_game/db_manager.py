"""Database manager for storing game results in SQLite.

Provides persistent storage for game run data including player results,
personas, models, and rewards.
"""

from typing import Dict, List, Optional
import sqlite3
from pathlib import Path
import json
import uuid
from datetime import datetime
import time
from .logger_setup import get_logger

logger = get_logger(__name__)


class DatabaseManager:
    """Manages SQLite database for storing game results.
    
    Creates and maintains a database table to store per-player game results
    including game_id, player_id, persona, model, and total_reward.
    
    Supports both read-only and read-write connections for concurrent access.
    Uses WAL (Write-Ahead Logging) mode for better concurrency.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        enabled: bool = True,
        access_mode: str = 'READ_WRITE'
    ):
        """Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file. If None, uses default:
                     data/game_results.db
            enabled: Whether database saving is enabled (default: True)
            access_mode: Connection access mode - 'READ_ONLY' or 'READ_WRITE' (default: 'READ_WRITE')
        """
        self.enabled = enabled
        self.access_mode = access_mode
        
        if not self.enabled:
            logger.info("Database manager is disabled")
            self.conn = None
            self.read_only_conn = None
            self.db_path = None
            return
        
        # Set default database path
        if db_path is None:
            db_path = "data/game_results.db"
        
        self.db_path = db_path
        
        # Ensure parent directory exists (only for write mode)
        if access_mode == 'READ_WRITE':
            db_file = Path(db_path)
            db_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Connect to SQLite
        try:
            read_only = (access_mode == 'READ_ONLY')
            
            if read_only:
                # Read-only connection using URI mode
                self.conn = sqlite3.connect(
                    f"file:{db_path}?mode=ro",
                    uri=True,
                    check_same_thread=False
                )
            else:
                # Read-write connection with timeout for automatic retry on locks
                self.conn = sqlite3.connect(
                    db_path,
                    check_same_thread=False,
                    timeout=5.0  # 5 second timeout for busy retries
                )
                # Enable WAL mode for better concurrency
                self.conn.execute("PRAGMA journal_mode=WAL")
                # Set busy timeout (in milliseconds) for automatic retries
                self.conn.execute("PRAGMA busy_timeout=5000")
                # Optimize for better performance with WAL mode
                self.conn.execute("PRAGMA synchronous=NORMAL")
            
            self.read_only_conn = None  # Lazy initialization for read-only connection
            
            # Only create tables if in write mode
            if not read_only:
                self._create_table()
                logger.info(f"Database manager initialized (READ_WRITE): {db_path}")
            else:
                logger.info(f"Database manager initialized (READ_ONLY): {db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}", exc_info=True)
            self.conn = None
            self.read_only_conn = None
            self.enabled = False

    def _column_exists(self, table_name: str, column_name: str) -> bool:
        """Check if a column exists in a table.
        
        Args:
            table_name: Name of the table
            column_name: Name of the column to check
            
        Returns:
            True if column exists, False otherwise
        """
        if not self.enabled or self.conn is None:
            return False
        
        try:
            # PRAGMA table_info returns: (cid, name, type, notnull, default_value, pk)
            cursor = self.conn.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            column_names = [col[1] for col in columns]  # Column name is at index 1
            return column_name in column_names
        except Exception:
            return False

    def _table_exists(self, table_name: str) -> bool:
        """Check if a table exists.
        
        Args:
            table_name: Name of the table to check
            
        Returns:
            True if table exists, False otherwise
        """
        if not self.enabled or self.conn is None:
            return False
        
        try:
            # Check sqlite_master for the table
            cursor = self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                [table_name]
            )
            return cursor.fetchone() is not None
        except Exception:
            return False

    def _create_table(self):
        """Create game_results table if it doesn't exist."""
        if not self.enabled or self.conn is None:
            return
        
        try:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS game_results (
                    game_id TEXT NOT NULL,
                    player_id INTEGER NOT NULL,
                    player_uuid TEXT,
                    persona TEXT NOT NULL,
                    model TEXT NOT NULL,
                    total_reward REAL NOT NULL,
                    experiment_id TEXT,
                    timestamp TIMESTAMP,
                    PRIMARY KEY (game_id, player_id)
                )
            """)
            
            # Add new columns if they don't exist (for migration from old schema)
            # SQLite doesn't support IF NOT EXISTS in ALTER TABLE, so check first
            if not self._column_exists("game_results", "experiment_id"):
                try:
                    self.conn.execute("ALTER TABLE game_results ADD COLUMN experiment_id TEXT")
                except Exception:
                    pass  # Column might already exist or error is expected
            
            if not self._column_exists("game_results", "timestamp"):
                try:
                    self.conn.execute("ALTER TABLE game_results ADD COLUMN timestamp TIMESTAMP")
                except Exception:
                    pass  # Column might already exist or error is expected
            
            if not self._column_exists("game_results", "player_uuid"):
                try:
                    self.conn.execute("ALTER TABLE game_results ADD COLUMN player_uuid TEXT")
                except Exception:
                    pass  # Column might already exist or error is expected
            
            # Check if experiments table exists and what columns it has
            table_exists = self._table_exists("experiments")
            has_max_fishes = False
            has_Max_fish = False
            if table_exists:
                # PRAGMA table_info returns: (cid, name, type, notnull, default_value, pk)
                cursor = self.conn.execute("PRAGMA table_info(experiments)")
                columns = cursor.fetchall()
                column_names = [col[1] for col in columns]  # Column name is at index 1
                has_max_fishes = "max_fishes" in column_names
                has_Max_fish = "Max_fish" in column_names
            
            # Create experiment tables
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    n_players INTEGER NOT NULL,
                    max_steps INTEGER NOT NULL,
                    initial_resource INTEGER NOT NULL,
                    regeneration_rate REAL NOT NULL,
                    max_extraction INTEGER NOT NULL,
                    Max_fish INTEGER NOT NULL,
                    number_of_games INTEGER NOT NULL,
                    number_of_players_per_game INTEGER NOT NULL
                )
            """)
            
            # Migration: Handle max_fishes -> Max_fish rename if table already exists
            if table_exists:
                try:
                    # Re-check columns after CREATE TABLE (in case it was just created)
                    cursor = self.conn.execute("PRAGMA table_info(experiments)")
                    columns = cursor.fetchall()
                    column_names = [col[1] for col in columns]  # Column name is at index 1
                    has_max_fishes = "max_fishes" in column_names
                    has_Max_fish = "Max_fish" in column_names
                    
                    if has_max_fishes and not has_Max_fish:
                        logger.info("Migrating: adding Max_fish column and copying data from max_fishes")
                        # Add new column
                        self.conn.execute("ALTER TABLE experiments ADD COLUMN Max_fish INTEGER")
                        # Copy data
                        self.conn.execute("UPDATE experiments SET Max_fish = max_fishes")
                        # Drop old column
                        self.conn.execute("ALTER TABLE experiments DROP COLUMN max_fishes")
                        logger.info("Migration complete: max_fishes renamed to Max_fish")
                    elif has_max_fishes and has_Max_fish:
                        # Both exist - copy data if needed and drop old column
                        logger.info("Migrating: copying max_fishes to Max_fish and dropping old column")
                        self.conn.execute("UPDATE experiments SET Max_fish = max_fishes WHERE Max_fish IS NULL OR Max_fish = 0")
                        self.conn.execute("ALTER TABLE experiments DROP COLUMN max_fishes")
                except Exception as e:
                    logger.warning(f"Migration warning: {e}")
                    # Continue anyway - might be a duplicate column error or rename might have already happened
            
            # Migration: Check if we need to add parameter columns
            # Check if columns exist using PRAGMA
            needs_migration = False
            if table_exists:
                columns = self.conn.execute("PRAGMA table_info(experiments)").fetchall()
                column_names = [col[1] for col in columns]  # Column name is at index 1
                if "n_players" not in column_names:
                    needs_migration = True
            
            if needs_migration:
                # New columns don't exist - need to add them
                logger.info("Migrating experiments table: adding parameter columns")
                try:
                    if not self._column_exists("experiments", "n_players"):
                        self.conn.execute("ALTER TABLE experiments ADD COLUMN n_players INTEGER")
                    if not self._column_exists("experiments", "max_steps"):
                        self.conn.execute("ALTER TABLE experiments ADD COLUMN max_steps INTEGER")
                    if not self._column_exists("experiments", "initial_resource"):
                        self.conn.execute("ALTER TABLE experiments ADD COLUMN initial_resource INTEGER")
                    if not self._column_exists("experiments", "regeneration_rate"):
                        self.conn.execute("ALTER TABLE experiments ADD COLUMN regeneration_rate REAL")
                    if not self._column_exists("experiments", "max_extraction"):
                        self.conn.execute("ALTER TABLE experiments ADD COLUMN max_extraction INTEGER")
                    if not self._column_exists("experiments", "Max_fish"):
                        self.conn.execute("ALTER TABLE experiments ADD COLUMN Max_fish INTEGER")
                    if not self._column_exists("experiments", "number_of_games"):
                        self.conn.execute("ALTER TABLE experiments ADD COLUMN number_of_games INTEGER")
                    if not self._column_exists("experiments", "number_of_players_per_game"):
                        self.conn.execute("ALTER TABLE experiments ADD COLUMN number_of_players_per_game INTEGER")
                    
                    # Try to migrate existing data from JSON to columns (if any exists)
                    try:
                        # Check if parameters column exists
                        cursor = self.conn.execute("PRAGMA table_info(experiments)")
                        columns = cursor.fetchall()
                        column_names = [col[1] for col in columns]  # Column name is at index 1
                        if "parameters" in column_names:
                            cursor = self.conn.execute("SELECT experiment_id, parameters FROM experiments WHERE parameters IS NOT NULL")
                            existing = cursor.fetchall()
                            for exp_id, params_json in existing:
                                try:
                                    params = json.loads(params_json)
                                    self.conn.execute("""
                                        UPDATE experiments SET
                                            n_players = ?,
                                            max_steps = ?,
                                            initial_resource = ?,
                                            regeneration_rate = ?,
                                            max_extraction = ?,
                                            Max_fish = ?,
                                            number_of_games = ?,
                                            number_of_players_per_game = ?
                                        WHERE experiment_id = ?
                                    """, (
                                        params.get("n_players"),
                                        params.get("max_steps"),
                                        params.get("initial_resource"),
                                        params.get("regeneration_rate"),
                                        params.get("max_extraction"),
                                        params.get("Max_fish") or params.get("max_fishes"),  # Support both names
                                        params.get("number_of_games"),
                                        params.get("number_of_players_per_game"),
                                        exp_id
                                    ))
                                except (json.JSONDecodeError, KeyError) as e:
                                    logger.warning(f"Could not migrate parameters for experiment {exp_id}: {e}")
                                    continue
                            logger.info("Migration complete: existing data migrated to new columns")
                    except Exception as e:
                        logger.warning(f"Could not migrate existing parameter data: {e}")
                except Exception as e:
                    logger.warning(f"Migration warning: {e}")
                    # Columns might already exist
            
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS experiment_players (
                    experiment_id TEXT NOT NULL,
                    player_index INTEGER NOT NULL,
                    player_uuid TEXT NOT NULL,
                    persona TEXT NOT NULL,
                    model TEXT NOT NULL,
                    PRIMARY KEY (experiment_id, player_index),
                    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
                )
            """)
            
            # Add player_uuid column if it doesn't exist (for migration)
            if not self._column_exists("experiment_players", "player_uuid"):
                try:
                    self.conn.execute("ALTER TABLE experiment_players ADD COLUMN player_uuid TEXT")
                    # Generate UUIDs for existing players that don't have them
                    # SQLite doesn't have gen_random_uuid(), so we'll generate in Python
                    existing_players = self.conn.execute("""
                        SELECT experiment_id, player_index 
                        FROM experiment_players 
                        WHERE player_uuid IS NULL OR player_uuid = ''
                    """).fetchall()
                    for exp_id, player_idx in existing_players:
                        new_uuid = str(uuid.uuid4())
                        self.conn.execute("""
                            UPDATE experiment_players 
                            SET player_uuid = ? 
                            WHERE experiment_id = ? AND player_index = ?
                        """, (new_uuid, exp_id, player_idx))
                    # Note: SQLite doesn't support ALTER COLUMN SET NOT NULL directly,
                    # so we'll handle NULLs in application code for now
                except Exception as e:
                    logger.debug(f"Migration note (may be expected): {e}")
                    pass  # Column might already exist or error is expected
            
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS experiment_results (
                    experiment_id TEXT NOT NULL,
                    game_id TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    PRIMARY KEY (experiment_id, game_id),
                    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
                )
            """)
            
            # Create table to store individual player results per game
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS experiment_game_players (
                    experiment_id TEXT NOT NULL,
                    game_id TEXT NOT NULL,
                    player_uuid TEXT NOT NULL,
                    player_index INTEGER NOT NULL,
                    persona TEXT NOT NULL,
                    model TEXT NOT NULL,
                    total_reward REAL NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    PRIMARY KEY (experiment_id, game_id, player_uuid),
                    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
                )
            """)
            
            # Add new columns if they don't exist (for migration from old schema)
            # SQLite doesn't support IF NOT EXISTS in ALTER TABLE, so check first
            if not self._column_exists("experiment_results", "winning_player_uuid"):
                try:
                    self.conn.execute("ALTER TABLE experiment_results ADD COLUMN winning_player_uuid TEXT")
                except Exception:
                    pass  # Column might already exist or error is expected
            
            # Migration: Rename winning_player_id to winning_player_uuid if old column exists
            try:
                cursor = self.conn.execute("PRAGMA table_info(experiment_results)")
                columns = cursor.fetchall()
                column_names = [col[1] for col in columns]  # Column name is at index 1
                if "winning_player_id" in column_names and "winning_player_uuid" not in column_names:
                    logger.info("Migrating: adding winning_player_uuid column")
                    # Add new column first
                    self.conn.execute("ALTER TABLE experiment_results ADD COLUMN winning_player_uuid TEXT")
                    # Data migration will happen in save_experiment_result based on selected_players
                    # Old winning_player_id column will be kept for backward compatibility for now
            except Exception as e:
                logger.debug(f"Migration note (may be expected): {e}")
                pass
            
            if not self._column_exists("experiment_results", "winning_payoff"):
                try:
                    self.conn.execute("ALTER TABLE experiment_results ADD COLUMN winning_payoff REAL")
                except Exception:
                    pass
            
            if not self._column_exists("experiment_results", "cumulative_payoff_sum"):
                try:
                    self.conn.execute("ALTER TABLE experiment_results ADD COLUMN cumulative_payoff_sum REAL")
                except Exception:
                    pass
            
            if not self._column_exists("experiment_results", "total_rounds"):
                try:
                    self.conn.execute("ALTER TABLE experiment_results ADD COLUMN total_rounds INTEGER")
                except Exception:
                    pass
            
            if not self._column_exists("experiment_results", "final_resource_level"):
                try:
                    self.conn.execute("ALTER TABLE experiment_results ADD COLUMN final_resource_level REAL")
                except Exception:
                    pass
            
            if not self._column_exists("experiment_results", "tragedy_occurred"):
                try:
                    self.conn.execute("ALTER TABLE experiment_results ADD COLUMN tragedy_occurred BOOLEAN")
                except Exception:
                    pass
            
            logger.debug("Game results and experiment tables created or already exist")
        except Exception as e:
            logger.error(f"Failed to create table: {e}", exc_info=True)
            raise

    def get_read_connection(self):
        """Get or create read-only connection for queries.
        
        Returns:
            SQLite connection in READ_ONLY mode
        """
        if not self.enabled:
            return None
        
        # If already in read-only mode, return main connection
        if self.access_mode == 'READ_ONLY':
            return self.conn
        
        # Otherwise, create a separate read-only connection
        if self.read_only_conn is None:
            try:
                self.read_only_conn = sqlite3.connect(
                    f"file:{self.db_path}?mode=ro",
                    uri=True,
                    check_same_thread=False
                )
                logger.debug("Created read-only connection for queries")
            except Exception as e:
                logger.warning(f"Failed to create read-only connection: {e}, falling back to main connection")
                return self.conn
        
        return self.read_only_conn

    def get_write_connection(self):
        """Get write connection for INSERT/UPDATE/DELETE operations.
        
        Returns:
            SQLite connection in READ_WRITE mode
        """
        if not self.enabled:
            return None
        
        if self.access_mode != 'READ_WRITE':
            logger.error("Cannot get write connection - DatabaseManager is in READ_ONLY mode")
            return None
        
        return self.conn

    def _execute_write_with_retry(self, operation, max_retries=3, retry_delay=0.5):
        """Execute a write operation with retry logic for lock conflicts.
        
        Note: SQLite's busy_timeout handles most retries automatically,
        but this method provides additional retry logic for edge cases.
        
        Args:
            operation: Callable that performs the write operation
            max_retries: Maximum number of retry attempts (default: 3)
            retry_delay: Delay between retries in seconds (default: 0.5)
            
        Returns:
            Result of the operation, or None if all retries failed
        """
        conn = self.get_write_connection()
        if conn is None:
            return None
        
        for attempt in range(max_retries):
            try:
                return operation(conn)
            except sqlite3.OperationalError as e:
                error_msg = str(e).lower()
                # Check if it's a lock-related error (SQLite uses "database is locked")
                if 'lock' in error_msg or 'locked' in error_msg or 'busy' in error_msg:
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Database locked, retrying in {retry_delay}s "
                            f"(attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(retry_delay)
                        continue
                    else:
                        logger.error(f"Database locked after {max_retries} attempts: {e}")
                        raise
                else:
                    # Not a lock error, re-raise immediately
                    raise
            except Exception as e:
                # For non-OperationalError exceptions, re-raise immediately
                raise
        
        return None

    def save_game_results(
        self,
        game_id: str,
        agents: List,
        summary: Dict,
        config: Optional[Dict] = None,
        experiment_id: Optional[str] = None,
        timestamp: Optional[str] = None
    ):
        """Save game results to database.
        
        Args:
            game_id: Unique identifier for the game
            agents: List of agent objects with player_id, persona, llm_model
            summary: Game summary dict with 'cumulative_payoffs' list
            config: Optional config dict to get model if not in agent
            experiment_id: Optional experiment identifier
            timestamp: Optional timestamp (ISO format string or will use current time)
        """
        if not self.enabled:
            return
        
        conn = self.get_write_connection()
        if conn is None:
            logger.warning("Cannot save game results - no write connection available")
            return
        
        try:
            cumulative_payoffs = summary.get("cumulative_payoffs", [])
            
            if not cumulative_payoffs:
                logger.warning(f"No cumulative_payoffs in summary for game {game_id}")
                return
            
            # Prepare data for insertion
            rows = []
            for i, agent in enumerate(agents):
                if i >= len(cumulative_payoffs):
                    logger.warning(
                        f"Player {i} has no reward data in summary for game {game_id}"
                    )
                    continue
                
                player_id = getattr(agent, 'player_id', i)
                # Get player UUID if available (for experiments)
                player_uuid = getattr(agent, 'player_uuid', None)
                persona = getattr(agent, 'persona', 'unknown')
                
                # Get model from agent or config
                model = getattr(agent, 'llm_model', None)
                if model is None and config:
                    model = config.get('llm_model', 'unknown')
                elif model is None:
                    model = 'unknown'
                
                total_reward = float(cumulative_payoffs[i])
                
                # Use provided timestamp or current time
                if timestamp is None:
                    from datetime import datetime
                    timestamp = datetime.now().isoformat()
                
                rows.append((game_id, player_id, player_uuid, persona, model, total_reward, experiment_id, timestamp))
            
            if not rows:
                logger.warning(f"No valid player data to save for game {game_id}")
                return
            
            # Insert data (using ON CONFLICT for idempotency)
            conn.executemany("""
                INSERT INTO game_results 
                (game_id, player_id, player_uuid, persona, model, total_reward, experiment_id, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (game_id, player_id) 
                DO UPDATE SET 
                    player_uuid = EXCLUDED.player_uuid,
                    persona = EXCLUDED.persona,
                    model = EXCLUDED.model,
                    total_reward = EXCLUDED.total_reward,
                    experiment_id = EXCLUDED.experiment_id,
                    timestamp = EXCLUDED.timestamp
            """, rows)
            
            conn.commit()
            logger.info(f"Saved {len(rows)} player results for game {game_id}")
            
            # Verify the save was successful (use read connection for verification)
            read_conn = self.get_read_connection()
            verify_count = 0
            if read_conn:
                cursor = read_conn.execute(
                    "SELECT COUNT(*) FROM game_results WHERE game_id = ?",
                    [game_id]
                )
                verify_count = cursor.fetchone()[0]
            
            if verify_count != len(rows):
                logger.warning(
                    f"Save verification failed for game {game_id}: "
                    f"expected {len(rows)} rows, found {verify_count}"
                )
            
        except Exception as e:
            logger.error(
                f"Failed to save game results for {game_id}: {e}",
                exc_info=True
            )
            # Don't raise - database errors shouldn't crash the game

    def query_results(
        self,
        query: str,
        parameters: Optional[list] = None
    ) -> List[tuple]:
        """Execute a query and return results.
        
        Uses read-only connection for SELECT queries to allow concurrent access.
        
        Args:
            query: SQL query string
            parameters: Optional query parameters (list or tuple)
            
        Returns:
            List of result tuples
        """
        if not self.enabled:
            logger.warning("Database is not enabled")
            return []
        
        conn = self.get_read_connection()
        if conn is None:
            logger.warning("Database is not initialized")
            return []
        
        try:
            if parameters:
                cursor = conn.execute(query, parameters)
            else:
                cursor = conn.execute(query)
            return cursor.fetchall()
        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)
            return []

    def get_all_results(self) -> List[tuple]:
        """Get all game results.
        
        Returns:
            List of tuples (game_id, player_id, persona, model, total_reward)
        """
        return self.query_results("SELECT * FROM game_results ORDER BY game_id, player_id")

    def verify_game_saved(self, game_id: str) -> bool:
        """Verify that a game was saved to the database.
        
        Args:
            game_id: Game ID to check
            
        Returns:
            True if game exists in database, False otherwise
        """
        if not self.enabled:
            return False
        
        conn = self.get_read_connection()
        if conn is None:
            return False
        
        try:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM game_results WHERE game_id = ?",
                [game_id]
            )
            result = cursor.fetchone()
            return result is not None and result[0] > 0
        except Exception as e:
            logger.error(f"Failed to verify game save: {e}", exc_info=True)
            return False

    def get_player_stats(self, player_id: Optional[int] = None) -> List[tuple]:
        """Get statistics for a specific player or all players.
        
        Args:
            player_id: Optional player ID to filter by
            
        Returns:
            List of tuples with player statistics
        """
        if player_id is not None:
            query = """
                SELECT 
                    player_id,
                    persona,
                    model,
                    COUNT(*) as game_count,
                    AVG(total_reward) as avg_reward,
                    MIN(total_reward) as min_reward,
                    MAX(total_reward) as max_reward,
                    SUM(total_reward) as total_reward_sum
                FROM game_results
                WHERE player_id = ?
                GROUP BY player_id, persona, model
            """
            return self.query_results(query, [player_id])
        else:
            query = """
                SELECT 
                    player_id,
                    persona,
                    model,
                    COUNT(*) as game_count,
                    AVG(total_reward) as avg_reward,
                    MIN(total_reward) as min_reward,
                    MAX(total_reward) as max_reward,
                    SUM(total_reward) as total_reward_sum
                FROM game_results
                GROUP BY player_id, persona, model
                ORDER BY player_id
            """
            return self.query_results(query)

    def close(self):
        """Close all database connections."""
        if self.read_only_conn is not None:
            try:
                self.read_only_conn.close()
                logger.debug("Read-only connection closed")
            except Exception as e:
                logger.error(f"Error closing read-only connection: {e}", exc_info=True)
            finally:
                self.read_only_conn = None
        
        if self.conn is not None:
            try:
                self.conn.close()
                logger.info("Database connection closed")
            except Exception as e:
                logger.error(f"Error closing database: {e}", exc_info=True)
            finally:
                self.conn = None

    def close_all(self):
        """Close all connections (alias for close)."""
        self.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    # ============================================================================
    # Experiment Management Methods
    # ============================================================================

    def save_experiment(
        self,
        experiment_id: str,
        name: str,
        players: List[Dict[str, str]],
        parameters: Dict
    ) -> bool:
        """Save an experiment definition to the database.
        
        Args:
            experiment_id: Unique identifier for the experiment
            name: Human-readable experiment name
            players: List of player dicts with 'persona' and 'model' keys
            parameters: Dictionary of experiment parameters
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or self.conn is None:
            logger.warning("Database is not enabled or not initialized")
            return False
        
        try:
            # Extract parameters
            n_players = parameters.get("n_players")
            max_steps = parameters.get("max_steps")
            initial_resource = parameters.get("initial_resource")
            regeneration_rate = parameters.get("regeneration_rate")
            max_extraction = parameters.get("max_extraction")
            Max_fish = parameters.get("Max_fish") or parameters.get("max_fishes", 1000)  # Support both, default 1000
            
            # Get write connection
            conn = self.get_write_connection()
            if conn is None:
                logger.warning("Cannot save experiment - no write connection available")
                return False
            
            # Check if old max_fishes column still exists (for migration compatibility)
            has_max_fishes_col = self._column_exists("experiments", "max_fishes")
            number_of_games = parameters.get("number_of_games")
            number_of_players_per_game = parameters.get("number_of_players_per_game")
            
            # Check if parameters column exists (for migration compatibility)
            has_old_parameters_column = self._column_exists("experiments", "parameters")
            
            # Build INSERT statement - handle migration columns (parameters, max_fishes)
            # If old columns exist, we need to include them to satisfy NOT NULL constraints
            base_columns = ["experiment_id", "name", "status", "created_at"]
            base_values = [experiment_id, name, "pending", datetime.now()]
            
            if has_old_parameters_column:
                base_columns.append("parameters")
                base_values.append("")
            
            param_columns = [
                "n_players", "max_steps", "initial_resource", "regeneration_rate",
                "max_extraction"
            ]
            param_values = [
                n_players, max_steps, initial_resource, regeneration_rate,
                max_extraction
            ]
            
            # Handle Max_fish and max_fishes columns
            if has_max_fishes_col:
                # Both columns exist - set both to the same value
                param_columns.extend(["max_fishes", "Max_fish"])
                param_values.extend([Max_fish, Max_fish])
            else:
                # Only Max_fish exists
                param_columns.append("Max_fish")
                param_values.append(Max_fish)
            
            param_columns.extend(["number_of_games", "number_of_players_per_game"])
            param_values.extend([number_of_games, number_of_players_per_game])
            
            all_columns = base_columns + param_columns
            all_values = base_values + param_values
            placeholders = ", ".join(["?"] * len(all_values))
            
            # Build UPDATE clause
            update_clause = ", ".join([
                f"{col} = EXCLUDED.{col}" 
                for col in all_columns 
                if col not in ["experiment_id", "created_at"]
            ])
            
            query = f"""
                INSERT INTO experiments ({", ".join(all_columns)})
                VALUES ({placeholders})
                ON CONFLICT (experiment_id) 
                DO UPDATE SET {update_clause}
            """
            
            conn.execute(query, all_values)
            
            # Delete existing players for this experiment (for updates)
            conn.execute(
                "DELETE FROM experiment_players WHERE experiment_id = ?",
                [experiment_id]
            )
            
            # Insert players with UUIDs
            player_rows = [
                (experiment_id, i, str(uuid.uuid4()), player["persona"], player["model"])
                for i, player in enumerate(players)
            ]
            
            if player_rows:
                conn.executemany("""
                    INSERT INTO experiment_players (experiment_id, player_index, player_uuid, persona, model)
                    VALUES (?, ?, ?, ?, ?)
                """, player_rows)
            
            conn.commit()
            logger.info(f"Saved experiment {experiment_id} with {len(players)} players")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save experiment {experiment_id}: {e}", exc_info=True)
            return False

    def load_experiment(self, experiment_id: str) -> Optional[Dict]:
        """Load an experiment definition from the database.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            Dictionary with experiment data or None if not found
        """
        if not self.enabled:
            logger.warning("Database is not enabled")
            return None
        
        conn = self.get_read_connection()
        if conn is None:
            logger.warning("Database is not initialized")
            return None
        
        try:
            # Try to load with new schema (parameter columns)
            try:
                cursor = conn.execute("""
                    SELECT name, status, created_at,
                           n_players, max_steps, initial_resource, regeneration_rate,
                           max_extraction, Max_fish, number_of_games, number_of_players_per_game
                    FROM experiments WHERE experiment_id = ?
                """, [experiment_id])
                exp_result = cursor.fetchone()
                
                if exp_result is None:
                    return None
                
                name, status, created_at, n_players, max_steps, initial_resource, regeneration_rate, \
                    max_extraction, Max_fish, number_of_games, number_of_players_per_game = exp_result
            except Exception:
                # Fallback to old schema (parameters JSON column)
                cursor = conn.execute(
                    "SELECT name, status, created_at, parameters FROM experiments WHERE experiment_id = ?",
                    [experiment_id]
                )
                exp_result = cursor.fetchone()
                
                if exp_result is None:
                    return None
                
                name, status, created_at, parameters_json = exp_result
                parameters_dict = json.loads(parameters_json)
                n_players = parameters_dict.get("n_players")
                max_steps = parameters_dict.get("max_steps")
                initial_resource = parameters_dict.get("initial_resource")
                regeneration_rate = parameters_dict.get("regeneration_rate")
                max_extraction = parameters_dict.get("max_extraction")
                Max_fish = parameters_dict.get("Max_fish") or parameters_dict.get("max_fishes", 1000)
                number_of_games = parameters_dict.get("number_of_games")
                number_of_players_per_game = parameters_dict.get("number_of_players_per_game")
            
            # Reconstruct parameters dict for backward compatibility
            parameters = {
                "n_players": n_players,
                "max_steps": max_steps,
                "initial_resource": initial_resource,
                "regeneration_rate": regeneration_rate,
                "max_extraction": max_extraction,
                "Max_fish": Max_fish,  # Use new name
                "number_of_games": number_of_games,
                "number_of_players_per_game": number_of_players_per_game,
            }
            
            # Load players (with UUIDs)
            try:
                # Try new schema with player_uuid
                cursor = conn.execute(
                    """SELECT player_index, player_uuid, persona, model 
                       FROM experiment_players 
                       WHERE experiment_id = ? 
                       ORDER BY player_index""",
                    [experiment_id]
                )
                player_results = cursor.fetchall()
                
                players = [
                    {"player_uuid": player_uuid, "persona": persona, "model": model}
                    for _, player_uuid, persona, model in player_results
                ]
            except Exception:
                # Fallback to old schema (without UUIDs) - generate UUIDs on the fly
                cursor = conn.execute(
                    """SELECT player_index, persona, model 
                       FROM experiment_players 
                       WHERE experiment_id = ? 
                       ORDER BY player_index""",
                    [experiment_id]
                )
                player_results = cursor.fetchall()
                
                players = [
                    {"player_uuid": str(uuid.uuid4()), "persona": persona, "model": model}
                    for _, persona, model in player_results
                ]
            
            return {
                "experiment_id": experiment_id,
                "name": name,
                "status": status,
                "created_at": created_at,
                "players": players,
                "parameters": parameters
            }
            
        except Exception as e:
            logger.error(f"Failed to load experiment {experiment_id}: {e}", exc_info=True)
            return None

    def list_experiments(self) -> List[Dict]:
        """List all experiments in the database.
        
        Returns:
            List of experiment dictionaries with basic info
        """
        if not self.enabled:
            logger.warning("Database is not enabled")
            return []
        
        conn = self.get_read_connection()
        if conn is None:
            logger.warning("Database is not initialized")
            return []
        
        try:
            cursor = conn.execute("""
                SELECT 
                    e.experiment_id,
                    e.name,
                    e.status,
                    e.created_at,
                    COUNT(DISTINCT ep.player_index) as player_count,
                    COUNT(DISTINCT er.game_id) as game_count
                FROM experiments e
                LEFT JOIN experiment_players ep ON e.experiment_id = ep.experiment_id
                LEFT JOIN experiment_results er ON e.experiment_id = er.experiment_id
                GROUP BY e.experiment_id, e.name, e.status, e.created_at
                ORDER BY e.created_at DESC
            """)
            results = cursor.fetchall()
            
            experiments = []
            for row in results:
                experiments.append({
                    "experiment_id": row[0],
                    "name": row[1],
                    "status": row[2],
                    "created_at": row[3],
                    "player_count": row[4] or 0,
                    "game_count": row[5] or 0
                })
            
            return experiments
            
        except Exception as e:
            logger.error(f"Failed to list experiments: {e}", exc_info=True)
            return []

    def update_experiment_status(self, experiment_id: str, status: str) -> bool:
        """Update the status of an experiment.
        
        Args:
            experiment_id: Experiment identifier
            status: New status (pending, running, completed, failed)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            logger.warning("Database is not enabled")
            return False
        
        conn = self.get_write_connection()
        if conn is None:
            logger.warning("Cannot update experiment status - no write connection available")
            return False
        
        try:
            conn.execute(
                "UPDATE experiments SET status = ? WHERE experiment_id = ?",
                [status, experiment_id]
            )
            conn.commit()
            logger.info(f"Updated experiment {experiment_id} status to {status}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update experiment status: {e}", exc_info=True)
            return False

    def reset_all_experiments_to_pending(self) -> int:
        """Reset all experiments to 'pending' status.
        
        Useful for resetting stuck or completed experiments to be reprocessed.
        
        Returns:
            Number of experiments that were reset, or -1 if failed
        """
        if not self.enabled:
            logger.warning("Database is not enabled")
            return -1
        
        conn = self.get_write_connection()
        if conn is None:
            logger.warning("Cannot reset experiments - no write connection available")
            return -1
        
        try:
            cursor = conn.execute(
                "UPDATE experiments SET status = 'pending' WHERE status != 'pending'",
                []
            )
            rows_affected = cursor.rowcount
            conn.commit()
            logger.info(f"Reset {rows_affected} experiments to 'pending' status")
            return rows_affected
            
        except Exception as e:
            logger.error(f"Failed to reset experiments to pending: {e}", exc_info=True)
            return -1

    def save_experiment_result(
        self,
        experiment_id: str,
        game_id: str,
        summary: Dict,
        selected_players: Optional[List[Dict]] = None
    ) -> bool:
        """Save a game result for an experiment.
        
        Args:
            experiment_id: Experiment identifier
            game_id: Game identifier
            summary: Game summary dictionary
            selected_players: Optional list of selected players with player_uuid, persona, model
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            logger.warning("Database is not enabled")
            return False
        
        # Check if we can get a write connection (don't check self.conn directly 
        # as it might be read-only, but get_write_connection will handle it)
        if self.access_mode != 'READ_WRITE':
            logger.warning("Database manager is in READ_ONLY mode, cannot save experiment result")
            return False
        
        try:
            summary_json = json.dumps(summary, default=str)
            timestamp = datetime.now()
            
            # Extract and calculate required fields from summary
            cumulative_payoffs = summary.get("cumulative_payoffs", [])
            
            # Calculate winning player and payoff
            winning_player_uuid = None
            winning_payoff = None
            cumulative_payoff_sum = None
            
            if cumulative_payoffs and selected_players:
                # Find player with highest payoff
                max_payoff = max(cumulative_payoffs)
                winning_player_index = cumulative_payoffs.index(max_payoff)
                # Get UUID of winning player
                if winning_player_index < len(selected_players):
                    winning_player_uuid = selected_players[winning_player_index].get("player_uuid")
                winning_payoff = float(max_payoff)
                cumulative_payoff_sum = float(sum(cumulative_payoffs))
            elif cumulative_payoffs:
                # Fallback if selected_players not provided - still calculate payoff
                max_payoff = max(cumulative_payoffs)
                winning_payoff = float(max_payoff)
                cumulative_payoff_sum = float(sum(cumulative_payoffs))
            
            # Extract other fields
            total_rounds = summary.get("total_rounds")
            final_resource_level = summary.get("final_resource_level")
            tragedy_occurred = summary.get("tragedy_occurred", False)
            
            # Convert to appropriate types
            if total_rounds is not None:
                total_rounds = int(total_rounds)
            if final_resource_level is not None:
                final_resource_level = float(final_resource_level)
            tragedy_occurred = bool(tragedy_occurred)
            
            # Get write connection
            conn = self.get_write_connection()
            if conn is None:
                logger.warning("Cannot save experiment result - no write connection available")
                return False
            
            conn.execute("""
                INSERT INTO experiment_results (
                    experiment_id, game_id, summary, timestamp,
                    winning_player_uuid, winning_payoff, cumulative_payoff_sum,
                    total_rounds, final_resource_level, tragedy_occurred
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (experiment_id, game_id) 
                DO UPDATE SET 
                    summary = EXCLUDED.summary,
                    timestamp = EXCLUDED.timestamp,
                    winning_player_uuid = EXCLUDED.winning_player_uuid,
                    winning_payoff = EXCLUDED.winning_payoff,
                    cumulative_payoff_sum = EXCLUDED.cumulative_payoff_sum,
                    total_rounds = EXCLUDED.total_rounds,
                    final_resource_level = EXCLUDED.final_resource_level,
                    tragedy_occurred = EXCLUDED.tragedy_occurred
            """, (
                experiment_id, game_id, summary_json, timestamp,
                winning_player_uuid, winning_payoff, cumulative_payoff_sum,
                total_rounds, final_resource_level, tragedy_occurred
            ))
            
            # Save individual player results if selected_players provided
            if selected_players and cumulative_payoffs:
                player_rows = []
                for i, player_info in enumerate(selected_players):
                    if i >= len(cumulative_payoffs):
                        logger.warning(f"Player {i} has no reward data in summary for game {game_id}")
                        continue
                    
                    # Get player UUID (must exist for experiments)
                    player_uuid = player_info.get("player_uuid")
                    if not player_uuid:
                        logger.warning(f"Player {i} in game {game_id} missing player_uuid, skipping")
                        continue
                    
                    persona = player_info.get("persona", "unknown")
                    model = player_info.get("model", "unknown")
                    total_reward = float(cumulative_payoffs[i])
                    
                    player_rows.append((
                        experiment_id, game_id, player_uuid, i, persona, model, total_reward, timestamp
                    ))
                
                if player_rows:
                    conn.executemany("""
                        INSERT INTO experiment_game_players (
                            experiment_id, game_id, player_uuid, player_index,
                            persona, model, total_reward, timestamp
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT (experiment_id, game_id, player_uuid) 
                        DO UPDATE SET 
                            player_index = EXCLUDED.player_index,
                            persona = EXCLUDED.persona,
                            model = EXCLUDED.model,
                            total_reward = EXCLUDED.total_reward,
                            timestamp = EXCLUDED.timestamp
                    """, player_rows)
                    logger.debug(f"Saved {len(player_rows)} individual player results for game {game_id}")
            
            conn.commit()
            logger.debug(f"Saved result for game {game_id} in experiment {experiment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save experiment result: {e}", exc_info=True)
            return False

    def get_experiment_results(self, experiment_id: str) -> List[Dict]:
        """Get all game results for an experiment.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            List of game result dictionaries
        """
        if not self.enabled:
            logger.warning("Database is not enabled")
            return []
        
        conn = self.get_read_connection()
        if conn is None:
            logger.warning("Database is not initialized")
            return []
        
        try:
            cursor = conn.execute(
                "SELECT game_id, summary, timestamp FROM experiment_results WHERE experiment_id = ? ORDER BY timestamp",
                [experiment_id]
            )
            results = cursor.fetchall()
            
            game_results = []
            for game_id, summary_json, timestamp in results:
                try:
                    summary = json.loads(summary_json)
                    game_results.append({
                        "game_id": game_id,
                        "summary": summary,
                        "timestamp": timestamp
                    })
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse summary for game {game_id}: {e}")
                    continue
            
            return game_results
            
        except Exception as e:
            logger.error(f"Failed to get experiment results: {e}", exc_info=True)
            return []

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment and all associated data.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            logger.warning("Database is not enabled")
            return False
        
        conn = self.get_write_connection()
        if conn is None:
            logger.warning("Cannot delete experiment - no write connection available")
            return False
        
        try:
            # Manual cascade delete (SQLite supports ON DELETE CASCADE, but we do it manually for clarity)
            # Delete in order: results -> players -> experiment
            conn.execute(
                "DELETE FROM experiment_results WHERE experiment_id = ?",
                [experiment_id]
            )
            conn.execute(
                "DELETE FROM experiment_players WHERE experiment_id = ?",
                [experiment_id]
            )
            conn.execute(
                "DELETE FROM experiments WHERE experiment_id = ?",
                [experiment_id]
            )
            conn.commit()
            logger.info(f"Deleted experiment {experiment_id} and all associated data")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete experiment: {e}", exc_info=True)
            return False

    def get_game_result(self, experiment_id: str, game_id: str) -> Optional[Dict]:
        """Get a single game result for an experiment.
        
        Args:
            experiment_id: Experiment identifier
            game_id: Game identifier
            
        Returns:
            Dictionary with game result data or None if not found
        """
        if not self.enabled:
            logger.warning("Database is not enabled")
            return None
        
        conn = self.get_read_connection()
        if conn is None:
            logger.warning("Database is not initialized")
            return None
        
        try:
            # Check which column exists (winning_player_uuid or winning_player_id)
            cursor = conn.execute("PRAGMA table_info(experiment_results)")
            columns = cursor.fetchall()
            column_names = [col[1] for col in columns]  # Column name is at index 1
            has_uuid_col = "winning_player_uuid" in column_names
            has_id_col = "winning_player_id" in column_names
            
            # Build SELECT query based on available columns
            if has_uuid_col:
                select_cols = "game_id, summary, timestamp, winning_player_uuid, winning_payoff, cumulative_payoff_sum, total_rounds, final_resource_level, tragedy_occurred"
            elif has_id_col:
                select_cols = "game_id, summary, timestamp, winning_player_id, winning_payoff, cumulative_payoff_sum, total_rounds, final_resource_level, tragedy_occurred"
            else:
                # Fallback if neither exists
                select_cols = "game_id, summary, timestamp, winning_payoff, cumulative_payoff_sum, total_rounds, final_resource_level, tragedy_occurred"
            
            cursor = conn.execute(
                f"SELECT {select_cols} FROM experiment_results WHERE experiment_id = ? AND game_id = ?",
                [experiment_id, game_id]
            )
            result = cursor.fetchone()
            
            if result is None:
                return None
            
            # Parse result based on which columns were selected
            if has_uuid_col:
                game_id, summary_json, timestamp, winning_player_uuid, winning_payoff, \
                    cumulative_payoff_sum, total_rounds, final_resource_level, tragedy_occurred = result
            elif has_id_col:
                game_id, summary_json, timestamp, winning_player_id, winning_payoff, \
                    cumulative_payoff_sum, total_rounds, final_resource_level, tragedy_occurred = result
                winning_player_uuid = None  # Convert ID to UUID if needed (would need player lookup)
            else:
                game_id, summary_json, timestamp, winning_payoff, \
                    cumulative_payoff_sum, total_rounds, final_resource_level, tragedy_occurred = result
                winning_player_uuid = None
            
            try:
                summary = json.loads(summary_json)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse summary for game {game_id}: {e}")
                summary = {}
            
            return {
                "game_id": game_id,
                "summary": summary,
                "timestamp": timestamp,
                "winning_player_uuid": winning_player_uuid,
                "winning_payoff": winning_payoff,
                "cumulative_payoff_sum": cumulative_payoff_sum,
                "total_rounds": total_rounds,
                "final_resource_level": final_resource_level,
                "tragedy_occurred": tragedy_occurred
            }
            
        except Exception as e:
            logger.error(f"Failed to get game result: {e}", exc_info=True)
            return None

    def get_game_players(self, experiment_id: str, game_id: str) -> List[Dict]:
        """Get all players for a specific game.
        
        Args:
            experiment_id: Experiment identifier
            game_id: Game identifier
            
        Returns:
            List of player dictionaries with persona, model, total_reward, etc.
        """
        if not self.enabled:
            logger.warning("Database is not enabled")
            return []
        
        conn = self.get_read_connection()
        if conn is None:
            logger.warning("Database is not initialized")
            return []
        
        try:
            cursor = conn.execute(
                """SELECT player_uuid, player_index, persona, model, total_reward, timestamp
                   FROM experiment_game_players
                   WHERE experiment_id = ? AND game_id = ?
                   ORDER BY player_index""",
                [experiment_id, game_id]
            )
            results = cursor.fetchall()
            
            players = []
            for player_uuid, player_index, persona, model, total_reward, timestamp in results:
                players.append({
                    "player_uuid": player_uuid,
                    "player_index": player_index,
                    "persona": persona,
                    "model": model,
                    "total_reward": total_reward,
                    "timestamp": timestamp
                })
            
            return players
            
        except Exception as e:
            logger.error(f"Failed to get game players: {e}", exc_info=True)
            return []

