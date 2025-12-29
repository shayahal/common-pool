"""Experiment worker that watches the database and processes pending experiments.

This script continuously polls the database for experiments with status="pending"
and processes them using a configurable number of workers.

Usage:
    python experiment_worker.py --workers 4
    python experiment_worker.py --workers 4 --poll-interval 10
    python experiment_worker.py --workers 4 --use-mock
"""

import sys
import os
import argparse
import time
import signal
from pathlib import Path
from typing import Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
import threading

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from cpr_game.db_manager import DatabaseManager
from cpr_game.config import CONFIG
from cpr_game.logger_setup import get_logger, setup_logging

# Import run_experiment from main.py
from main import run_experiment

logger = get_logger(__name__)

# Global flag for graceful shutdown
shutdown_requested = False
shutdown_lock = threading.Lock()


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    with shutdown_lock:
        shutdown_requested = True
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    # On Windows, we can't raise KeyboardInterrupt from signal handler
    # The main thread will check the flag periodically


def is_shutdown_requested() -> bool:
    """Check if shutdown has been requested."""
    with shutdown_lock:
        return shutdown_requested


def interruptible_sleep(duration: float, check_interval: float = 0.1) -> None:
    """Sleep for the specified duration, but check for shutdown periodically.
    
    Args:
        duration: Total seconds to sleep
        check_interval: How often to check for shutdown (default: 0.1 seconds)
    
    Raises:
        KeyboardInterrupt: If CTRL-C is pressed during sleep
    """
    elapsed = 0.0
    while elapsed < duration:
        if is_shutdown_requested():
            break
        sleep_time = min(check_interval, duration - elapsed)
        try:
            time.sleep(sleep_time)
        except KeyboardInterrupt:
            # Propagate KeyboardInterrupt immediately
            with shutdown_lock:
                shutdown_requested = True
            raise
        elapsed += sleep_time


def get_pending_experiments(db_manager: DatabaseManager, reset_stale_running: bool = True, stale_timeout_hours: int = 24) -> List[str]:
    """Get list of experiment IDs with status='pending'.
    
    Also resets experiments that have been in 'running' status for too long back to 'pending'.
    
    Args:
        db_manager: Database manager instance
        reset_stale_running: If True, reset stale 'running' experiments to 'pending'
        stale_timeout_hours: Hours after which a 'running' experiment is considered stale
        
    Returns:
        List of experiment IDs that are pending
    """
    if not db_manager.enabled:
        return []
    
    try:
        conn = db_manager.get_read_connection()
        if conn is None:
            return []
        
        # Reset stale 'running' experiments back to 'pending'
        if reset_stale_running:
            try:
                write_conn = db_manager.get_write_connection()
                if write_conn:
                    # Reset experiments that have been running for more than stale_timeout_hours
                    # We check created_at since we don't have an 'updated_at' field
                    # This is approximate but should work for most cases
                    # Note: This assumes experiments in 'running' status that were created
                    # more than stale_timeout_hours ago are stale (orphaned from crashed workers)
                    # SQLite datetime subtraction: datetime('now', '-' || hours || ' hours')
                    cursor = write_conn.execute(
                        f"""
                        UPDATE experiments 
                        SET status = 'pending' 
                        WHERE status = 'running' 
                        AND datetime(created_at) < datetime('now', '-{stale_timeout_hours} hours')
                        """
                    )
                    reset_count = cursor.rowcount
                    write_conn.commit()
                    if reset_count > 0:
                        logger.info(f"Reset {reset_count} stale 'running' experiments back to 'pending' status")
            except Exception as e:
                logger.error(f"Failed to reset stale running experiments: {e}", exc_info=True)
                raise
        
        # Get pending experiments
        cursor = conn.execute(
            "SELECT experiment_id FROM experiments WHERE status = 'pending' ORDER BY created_at ASC",
            []
        )
        results = cursor.fetchall()
        return [row[0] for row in results]
    except Exception as e:
        logger.error(f"Failed to query pending experiments: {e}", exc_info=True)
        raise


def claim_experiment(db_manager: DatabaseManager, experiment_id: str) -> bool:
    """Atomically claim an experiment by updating its status from 'pending' to 'running'.
    
    This uses an atomic UPDATE with WHERE clause to prevent race conditions
    when multiple workers try to claim the same experiment.
    
    Args:
        db_manager: Database manager instance
        experiment_id: Experiment ID to claim
        
    Returns:
        True if successfully claimed, False if already claimed by another worker
    """
    if not db_manager.enabled:
        return False
    
    conn = db_manager.get_write_connection()
    if conn is None:
        logger.warning("Cannot claim experiment - no write connection available")
        return False
    
    try:
        # Atomic update: only update if status is still 'pending'
        cursor = conn.execute(
            "UPDATE experiments SET status = 'running' WHERE experiment_id = ? AND status = 'pending'",
            [experiment_id]
        )
        conn.commit()
        
        # Check if the update actually affected a row
        rows_affected = cursor.rowcount
        if rows_affected > 0:
            logger.debug(f"Claimed experiment {experiment_id} (status -> running)")
            return True
        else:
            # Another worker must have claimed it
            logger.debug(f"Failed to claim experiment {experiment_id} (already claimed or not pending)")
            return False
    except Exception as e:
        logger.error(f"Failed to claim experiment {experiment_id}: {e}", exc_info=True)
        raise


def process_experiment(
    experiment_id: str,
    worker_id: Optional[int] = None,
    max_workers: Optional[int] = None,
    already_claimed: bool = False,
    debug: bool = False
) -> bool:
    """Process a single experiment.
    
    This function processes an experiment. If already_claimed is False, it will
    atomically claim the experiment (updates status from 'pending' to 'running').
    The run_experiment function will handle updating the status to 'completed' or 'failed' at the end.
    
    Args:
        experiment_id: Experiment ID to process
        worker_id: Optional worker ID for logging
        max_workers: Maximum number of parallel workers for game execution
        already_claimed: If True, assumes experiment is already claimed (status='running')
        
    Returns:
        True if experiment completed successfully, False otherwise
    """
    worker_prefix = f"[Worker {worker_id}]" if worker_id is not None else "[Worker]"
    logger.info(f"{worker_prefix} Processing experiment {experiment_id}")
    
    db_path = CONFIG.get("db_path", "data/game_results.db")
    db_enabled = CONFIG.get("db_enabled", True)
    
    try:
        # Only claim if not already claimed
        if not already_claimed:
            db_manager = DatabaseManager(db_path=db_path, enabled=db_enabled)
            
            if not claim_experiment(db_manager, experiment_id):
                logger.warning(f"{worker_prefix} Could not claim experiment {experiment_id} (already claimed or not pending), skipping")
                db_manager.close()
                return False
            
            db_manager.close()
        
        # Check for shutdown before starting long-running experiment
        if is_shutdown_requested():
            logger.info(f"{worker_prefix} Shutdown requested, aborting experiment {experiment_id}")
            # Reset experiment to pending so another worker can pick it up
            try:
                db_manager = DatabaseManager(db_path=db_path, enabled=db_enabled)
                db_manager.update_experiment_status(experiment_id, "pending")
                db_manager.close()
            except Exception as reset_error:
                logger.error(f"{worker_prefix} Failed to reset experiment {experiment_id} to pending: {reset_error}", exc_info=True)
                raise RuntimeError(f"Failed to reset experiment {experiment_id} to pending: {reset_error}") from reset_error
        
        # Now run the experiment (run_experiment will update status to 'completed' or 'failed' at the end)
        # Note: run_experiment also tries to update status to 'running', but since we've already claimed it,
        # this is idempotent and harmless
        # Always use real agents (no mock agents)
        # In debug mode, force max_workers to 1 for single-threaded execution
        effective_max_workers = 1 if debug else max_workers
        success = run_experiment(
            experiment_id=experiment_id,
            use_mock_agents=False,
            max_workers=effective_max_workers,
            debug=debug
        )
        
        logger.info(f"{worker_prefix} Completed experiment {experiment_id}: {'SUCCESS' if success else 'FAILED'}")
        return success
        
    except Exception as e:
        logger.error(f"{worker_prefix} Error processing experiment {experiment_id}: {e}", exc_info=True)
        
        # Try to update status to 'failed' if there was an error
        try:
            db_manager = DatabaseManager(db_path=db_path, enabled=db_enabled)
            db_manager.update_experiment_status(experiment_id, "failed")
            db_manager.close()
        except Exception as update_error:
            logger.error(f"{worker_prefix} Failed to update experiment {experiment_id} status to 'failed': {update_error}", exc_info=True)
            # Reraise the update error - this is a secondary failure
            raise
        
        # Reraise the original error
        raise


def worker_loop(
    worker_id: int,
    game_workers: Optional[int],
    poll_interval: float,
    stale_timeout_hours: int = 1,
    debug: bool = False
):
    """Main loop for a worker thread.
    
    Args:
        worker_id: Unique identifier for this worker
        game_workers: Maximum number of parallel workers for game execution
        poll_interval: Seconds to wait between polling cycles
        stale_timeout_hours: Hours after which a 'running' experiment is considered stale (default: 1)
    """
    logger.info(f"[Worker {worker_id}] Starting worker loop")
    
    db_path = CONFIG.get("db_path", "data/game_results.db")
    db_enabled = CONFIG.get("db_enabled", True)
    poll_count = 0
    
    while True:
        # Check for shutdown
        if is_shutdown_requested():
            logger.info(f"[Worker {worker_id}] Shutdown requested, exiting")
            break
        
        try:
            # Initialize database manager for this iteration
            db_manager = DatabaseManager(db_path=db_path, enabled=db_enabled)
            
            if not db_manager.enabled:
                logger.error(f"[Worker {worker_id}] Database is not enabled, waiting...")
                db_manager.close()
                interruptible_sleep(poll_interval)
                continue
            
            # Get pending experiments (also resets stale running experiments)
            # Only reset stale experiments on worker 0 to avoid race conditions
            reset_stale = (worker_id == 0)
            pending_experiments = get_pending_experiments(db_manager, reset_stale_running=reset_stale, stale_timeout_hours=stale_timeout_hours)
            db_manager.close()
            
            poll_count += 1
            
            if not pending_experiments:
                # Log at INFO level periodically so user can see the worker is active and polling
                # Only log every 6th poll (every 30s with 5s interval) to reduce log spam
                if poll_count % 6 == 0:
                    logger.info(f"[Worker {worker_id}] No pending experiments found (polling every {poll_interval}s)...")
                interruptible_sleep(poll_interval)
                continue
            
            # Try to claim and process one experiment
            claimed = False
            for experiment_id in pending_experiments:
                # Double-check shutdown before processing
                if is_shutdown_requested():
                    logger.info(f"[Worker {worker_id}] Shutdown requested during experiment claim")
                    return
                
                # Try to claim this experiment
                db_manager = DatabaseManager(db_path=db_path, enabled=db_enabled)
                if claim_experiment(db_manager, experiment_id):
                    db_manager.close()
                    claimed = True
                    logger.debug(f"[Worker {worker_id}] Successfully claimed experiment {experiment_id}, starting processing...")
                    # Check shutdown one more time before starting long-running process
                    if is_shutdown_requested():
                        logger.info(f"[Worker {worker_id}] Shutdown requested, skipping experiment {experiment_id}")
                        # Reset experiment to pending so another worker can pick it up
                        try:
                            db_manager = DatabaseManager(db_path=db_path, enabled=db_enabled)
                            db_manager.update_experiment_status(experiment_id, "pending")
                            db_manager.close()
                        except Exception as reset_error:
                            logger.error(f"[Worker {worker_id}] Failed to reset experiment {experiment_id} to pending: {reset_error}", exc_info=True)
                            raise
                        break
                    # Process the experiment (already claimed, so pass already_claimed=True)
                    process_experiment(
                        experiment_id=experiment_id,
                        worker_id=worker_id,
                        max_workers=game_workers,
                        already_claimed=True,
                        debug=debug
                    )
                    break  # Process one experiment per iteration
                else:
                    db_manager.close()
                    # Try next experiment
                    continue
            
            if not claimed:
                # Could not claim any experiment (all were already claimed by other workers)
                logger.debug(f"[Worker {worker_id}] Could not claim any experiment (already claimed by other workers), waiting {poll_interval}s...")
                interruptible_sleep(poll_interval)
                
        except Exception as e:
            logger.error(f"[Worker {worker_id}] Error in worker loop: {e}", exc_info=True)
            raise
    
    logger.info(f"[Worker {worker_id}] Worker loop ended")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Watch database and process pending experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python experiment_worker.py --workers 4
  python experiment_worker.py --workers 4 --poll-interval 10
  python experiment_worker.py --workers 4 --game-workers 5
        """
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker threads to process experiments in parallel (default: 1)"
    )
    
    parser.add_argument(
        "--game-workers",
        type=int,
        default=None,
        help="Maximum number of parallel workers for game execution within each experiment (default: min(10, number_of_games))"
    )
    
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Seconds to wait between polling cycles (default: 5.0)"
    )
    
    parser.add_argument(
        "--stale-timeout-hours",
        type=int,
        default=1,
        help="Hours after which a 'running' experiment is considered stale and reset to 'pending' (default: 1)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode: single-threaded execution in main thread (no parallel workers)"
    )
    
    args = parser.parse_args()
    
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    
    if args.poll_interval < 0.1:
        parser.error("--poll-interval must be at least 0.1 seconds")
    
    # Setup logging
    log_dir = CONFIG.get("log_dir", "logs")
    setup_logging(log_dir=log_dir)
    logger.info(f"Logging initialized (logs directory: {log_dir})")
    
    # Setup signal handlers for graceful shutdown
    # Note: On Windows, signal handlers can interfere with KeyboardInterrupt
    # So we only set them on Unix-like systems
    import platform
    if platform.system() != 'Windows':
        try:
            signal.signal(signal.SIGINT, signal_handler)
        except (AttributeError, ValueError):
            pass
        try:
            signal.signal(signal.SIGTERM, signal_handler)
        except (AttributeError, ValueError):
            pass
    # On Windows, we rely on KeyboardInterrupt being raised directly

    debug_mode = args.debug or os.getenv("DEBUG", "false").lower() == "true"
    
    logger.info("=" * 60)
    logger.info("EXPERIMENT WORKER STARTING")
    logger.info("=" * 60)
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Game workers per experiment: {args.game_workers or 'auto'}")
    logger.info(f"Poll interval: {args.poll_interval}s")
    logger.info(f"Stale timeout: {args.stale_timeout_hours} hours")
    logger.info(f"Agent mode: REAL (API calls enabled)")
    logger.info(f"Debug mode: {'ENABLED (single-threaded)' if debug_mode else 'DISABLED (parallel)'}")
    logger.info("=" * 60)
    
    # In debug mode, run in main thread
    if debug_mode:
        logger.info("Running in DEBUG mode: single-threaded execution in main thread")
        # Override game_workers to 1 for single-threaded game execution
        debug_game_workers = 1
        logger.info(f"Game workers set to 1 for debug mode")
        
        # Run worker loop directly in main thread
        try:
            worker_loop(
                worker_id=0,
                game_workers=debug_game_workers,
                poll_interval=args.poll_interval,
                stale_timeout_hours=args.stale_timeout_hours,
                debug=True
            )
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received (CTRL-C), exiting...")
        except Exception as e:
            logger.error(f"Error in debug mode: {e}", exc_info=True)
            raise
        
        logger.info("=" * 60)
        logger.info("EXPERIMENT WORKER SHUTDOWN COMPLETE")
        logger.info("=" * 60)
        return
    
    # Start worker threads (normal mode)
    worker_threads = []
    for worker_id in range(args.workers):
        thread = threading.Thread(
            target=worker_loop,
            args=(worker_id, args.game_workers, args.poll_interval, args.stale_timeout_hours, False),
            name=f"Worker-{worker_id}",
            daemon=False  # Don't allow daemon threads so they can finish gracefully
        )
        thread.start()
        worker_threads.append(thread)
        logger.info(f"Started worker {worker_id}")
    
    try:
        # Wait for all worker threads, but check for shutdown periodically
        while True:
            # Check if shutdown was requested
            if is_shutdown_requested():
                logger.info("Shutdown requested, exiting immediately...")
                os._exit(0)
            
            # Check if all threads are done
            all_done = all(not thread.is_alive() for thread in worker_threads)
            if all_done:
                logger.info("All workers finished")
                break
            
            # Wait a short time and check again
            # Use try/except to catch KeyboardInterrupt immediately
            try:
                interruptible_sleep(0.5)
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received (CTRL-C), exiting immediately...")
                # Exit immediately without waiting for threads
                os._exit(0)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received (CTRL-C), exiting immediately...")
        # Exit immediately without waiting for threads
        os._exit(0)
    
    logger.info("=" * 60)
    logger.info("EXPERIMENT WORKER SHUTDOWN COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

