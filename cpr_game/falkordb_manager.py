"""FalkorDB Docker container management utility.

Provides functions to start, stop, and check the status of the FalkorDB container.
Automatically ensures FalkorDB is running when needed.
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

from .logger_setup import get_logger

logger = get_logger(__name__)

# Container name used in docker-compose
FALKORDB_CONTAINER_NAME = "falkordb"


def _run_docker_command(cmd: list[str], check: bool = False) -> Tuple[bool, str]:
    """Run a docker command and return success status and output.
    
    Args:
        cmd: Docker command as list of strings
        check: If True, raise exception on non-zero exit code
        
    Returns:
        Tuple of (success: bool, output: str)
    """
    try:
        result = subprocess.run(
            ["docker"] + cmd,
            capture_output=True,
            text=True,
            check=check
        )
        return result.returncode == 0, result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return False, e.stderr.strip()
    except FileNotFoundError:
        logger.error("Docker is not installed or not in PATH")
        return False, "Docker not found"


def _get_project_root() -> Path:
    """Get the project root directory (where docker-compose.yml is located)."""
    # Try to find project root by looking for docker-compose.yml
    current = Path(__file__).parent
    while current != current.parent:
        if (current / "docker-compose.yml").exists():
            return current
        current = current.parent
    # Fallback to current working directory
    return Path.cwd()


def _run_docker_compose_command(cmd: list[str], check: bool = False) -> Tuple[bool, str]:
    """Run a docker-compose command and return success status and output.
    
    Args:
        cmd: Docker compose command as list of strings
        check: If True, raise exception on non-zero exit code
        
    Returns:
        Tuple of (success: bool, output: str)
    """
    try:
        project_root = _get_project_root()
        result = subprocess.run(
            ["docker", "compose"] + cmd,
            capture_output=True,
            text=True,
            check=check,
            cwd=str(project_root)
        )
        return result.returncode == 0, result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return False, e.stderr.strip()
    except FileNotFoundError:
        logger.error("Docker Compose is not installed or not in PATH")
        return False, "Docker Compose not found"


def is_falkordb_running() -> bool:
    """Check if FalkorDB container is running.
    
    Returns:
        True if container is running, False otherwise
    """
    success, output = _run_docker_command(
        ["ps", "--filter", f"name={FALKORDB_CONTAINER_NAME}", "--format", "{{.Names}}"]
    )
    if not success:
        return False
    
    return FALKORDB_CONTAINER_NAME in output


def get_falkordb_status() -> str:
    """Get the status of FalkorDB container.
    
    Returns:
        Status string: "running", "stopped", "not_found", or "error"
    """
    # Check if container exists (running or stopped)
    success, output = _run_docker_command(
        ["ps", "-a", "--filter", f"name={FALKORDB_CONTAINER_NAME}", "--format", "{{.Status}}"]
    )
    
    if not success:
        return "error"
    
    if not output:
        return "not_found"
    
    if "Up" in output:
        return "running"
    else:
        return "stopped"


def start_falkordb(use_compose: bool = True) -> bool:
    """Start FalkorDB container.
    
    Args:
        use_compose: If True, use docker-compose. If False, use docker directly.
        
    Returns:
        True if started successfully, False otherwise
    """
    if is_falkordb_running():
        logger.info("FalkorDB is already running")
        return True
    
    if use_compose:
        logger.info("Starting FalkorDB using docker-compose...")
        success, output = _run_docker_compose_command(["up", "-d", "falkordb"])
        if success:
            logger.info("FalkorDB started successfully via docker-compose")
        else:
            logger.warning(f"Failed to start via docker-compose: {output}")
            # Fall back to direct docker command
            use_compose = False
    
    if not use_compose:
        logger.info("Starting FalkorDB using docker directly...")
        status = get_falkordb_status()
        
        if status == "stopped":
            # Container exists but is stopped
            success, output = _run_docker_command(["start", FALKORDB_CONTAINER_NAME])
            if success:
                logger.info("FalkorDB container started")
            else:
                logger.error(f"Failed to start container: {output}")
                return False
        elif status == "not_found":
            # Container doesn't exist, create it
            logger.info("Creating new FalkorDB container...")
            success, output = _run_docker_command([
                "run", "-d",
                "--name", FALKORDB_CONTAINER_NAME,
                "-p", "6379:6379",
                "-p", "3000:3000",
                "--restart", "unless-stopped",
                "falkordb/falkordb:latest"
            ])
            if success:
                logger.info("FalkorDB container created and started")
            else:
                logger.error(f"Failed to create container: {output}")
                return False
        else:
            logger.info("FalkorDB is already running")
            return True
    
    # Verify it's actually running
    if is_falkordb_running():
        logger.info("FalkorDB is now running")
        return True
    else:
        logger.error("FalkorDB start command succeeded but container is not running")
        return False


def stop_falkordb(use_compose: bool = True) -> bool:
    """Stop FalkorDB container.
    
    Args:
        use_compose: If True, use docker-compose. If False, use docker directly.
        
    Returns:
        True if stopped successfully, False otherwise
    """
    if not is_falkordb_running():
        logger.info("FalkorDB is not running")
        return True
    
    if use_compose:
        logger.info("Stopping FalkorDB using docker-compose...")
        success, output = _run_docker_compose_command(["stop", "falkordb"])
        if success:
            logger.info("FalkorDB stopped successfully via docker-compose")
        else:
            logger.warning(f"Failed to stop via docker-compose: {output}")
            # Fall back to direct docker command
            use_compose = False
    
    if not use_compose:
        logger.info("Stopping FalkorDB using docker directly...")
        success, output = _run_docker_command(["stop", FALKORDB_CONTAINER_NAME])
        if success:
            logger.info("FalkorDB container stopped")
        else:
            logger.error(f"Failed to stop container: {output}")
            return False
    
    # Verify it's actually stopped
    if not is_falkordb_running():
        logger.info("FalkorDB is now stopped")
        return True
    else:
        logger.error("FalkorDB stop command succeeded but container is still running")
        return False


def ensure_falkordb_running() -> bool:
    """Ensure FalkorDB is running, starting it if necessary.
    
    This is the main function to call when you need FalkorDB to be available.
    It will automatically start it if it's not running.
    
    Returns:
        True if FalkorDB is running (or was started), False otherwise
    """
    if is_falkordb_running():
        logger.debug("FalkorDB is already running")
        return True
    
    logger.info("FalkorDB is not running, starting it automatically...")
    return start_falkordb()

