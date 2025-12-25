"""FalkorDB exporter for OpenTelemetry traces using Graphiti.

This module provides a bridge between OpenTelemetry traces and FalkorDB
by converting traces to Graphiti episodes and storing them in a knowledge graph.
"""

import asyncio
import json
import logging
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Sequence

try:
    from graphiti_core import Graphiti
    from graphiti_core.driver.falkordb_driver import FalkorDriver
    from graphiti_core.nodes import EpisodeType
    GRAPHITI_AVAILABLE = True
except ImportError:
    GRAPHITI_AVAILABLE = False

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from .logger_setup import get_logger
from .falkordb_manager import ensure_falkordb_running

logger = get_logger(__name__)


class FalkorDBExporter(SpanExporter):
    """OpenTelemetry span exporter that sends traces to FalkorDB via Graphiti.
    
    Converts OpenTelemetry spans to Graphiti episodes and stores them
    in a FalkorDB knowledge graph for querying and analysis.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        username: Optional[str] = None,
        password: Optional[str] = None,
        group_id: str = "cpr-game-traces",
        enabled: bool = True,
        max_retries: int = 10,  # Increased for rate limits
        base_retry_delay: float = 5.0,  # Increased base delay
        max_retry_delay: float = 300.0,  # 5 minutes max delay for rate limits
        export_timeout: float = 60 * 60.0,  # 60 minutes default (increased for rate limits)
        episode_rate_limit: float = 1.0  # Minimum seconds between episode additions (throttling to prevent rate limits)
    ):
        """Initialize FalkorDB exporter.
        
        Args:
            host: FalkorDB host
            port: FalkorDB port
            username: FalkorDB username (optional)
            password: FalkorDB password (optional)
            group_id: Group ID for organizing episodes in the graph
            enabled: Whether the exporter is enabled
            max_retries: Maximum number of retries for transient errors (default: 5)
            base_retry_delay: Base delay in seconds for exponential backoff (default: 2.0)
            max_retry_delay: Maximum delay in seconds between retries (default: 60.0)
            export_timeout: Maximum time in seconds for entire export operation (default: 30 minutes)
        """
        self.enabled = enabled and GRAPHITI_AVAILABLE
        self.group_id = group_id
        self.graphiti = None
        self._initialized = False
        self._executor = None
        self._loop = None
        self._loop_thread = None
        self.max_retries = max_retries
        self.base_retry_delay = base_retry_delay
        self.max_retry_delay = max_retry_delay
        self.export_timeout = export_timeout
        self.episode_rate_limit = episode_rate_limit
        self._last_episode_time = 0.0  # Track last episode addition time for throttling
        
        if not enabled:
            return
        
        if not GRAPHITI_AVAILABLE:
            raise ImportError(
                "Graphiti is not available. Install with: pip install graphiti-core[falkordb]"
            )
        
        # Store connection parameters (will create Graphiti in background loop)
        self._host = host
        self._port = port
        self._username = username
        self._password = password
        
        # Ensure FalkorDB container is running before attempting connection
        if not ensure_falkordb_running():
            error_msg = (
                "Failed to start FalkorDB container. "
                "Please ensure Docker is running and try again."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Initialize Graphiti connection (test that it works)
        try:
            falkor_driver = FalkorDriver(
                host=host,
                port=str(port),
                username=username,
                password=password,
            )
            # Just test that Graphiti can be created (will recreate in background loop)
            test_graphiti = Graphiti(graph_driver=falkor_driver)
            logger.info(f"FalkorDB exporter initialized: {host}:{port}")
            
            # Start background event loop for async exports
            self._start_background_loop()
        except Exception as e:
            # Check if it's a missing API key error
            error_str = str(e).lower()
            if "api_key" in error_str or "api key" in error_str:
                logger.warning(
                    f"FalkorDB exporter requires OPENAI_API_KEY environment variable. "
                    f"FalkorDB export will be disabled. Error: {e}"
                )
                self.enabled = False
                self.graphiti = None
            else:
                # For other errors, re-raise
                raise
    
    def _start_background_loop(self):
        """Start a background thread with its own event loop for async exports."""
        if self._loop_thread is not None and self._loop_thread.is_alive():
            return
        
        def run_loop():
            """Run event loop in background thread."""
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            
            # Create Graphiti instance in this event loop
            # NO SILENT FAILURES - if this fails, we must know about it
            try:
                falkor_driver = FalkorDriver(
                    host=self._host,
                    port=str(self._port),
                    username=self._username,
                    password=self._password,
                )
                self.graphiti = Graphiti(graph_driver=falkor_driver)
                logger.info("Graphiti instance created in background event loop")
            except Exception as e:
                # This is a FATAL error - log it and disable exporter
                error_msg = f"FATAL: Failed to create Graphiti in background loop: {e}"
                logger.error(error_msg, exc_info=True)
                self.enabled = False
                self.graphiti = None
                # Note: We can't raise here because we're in a background thread
                # But the export() method will check and raise if graphiti is None
            
            self._loop.run_forever()
        
        self._loop_thread = threading.Thread(target=run_loop, daemon=True, name="FalkorDBExporter")
        self._loop_thread.start()
        
        # Wait for the loop to start and Graphiti to be created
        import time
        max_wait = 5.0  # Wait up to 5 seconds
        wait_interval = 0.1
        waited = 0.0
        while waited < max_wait:
            if self._loop is not None and self._loop.is_running() and self.graphiti is not None:
                logger.info("Graphiti instance ready in background event loop")
                break
            time.sleep(wait_interval)
            waited += wait_interval
        
        if self.graphiti is None:
            # This is a problem - Graphiti should be ready by now
            error_msg = (
                f"FATAL: Graphiti instance not created in background loop after {waited:.1f}s. "
                "FalkorDB export will fail. Check logs for initialization errors."
            )
            logger.error(error_msg)
            # Don't raise here - let export() method handle it and raise with better context
        else:
            logger.info("FalkorDB exporter background loop ready")
    
    async def _initialize_indices(self):
        """Initialize Graphiti indices and constraints (async).
        
        NO SILENT FAILURES - all errors are logged.
        Index initialization failures are logged but don't block export (indices might already exist).
        """
        if not self.enabled or self._initialized:
            return
        
        try:
            await self.graphiti.build_indices_and_constraints()
            self._initialized = True
            logger.info("Graphiti indices and constraints built successfully")
        except Exception as e:
            # Index initialization can fail for various reasons:
            # 1. Indices already exist (some Graphiti versions handle this gracefully, others don't)
            # 2. RediSearch syntax errors (version incompatibility between Graphiti and FalkorDB)
            # 3. Permission issues or FalkorDB configuration problems
            # 
            # We log the error at ERROR level (NO SILENT FAILURES) but allow export to continue
            # because:
            # - Indices might already exist from previous runs
            # - Episodes can still be added without indices (just slower queries)
            # - This is a Graphiti/FalkorDB compatibility issue, not our code
            
            error_str = str(e).lower()
            error_msg = f"Graphiti index initialization failed: {e}"
            
            if "already exists" in error_str:
                # Indices already exist - this is usually fine
                logger.info(f"{error_msg}. Indices already exist, continuing with export.")
                self._initialized = True
            elif "syntax error" in error_str or "redisearch" in error_str:
                # RediSearch syntax error - likely version incompatibility
                logger.error(
                    f"{error_msg}. This is likely a Graphiti/FalkorDB version incompatibility. "
                    f"Export will continue but queries may be slow. "
                    f"Check Graphiti and FalkorDB versions for compatibility.",
                    exc_info=True
                )
                self._initialized = True  # Mark as initialized to avoid retry loops
            else:
                # Other errors - log as error but continue
                logger.error(
                    f"{error_msg}. Export will continue but indices may not be properly set up. "
                    f"Check FalkorDB logs and Graphiti version compatibility.",
                    exc_info=True
                )
                self._initialized = True  # Mark as initialized to avoid infinite retries
    
    def _span_to_episode(self, span: ReadableSpan) -> Dict[str, Any]:
        """Convert OpenTelemetry span to Graphiti episode format.
        
        Args:
            span: OpenTelemetry span
            
        Returns:
            Dictionary with episode data
        """
        # Extract span information
        span_name = span.name
        trace_id = format(span.context.trace_id, '032x')
        span_id = format(span.context.span_id, '016x')
        parent_span_id = format(span.parent.span_id, '016x') if span.parent else None
        
        # Build episode content
        episode_data = {
            "span_name": span_name,
            "trace_id": trace_id,
            "span_id": span_id,
            "parent_span_id": parent_span_id,
            "start_time": span.start_time / 1e9,  # Convert nanoseconds to seconds
            "end_time": span.end_time / 1e9 if span.end_time else None,
            "duration_ms": (span.end_time - span.start_time) / 1e6 if span.end_time else None,
            "status_code": span.status.status_code.name if span.status else None,
            "status_message": span.status.description if span.status else None,
        }
        
        # Add attributes
        if span.attributes:
            episode_data["attributes"] = dict(span.attributes)
        
        # Add events (prompts, responses, etc.)
        if span.events:
            events = []
            for event in span.events:
                event_data = {
                    "name": event.name,
                    "timestamp": event.timestamp / 1e9,
                    "attributes": dict(event.attributes) if event.attributes else {}
                }
                events.append(event_data)
            episode_data["events"] = events
        
        # Create a human-readable description
        description_parts = [f"Span: {span_name}"]
        if span.attributes:
            # Include key attributes in description
            if "game.id" in span.attributes:
                description_parts.append(f"Game: {span.attributes['game.id']}")
            if "round.number" in span.attributes:
                description_parts.append(f"Round: {span.attributes['round.number']}")
            if "player.id" in span.attributes:
                description_parts.append(f"Player: {span.attributes['player.id']}")
        
        description = " | ".join(description_parts)
        
        return {
            "content": json.dumps(episode_data),
            "type": EpisodeType.json,
            "description": description,
            "name": f"{span_name}_{span_id[:8]}",
        }
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if an error is retryable.
        
        Args:
            error: The exception to check
            
        Returns:
            True if the error is retryable, False otherwise
        """
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # Retryable errors
        retryable_indicators = [
            "rate limit",
            "rate_limit",
            "429",
            "too many requests",
            "request timed out",
            "timeout",
            "temporarily unavailable",
            "service unavailable",
            "503",
            "502",
            "connection",
            "network",
            "temporary",
        ]
        
        # Non-retryable errors (permanent failures)
        non_retryable_indicators = [
            "redisearch",
            "syntax error",
            "invalid",
            "not found",
            "404",
            "401",
            "403",
            "authentication",
            "authorization",
        ]
        
        # Check for non-retryable errors first
        for indicator in non_retryable_indicators:
            if indicator in error_str:
                return False
        
        # Check for retryable errors
        for indicator in retryable_indicators:
            if indicator in error_str or indicator in error_type:
                return True
        
        # Default: retry unknown errors (they might be transient)
        return True
    
    async def _add_episode_with_retry(
        self,
        episode: Dict[str, Any],
        reference_time: datetime,
        episode_name: str
    ) -> bool:
        """Add an episode to Graphiti with retry logic and exponential backoff.
        
        Args:
            episode: Episode data dictionary
            reference_time: Reference time for the episode
            episode_name: Name of the episode (for logging)
            
        Returns:
            True if successful, False if all retries failed
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                await self.graphiti.add_episode(
                    name=episode["name"],
                    episode_body=episode["content"],
                    source=episode["type"],
                    source_description=episode["description"],
                    reference_time=reference_time,
                    group_id=self.group_id
                )
                if attempt > 0:
                    logger.info(
                        f"Successfully added episode {episode_name} after {attempt + 1} attempts"
                    )
                return True
                
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                
                # Check if error is retryable
                if not self._is_retryable_error(e):
                    # Non-retryable error - log and fail immediately
                    if "redisearch" in error_str or "syntax error" in error_str:
                        error_msg = (
                            f"Failed to add episode {episode_name}: {e}. "
                            f"This is a Graphiti/FalkorDB compatibility issue (RediSearch syntax error). "
                            f"Check Graphiti and FalkorDB versions. Episode may still be stored."
                        )
                    else:
                        error_msg = f"Failed to add episode {episode_name}: {e} (non-retryable error)"
                    logger.error(error_msg, exc_info=True)
                    return False
                
                # Retryable error - check if we should retry
                if attempt < self.max_retries - 1:
                    # Calculate exponential backoff with jitter
                    delay = min(self.base_retry_delay * (2 ** attempt), self.max_retry_delay)
                    # For rate limits, use longer delays (double the calculated delay)
                    if "rate limit" in error_str or "429" in error_str:
                        delay = min(delay * 2, self.max_retry_delay)
                    jitter = random.uniform(0, delay * 0.1)  # Add up to 10% jitter
                    total_delay = delay + jitter
                    
                    # Determine error type for logging
                    if "rate limit" in error_str or "429" in error_str:
                        error_type = "rate limit"
                    elif "timeout" in error_str or "timed out" in error_str:
                        error_type = "timeout"
                    else:
                        error_type = "transient error"
                    
                    logger.warning(
                        f"Episode {episode_name} failed with {error_type} (attempt {attempt + 1}/{self.max_retries}): {e}. "
                        f"Retrying in {total_delay:.2f}s..."
                    )
                    await asyncio.sleep(total_delay)
                else:
                    # Max retries reached
                    if "rate limit" in error_str:
                        error_msg = (
                            f"Failed to add episode {episode_name} after {self.max_retries} attempts: {e}. "
                            f"OpenAI rate limit exceeded. All retries exhausted."
                        )
                    elif "timeout" in error_str or "timed out" in error_str:
                        error_msg = (
                            f"Failed to add episode {episode_name} after {self.max_retries} attempts: {e}. "
                            f"Request timeout. All retries exhausted."
                        )
                    else:
                        error_msg = (
                            f"Failed to add episode {episode_name} after {self.max_retries} attempts: {e}. "
                            f"All retries exhausted."
                        )
                    logger.error(error_msg, exc_info=True)
                    return False
        
        # Should not reach here, but just in case
        logger.error(f"Failed to add episode {episode_name}: {last_error}", exc_info=True)
        return False
    
    async def _export_spans_async(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Export spans to FalkorDB asynchronously.
        
        Args:
            spans: List of spans to export
            
        Returns:
            SpanExportResult indicating success or failure
            
        Raises:
            RuntimeError: If export fails (NO SILENT FAILURES)
        """
        if not self.enabled:
            # Only silently succeed if explicitly disabled
            return SpanExportResult.SUCCESS
        
        if not self.graphiti:
            error_msg = "FalkorDB exporter is enabled but Graphiti instance is not available - cannot export spans"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Initialize indices if needed (non-blocking - failures are logged but don't stop export)
        await self._initialize_indices()
        
        # Convert spans to episodes and add to graph
        # NO SILENT FAILURES - all failures are logged at ERROR level
        # Continue processing all episodes even if some fail
        failed_episodes = []
        successful_count = 0
        
        for i, span in enumerate(spans):
            episode = self._span_to_episode(span)
            
            # Get reference time from span
            reference_time = datetime.fromtimestamp(
                span.start_time / 1e9,
                tz=timezone.utc
            )
            
            # Rate limiting: throttle episode additions to prevent API rate limits
            # Graphiti's add_episode makes LLM API calls, so we need to space them out
            if self.episode_rate_limit > 0:
                import time
                current_time = time.time()
                time_since_last = current_time - self._last_episode_time
                if time_since_last < self.episode_rate_limit:
                    wait_time = self.episode_rate_limit - time_since_last
                    logger.debug(f"Throttling episode {i+1}/{len(spans)}: waiting {wait_time:.2f}s to prevent rate limits")
                    await asyncio.sleep(wait_time)
                self._last_episode_time = time.time()
            
            logger.info(f"Adding episode {i+1}/{len(spans)}: {episode['name']}")
            
            # Add episode to graph with retry logic
            # NO SILENT FAILURES - all failures are logged at ERROR level
            success = await self._add_episode_with_retry(
                episode=episode,
                reference_time=reference_time,
                episode_name=episode['name']
            )
            
            if success:
                logger.info(f"Successfully added episode: {episode['name']}")
                successful_count += 1
            else:
                # Failure already logged in _add_episode_with_retry
                failed_episodes.append((episode['name'], "Failed after retries"))
                # Continue with other episodes instead of failing immediately
        
        # Report results - NO SILENT FAILURES
        if failed_episodes:
            error_summary = (
                f"FalkorDB export completed with {len(failed_episodes)}/{len(spans)} failures. "
                f"Successful: {successful_count}, Failed: {len(failed_episodes)}. "
                f"Failed episodes: {[name for name, _ in failed_episodes[:5]]}"
            )
            logger.error(error_summary)
            # Raise error to indicate partial failure
            raise RuntimeError(
                f"FalkorDB export partially failed: {len(failed_episodes)}/{len(spans)} episodes failed. "
                f"First failure: {failed_episodes[0][1]}"
            )
        else:
            logger.info(f"Successfully exported all {len(spans)} spans to FalkorDB")
            return SpanExportResult.SUCCESS
    
    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Export spans to FalkorDB (synchronous wrapper).
        
        IMPORTANT: This method schedules the export asynchronously and returns immediately
        to avoid blocking other exporters (like Langfuse). The actual export happens in
        the background. This ensures Langfuse traces are sent immediately without waiting
        for FalkorDB to complete.
        
        Args:
            spans: List of spans to export
            
        Returns:
            SpanExportResult indicating success or failure (always SUCCESS to avoid blocking)
        """
        if not self.enabled:
            logger.debug(f"FalkorDB exporter disabled, skipping {len(spans)} spans")
            return SpanExportResult.SUCCESS
        
        if not spans:
            logger.debug("No spans to export")
            return SpanExportResult.SUCCESS
        
        logger.info(f"Scheduling export of {len(spans)} spans to FalkorDB (non-blocking)")
        
        # Wait for Graphiti to be ready if not yet (quick check, don't block long)
        if self.graphiti is None:
            import time
            logger.warning("Graphiti instance not ready, waiting up to 2 seconds...")
            for i in range(20):  # Wait up to 2 seconds (reduced from 5)
                if self.graphiti is not None:
                    logger.info(f"Graphiti instance ready after {i * 0.1:.1f}s")
                    break
                time.sleep(0.1)
            
            if self.graphiti is None:
                error_msg = (
                    "FATAL: Graphiti instance still not ready after waiting 2 seconds. "
                    "FalkorDB export is enabled but cannot export spans. "
                    "This indicates a serious initialization problem."
                )
                logger.error(error_msg)
                # Don't raise - schedule the export anyway, it will fail and be logged
                # This prevents blocking other exporters
        
        # Schedule export in background and return immediately (non-blocking)
        # This allows Langfuse and other exporters to flush without waiting
        if self._loop is not None and self._loop.is_running():
            # Wrap the async export to catch and log any exceptions
            async def export_with_error_handling():
                try:
                    return await self._export_spans_async(spans)
                except Exception as e:
                    # Log the error - NO SILENT FAILURES
                    logger.error(
                        f"FalkorDB async export failed: {e}. "
                        f"This error occurred in background export and was logged.",
                        exc_info=True
                    )
                    # Return failure but don't raise - we're in background task
                    return SpanExportResult.FAILURE
            
            # Schedule task in background loop but DON'T wait for it
            # The export will happen asynchronously
            task = asyncio.run_coroutine_threadsafe(
                export_with_error_handling(),
                self._loop
            )
            
            # Store the task so we can check its status later if needed
            # But return SUCCESS immediately to avoid blocking
            logger.debug(f"FalkorDB export scheduled asynchronously (task: {task})")
            return SpanExportResult.SUCCESS
        else:
            # Fallback: if no event loop, we need to run it
            # But this shouldn't happen normally
            logger.warning("No event loop available, running export synchronously (this may block)")
            try:
                return asyncio.run(self._export_spans_async(spans))
            except Exception as e:
                error_msg = f"FalkorDB export failed in new event loop: {e}"
                logger.error(error_msg, exc_info=True)
                # Return SUCCESS anyway to avoid blocking other exporters
                return SpanExportResult.SUCCESS
    
    def shutdown(self):
        """Shutdown the exporter."""
        # Stop background event loop
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._loop_thread is not None:
                self._loop_thread.join(timeout=5.0)
                if self._loop_thread.is_alive():
                    logger.warning("Background event loop thread did not stop within timeout")
        # Graphiti/FalkorDB connection cleanup if needed
        pass

