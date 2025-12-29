"""CSV parser for Langfuse exports.

Handles parsing of Langfuse CSV exports containing traces, sessions, generations, and scores.
"""

import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator
from datetime import datetime
import pandas as pd

import logging

logger = logging.getLogger(__name__)


class LangfuseCSVParser:
    """Parser for Langfuse CSV exports."""
    
    # Common column name mappings for different Langfuse export formats
    COLUMN_MAPPINGS = {
        "trace": {
            "id": ["id", "trace_id", "traceId"],
            "name": ["name", "trace_name"],
            "session_id": ["session_id", "sessionId", "session"],
            "timestamp": ["timestamp", "created_at", "createdAt"],
            "user_id": ["user_id", "userId", "user"],
            "metadata": ["metadata", "meta"],
            "input": ["input", "inputs"],
            "output": ["output", "outputs"],
            "release": ["release", "version"],
            "tags": ["tags"],
            "public": ["public"],
        },
        "session": {
            "id": ["id", "session_id", "sessionId"],
            "name": ["name", "session_name"],
            "user_id": ["user_id", "userId", "user"],
            "created_at": ["created_at", "createdAt", "timestamp"],
            "updated_at": ["updated_at", "updatedAt"],
            "metadata": ["metadata", "meta"],
        },
        "generation": {
            "id": ["id", "generation_id", "generationId"],
            "trace_id": ["trace_id", "traceId", "trace"],
            "span_id": ["span_id", "spanId", "span"],
            "model": ["model", "model_name", "modelName"],
            "prompt": ["prompt", "input", "inputs"],
            "response": ["response", "output", "outputs", "completion"],
            "system_prompt": ["system_prompt", "systemPrompt", "system"],
            "reasoning": ["reasoning", "chain_of_thought", "chainOfThought"],
            "tokens_input": ["tokens_input", "input_tokens", "inputTokens", "prompt_tokens"],
            "tokens_output": ["tokens_output", "output_tokens", "outputTokens", "completion_tokens"],
            "cost": ["cost", "total_cost", "totalCost"],
            "latency_ms": ["latency_ms", "latency", "duration_ms", "durationMs"],
            "temperature": ["temperature", "temp"],
            "metadata": ["metadata", "meta"],
        },
        "score": {
            "id": ["id", "score_id", "scoreId"],
            "trace_id": ["trace_id", "traceId", "trace"],
            "span_id": ["span_id", "spanId", "span"],
            "name": ["name", "score_name", "metric_name"],
            "value": ["value", "score_value"],
            "comment": ["comment", "description"],
            "timestamp": ["timestamp", "created_at", "createdAt"],
        },
        "span": {
            "id": ["id", "span_id", "spanId", "observationId"],
            "trace_id": ["trace_id", "traceId", "trace"],
            "parent_id": ["parent_id", "parentId", "parentObservationId", "parent_observation_id"],
            "name": ["name", "span_name"],
            "type": ["type", "span_type", "observationType"],
            "start_time": ["start_time", "startTime", "start", "createdAt"],
            "end_time": ["end_time", "endTime", "end", "completedAt"],
            "duration_ms": ["duration_ms", "duration", "durationMs", "latency"],
            "status": ["status", "state", "level"],
            "input": ["input", "inputs"],
            "output": ["output", "outputs"],
            "metadata": ["metadata", "meta"],
        },
    }
    
    def __init__(self, csv_path: str):
        """Initialize parser with CSV file path.
        
        Args:
            csv_path: Path to Langfuse CSV export file
        """
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        logger.debug(f"Initialized CSV parser for: {csv_path}")
    
    def detect_csv_type(self) -> str:
        """Detect the type of CSV (trace, session, generation, score, span).
        
        Returns:
            CSV type string
        """
        try:
            df = pd.read_csv(self.csv_path, nrows=1)
            columns = [col.lower() for col in df.columns]
            
            # Check for key identifying columns
            if any(col in columns for col in ["generation_id", "generationid", "model"]):
                return "generation"
            elif any(col in columns for col in ["span_id", "spanid", "span_type"]):
                return "span"
            elif any(col in columns for col in ["score_id", "scoreid", "score_name"]):
                return "score"
            elif any(col in columns for col in ["session_id", "sessionid"]):
                if any(col in columns for col in ["trace_id", "traceid"]):
                    return "trace"
                else:
                    return "session"
            elif any(col in columns for col in ["trace_id", "traceid"]):
                return "trace"
            else:
                # Default to trace if we can't determine
                logger.warning(f"Could not determine CSV type, defaulting to 'trace'. Columns: {columns}")
                return "trace"
        except Exception as e:
            logger.error(f"Error detecting CSV type: {e}", exc_info=True)
            raise
    
    def _normalize_column_name(self, col: str, csv_type: str) -> Optional[str]:
        """Normalize column name to standard format.
        
        Args:
            col: Original column name
            csv_type: Type of CSV (trace, session, etc.)
        
        Returns:
            Normalized column name or None if not found
        """
        col_lower = col.lower()
        mappings = self.COLUMN_MAPPINGS.get(csv_type, {})
        
        for standard_name, variants in mappings.items():
            if col_lower in [v.lower() for v in variants]:
                return standard_name
        
        # Return original if no mapping found
        return col.lower()
    
    def _parse_metadata(self, value: Any) -> Optional[str]:
        """Parse metadata field (may be JSON string or dict).
        
        Args:
            value: Metadata value
        
        Returns:
            JSON string or None
        """
        if value is None or pd.isna(value):
            return None
        
        if isinstance(value, str):
            try:
                # Try to parse and re-serialize to ensure valid JSON
                parsed = json.loads(value)
                return json.dumps(parsed)
            except (json.JSONDecodeError, TypeError):
                # If not JSON, wrap as string
                return json.dumps({"raw": str(value)})
        elif isinstance(value, dict):
            return json.dumps(value)
        else:
            return json.dumps({"raw": str(value)})
    
    def _parse_datetime(self, value: Any) -> Optional[datetime]:
        """Parse datetime field.
        
        Args:
            value: Datetime value (string, timestamp, etc.)
        
        Returns:
            datetime object or None
        """
        if value is None or pd.isna(value):
            return None
        
        if isinstance(value, datetime):
            return value
        
        if isinstance(value, (int, float)):
            # Assume Unix timestamp
            try:
                return datetime.fromtimestamp(value)
            except (ValueError, OSError):
                return None
        
        if isinstance(value, str):
            # Try common datetime formats
            formats = [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%d %H:%M:%S.%f",
            ]
            for fmt in formats:
                try:
                    return datetime.strptime(value, fmt)
                except ValueError:
                    continue
        
        return None
    
    def parse(self) -> List[Dict[str, Any]]:
        """Parse CSV file and return list of normalized records.
        
        Returns:
            List of dictionaries with normalized field names
        """
        csv_type = self.detect_csv_type()
        logger.info(f"Parsing {csv_type} CSV: {self.csv_path}")
        
        try:
            df = pd.read_csv(self.csv_path)
            logger.debug(f"Loaded {len(df)} rows from CSV")
            
            # Normalize column names
            column_mapping = {}
            for col in df.columns:
                normalized = self._normalize_column_name(col, csv_type)
                if normalized:
                    column_mapping[col] = normalized
            
            # Rename columns
            df = df.rename(columns=column_mapping)
            
            # Parse records
            records = []
            for idx, row in df.iterrows():
                record = {"_csv_type": csv_type, "_row_index": idx}
                
                for col, value in row.items():
                    if pd.isna(value):
                        continue
                    
                    # Handle special parsing for certain fields
                    if "metadata" in col.lower():
                        record[col] = self._parse_metadata(value)
                    elif any(dt_keyword in col.lower() for dt_keyword in ["time", "date", "timestamp", "created", "updated"]):
                        dt = self._parse_datetime(value)
                        if dt:
                            record[col] = dt
                        else:
                            record[col] = value
                    else:
                        record[col] = value
                
                records.append(record)
            
            logger.info(f"Parsed {len(records)} {csv_type} records")
            return records
            
        except Exception as e:
            logger.error(f"Error parsing CSV: {e}", exc_info=True)
            raise
    
    def parse_streaming(self) -> Iterator[Dict[str, Any]]:
        """Parse CSV file in streaming fashion (for large files).
        
        Yields:
            Dictionary with normalized field names
        """
        csv_type = self.detect_csv_type()
        logger.info(f"Streaming parse of {csv_type} CSV: {self.csv_path}")
        
        try:
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                # Build column mapping
                column_mapping = {}
                for col in reader.fieldnames or []:
                    normalized = self._normalize_column_name(col, csv_type)
                    if normalized:
                        column_mapping[col] = normalized
                
                for idx, row in enumerate(reader):
                    record = {"_csv_type": csv_type, "_row_index": idx}
                    
                    for col, value in row.items():
                        if not value or value.strip() == "":
                            continue
                        
                        normalized_col = column_mapping.get(col, col.lower())
                        
                        # Handle special parsing
                        if "metadata" in normalized_col:
                            record[normalized_col] = self._parse_metadata(value)
                        elif any(dt_keyword in normalized_col for dt_keyword in ["time", "date", "timestamp", "created", "updated"]):
                            dt = self._parse_datetime(value)
                            if dt:
                                record[normalized_col] = dt
                            else:
                                record[normalized_col] = value
                        else:
                            record[normalized_col] = value
                    
                    yield record
                    
        except Exception as e:
            logger.error(f"Error streaming CSV: {e}", exc_info=True)
            raise
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics about the CSV file.
        
        Returns:
            Dictionary with summary information
        """
        csv_type = self.detect_csv_type()
        
        try:
            df = pd.read_csv(self.csv_path)
            
            return {
                "csv_type": csv_type,
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": list(df.columns),
                "file_size": os.path.getsize(self.csv_path),
            }
        except Exception as e:
            logger.error(f"Error getting CSV summary: {e}", exc_info=True)
            raise

