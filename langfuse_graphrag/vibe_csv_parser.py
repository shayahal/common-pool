"""CSV parser for Vibe Bench Langfuse exports.

Handles parsing of Vibe Bench CSV exports containing agent traces and operations.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


class VibeBenchCSVParser:
    """Parser for Vibe Bench Langfuse CSV exports."""
    
    def __init__(self, csv_path: str):
        """Initialize parser with CSV file path.
        
        Args:
            csv_path: Path to Vibe Bench CSV export file
        """
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        logger.debug(f"Initialized Vibe Bench CSV parser for: {csv_path}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of CSV file.
        
        Returns:
            Dictionary with summary information
        """
        try:
            df = pd.read_csv(self.csv_path, nrows=100)
            return {
                "total_rows_estimated": len(pd.read_csv(self.csv_path)),
                "columns": list(df.columns),
                "column_count": len(df.columns),
                "sample_names": df['name'].value_counts().head(10).to_dict() if 'name' in df.columns else {},
            }
        except Exception as e:
            logger.error(f"Error getting CSV summary: {e}", exc_info=True)
            raise
    
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
        
        if isinstance(value, str):
            # Try ISO format
            try:
                return datetime.fromisoformat(value.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                pass
            
            # Try common formats
            formats = [
                '%Y-%m-%dT%H:%M:%S.%fZ',
                '%Y-%m-%dT%H:%M:%SZ',
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d',
            ]
            for fmt in formats:
                try:
                    return datetime.strptime(value, fmt)
                except ValueError:
                    continue
        
        return None
    
    def _strip_quotes(self, value: Any) -> Any:
        """Strip surrounding quotes from string if present.
        
        Args:
            value: Value to strip quotes from
            
        Returns:
            Stripped value
        """
        if not isinstance(value, str):
            return value
        
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            return value[1:-1]
        return value
    
    def parse(self) -> List[Dict[str, Any]]:
        """Parse CSV file and return records.
        
        Returns:
            List of record dictionaries
        """
        try:
            logger.info(f"Parsing Vibe Bench CSV: {self.csv_path}")
            
            # Read CSV
            df = pd.read_csv(self.csv_path)
            logger.info(f"Read {len(df)} rows from CSV")
            
            # Critical fields to preserve original values for
            critical_fields = ['id', 'sessionId', 'name']
            original_values = {}
            for field in critical_fields:
                if field in df.columns:
                    original_values[field] = df[field].copy()
            
            # Parse records
            records = []
            for idx, row in df.iterrows():
                record = {"_csv_type": "trace", "_row_index": idx}
                
                # Preserve original values for critical fields
                for field in critical_fields:
                    if field in original_values:
                        record[f"_original_{field}"] = original_values[field][idx]
                
                # Process each column
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
                    elif col in ["input", "output"]:
                        # Strip quotes from input/output if they're JSON strings
                        record[col] = self._strip_quotes(value)
                    elif col == "tags":
                        # Parse tags if it's a JSON string
                        if isinstance(value, str):
                            try:
                                record[col] = json.loads(value)
                            except json.JSONDecodeError:
                                record[col] = value
                        else:
                            record[col] = value
                    else:
                        record[col] = value
                
                records.append(record)
            
            logger.info(f"Parsed {len(records)} records")
            return records
            
        except Exception as e:
            logger.error(f"Error parsing CSV: {e}", exc_info=True)
            raise

