"""
Change detection system for different data types.

This module provides intelligent change detection algorithms that can identify
when data has been modified and determine the nature of the changes.
"""

import logging
import hashlib
import json
from datetime import datetime, date
from typing import Dict, Any, List, Optional, Union, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """Types of data changes."""
    NO_CHANGE = "no_change"
    NEW_DATA = "new_data"
    UPDATED_VALUES = "updated_values"
    STRUCTURAL_CHANGE = "structural_change"
    DELETED_DATA = "deleted_data"
    SCHEMA_CHANGE = "schema_change"
    REBALANCING = "rebalancing"


@dataclass
class DataChange:
    """Represents a detected change in data."""
    change_type: ChangeType
    affected_fields: List[str]
    change_summary: str
    confidence: float
    metadata: Dict[str, Any]
    detected_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'change_type': self.change_type.value,
            'affected_fields': self.affected_fields,
            'change_summary': self.change_summary,
            'confidence': self.confidence,
            'metadata': self.metadata,
            'detected_at': self.detected_at.isoformat()
        }


class ChangeDetector:
    """
    Advanced change detection system for different data types.

    Features:
    - Schema change detection
    - Value-level change analysis
    - Statistical change detection
    - Confidence scoring
    - Change summarization
    """

    def __init__(self, sensitivity_level: float = 0.1):
        """
        Initialize change detector.

        Args:
            sensitivity_level: Sensitivity threshold for detecting changes (0.0 to 1.0)
        """
        self.sensitivity_level = sensitivity_level
        self.change_history: List[DataChange] = []

    def detect_changes(self,
                      old_data: Any,
                      new_data: Any,
                      data_type: str = "unknown") -> List[DataChange]:
        """
        Detect changes between old and new data.

        Args:
            old_data: Previous version of data
            new_data: New version of data
            data_type: Type of data for specialized detection

        Returns:
            List of detected changes
        """
        changes = []

        # Handle None cases
        if old_data is None and new_data is None:
            return changes

        if old_data is None:
            changes.append(DataChange(
                change_type=ChangeType.NEW_DATA,
                affected_fields=[],
                change_summary="New data detected",
                confidence=1.0,
                metadata={'data_type': data_type},
                detected_at=datetime.utcnow()
            ))
            return changes

        if new_data is None:
            changes.append(DataChange(
                change_type=ChangeType.DELETED_DATA,
                affected_fields=[],
                change_summary="Data has been deleted",
                confidence=1.0,
                metadata={'data_type': data_type},
                detected_at=datetime.utcnow()
            ))
            return changes

        # Detect changes based on data type
        if isinstance(old_data, pd.DataFrame) and isinstance(new_data, pd.DataFrame):
            changes.extend(self._detect_dataframe_changes(old_data, new_data, data_type))
        elif isinstance(old_data, dict) and isinstance(new_data, dict):
            changes.extend(self._detect_dict_changes(old_data, new_data, data_type))
        elif isinstance(old_data, (list, tuple)) and isinstance(new_data, (list, tuple)):
            changes.extend(self._detect_list_changes(old_data, new_data, data_type))
        else:
            changes.extend(self._detect_generic_changes(old_data, new_data, data_type))

        # Store changes in history
        self.change_history.extend(changes)

        return changes

    def _detect_dataframe_changes(self,
                                 old_df: pd.DataFrame,
                                 new_df: pd.DataFrame,
                                 data_type: str) -> List[DataChange]:
        """Detect changes in pandas DataFrames."""
        changes = []

        # Check schema changes
        schema_changes = self._check_dataframe_schema_changes(old_df, new_df)
        if schema_changes:
            changes.append(schema_changes)

        # Check size changes
        size_changes = self._check_dataframe_size_changes(old_df, new_df)
        if size_changes:
            changes.append(size_changes)

        # Check value changes if schemas are compatible
        if old_df.columns.equals(new_df.columns):
            value_changes = self._check_dataframe_value_changes(old_df, new_df, data_type)
            changes.extend(value_changes)

        return changes

    def _check_dataframe_schema_changes(self,
                                       old_df: pd.DataFrame,
                                       new_df: pd.DataFrame) -> Optional[DataChange]:
        """Check for schema changes in DataFrames."""
        old_columns = set(old_df.columns)
        new_columns = set(new_df.columns)

        added_columns = new_columns - old_columns
        removed_columns = old_columns - new_columns

        if added_columns or removed_columns:
            summary_parts = []
            if added_columns:
                summary_parts.append(f"Added columns: {list(added_columns)}")
            if removed_columns:
                summary_parts.append(f"Removed columns: {list(removed_columns)}")

            return DataChange(
                change_type=ChangeType.SCHEMA_CHANGE,
                affected_fields=list(added_columns | removed_columns),
                change_summary="; ".join(summary_parts),
                confidence=1.0,
                metadata={
                    'added_columns': list(added_columns),
                    'removed_columns': list(removed_columns)
                },
                detected_at=datetime.utcnow()
            )

        # Check data type changes
        dtype_changes = []
        for col in old_columns & new_columns:
            if old_df[col].dtype != new_df[col].dtype:
                dtype_changes.append(col)

        if dtype_changes:
            return DataChange(
                change_type=ChangeType.SCHEMA_CHANGE,
                affected_fields=dtype_changes,
                change_summary=f"Data type changes in columns: {dtype_changes}",
                confidence=0.9,
                metadata={'dtype_changes': dtype_changes},
                detected_at=datetime.utcnow()
            )

        return None

    def _check_dataframe_size_changes(self,
                                     old_df: pd.DataFrame,
                                     new_df: pd.DataFrame) -> Optional[DataChange]:
        """Check for size changes in DataFrames."""
        old_shape = old_df.shape
        new_shape = new_df.shape

        if old_shape != new_shape:
            size_change = new_shape[0] - old_shape[0]
            change_type = ChangeType.NEW_DATA if size_change > 0 else ChangeType.DELETED_DATA

            return DataChange(
                change_type=change_type,
                affected_fields=[],
                change_summary=f"Shape changed from {old_shape} to {new_shape}",
                confidence=1.0,
                metadata={
                    'old_shape': old_shape,
                    'new_shape': new_shape,
                    'row_change': size_change
                },
                detected_at=datetime.utcnow()
            )

        return None

    def _check_dataframe_value_changes(self,
                                      old_df: pd.DataFrame,
                                      new_df: pd.DataFrame,
                                      data_type: str) -> List[DataChange]:
        """Check for value changes in DataFrames."""
        changes = []

        # Align DataFrames by index
        try:
            common_index = old_df.index.intersection(new_df.index)
            if len(common_index) == 0:
                # No common index - all new data
                changes.append(DataChange(
                    change_type=ChangeType.NEW_DATA,
                    affected_fields=[],
                    change_summary="No common index found - all data is new",
                    confidence=1.0,
                    metadata={'data_type': data_type},
                    detected_at=datetime.utcnow()
                ))
                return changes

            old_subset = old_df.loc[common_index]
            new_subset = new_df.loc[common_index]

            # Check for value differences
            changed_fields = []
            for col in old_subset.columns:
                if col in new_subset.columns:
                    # Handle different data types safely
                    try:
                        if not old_subset[col].equals(new_subset[col]):
                            # Calculate the percentage of changed values
                            if pd.api.types.is_numeric_dtype(old_subset[col]) and pd.api.types.is_numeric_dtype(new_subset[col]):
                                diff_mask = ~np.isclose(old_subset[col].fillna(0), new_subset[col].fillna(0),
                                                       rtol=self.sensitivity_level)
                            else:
                                diff_mask = old_subset[col].fillna('') != new_subset[col].fillna('')

                            changed_ratio = diff_mask.sum() / len(diff_mask)
                            if changed_ratio > self.sensitivity_level:
                                changed_fields.append({
                                    'column': col,
                                    'changed_ratio': changed_ratio,
                                    'changed_count': diff_mask.sum()
                                })
                    except Exception as e:
                        logger.warning(f"Error comparing column {col}: {e}")
                        changed_fields.append({
                            'column': col,
                            'changed_ratio': 1.0,
                            'error': str(e)
                        })

            if changed_fields:
                affected_columns = [field['column'] for field in changed_fields]
                total_change_ratio = sum(field['changed_ratio'] for field in changed_fields) / len(changed_fields)

                changes.append(DataChange(
                    change_type=ChangeType.UPDATED_VALUES,
                    affected_fields=affected_columns,
                    change_summary=f"Value changes detected in {len(affected_columns)} columns",
                    confidence=min(total_change_ratio, 1.0),
                    metadata={
                        'changed_fields': changed_fields,
                        'total_change_ratio': total_change_ratio,
                        'data_type': data_type
                    },
                    detected_at=datetime.utcnow()
                ))

        except Exception as e:
            logger.error(f"Error detecting DataFrame value changes: {e}")
            changes.append(DataChange(
                change_type=ChangeType.UPDATED_VALUES,
                affected_fields=[],
                change_summary=f"Error during comparison: {e}",
                confidence=0.5,
                metadata={'error': str(e), 'data_type': data_type},
                detected_at=datetime.utcnow()
            ))

        return changes

    def _detect_dict_changes(self,
                           old_dict: Dict[str, Any],
                           new_dict: Dict[str, Any],
                           data_type: str) -> List[DataChange]:
        """Detect changes in dictionaries."""
        changes = []

        old_keys = set(old_dict.keys())
        new_keys = set(new_dict.keys())

        added_keys = new_keys - old_keys
        removed_keys = old_keys - new_keys
        common_keys = old_keys & new_keys

        # Check for structural changes
        if added_keys or removed_keys:
            summary_parts = []
            if added_keys:
                summary_parts.append(f"Added keys: {list(added_keys)}")
            if removed_keys:
                summary_parts.append(f"Removed keys: {list(removed_keys)}")

            changes.append(DataChange(
                change_type=ChangeType.STRUCTURAL_CHANGE,
                affected_fields=list(added_keys | removed_keys),
                change_summary="; ".join(summary_parts),
                confidence=1.0,
                metadata={
                    'added_keys': list(added_keys),
                    'removed_keys': list(removed_keys)
                },
                detected_at=datetime.utcnow()
            ))

        # Check for value changes in common keys
        changed_keys = []
        for key in common_keys:
            try:
                if old_dict[key] != new_dict[key]:
                    changed_keys.append(key)
            except Exception:
                # Handle unhashable types or complex comparisons
                if str(old_dict[key]) != str(new_dict[key]):
                    changed_keys.append(key)

        if changed_keys:
            changes.append(DataChange(
                change_type=ChangeType.UPDATED_VALUES,
                affected_fields=changed_keys,
                change_summary=f"Value changes in {len(changed_keys)} keys",
                confidence=0.9,
                metadata={
                    'changed_keys': changed_keys,
                    'data_type': data_type
                },
                detected_at=datetime.utcnow()
            ))

        return changes

    def _detect_list_changes(self,
                           old_list: List[Any],
                           new_list: List[Any],
                           data_type: str) -> List[DataChange]:
        """Detect changes in lists."""
        changes = []

        if len(old_list) != len(new_list):
            changes.append(DataChange(
                change_type=ChangeType.STRUCTURAL_CHANGE,
                affected_fields=[],
                change_summary=f"Length changed from {len(old_list)} to {len(new_list)}",
                confidence=1.0,
                metadata={
                    'old_length': len(old_list),
                    'new_length': len(new_list),
                    'data_type': data_type
                },
                detected_at=datetime.utcnow()
            ))

        # Check for content changes
        min_length = min(len(old_list), len(new_list))
        changed_indices = []

        for i in range(min_length):
            try:
                if old_list[i] != new_list[i]:
                    changed_indices.append(i)
            except Exception:
                if str(old_list[i]) != str(new_list[i]):
                    changed_indices.append(i)

        if changed_indices:
            change_ratio = len(changed_indices) / min_length
            changes.append(DataChange(
                change_type=ChangeType.UPDATED_VALUES,
                affected_fields=[str(i) for i in changed_indices],
                change_summary=f"Content changes at {len(changed_indices)} positions",
                confidence=change_ratio,
                metadata={
                    'changed_indices': changed_indices,
                    'change_ratio': change_ratio,
                    'data_type': data_type
                },
                detected_at=datetime.utcnow()
            ))

        return changes

    def _detect_generic_changes(self,
                              old_data: Any,
                              new_data: Any,
                              data_type: str) -> List[DataChange]:
        """Detect changes in generic data types."""
        changes = []

        # Simple equality check
        try:
            if old_data != new_data:
                changes.append(DataChange(
                    change_type=ChangeType.UPDATED_VALUES,
                    affected_fields=[],
                    change_summary="Data has changed",
                    confidence=0.8,
                    metadata={'data_type': data_type},
                    detected_at=datetime.utcnow()
                ))
        except Exception:
            # Handle complex types by comparing string representations
            if str(old_data) != str(new_data):
                changes.append(DataChange(
                    change_type=ChangeType.UPDATED_VALUES,
                    affected_fields=[],
                    change_summary="Data has changed (string comparison)",
                    confidence=0.6,
                    metadata={'data_type': data_type},
                    detected_at=datetime.utcnow()
                ))

        return changes

    def calculate_data_hash(self, data: Any) -> str:
        """
        Calculate a consistent hash for data to detect changes.

        Args:
            data: Data to hash

        Returns:
            Hexadecimal hash string
        """
        try:
            if isinstance(data, pd.DataFrame):
                # For DataFrames, hash the string representation of sorted values
                content = data.sort_index().sort_index(axis=1).to_string()
            elif isinstance(data, dict):
                # For dictionaries, sort keys and create consistent representation
                content = json.dumps(data, sort_keys=True, default=str)
            elif isinstance(data, (list, tuple)):
                # For lists/tuples, convert to string
                content = str(sorted(data) if all(isinstance(x, (str, int, float)) for x in data) else data)
            else:
                content = str(data)

            return hashlib.sha256(content.encode('utf-8')).hexdigest()

        except Exception as e:
            logger.warning(f"Error calculating hash: {e}")
            # Fallback to simple string hash
            return hashlib.sha256(str(data).encode('utf-8')).hexdigest()

    def has_significant_changes(self, changes: List[DataChange], threshold: float = 0.1) -> bool:
        """
        Determine if changes are significant enough to warrant an update.

        Args:
            changes: List of detected changes
            threshold: Significance threshold

        Returns:
            True if changes are significant
        """
        if not changes:
            return False

        # Any schema or structural changes are significant
        significant_types = {ChangeType.SCHEMA_CHANGE, ChangeType.STRUCTURAL_CHANGE,
                           ChangeType.NEW_DATA, ChangeType.DELETED_DATA}

        for change in changes:
            if change.change_type in significant_types:
                return True
            if change.change_type == ChangeType.UPDATED_VALUES and change.confidence > threshold:
                return True

        return False

    def summarize_changes(self, changes: List[DataChange]) -> str:
        """
        Create a human-readable summary of changes.

        Args:
            changes: List of changes to summarize

        Returns:
            Summary string
        """
        if not changes:
            return "No changes detected"

        summary_parts = []
        change_counts = {}

        for change in changes:
            change_type = change.change_type.value
            change_counts[change_type] = change_counts.get(change_type, 0) + 1

        for change_type, count in change_counts.items():
            if count == 1:
                summary_parts.append(f"1 {change_type.replace('_', ' ')}")
            else:
                summary_parts.append(f"{count} {change_type.replace('_', ' ')} changes")

        return "; ".join(summary_parts)

    def get_change_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about detected changes.

        Returns:
            Dictionary containing change statistics
        """
        if not self.change_history:
            return {
                'total_changes': 0,
                'change_types': {},
                'confidence_stats': {}
            }

        change_type_counts = {}
        confidences = []

        for change in self.change_history:
            change_type = change.change_type.value
            change_type_counts[change_type] = change_type_counts.get(change_type, 0) + 1
            confidences.append(change.confidence)

        return {
            'total_changes': len(self.change_history),
            'change_types': change_type_counts,
            'confidence_stats': {
                'mean': np.mean(confidences) if confidences else 0,
                'std': np.std(confidences) if confidences else 0,
                'min': min(confidences) if confidences else 0,
                'max': max(confidences) if confidences else 0
            }
        }