"""
Conflict resolution for overlapping data ranges.

This module provides sophisticated conflict resolution algorithms to handle
overlapping data ranges and conflicting updates in incremental downloads.
"""

import logging
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple, Set
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ConflictType(Enum):
    """Types of data conflicts."""
    TIMESTAMP_OVERLAP = "timestamp_overlap"
    DATA_MISMATCH = "data_mismatch"
    VERSION_CONFLICT = "version_conflict"
    DUPLICATE_DATA = "duplicate_data"
    SCHEMA_CONFLICT = "schema_conflict"
    INTEGRITY_VIOLATION = "integrity_violation"


class ResolutionStrategy(Enum):
    """Conflict resolution strategies."""
    LATEST_WINS = "latest_wins"           # Use most recent data
    MERGE_VALUES = "merge_values"         # Merge non-conflicting values
    MANUAL_REVIEW = "manual_review"       # Require manual intervention
    PRIORITY_BASED = "priority_based"     # Use data source priority
    VALIDATION_BASED = "validation_based" # Use most validated data


@dataclass
class DataRange:
    """Represents a range of data."""
    dataset_name: str
    symbol: Optional[str]
    start_date: date
    end_date: date
    data_hash: str
    version: int
    timestamp: datetime
    metadata: Dict[str, Any]

    def overlaps_with(self, other: 'DataRange') -> bool:
        """Check if this range overlaps with another."""
        return (self.dataset_name == other.dataset_name and
                self.symbol == other.symbol and
                not (self.end_date < other.start_date or self.start_date > other.end_date))

    def get_overlap_period(self, other: 'DataRange') -> Optional[Tuple[date, date]]:
        """Get the overlapping period with another range."""
        if not self.overlaps_with(other):
            return None

        overlap_start = max(self.start_date, other.start_date)
        overlap_end = min(self.end_date, other.end_date)
        return (overlap_start, overlap_end)


@dataclass
class Conflict:
    """Represents a data conflict."""
    conflict_id: str
    conflict_type: ConflictType
    ranges: List[DataRange]
    overlap_period: Optional[Tuple[date, date]]
    resolution_strategy: Optional[ResolutionStrategy]
    resolved: bool
    created_at: datetime
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ConflictResolver:
    """
    Advanced conflict resolution system for overlapping data ranges.

    Features:
    - Multiple resolution strategies
    - Automatic conflict detection
    - Priority-based resolution
    - Manual intervention support
    - Conflict history tracking
    """

    def __init__(self, default_strategy: ResolutionStrategy = ResolutionStrategy.LATEST_WINS):
        """
        Initialize conflict resolver.

        Args:
            default_strategy: Default resolution strategy
        """
        self.default_strategy = default_strategy
        self.conflicts: Dict[str, Conflict] = {}
        self.resolution_rules: Dict[str, ResolutionStrategy] = {}

    def detect_conflicts(self, ranges: List[DataRange]) -> List[Conflict]:
        """
        Detect conflicts among data ranges.

        Args:
            ranges: List of data ranges to check

        Returns:
            List of detected conflicts
        """
        conflicts = []
        processed_pairs = set()

        for i, range1 in enumerate(ranges):
            for j, range2 in enumerate(ranges[i + 1:], i + 1):
                pair_id = tuple(sorted([id(range1), id(range2)]))
                if pair_id in processed_pairs:
                    continue
                processed_pairs.add(pair_id)

                if range1.overlaps_with(range2):
                    conflict = self._analyze_conflict(range1, range2)
                    if conflict:
                        conflicts.append(conflict)
                        self.conflicts[conflict.conflict_id] = conflict

        return conflicts

    def resolve_conflict(self,
                        conflict_id: str,
                        strategy: Optional[ResolutionStrategy] = None,
                        manual_resolution: Optional[Dict[str, Any]] = None) -> bool:
        """
        Resolve a specific conflict.

        Args:
            conflict_id: ID of conflict to resolve
            strategy: Resolution strategy to use
            manual_resolution: Manual resolution data

        Returns:
            True if successfully resolved
        """
        conflict = self.conflicts.get(conflict_id)
        if not conflict:
            logger.error(f"Conflict {conflict_id} not found")
            return False

        if conflict.resolved:
            logger.warning(f"Conflict {conflict_id} is already resolved")
            return True

        strategy = strategy or conflict.resolution_strategy or self.default_strategy

        try:
            if strategy == ResolutionStrategy.LATEST_WINS:
                result = self._resolve_latest_wins(conflict)
            elif strategy == ResolutionStrategy.MERGE_VALUES:
                result = self._resolve_merge_values(conflict)
            elif strategy == ResolutionStrategy.PRIORITY_BASED:
                result = self._resolve_priority_based(conflict)
            elif strategy == ResolutionStrategy.VALIDATION_BASED:
                result = self._resolve_validation_based(conflict)
            elif strategy == ResolutionStrategy.MANUAL_REVIEW:
                result = self._resolve_manual_review(conflict, manual_resolution)
            else:
                logger.error(f"Unknown resolution strategy: {strategy}")
                return False

            if result:
                conflict.resolved = True
                conflict.resolved_at = datetime.utcnow()
                conflict.resolution_strategy = strategy
                logger.info(f"Resolved conflict {conflict_id} using {strategy.value}")

            return result

        except Exception as e:
            logger.error(f"Error resolving conflict {conflict_id}: {e}")
            return False

    def set_resolution_rule(self,
                           dataset_pattern: str,
                           strategy: ResolutionStrategy) -> None:
        """
        Set a resolution rule for dataset patterns.

        Args:
            dataset_pattern: Dataset name pattern (supports wildcards)
            strategy: Resolution strategy to use
        """
        self.resolution_rules[dataset_pattern] = strategy
        logger.info(f"Set resolution rule for '{dataset_pattern}': {strategy.value}")

    def get_resolution_strategy(self, dataset_name: str) -> ResolutionStrategy:
        """
        Get resolution strategy for a dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Resolution strategy to use
        """
        # Check exact match first
        if dataset_name in self.resolution_rules:
            return self.resolution_rules[dataset_name]

        # Check pattern matches
        for pattern, strategy in self.resolution_rules.items():
            if self._matches_pattern(dataset_name, pattern):
                return strategy

        return self.default_strategy

    def get_conflicts(self,
                     dataset_name: Optional[str] = None,
                     resolved: Optional[bool] = None) -> List[Conflict]:
        """
        Get conflicts, optionally filtered.

        Args:
            dataset_name: Optional dataset filter
            resolved: Optional resolution status filter

        Returns:
            List of matching conflicts
        """
        conflicts = []

        for conflict in self.conflicts.values():
            # Filter by dataset name
            if dataset_name is not None:
                if not any(r.dataset_name == dataset_name for r in conflict.ranges):
                    continue

            # Filter by resolution status
            if resolved is not None and conflict.resolved != resolved:
                continue

            conflicts.append(conflict)

        return conflicts

    def _analyze_conflict(self, range1: DataRange, range2: DataRange) -> Optional[Conflict]:
        """Analyze two overlapping ranges for conflicts."""
        if not range1.overlaps_with(range2):
            return None

        conflict_id = f"conflict_{hash((range1.data_hash, range2.data_hash)) % 1000000}"
        overlap_period = range1.get_overlap_period(range2)

        # Determine conflict type
        conflict_type = self._determine_conflict_type(range1, range2)

        # Determine appropriate resolution strategy
        strategy = self.get_resolution_strategy(range1.dataset_name)

        conflict = Conflict(
            conflict_id=conflict_id,
            conflict_type=conflict_type,
            ranges=[range1, range2],
            overlap_period=overlap_period,
            resolution_strategy=strategy,
            resolved=False,
            created_at=datetime.utcnow(),
            metadata={
                'range1_hash': range1.data_hash,
                'range2_hash': range2.data_hash,
                'overlap_days': (overlap_period[1] - overlap_period[0]).days + 1 if overlap_period else 0
            }
        )

        return conflict

    def _determine_conflict_type(self, range1: DataRange, range2: DataRange) -> ConflictType:
        """Determine the type of conflict between ranges."""
        # Check for data hash mismatch in overlapping period
        if range1.data_hash != range2.data_hash:
            return ConflictType.DATA_MISMATCH

        # Check for version conflicts
        if range1.version != range2.version:
            return ConflictType.VERSION_CONFLICT

        # Check for timestamp issues
        if range1.timestamp == range2.timestamp:
            return ConflictType.DUPLICATE_DATA

        return ConflictType.TIMESTAMP_OVERLAP

    def _resolve_latest_wins(self, conflict: Conflict) -> bool:
        """Resolve conflict by using the latest data."""
        try:
            # Find the range with the latest timestamp
            latest_range = max(conflict.ranges, key=lambda r: r.timestamp)

            conflict.resolution_notes = f"Selected range with timestamp {latest_range.timestamp} (hash: {latest_range.data_hash})"
            conflict.metadata['winning_range'] = {
                'hash': latest_range.data_hash,
                'timestamp': latest_range.timestamp.isoformat(),
                'version': latest_range.version
            }

            return True

        except Exception as e:
            logger.error(f"Error in latest_wins resolution: {e}")
            return False

    def _resolve_merge_values(self, conflict: Conflict) -> bool:
        """Resolve conflict by merging non-conflicting values."""
        try:
            # This is a complex operation that would require actual data access
            # For now, we'll mark it as requiring manual review
            conflict.resolution_notes = "Merge resolution requires manual data analysis"
            conflict.resolution_strategy = ResolutionStrategy.MANUAL_REVIEW

            return False  # Requires manual intervention

        except Exception as e:
            logger.error(f"Error in merge_values resolution: {e}")
            return False

    def _resolve_priority_based(self, conflict: Conflict) -> bool:
        """Resolve conflict based on data source priority."""
        try:
            # Use version number as priority (higher version wins)
            highest_version_range = max(conflict.ranges, key=lambda r: r.version)

            conflict.resolution_notes = f"Selected range with highest version {highest_version_range.version}"
            conflict.metadata['winning_range'] = {
                'hash': highest_version_range.data_hash,
                'version': highest_version_range.version,
                'priority_score': highest_version_range.version
            }

            return True

        except Exception as e:
            logger.error(f"Error in priority_based resolution: {e}")
            return False

    def _resolve_validation_based(self, conflict: Conflict) -> bool:
        """Resolve conflict based on data validation scores."""
        try:
            # Use metadata validation scores if available
            best_range = None
            best_score = -1

            for range_obj in conflict.ranges:
                validation_score = range_obj.metadata.get('validation_score', 0)
                if validation_score > best_score:
                    best_score = validation_score
                    best_range = range_obj

            if best_range:
                conflict.resolution_notes = f"Selected range with highest validation score {best_score}"
                conflict.metadata['winning_range'] = {
                    'hash': best_range.data_hash,
                    'validation_score': best_score
                }
                return True
            else:
                # Fallback to latest wins
                return self._resolve_latest_wins(conflict)

        except Exception as e:
            logger.error(f"Error in validation_based resolution: {e}")
            return False

    def _resolve_manual_review(self,
                              conflict: Conflict,
                              manual_resolution: Optional[Dict[str, Any]]) -> bool:
        """Handle manual resolution of conflicts."""
        if not manual_resolution:
            conflict.resolution_notes = "Marked for manual review - no resolution provided"
            return False

        try:
            selected_hash = manual_resolution.get('selected_hash')
            if not selected_hash:
                conflict.resolution_notes = "Manual review requires selected_hash"
                return False

            # Verify the selected hash exists in the conflict ranges
            selected_range = None
            for range_obj in conflict.ranges:
                if range_obj.data_hash == selected_hash:
                    selected_range = range_obj
                    break

            if not selected_range:
                conflict.resolution_notes = f"Selected hash {selected_hash} not found in conflict ranges"
                return False

            conflict.resolution_notes = f"Manually selected range with hash {selected_hash}. Reason: {manual_resolution.get('reason', 'No reason provided')}"
            conflict.metadata['winning_range'] = {
                'hash': selected_hash,
                'manual_reason': manual_resolution.get('reason', ''),
                'resolved_by': manual_resolution.get('resolved_by', 'unknown')
            }

            return True

        except Exception as e:
            logger.error(f"Error in manual_review resolution: {e}")
            return False

    def _matches_pattern(self, text: str, pattern: str) -> bool:
        """Check if text matches a pattern (simple wildcard support)."""
        if '*' not in pattern:
            return text == pattern

        # Simple wildcard matching
        import re
        regex_pattern = pattern.replace('*', '.*')
        return bool(re.match(f'^{regex_pattern}$', text))

    def get_statistics(self) -> Dict[str, Any]:
        """Get conflict resolution statistics."""
        if not self.conflicts:
            return {
                'total_conflicts': 0,
                'resolved_conflicts': 0,
                'pending_conflicts': 0,
                'resolution_rate': 0.0,
                'conflicts_by_type': {},
                'conflicts_by_strategy': {}
            }

        total_conflicts = len(self.conflicts)
        resolved_conflicts = sum(1 for c in self.conflicts.values() if c.resolved)
        pending_conflicts = total_conflicts - resolved_conflicts

        # Count by type
        type_counts = {}
        for conflict in self.conflicts.values():
            conflict_type = conflict.conflict_type.value
            type_counts[conflict_type] = type_counts.get(conflict_type, 0) + 1

        # Count by resolution strategy
        strategy_counts = {}
        for conflict in self.conflicts.values():
            if conflict.resolved and conflict.resolution_strategy:
                strategy = conflict.resolution_strategy.value
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        return {
            'total_conflicts': total_conflicts,
            'resolved_conflicts': resolved_conflicts,
            'pending_conflicts': pending_conflicts,
            'resolution_rate': resolved_conflicts / total_conflicts if total_conflicts > 0 else 0.0,
            'conflicts_by_type': type_counts,
            'conflicts_by_strategy': strategy_counts,
            'resolution_rules_count': len(self.resolution_rules)
        }

    def cleanup_old_conflicts(self, days_to_keep: int = 30) -> int:
        """
        Clean up old resolved conflicts.

        Args:
            days_to_keep: Number of days to keep resolved conflicts

        Returns:
            Number of conflicts cleaned up
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        conflicts_to_remove = []

        for conflict_id, conflict in self.conflicts.items():
            if (conflict.resolved and
                conflict.resolved_at and
                conflict.resolved_at < cutoff_date):
                conflicts_to_remove.append(conflict_id)

        for conflict_id in conflicts_to_remove:
            del self.conflicts[conflict_id]

        if conflicts_to_remove:
            logger.info(f"Cleaned up {len(conflicts_to_remove)} old resolved conflicts")

        return len(conflicts_to_remove)