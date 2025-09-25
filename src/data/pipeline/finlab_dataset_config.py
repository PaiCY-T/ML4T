"""
FinLab Dataset Configuration and Management.

This module provides comprehensive configuration for all FinLab datasets,
mapping field names, data types, and update strategies for optimal pipeline performance.
"""

import logging
from typing import Dict, List, Optional, Set, Any, Tuple
from enum import Enum
from dataclasses import dataclass, field
from datetime import timedelta
import pandas as pd
from pathlib import Path

from ..core.temporal import DataType

logger = logging.getLogger(__name__)


class FinLabDatasetType(Enum):
    """FinLab dataset categories."""
    ETL = "etl"  # Price, volume, technical indicators
    FINANCIAL_STATEMENT = "financial_statement"  # Financial statements
    FUNDAMENTAL_FEATURES = "fundamental_features"  # Calculated ratios
    BROKER_TRANSACTIONS = "broker_transactions"  # Institutional flow
    MARKET_DATA = "market_data"  # Market microstructure


@dataclass
class FinLabField:
    """FinLab field configuration."""
    name: str
    key: str  # FinLab API key
    data_type: str  # Python data type
    dataset_type: FinLabDatasetType
    temporal_type: DataType
    update_frequency: str = "daily"  # daily, weekly, monthly, quarterly
    lag_days: int = 0  # Expected data lag
    required: bool = True  # Required for pipeline
    validation_rules: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FinLabDatasetConfig:
    """Complete FinLab dataset configuration."""

    # Core ETL datasets
    ETL_FIELDS = {
        # Price data (daily updates, minimal lag)
        "adj_close": FinLabField("adj_close", "etl:adj_close", "float",
                                FinLabDatasetType.ETL, DataType.PRICE, lag_days=0),
        "adj_high": FinLabField("adj_high", "etl:adj_high", "float",
                               FinLabDatasetType.ETL, DataType.PRICE, lag_days=0),
        "adj_low": FinLabField("adj_low", "etl:adj_low", "float",
                              FinLabDatasetType.ETL, DataType.PRICE, lag_days=0),
        "adj_open": FinLabField("adj_open", "etl:adj_open", "float",
                               FinLabDatasetType.ETL, DataType.PRICE, lag_days=0),

        # Broker transactions (daily updates, T+1 lag)
        "buy": FinLabField("buy", "etl:broker_transactions:top15_buy", "int",
                          FinLabDatasetType.BROKER_TRANSACTIONS, DataType.MARKET_DATA, lag_days=1),
        "sell": FinLabField("sell", "etl:broker_transactions:top15_sell", "int",
                           FinLabDatasetType.BROKER_TRANSACTIONS, DataType.MARKET_DATA, lag_days=1),
    }

    # Financial statement datasets (quarterly updates, 60-day lag)
    FINANCIAL_STATEMENT_FIELDS = {
        # Balance sheet items
        "total_assets": FinLabField("資產總額", "financial_statement:資產總額", "float",
                                   FinLabDatasetType.FINANCIAL_STATEMENT, DataType.FUNDAMENTAL,
                                   update_frequency="quarterly", lag_days=60),
        "current_assets": FinLabField("流動資產", "financial_statement:流動資產", "float",
                                     FinLabDatasetType.FINANCIAL_STATEMENT, DataType.FUNDAMENTAL,
                                     update_frequency="quarterly", lag_days=60),
        "non_current_assets": FinLabField("非流動資產", "financial_statement:非流動資產", "float",
                                         FinLabDatasetType.FINANCIAL_STATEMENT, DataType.FUNDAMENTAL,
                                         update_frequency="quarterly", lag_days=60),
        "current_liabilities": FinLabField("流動負債", "financial_statement:流動負債", "float",
                                          FinLabDatasetType.FINANCIAL_STATEMENT, DataType.FUNDAMENTAL,
                                          update_frequency="quarterly", lag_days=60),
        "non_current_liabilities": FinLabField("非流動負債", "financial_statement:非流動負債", "float",
                                              FinLabDatasetType.FINANCIAL_STATEMENT, DataType.FUNDAMENTAL,
                                              update_frequency="quarterly", lag_days=60),
        "total_liabilities": FinLabField("負債總額", "financial_statement:負債總額", "float",
                                        FinLabDatasetType.FINANCIAL_STATEMENT, DataType.FUNDAMENTAL,
                                        update_frequency="quarterly", lag_days=60),
        "shareholders_equity": FinLabField("股東權益總額", "financial_statement:股東權益總額", "float",
                                          FinLabDatasetType.FINANCIAL_STATEMENT, DataType.FUNDAMENTAL,
                                          update_frequency="quarterly", lag_days=60),
        "share_capital": FinLabField("股本", "financial_statement:股本", "float",
                                    FinLabDatasetType.FINANCIAL_STATEMENT, DataType.FUNDAMENTAL,
                                    update_frequency="quarterly", lag_days=60),
        "retained_earnings": FinLabField("保留盈餘", "financial_statement:保留盈餘", "float",
                                        FinLabDatasetType.FINANCIAL_STATEMENT, DataType.FUNDAMENTAL,
                                        update_frequency="quarterly", lag_days=60),
        "cash_and_equivalents": FinLabField("現金及約當現金", "financial_statement:現金及約當現金", "float",
                                           FinLabDatasetType.FINANCIAL_STATEMENT, DataType.FUNDAMENTAL,
                                           update_frequency="quarterly", lag_days=60),

        # Income statement items
        "revenue": FinLabField("營業收入淨額", "financial_statement:營業收入淨額", "float",
                              FinLabDatasetType.FINANCIAL_STATEMENT, DataType.FUNDAMENTAL,
                              update_frequency="quarterly", lag_days=60),
        "operating_costs": FinLabField("營業成本", "financial_statement:營業成本", "float",
                                      FinLabDatasetType.FINANCIAL_STATEMENT, DataType.FUNDAMENTAL,
                                      update_frequency="quarterly", lag_days=60),
        "gross_profit": FinLabField("營業毛利", "financial_statement:營業毛利", "float",
                                   FinLabDatasetType.FINANCIAL_STATEMENT, DataType.FUNDAMENTAL,
                                   update_frequency="quarterly", lag_days=60),
        "operating_income": FinLabField("營業利益", "financial_statement:營業利益", "float",
                                       FinLabDatasetType.FINANCIAL_STATEMENT, DataType.FUNDAMENTAL,
                                       update_frequency="quarterly", lag_days=60),
        "pre_tax_income": FinLabField("稅前淨利", "financial_statement:稅前淨利", "float",
                                     FinLabDatasetType.FINANCIAL_STATEMENT, DataType.FUNDAMENTAL,
                                     update_frequency="quarterly", lag_days=60),
        "net_income": FinLabField("歸屬母公司淨利_損", "financial_statement:歸屬母公司淨利_損", "float",
                                 FinLabDatasetType.FINANCIAL_STATEMENT, DataType.FUNDAMENTAL,
                                 update_frequency="quarterly", lag_days=60),
        "eps": FinLabField("每股盈餘", "financial_statement:每股盈餘", "float",
                          FinLabDatasetType.FINANCIAL_STATEMENT, DataType.FUNDAMENTAL,
                          update_frequency="quarterly", lag_days=60),

        # Cash flow statement items
        "operating_cash_flow": FinLabField("營業活動之淨現金流入_流出", "financial_statement:營業活動之淨現金流入_流出", "float",
                                          FinLabDatasetType.FINANCIAL_STATEMENT, DataType.FUNDAMENTAL,
                                          update_frequency="quarterly", lag_days=60),
        "investing_cash_flow": FinLabField("投資活動之淨現金流入_流出", "financial_statement:投資活動之淨現金流入_流出", "float",
                                          FinLabDatasetType.FINANCIAL_STATEMENT, DataType.FUNDAMENTAL,
                                          update_frequency="quarterly", lag_days=60),
        "financing_cash_flow": FinLabField("籌資活動之淨現金流入_流出", "financial_statement:籌資活動之淨現金流入_流出", "float",
                                          FinLabDatasetType.FINANCIAL_STATEMENT, DataType.FUNDAMENTAL,
                                          update_frequency="quarterly", lag_days=60),
    }

    # Fundamental features (calculated ratios, quarterly updates)
    FUNDAMENTAL_FEATURES_FIELDS = {
        "roa": FinLabField("ROA綜合損益", "fundamental_features:ROA綜合損益", "float",
                          FinLabDatasetType.FUNDAMENTAL_FEATURES, DataType.FUNDAMENTAL,
                          update_frequency="quarterly", lag_days=65),
        "roe": FinLabField("ROE綜合損益", "fundamental_features:ROE綜合損益", "float",
                          FinLabDatasetType.FUNDAMENTAL_FEATURES, DataType.FUNDAMENTAL,
                          update_frequency="quarterly", lag_days=65),
        "current_ratio": FinLabField("流動比率", "fundamental_features:流動比率", "float",
                                    FinLabDatasetType.FUNDAMENTAL_FEATURES, DataType.FUNDAMENTAL,
                                    update_frequency="quarterly", lag_days=65),
        "gross_margin": FinLabField("營業毛利率", "fundamental_features:營業毛利率", "float",
                                   FinLabDatasetType.FUNDAMENTAL_FEATURES, DataType.FUNDAMENTAL,
                                   update_frequency="quarterly", lag_days=65),
        "operating_margin": FinLabField("營業利益率", "fundamental_features:營業利益率", "float",
                                       FinLabDatasetType.FUNDAMENTAL_FEATURES, DataType.FUNDAMENTAL,
                                       update_frequency="quarterly", lag_days=65),
        "debt_to_assets": FinLabField("淨值除資產", "fundamental_features:淨值除資產", "float",
                                     FinLabDatasetType.FUNDAMENTAL_FEATURES, DataType.FUNDAMENTAL,
                                     update_frequency="quarterly", lag_days=65),
        "revenue_growth": FinLabField("營收成長率", "fundamental_features:營收成長率", "float",
                                     FinLabDatasetType.FUNDAMENTAL_FEATURES, DataType.FUNDAMENTAL,
                                     update_frequency="quarterly", lag_days=65),
        "earnings_growth": FinLabField("營業利益成長率", "fundamental_features:營業利益成長率", "float",
                                      FinLabDatasetType.FUNDAMENTAL_FEATURES, DataType.FUNDAMENTAL,
                                      update_frequency="quarterly", lag_days=65),
        "ebitda": FinLabField("EBITDA", "fundamental_features:EBITDA", "float",
                             FinLabDatasetType.FUNDAMENTAL_FEATURES, DataType.FUNDAMENTAL,
                             update_frequency="quarterly", lag_days=65),
    }

    @classmethod
    def get_all_fields(cls) -> Dict[str, FinLabField]:
        """Get all configured FinLab fields."""
        all_fields = {}
        all_fields.update(cls.ETL_FIELDS)
        all_fields.update(cls.FINANCIAL_STATEMENT_FIELDS)
        all_fields.update(cls.FUNDAMENTAL_FEATURES_FIELDS)
        return all_fields

    @classmethod
    def get_fields_by_dataset_type(cls, dataset_type: FinLabDatasetType) -> Dict[str, FinLabField]:
        """Get fields by dataset type."""
        all_fields = cls.get_all_fields()
        return {k: v for k, v in all_fields.items() if v.dataset_type == dataset_type}

    @classmethod
    def get_fields_by_temporal_type(cls, temporal_type: DataType) -> Dict[str, FinLabField]:
        """Get fields by temporal data type."""
        all_fields = cls.get_all_fields()
        return {k: v for k, v in all_fields.items() if v.temporal_type == temporal_type}

    @classmethod
    def get_fields_by_update_frequency(cls, frequency: str) -> Dict[str, FinLabField]:
        """Get fields by update frequency."""
        all_fields = cls.get_all_fields()
        return {k: v for k, v in all_fields.items() if v.update_frequency == frequency}

    @classmethod
    def get_high_priority_fields(cls) -> Dict[str, FinLabField]:
        """Get high-priority fields for daily updates."""
        all_fields = cls.get_all_fields()
        return {k: v for k, v in all_fields.items()
                if v.update_frequency == "daily" or v.required}

    @classmethod
    def get_field_by_key(cls, key: str) -> Optional[FinLabField]:
        """Get field configuration by FinLab key."""
        all_fields = cls.get_all_fields()
        for field in all_fields.values():
            if field.key == key:
                return field
        return None

    @classmethod
    def load_from_csv(cls, csv_path: Path) -> 'FinLabDatasetConfig':
        """Load additional field configurations from CSV file."""
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
            logger.info(f"Loaded {len(df)} field definitions from {csv_path}")

            # Process CSV data to create field configurations
            # This allows for dynamic field loading beyond the hardcoded ones
            additional_fields = {}

            for _, row in df.iterrows():
                field_name = row.get('資料集名稱', '').strip()
                field_key = row.get('下載方式及key', '').strip()
                data_type = row.get('數據類型', 'float').strip()

                if field_name and field_key:
                    # Determine dataset type from key
                    dataset_type = cls._infer_dataset_type(field_key)
                    temporal_type = cls._infer_temporal_type(dataset_type)

                    # Create field configuration
                    field_config = FinLabField(
                        name=field_name,
                        key=field_key,
                        data_type=data_type,
                        dataset_type=dataset_type,
                        temporal_type=temporal_type,
                        update_frequency=cls._infer_update_frequency(dataset_type),
                        lag_days=cls._infer_lag_days(dataset_type)
                    )

                    additional_fields[field_name] = field_config

            logger.info(f"Created {len(additional_fields)} additional field configurations")
            return additional_fields

        except Exception as e:
            logger.error(f"Error loading field configurations from CSV: {e}")
            return {}

    @classmethod
    def _infer_dataset_type(cls, key: str) -> FinLabDatasetType:
        """Infer dataset type from FinLab key."""
        if key.startswith('etl:'):
            return FinLabDatasetType.ETL
        elif key.startswith('financial_statement:'):
            return FinLabDatasetType.FINANCIAL_STATEMENT
        elif key.startswith('fundamental_features:'):
            return FinLabDatasetType.FUNDAMENTAL_FEATURES
        elif 'broker_transactions' in key:
            return FinLabDatasetType.BROKER_TRANSACTIONS
        else:
            return FinLabDatasetType.MARKET_DATA

    @classmethod
    def _infer_temporal_type(cls, dataset_type: FinLabDatasetType) -> DataType:
        """Infer temporal data type from dataset type."""
        if dataset_type == FinLabDatasetType.ETL:
            return DataType.PRICE
        elif dataset_type in [FinLabDatasetType.FINANCIAL_STATEMENT,
                             FinLabDatasetType.FUNDAMENTAL_FEATURES]:
            return DataType.FUNDAMENTAL
        else:
            return DataType.MARKET_DATA

    @classmethod
    def _infer_update_frequency(cls, dataset_type: FinLabDatasetType) -> str:
        """Infer update frequency from dataset type."""
        if dataset_type == FinLabDatasetType.ETL:
            return "daily"
        elif dataset_type in [FinLabDatasetType.FINANCIAL_STATEMENT,
                             FinLabDatasetType.FUNDAMENTAL_FEATURES]:
            return "quarterly"
        else:
            return "daily"

    @classmethod
    def _infer_lag_days(cls, dataset_type: FinLabDatasetType) -> int:
        """Infer expected data lag from dataset type."""
        if dataset_type == FinLabDatasetType.ETL:
            return 0
        elif dataset_type == FinLabDatasetType.BROKER_TRANSACTIONS:
            return 1
        elif dataset_type == FinLabDatasetType.FINANCIAL_STATEMENT:
            return 60
        elif dataset_type == FinLabDatasetType.FUNDAMENTAL_FEATURES:
            return 65
        else:
            return 1


class DatasetUpdateStrategy:
    """Strategy for updating different dataset types."""

    def __init__(self, config: FinLabDatasetConfig):
        self.config = config
        self.update_priorities = {
            "critical": ["adj_close", "adj_high", "adj_low", "adj_open"],
            "high": ["buy", "sell", "revenue", "net_income"],
            "medium": ["roa", "roe", "current_ratio"],
            "low": ["ebitda", "revenue_growth"]
        }

    def get_update_schedule(self) -> Dict[str, List[str]]:
        """Get optimized update schedule by frequency."""
        daily_fields = []
        weekly_fields = []
        monthly_fields = []
        quarterly_fields = []

        all_fields = self.config.get_all_fields()

        for field_name, field_config in all_fields.items():
            if field_config.update_frequency == "daily":
                daily_fields.append(field_name)
            elif field_config.update_frequency == "weekly":
                weekly_fields.append(field_name)
            elif field_config.update_frequency == "monthly":
                monthly_fields.append(field_name)
            elif field_config.update_frequency == "quarterly":
                quarterly_fields.append(field_name)

        return {
            "daily": daily_fields,
            "weekly": weekly_fields,
            "monthly": monthly_fields,
            "quarterly": quarterly_fields
        }

    def get_priority_groups(self) -> Dict[str, List[str]]:
        """Get fields grouped by update priority."""
        return self.update_priorities

    def should_update_field(self, field_name: str, last_update: Optional[timedelta] = None) -> bool:
        """Determine if a field should be updated based on its configuration."""
        all_fields = self.config.get_all_fields()
        field_config = all_fields.get(field_name)

        if not field_config:
            return False

        if last_update is None:
            return True

        # Check if enough time has passed based on update frequency
        if field_config.update_frequency == "daily" and last_update.days >= 1:
            return True
        elif field_config.update_frequency == "weekly" and last_update.days >= 7:
            return True
        elif field_config.update_frequency == "monthly" and last_update.days >= 30:
            return True
        elif field_config.update_frequency == "quarterly" and last_update.days >= 90:
            return True

        return False


# Global configuration instance
finlab_config = FinLabDatasetConfig()

# Load additional configurations from CSV if available
csv_path = Path(__file__).parent.parent.parent / "example" / "finlab_database_cleaned.csv"
if csv_path.exists():
    additional_fields = FinLabDatasetConfig.load_from_csv(csv_path)
    # Note: In a real implementation, you would merge these with the main config