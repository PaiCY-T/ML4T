#!/usr/bin/env python3
"""
Demo script for FinLab API Integration (Issue #67).

This script demonstrates the basic usage of the FinLab API client
with authentication, rate limiting, and data downloading capabilities.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from finlab_downloader import (
    FinLabClientFactory,
    DatasetCatalogFactory,
    IntegratedDownloaderFactory,
    DatasetSpecification,
    FinLabDownloaderError
)
from finlab_downloader.core.dataset import DownloadMethod, DataType
from finlab_downloader.utils.logger import get_logger


def demo_client_with_defaults():
    """Demonstrate client creation with default configuration."""
    print("\n=== Demo: Client with Defaults ===")

    # Replace with your actual FinLab API token
    api_token = os.getenv('FINLAB_API_TOKEN', 'your_token_here')

    if api_token == 'your_token_here':
        print("⚠️  Please set FINLAB_API_TOKEN environment variable or update the script")
        return

    try:
        # Create client with defaults
        client = FinLabClientFactory.create_client_with_defaults(api_token)

        print(f"✅ Created FinLab client successfully")
        print(f"   Connected: {client.is_connected()}")

        # Test connection
        with client:
            print(f"   Connected: {client.is_connected()}")
            print(f"   Authenticated: {client.is_authenticated()}")

            # Get statistics
            stats = client.get_statistics()
            print(f"   Request count: {stats['request_count']}")
            print(f"   Error rate: {stats['error_rate']:.2%}")

    except FinLabDownloaderError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


def demo_dataset_specification():
    """Demonstrate dataset specification creation."""
    print("\n=== Demo: Dataset Specifications ===")

    # Create various dataset specifications
    datasets = [
        DatasetSpecification(
            name="Taiwan Stock Prices",
            download_method=DownloadMethod.ETL,
            download_key="etl:price",
            data_type=DataType.FLOAT,
            english_name="Taiwan Stock Prices",
            chinese_name="台股價格",
            description="Daily stock prices for Taiwan market"
        ),
        DatasetSpecification(
            name="Financial Statements",
            download_method=DownloadMethod.FINANCIAL_STATEMENT,
            download_key="revenue",
            data_type=DataType.FLOAT,
            description="Company financial statement data"
        ),
        DatasetSpecification(
            name="Fundamental Features",
            download_method=DownloadMethod.FUNDAMENTAL_FEATURES,
            download_key="fundamental_features:pe_ratio",
            data_type=DataType.FLOAT,
            description="Fundamental analysis features"
        )
    ]

    for i, dataset in enumerate(datasets, 1):
        print(f"\n📊 Dataset {i}: {dataset.name}")
        print(f"   Download method: {dataset.download_method.value}")
        print(f"   Download key: {dataset.download_key}")
        print(f"   Data type: {dataset.data_type.value}")
        print(f"   Category: {dataset.category}")
        if dataset.english_name and dataset.chinese_name:
            print(f"   Names: {dataset.english_name} / {dataset.chinese_name}")


def demo_download_simulation():
    """Demonstrate download simulation (without actual API calls)."""
    print("\n=== Demo: Download Simulation ===")

    # Create a mock dataset specification
    dataset_spec = DatasetSpecification(
        name="Demo Dataset",
        download_method=DownloadMethod.ETL,
        download_key="etl:demo_key",
        data_type=DataType.FLOAT,
        description="Demo dataset for testing"
    )

    print(f"📥 Simulating download for: {dataset_spec.name}")
    print(f"   Method: {dataset_spec.download_method.value}")
    print(f"   Key: {dataset_spec.download_key}")
    print(f"   Type: {dataset_spec.data_type.value}")

    # In a real scenario, you would:
    # 1. Create client with actual token
    # 2. Call client.download_dataset(dataset_spec)
    # 3. Process the returned data

    print("   ✅ Download simulation completed")
    print("   📝 Note: Replace with actual API token for real downloads")


def demo_rate_limiting():
    """Demonstrate rate limiting configuration."""
    print("\n=== Demo: Rate Limiting Configuration ===")

    from finlab_downloader.core.client import RateLimitConfig, ProgressConfig

    # Create custom rate limiting configuration
    rate_config = RateLimitConfig(
        max_requests_per_minute=30,
        max_requests_per_hour=500,
        max_requests_per_day=5000,
        backoff_factor=2.0,
        max_retries=5
    )

    progress_config = ProgressConfig(
        show_progress=True,
        update_interval=0.5,
        chunk_size=1000,
        verbose=True
    )

    print("⚙️  Rate Limiting Configuration:")
    print(f"   Max requests per minute: {rate_config.max_requests_per_minute}")
    print(f"   Max requests per hour: {rate_config.max_requests_per_hour}")
    print(f"   Max requests per day: {rate_config.max_requests_per_day}")
    print(f"   Backoff factor: {rate_config.backoff_factor}")
    print(f"   Max retries: {rate_config.max_retries}")

    print("\n⚙️  Progress Configuration:")
    print(f"   Show progress: {progress_config.show_progress}")
    print(f"   Update interval: {progress_config.update_interval}s")
    print(f"   Chunk size: {progress_config.chunk_size}")
    print(f"   Verbose: {progress_config.verbose}")


def demo_error_handling():
    """Demonstrate error handling capabilities."""
    print("\n=== Demo: Error Handling ===")

    from finlab_downloader.core.exceptions import (
        ValidationError,
        AuthenticationError,
        DataSourceError,
        RateLimitError
    )

    error_examples = [
        ("ValidationError", "Invalid configuration parameters"),
        ("AuthenticationError", "Invalid API token or authentication failure"),
        ("DataSourceError", "FinLab API request failed or data unavailable"),
        ("RateLimitError", "API rate limits exceeded")
    ]

    print("🛡️  Error handling capabilities:")
    for error_type, description in error_examples:
        print(f"   {error_type}: {description}")

    print("\n✅ All errors include:")
    print("   - Detailed error messages")
    print("   - Error codes for programmatic handling")
    print("   - Context information for debugging")
    print("   - Automatic retry logic with exponential backoff")


def main():
    """Run all demos."""
    print("🚀 FinLab API Integration Demo (Issue #67)")
    print("=" * 50)

    # Set up logging
    logger = get_logger("FinLabDemo")
    logger.info("Starting FinLab API integration demo")

    try:
        # Run demos
        demo_client_with_defaults()
        demo_dataset_specification()
        demo_download_simulation()
        demo_rate_limiting()
        demo_error_handling()

        print("\n" + "=" * 50)
        print("✅ Demo completed successfully!")
        print("\n📋 Next steps:")
        print("   1. Set your FINLAB_API_TOKEN environment variable")
        print("   2. Create a configuration file for your setup")
        print("   3. Use the DatasetCatalog to browse available datasets")
        print("   4. Start downloading real data with the FinLabClient")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())