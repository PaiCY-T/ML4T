#!/usr/bin/env python3
"""
Demo script for the dataset parser functionality.

This script demonstrates parsing the FinLab CSV file and creating a dataset catalog.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from finlab_downloader.core.dataset import DatasetCatalog, DatasetParser, DownloadMethod
from finlab_downloader.core.validation import create_taiwan_market_context


def main():
    """Run the dataset parser demo."""
    print("FinLab Dataset Parser Demo")
    print("=" * 40)

    # Parse the CSV file
    csv_path = Path("example/finlab_database_cleaned.csv")

    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        print("Make sure the file exists before running this demo.")
        return

    try:
        print(f"Parsing CSV file: {csv_path}")
        catalog = DatasetCatalog.from_csv(csv_path)

        print(f"\nCatalog Statistics:")
        print(f"- Total datasets: {catalog.count()}")
        print(f"- Categories: {len(catalog.get_categories())}")
        print(f"- Download methods: {len(catalog.get_download_methods())}")

        print(f"\nCategories:")
        for category in catalog.get_categories():
            count = len(catalog.list_datasets(category=category))
            print(f"  - {category}: {count} datasets")

        print(f"\nDownload Methods:")
        for method in catalog.get_download_methods():
            count = len(catalog.list_datasets(download_method=method))
            print(f"  - {method.value}: {count} datasets")

        # Show some example datasets
        print(f"\nExample Datasets:")

        # ETL datasets
        etl_datasets = catalog.list_datasets(download_method=DownloadMethod.ETL)[:5]
        print(f"\nETL Datasets (first 5):")
        for ds in etl_datasets:
            print(f"  - {ds.name} ({ds.data_type.value})")

        # Financial statement datasets
        financial_datasets = catalog.list_datasets(download_method=DownloadMethod.FINANCIAL_STATEMENT)[:5]
        print(f"\nFinancial Statement Datasets (first 5):")
        for ds in financial_datasets:
            print(f"  - {ds.display_name} ({ds.data_type.value})")

        # Search functionality demo
        print(f"\nSearch Demo:")
        search_terms = ["營業收入", "adj_close", "revenue"]
        for term in search_terms:
            results = catalog.search(term)
            print(f"  Search '{term}': {len(results)} results")
            for result in results[:2]:  # Show first 2 results
                print(f"    - {result.display_name}")

        # Validation demo
        print(f"\nValidation Demo:")
        validation_context = create_taiwan_market_context()

        # Get a sample dataset
        sample_dataset = catalog.get_dataset("adj_close")
        if sample_dataset:
            print(f"  Testing validation for: {sample_dataset.name}")

            # Test valid values
            test_values = ["100.5", "0", "999.99"]
            for value in test_values:
                try:
                    validated = sample_dataset.validate_value(value)
                    print(f"    '{value}' -> {validated} ✓")
                except Exception as e:
                    print(f"    '{value}' -> Error: {e} ✗")

        print(f"\nDemo completed successfully!")

    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()