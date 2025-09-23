#!/usr/bin/env python3
"""
Database migration runner for ML4T temporal data system.

This script handles the creation and management of the database schema
for the point-in-time data management system.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
import psycopg2
from psycopg2.extras import RealDictCursor

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MigrationRunner:
    """Handles database migrations for the ML4T system."""
    
    def __init__(self, connection_params: Dict[str, str]):
        self.connection_params = connection_params
        self.migrations_dir = Path(__file__).parent.parent / "migrations"
        
    def get_connection(self):
        """Get database connection."""
        return psycopg2.connect(**self.connection_params)
    
    def create_migration_table(self):
        """Create migration tracking table if it doesn't exist."""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        version VARCHAR(20) PRIMARY KEY,
                        name VARCHAR(200) NOT NULL,
                        applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        checksum VARCHAR(64)
                    )
                """)
                conn.commit()
                logger.info("Migration tracking table ready")
    
    def get_applied_migrations(self) -> List[str]:
        """Get list of already applied migrations."""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT version FROM schema_migrations ORDER BY version")
                return [row[0] for row in cursor.fetchall()]
    
    def get_available_migrations(self) -> List[Path]:
        """Get list of available migration files."""
        migrations = []
        for file_path in self.migrations_dir.glob("*.sql"):
            if file_path.name.endswith('.sql'):
                migrations.append(file_path)
        return sorted(migrations)
    
    def calculate_checksum(self, content: str) -> str:
        """Calculate checksum for migration content."""
        import hashlib
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def apply_migration(self, migration_path: Path) -> bool:
        """Apply a single migration."""
        version = migration_path.stem
        content = migration_path.read_text(encoding='utf-8')
        checksum = self.calculate_checksum(content)
        
        logger.info(f"Applying migration: {version}")
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Execute migration
                    cursor.execute(content)
                    
                    # Record migration
                    cursor.execute(
                        """
                        INSERT INTO schema_migrations (version, name, checksum)
                        VALUES (%s, %s, %s)
                        """,
                        (version, migration_path.name, checksum)
                    )
                    
                    conn.commit()
                    logger.info(f"Successfully applied migration: {version}")
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to apply migration {version}: {e}")
            return False
    
    def run_migrations(self, target_version: Optional[str] = None) -> bool:
        """Run all pending migrations."""
        self.create_migration_table()
        applied = set(self.get_applied_migrations())
        available = self.get_available_migrations()
        
        success = True
        
        for migration_path in available:
            version = migration_path.stem
            
            # Skip if already applied
            if version in applied:
                logger.info(f"Migration {version} already applied, skipping")
                continue
            
            # Stop if we've reached target version
            if target_version and version > target_version:
                logger.info(f"Reached target version {target_version}, stopping")
                break
            
            # Apply migration
            if not self.apply_migration(migration_path):
                success = False
                break
        
        return success
    
    def validate_schema(self) -> bool:
        """Validate that the schema is properly set up."""
        logger.info("Validating database schema...")
        
        required_tables = [
            'pit_data',
            'settlement_calendar', 
            'taiwan_stock_info',
            'data_quality_log',
            'query_performance_log'
        ]
        
        required_indexes = [
            'idx_pit_symbol_asof_type',
            'idx_pit_value_date',
            'idx_settlement_date'
        ]
        
        required_functions = [
            'get_taiwan_settlement_date',
            'validate_temporal_consistency'
        ]
        
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    # Check tables
                    cursor.execute("""
                        SELECT table_name 
                        FROM information_schema.tables 
                        WHERE table_schema = 'ml4t'
                    """)
                    existing_tables = {row['table_name'] for row in cursor.fetchall()}
                    
                    missing_tables = set(required_tables) - existing_tables
                    if missing_tables:
                        logger.error(f"Missing tables: {missing_tables}")
                        return False
                    
                    # Check indexes
                    cursor.execute("""
                        SELECT indexname 
                        FROM pg_indexes 
                        WHERE schemaname = 'ml4t'
                    """)
                    existing_indexes = {row['indexname'] for row in cursor.fetchall()}
                    
                    missing_indexes = set(required_indexes) - existing_indexes
                    if missing_indexes:
                        logger.error(f"Missing indexes: {missing_indexes}")
                        return False
                    
                    # Check functions
                    cursor.execute("""
                        SELECT routine_name 
                        FROM information_schema.routines 
                        WHERE routine_schema = 'ml4t'
                    """)
                    existing_functions = {row['routine_name'] for row in cursor.fetchall()}
                    
                    missing_functions = set(required_functions) - existing_functions
                    if missing_functions:
                        logger.error(f"Missing functions: {missing_functions}")
                        return False
                    
                    logger.info("Schema validation passed!")
                    return True
                    
        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            return False
    
    def get_migration_status(self):
        """Show migration status."""
        self.create_migration_table()
        applied = self.get_applied_migrations()
        available = self.get_available_migrations()
        
        print("\nMigration Status:")
        print("=" * 50)
        
        for migration_path in available:
            version = migration_path.stem
            status = "✓ Applied" if version in applied else "✗ Pending"
            print(f"{version:<20} {status}")
        
        print(f"\nTotal: {len(available)} migrations")
        print(f"Applied: {len(applied)} migrations")
        print(f"Pending: {len(available) - len(applied)} migrations")


def get_connection_params() -> Dict[str, str]:
    """Get database connection parameters from environment."""
    return {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', '5432')),
        'dbname': os.getenv('DB_NAME', 'ml4t'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', ''),
    }


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description='ML4T Database Migration Runner')
    parser.add_argument('command', choices=['migrate', 'status', 'validate'], 
                       help='Command to execute')
    parser.add_argument('--target', help='Target migration version')
    parser.add_argument('--host', default='localhost', help='Database host')
    parser.add_argument('--port', type=int, default=5432, help='Database port')
    parser.add_argument('--dbname', default='ml4t', help='Database name')
    parser.add_argument('--user', default='postgres', help='Database user')
    parser.add_argument('--password', help='Database password')
    
    args = parser.parse_args()
    
    # Build connection params
    connection_params = {
        'host': args.host,
        'port': args.port,
        'dbname': args.dbname,
        'user': args.user,
    }
    
    if args.password:
        connection_params['password'] = args.password
    elif 'DB_PASSWORD' in os.environ:
        connection_params['password'] = os.environ['DB_PASSWORD']
    
    runner = MigrationRunner(connection_params)
    
    try:
        if args.command == 'migrate':
            print(f"Running migrations on database: {args.dbname}@{args.host}:{args.port}")
            success = runner.run_migrations(args.target)
            if success:
                print("✓ All migrations completed successfully!")
                return 0
            else:
                print("✗ Migration failed!")
                return 1
                
        elif args.command == 'status':
            runner.get_migration_status()
            return 0
            
        elif args.command == 'validate':
            success = runner.validate_schema()
            if success:
                print("✓ Schema validation passed!")
                return 0
            else:
                print("✗ Schema validation failed!")
                return 1
                
    except Exception as e:
        logger.error(f"Command failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())