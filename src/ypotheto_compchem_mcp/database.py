import logging
import time

import psycopg2

from ypotheto_compchem_mcp.config import settings


def get_connection():
    """Retrieve a connection to the managed PostgreSQL database with retries for slot saturation."""
    if not settings.database_url:
        return None
        
    max_retries = 6
    for attempt in range(max_retries):
        try:
            return psycopg2.connect(settings.database_url)
        except psycopg2.OperationalError as e:
            err_msg = str(e)
            # If the database ran out of connection slots, sleep and retry
            if "remaining connection slots" in err_msg or "too many connections" in err_msg:
                if attempt < max_retries - 1:
                    sleep_time = 1.5 * (attempt + 1)
                    logging.warning(
                        f"Database connection slots full. Retrying connection in {sleep_time:.1f}s "
                        f"(attempt {attempt + 1}/{max_retries})..."
                    )
                    time.sleep(sleep_time)
                    continue
            logging.error(f"Failed to connect to database: {err_msg}")
            raise e
        except Exception as e:
            logging.error(f"Failed to connect to database: {str(e)}")
            raise e

def initialize_database():
    """Initialize the schema and tables in the PostgreSQL database if configured."""
    conn = get_connection()
    if conn is None:
        logging.warning("Database URL is not configured. Database features are disabled.")
        return
        
    try:
        cur = conn.cursor()

        cur.execute("CREATE SCHEMA IF NOT EXISTS compchem;")

        # 1. Molecules Table (Searchable Archive)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS compchem.molecules (
                molecule_id VARCHAR(50) PRIMARY KEY,
                workspace_id VARCHAR(100) NOT NULL,
                name VARCHAR(255) NOT NULL,
                formula VARCHAR(100) NOT NULL,
                smiles TEXT NOT NULL,
                num_atoms INT NOT NULL,
                method VARCHAR(50) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB DEFAULT '{}'::jsonb
            );
        """)
        
        cur.execute("CREATE INDEX IF NOT EXISTS idx_molecules_workspace ON compchem.molecules(workspace_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_molecules_formula ON compchem.molecules(formula);")
        
        # 2. Durable Jobs Queue Table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS compchem.jobs (
                job_id VARCHAR(50) PRIMARY KEY,
                workspace_id VARCHAR(100) NOT NULL,
                status VARCHAR(50) NOT NULL,
                progress_message TEXT DEFAULT 'Job initialized.',
                estimated_time_seconds INT NOT NULL,
                func_name VARCHAR(255) NOT NULL,
                args JSONB DEFAULT '[]'::jsonb,
                kwargs JSONB DEFAULT '{}'::jsonb,
                results JSONB DEFAULT '{}'::jsonb,
                warnings JSONB DEFAULT '[]'::jsonb,
                error JSONB DEFAULT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP DEFAULT NULL,
                finished_at TIMESTAMP DEFAULT NULL,
                lease_timeout TIMESTAMP DEFAULT NULL
            );
        """)
        
        cur.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON compchem.jobs(status);")
        
        conn.commit()
        cur.close()
        conn.close()
        logging.info("PostgreSQL database tables successfully initialized.")
    except Exception as e:
        logging.error(f"Failed to initialize database tables: {str(e)}", exc_info=True)
