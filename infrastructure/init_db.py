import os
import logging
from urllib.parse import urlparse
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv

load_dotenv()


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DBInit")

def init_db():
    db_uri = os.getenv("POSTGRES_URI", "postgresql://postgres:password@localhost:5432/sports_db")
    parsed = urlparse(db_uri)
    db_user = parsed.username
    db_password = parsed.password
    db_host = parsed.hostname
    db_port = parsed.port
    db_name = parsed.path.lstrip('/') or 'sports_db'

    # 1. Connect to default 'postgres' database to create database
    try:
        conn = psycopg2.connect(
            dbname='postgres',
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        logger.info(f"Attempting to create database '{db_name}'...")
        try:
            cursor.execute(f"CREATE DATABASE {db_name};")
            logger.info(f"Database '{db_name}' created successfully.")
        except psycopg2.errors.DuplicateDatabase:
            logger.info(f"Database '{db_name}' already exists. Skipping creation.")
            
        cursor.close()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to connect to default postgres db: {e}")
        return

    # 2. Connect to newly created database and apply schema
    try:
        logger.info(f"Connecting to '{db_name}' to apply schema...")
        conn = psycopg2.connect(
            dbname=db_name,
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port
        )
        cursor = conn.cursor()
        
        schema_path = os.path.join(os.path.dirname(__file__), 'db_schema.sql')
        with open(schema_path, 'r') as f:
            schema_sql = f.read()
            
        logger.info("Executing db_schema.sql...")
        cursor.execute(schema_sql)
        
        # 3. Drop foreign key constraints temporarily to allow the backfill
        logger.info("Dropping foreign key constraints for initial backfill...")
        cursor.execute("ALTER TABLE player_game_logs DROP CONSTRAINT IF EXISTS player_game_logs_game_id_fkey;")
        cursor.execute("ALTER TABLE player_game_logs DROP CONSTRAINT IF EXISTS player_game_logs_player_id_fkey;")
        
        conn.commit()
        logger.info("Schema applied successfully.")
        
        cursor.close()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to apply schema: {e}")

if __name__ == "__main__":
    init_db()
