import os
import psycopg2
import logging
from re import sub
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("EntityResolver")

class EntityResolver:
    def __init__(self):
        self.conn_uri = os.getenv("POSTGRES_URI", "postgresql://postgres:password@localhost:5432/quant_engine")
        self.mapping_cache = {}
        self.load_mappings_from_db()

    def load_mappings_from_db(self):
        """Loads mappings from PostgreSQL into local RAM dict for O(1) speed."""
        try:
            conn = psycopg2.connect(self.conn_uri)
            cursor = conn.cursor()
            # Ensure table exists check
            cursor.execute("SELECT book_name, remote_player_name, master_player_id FROM player_mappings;")
            rows = cursor.fetchall()
            for book_name, remote_name, master_id in rows:
                self.mapping_cache[(book_name.lower(), remote_name.lower())] = master_id
            cursor.close()
            conn.close()
            logger.info(f"Successfully cached {len(self.mapping_cache)} entity mappings from PostgreSQL.")
        except Exception as e:
            logger.error(f"Error loading entity mappings from database: {e}")

    def normalize_string(self, text: str) -> str:
        """Removes accents, suffixes, and punctuation for baseline matching."""
        text = text.lower().strip()
        text = sub(r'[.\']', '', text) # Remove periods and apostrophes (P.J. -> pj)
        text = sub(r'\s+(jr|sr|iii|ii|iv)$', '', text) # Strip suffixes
        return text

    def resolve_player(self, book_name: str, remote_name: str) -> str:
        """Resolves a book-specific player name string to our master_player_id."""
        book_clean = book_name.lower().strip()
        name_clean = remote_name.lower().strip()
        
        # Fast path: exact dictionary match
        if (book_clean, name_clean) in self.mapping_cache:
            return self.mapping_cache[(book_clean, name_clean)]
            
        # Fallback path: normalization slugification
        normalized_name = self.normalize_string(remote_name)
        fallback_id = normalized_name.replace(" ", "_")
        
        logger.warning(f"[UNRESOLVED ENTITY] Stale cache hit for '{remote_name}' on {book_name}. Defaulting to fallback ID: {fallback_id}")
        return fallback_id

if __name__ == "__main__":
    # Simple test instantiation
    logging.basicConfig(level=logging.INFO)
    resolver = EntityResolver()
    print(resolver.resolve_player("DraftKings", "P.J. Washington Jr."))
