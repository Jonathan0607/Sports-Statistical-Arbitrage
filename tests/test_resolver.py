"""
Test Harness: Self-Healing Entity Resolution TDD Loop
-----------------------------------------------------
Verifies that aggressively stripped player names resolve to valid
master_player_ids instead of falling through to the slugified fallback.
"""
import sys
import os
import logging

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from infrastructure.entity_resolver import EntityResolver, clean_player_name

logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger("TestResolver")

# ── Pre-flight: verify clean_player_name works as expected ──────────────────
def test_clean_player_name():
    """Ensure the aggressive stripping function produces the expected keys."""
    cases = {
        "OG Anunoby":           "oganunoby",
        "Karl-Anthony Towns":   "karlanthonytowns",
        "P.J. Washington Jr.":  "pjwashington",
        "Domantas Sabonis":     "domantassabonis",
        "Shai Gilgeous-Alexander": "shaigilgeousalexander",
        "De’Aaron Fox":         "deaaronfox",
    }
    passed = 0
    for raw, expected in cases.items():
        result = clean_player_name(raw)
        status = "✅" if result == expected else "❌"
        if result != expected:
            logger.error(f"{status} clean_player_name('{raw}') = '{result}' (expected '{expected}')")
        else:
            logger.info(f"{status} clean_player_name('{raw}') = '{result}'")
            passed += 1
    logger.info(f"clean_player_name: {passed}/{len(cases)} passed")
    return passed == len(cases)


# ── Main TDD Loop: resolve aggressively stripped names via the DB cache ─────
def test_resolver_lookups():
    """
    Instantiate EntityResolver (connects to Postgres and loads caches),
    then attempt to resolve names that previously caused [UNRESOLVED ENTITY].
    """
    resolver = EntityResolver()

    # These are the exact strings that the Monte Carlo engine passes after
    # clean_player_name is applied by the scraper pipeline.
    test_cases = [
        ("PrizePicks", "oganunoby"),
        ("PrizePicks", "karlanthonytowns"),
        ("PrizePicks", "domantassabonis"),
    ]

    all_passed = True
    for book, name in test_cases:
        result = resolver.resolve_player(book, name)
        # A successful resolution returns a master_player_id (UUID or int).
        # A failure returns the slugified fallback (e.g., "oganunoby" or "og_anunoby").
        is_fallback = result == name or result == name.replace(" ", "_")
        status = "✅" if not is_fallback else "❌"
        if is_fallback:
            logger.error(f"{status} resolve_player('{book}', '{name}') -> FALLBACK '{result}'")
            all_passed = False
        else:
            logger.info(f"{status} resolve_player('{book}', '{name}') -> '{result}'")

    return all_passed


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("PHASE 1: Testing clean_player_name()")
    logger.info("=" * 60)
    cpn_ok = test_clean_player_name()

    logger.info("")
    logger.info("=" * 60)
    logger.info("PHASE 2: Testing EntityResolver.resolve_player()")
    logger.info("=" * 60)
    resolver_ok = test_resolver_lookups()

    logger.info("")
    logger.info("=" * 60)
    if cpn_ok and resolver_ok:
        logger.info("🟢 ALL TESTS PASSED")
        sys.exit(0)
    else:
        logger.error("🔴 SOME TESTS FAILED")
        sys.exit(1)
