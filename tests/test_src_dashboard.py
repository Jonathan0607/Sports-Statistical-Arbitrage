from streamlit.testing.v1 import AppTest

def test_src_dashboard_rendering():
    """
    Simulates a headless run of the Streamlit dashboard (src/dashboard.py)
    to ensure it compiles, executes without throwing exceptions, and renders
    the proper UI elements under mock data conditions.
    """
    # 1. Initialize and run the Streamlit app headlessly
    at = AppTest.from_file("src/dashboard.py").run(timeout=15)

    # 2. Verify there are no uncaught exceptions during setup or render
    assert not at.exception, f"App execution raised uncaught exceptions: {at.exception}"

    # 3. Verify UI Layout & Header
    header_found = any("StatArb Trading Terminal" in md.value for md in at.markdown)
    assert header_found, "Dashboard header 'StatArb Trading Terminal' was not found in the rendered markdown."

    # 4. Verify Tabs are initialized (Live Execution Board and Historical Analytics)
    assert len(at.tabs) >= 2, f"Expected at least 2 tabs, but found {len(at.tabs)}."
    
    # 5. Verify dataframes are rendered OR clean empty-state warning is shown
    has_df = len(at.dataframe) > 0
    has_info = any("System Active" in info.value for info in at.info)
    assert has_df or has_info, "UI failed to render correctly in empty state"

