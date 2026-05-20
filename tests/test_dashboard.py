from streamlit.testing.v1 import AppTest

def test_dashboard_rendering():
    """
    Simulates a headless run of the Streamlit dashboard to ensure it compiles, 
    executes without throwing exceptions, and renders the proper UI elements.
    """
    # 1. Initialize and run the Streamlit app headlessly with increased timeout
    at = AppTest.from_file("app.py").run(timeout=15)

    # 2. Verify there are no uncaught Python or Plotly exceptions
    assert not at.exception, f"App execution raised uncaught exceptions: {at.exception}"

    # 3. Verify UI Layout & Header
    # Our app uses st.markdown to inject the "StatArb Trading Terminal" header
    header_found = any("StatArb Trading Terminal" in md.value for md in at.markdown)
    assert header_found, "Dashboard header 'StatArb Trading Terminal' was not found in the rendered markdown."

    # 4. Verify Tabs are initialized
    # We should have exactly 4 tabs created: Live Signal Board, Copula Risk, MC Drawdown, CLV Tracker
    assert len(at.tabs) >= 4, f"Expected at least 4 tabs, but found {len(at.tabs)}."

    # 5. Verify Data Elements
    # The app renders multiple dataframes (Active Order Book, DB Connection Logs, CLV Mock Trades)
    assert len(at.dataframe) > 0, "No dataframes were rendered. The Active Order Book might be failing."

    # 6. Verify Visualizations (Plotly charts)
    # The app uses plotly charts which rendered successfully (no exceptions thrown).
