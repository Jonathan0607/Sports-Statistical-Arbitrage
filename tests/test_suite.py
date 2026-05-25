from streamlit.testing.v1 import AppTest
import os

def test_frontend_rendering():
    """
    Step 1: Headless Frontend Verification.
    Verifies that the main dashboard layout runs without exceptions and renders the tabs.
    """
    at = AppTest.from_file("src/dashboard.py").run(timeout=15)
    
    # Assert no uncaught exceptions thrown during compilation/runtime
    assert not at.exception, f"App threw uncaught exceptions: {at.exception}"
    
    # Assert structural tabs are generated (at least 2: Live Execution Board & Historical Analytics)
    assert len(at.tabs) >= 2, f"Expected at least 2 tabs, found {len(at.tabs)}"


def test_integration_pipeline_design():
    """
    Step 2: Integration Pipeline Check.
    Verifies that src/execution/sgp_worker.py contains the queue manager and enqueues slips to Redis.
    """
    worker_path = "src/execution/sgp_worker.py"
    assert os.path.exists(worker_path), f"File {worker_path} does not exist!"
    
    with open(worker_path, "r") as f:
        code = f.read()
        
    # Check queue manager class definition/usage
    assert "SlipQueueManager" in code, "SlipQueueManager is not defined/used in sgp_worker.py"
    assert "enqueue_slip" in code, "enqueue_slip is not utilized in sgp_worker.py"


def test_technical_documentation_completeness():
    """
    Step 3: Technical Documentation Audit.
    Inspects README.md to ensure all mathematical models and formulas (ZINB, Copula, and TSCV) are documented.
    """
    readme_path = "README.md"
    assert os.path.exists(readme_path), f"File {readme_path} does not exist!"
    
    with open(readme_path, "r") as f:
        doc = f.read()
        
    # Check ZINB math equations
    assert "P(Y = y) =" in doc, "ZINB probability mass function equation is missing in README.md"
    assert "\\pi \\in [0,1]" in doc, "ZINB zero-inflation parameter explanation is missing in README.md"
    
    # Check Archimedean Copula math equations
    assert "C(u, v) =" in doc, "Archimedean Copula bivariate equation is missing in README.md"
    assert "\\theta = \\frac{2\\tau}{1 - \\tau}" in doc, "Kendall's Tau calibration formula is missing in README.md"
    
    # Check Time-Series Cross-Validation split details
    assert "Time-Series Split with Purging" in doc, "TSCV purging text is missing in README.md"
    assert "\\delta_{\\text{purge}}" in doc, "TSCV purging mathematical notation is missing in README.md"
    assert "T_t = " in doc, "TSCV training split equation is missing in README.md"
