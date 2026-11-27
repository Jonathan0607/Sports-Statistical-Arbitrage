# Engineering Changelog

All notable production architecture, runtime, serving, and build changes are documented in this ledger.

## [2026-11-25] Phase 1: Production ML Serving, Feature Parity & C++20 Hardening

### Added
- **Unified Feature Parity Contract (`src/feature_contract.py`):**
  - Defined `FEATURE_SCHEMA_VERSION = "1.0.0"`, ordered `FEATURE_COLUMNS` (28 NBA features) and `WNBA_FEATURE_COLUMNS` (21+ WNBA features), and shared `predict_meta_ensemble()` / `prepare_xgb_lgb_frames()` adapters handling cold-start unseen categorical IDs safely via known training category extraction.
  - Enforced strictly causal (`0..T-1`) expanding player cohort tiering (`assign_player_tier()`), shifted season-partitioned EWMA routing (`_causal_tier_ewma()`), shifted `cum_min` for Bayesian shrinkage (`compute_target_rolling_features()`), and shared single-snapshot inference assembly (`build_online_feature_row()`).
- **Typed Inference & Portfolio Allocation Service (`src/serving/`):**
  - `src/serving/schemas.py`: Strict Pydantic v2 request/response contracts (`extra="forbid"`, `allow_inf_nan=False`, explicit finiteness validators rejecting `NaN`/`Inf`, bounded numeric domains, `LeagueEnum`, `StatKeyEnum`, `schema_version` validation, structured `ErrorResponse`).
  - `src/serving/predictor.py`: `InferencePredictor` orchestrating Redis history hydration (`player:history:{id}`), offline-identical Meta-Ensemble (XGBoost + LightGBM + Ridge) residual inference, ZINB survival pricing, Clayton/Gumbel/Frank/Gaussian Copula joint dependence modeling, Ledoit-Wolf shrinkage, C++ Eigen `ColPivHouseholderQR` Kelly optimization, numerical condition/rank telemetry, and positional-prior cold-start handling.
  - `src/serving/app.py`: FastAPI service exposing `GET /health`, `POST /v1/predict`, and `POST /v1/allocate` with JSON-safe HTTP 422 validation exception handling.
- **Verification & Profiling Suites:**
  - `tests/test_feature_parity.py`: Temporal leakage perturbation invariance (`rtol=0, atol=1e-12`), 50-snapshot batch vs. online inference parity (`atol=1e-6`), 200+ row vectorized vs. sequential evaluator parity, strict DNP `vacated_usage` test, SQLite-backed `get_co_stars()` historical mid-season trade test, and `train_seasons` isolation regression test.
  - `tests/test_cpp_bindings.py`: Deterministic C++20 unit tests covering `quant_core.calculate_shins_probs` (balanced/skewed/no-vig markets vs. Python reference `< 1e-6`) and `quant_core.optimize_multivariate_kelly` (`ColPivHouseholderQR` vs. SciPy Pivoted QR across well-conditioned, ill-conditioned, and rank-deficient covariance matrices).
  - `tests/test_serving_api.py`: FastAPI `TestClient` suite covering `/health`, `/v1/predict`, `/v1/allocate`, cold-start unseen players, singular/rank-deficient covariance matrices, strict `422` rejection of `NaN`/`Inf`/out-of-bounds/missing/extra fields, and async `httpx` poller integration.
  - `benchmarks/profile_inference.py`: Stage-by-stage `time.perf_counter_ns()` harness (100 warmup + 1,000 timed iterations) emitting `benchmarks/latest_latency_report.json` and Markdown latency/accuracy comparison tables across Redis hydration, Meta-Ensemble inference, ZINB+Copula pricing, C++ Eigen vs. NumPy `lstsq` / SciPy Pivoted QR across `N = 5, 15, 30, 50`, and API round-trips.

### Changed
- **Point-in-Time Leakage Remediation (`src/train_global_model.py`, `src/evaluate_fast.py`, `src/historical_replay_core.py`, `src/features.py`):**
  - Enforced `train_seasons` filtering and seasonal `target_date` SQL cutoffs prior to feature compilation in `src/train_global_model.py`, replaced row-wise CV with timestamp-grouped `_iter_grouped_time_splits()`, moved Optuna NB dispersion `alpha` calibration strictly onto OOF residuals, and recorded `feature_schema_version` / `train_seasons` in saved `*_meta.json` artifacts.
  - Refactored `src/evaluate_fast.py` (`evaluate_dataframe_vectorized()` + seasonal `eval_cutoff`) and `src/historical_replay_core.py` (`evaluate_dataframe_sequential()`, plus `.shift(1)` on legacy `roll_residual`) onto `src/feature_contract.py`.
  - Updated `src/features.py`: replaced mutable `players.current_team_id` in `get_co_stars()` with historical `player_game_logs JOIN games` (`gm.start_time < :target_date`) team inference, added causal per-row `team_id` schedule inference in `build_player_feature_matrix()`, preserved pre-supplied historical context IDs/fields, and restricted `vacated_usage` sidelined detection to strict DNP `minutes_played == 0.0` instead of `< 5.0` to prevent intra-game early injury/ejection leakage.
- **Poller Decoupling (`scrapers/hfp_poller.py`):**
  - Added optional backward-compatible `--inference-url` CLI flag, `INFERENCE_URL` env support, and async `predict_via_http()` routing to `UnifiedRestPoller` via `httpx.AsyncClient`.
- **C++20 Strict Eigen Build & Git Hygiene (`.gitignore`, `CMakeLists.txt`, `cpp_core/shins_method.cpp`, `cpp_core/nb_loss.cpp`):**
  - Upgraded `CMakeLists.txt` target options to `-std=c++20 -O3 -Wall -Wextra -Wpedantic -Werror` and fixed GCC 14.2 / Clang strict warnings (`-Wunused-variable` on `z_prev` and `tolerance` in `shins_method.cpp`, `-Wsign-compare` in `nb_loss.cpp`).
  - Added `CMakeFiles/` alongside `build/`, `*.so`, `*.dylib` in `.gitignore`.
- **Containerized Local Stack & Live Linux Smoke Test (`Dockerfile`, `docker-compose.yml`, `requirements.txt`):**
  - Built multi-stage Linux Docker images compiling C++20 `quant_core` under GCC 14.2.0 strict `-Werror` flags in `builder` and serving `uvicorn src.serving.app:app` in slim Python 3.11 runtime with `/health` healthcheck.
  - Resolved Linux container dependency incompatibilities in `requirements.txt` (`starlette==1.10.0` for `fastapi==0.165.0`, and `lifelines==0.30.3` for `scipy==1.17.1` compatibility after `scipy.integrate.trapz` removal).
  - Verified live container stack (`docker compose up -d --no-deps redis inference-api`) via HTTP `200` smoke tests against `GET /health` (`redis_connected: true`, `cpp_eigen_available: true`, `feature_schema_version: "1.0.0"`), `POST /v1/predict` (`23.26 ms`), and `POST /v1/allocate` (`cpp_eigen_colpiv_qr`, `1.72 ms`), followed by clean `docker compose down`.

---

## [2026-11-25] Phase 2: Drift Circuit Breaker, Model Registry, Test Suite Repair & CI/CD

### Added
- **Rolling Calibration & Covariate Drift Harness (`src/drift_monitor.py`):**
  - Unified SQLAlchemy telemetry store (`prediction_telemetry` supporting SQLite in-memory/file WAL and PostgreSQL) recording `prediction_id`, `timestamp`, `model_version`, `feature_schema_version`, `player_id`, `stat_key`, `predicted_mean`, `ewma_baseline`, `predicted_prob`, `line`, `feature_snapshot_json`, and nullable `realized_value` / `realized_hit` updated via `record_settlement()`.
  - Rolling-window calibration evaluator computing 10-bin **Expected Calibration Error (`compute_ece`)**, **Brier Score**, **Log-Loss**, and **Meta MAE vs. EWMA MAE** delta alongside **Population Stability Index (`compute_psi`)** and two-sample **Kolmogorov-Smirnov p-values** (`scipy.stats.ks_2samp`) against `models/registry/reference_stats.json` (generated from `55,055` clean Fold 2 training rows).
  - Automated `DriftThresholds` circuit breaker emitting `HEALTHY`, `WARN_RETRAIN`, or `CIRCUIT_BREAK_FALLBACK_EWMA`, with optional Discord webhook alerts and CLI inspection.
- **Champion/Challenger Model Registry (`src/model_registry.py`, `models/registry/`):**
  - JSON-backed release management under `models/registry/` (`registry.json`, `champion.json`, `releases/<version_id>.json`) tracking `vYYYY.MM.DD-<short_sha>`, `git_sha`, `feature_schema_version`, `train_seasons`, `training_samples`, per-stat SHA-256 checksums across all 28 artifacts, and benchmark metrics (`overall_mae`, `overall_crps`, `ewma_mae`).
  - Implemented `promote_challenger()` promotion gates (schema compatibility, SHA-256 integrity, baseline `ewma_mae` clearance, and incumbent champion comparison) plus atomic `rollback_to_previous()` using temp-file + `os.replace`.
  - Registered clean Phase 1 `models/wf_fold2` artifacts as initial champion release `v2026.11.25-4c884cd`.
- **CI/CD Workflow & Synthetic Test Fixture Seeding (`.github/workflows/ci.yml`, `scripts/create_ci_model_fixtures.py`, `pytest.ini`):**
  - Upgraded GitHub Actions pipeline with `cpp-and-python-tests` (C++20 `-Wall -Wextra -Wpedantic -Werror` Eigen build + `pytest -m "not db"`) and `docker-smoke-test` (`--build-arg GIT_SHA=${{ github.sha }}` container verification on `/health` and `/v1/drift`).
  - Added `pytest.ini` registering `@pytest.mark.db` and `tests/test_drift_and_registry.py` validating calibration metrics, drift circuit breaker escalation, FastAPI EWMA fallback, registry promotion gates, and atomic rollback.

### Changed
- **Serving & Container Governance Wiring (`src/serving/app.py`, `src/serving/predictor.py`, `src/serving/schemas.py`, `Dockerfile`):**
  - Exposed `GET /v1/drift` (`DriftReportResponse`), added `drift_status`, `fallback_active`, and `registry_champion_id` to `GET /health`, enabled prediction telemetry logging, wired automatic EWMA fallback in `predict_batch()` when `fallback_active` trips, resolved default `model_version` from `models/registry/champion.json`, and added `ARG GIT_SHA=unknown` / `ENV GIT_SHA=${GIT_SHA}` to `Dockerfile`.
- **Legacy Suite & SHAP Categorical Repair (`src/shap_analyzer.py`, `tests/`):**
  - Updated `src/shap_analyzer.py` to compute exact native XGBoost TreeSHAP (`pred_contribs=True` on `xgb.DMatrix(..., enable_categorical=True)`) and encode categorical axes for Beeswarm plotting.
  - Updated stale unit tests in `tests/test_shap_analyzer.py`, `tests/test_historical_replay.py` (marking live-Postgres integration test `@pytest.mark.db`), `tests/test_smart_cache_and_discord_handler.py`, `tests/test_src_dashboard.py`, and `tests/test_walk_forward.py` to match current production contracts.

---

## [2026-11-25] Phase 3: Context-Dependent Conditional Copula & Registry Calibration Enrichment

### Added
- **Conditional Copula Engine (`src/conditional_copula.py`, `models/registry/conditional_copula_fold2.json`):**
  - Implemented regularized context-dependent Kendall's $\hat{\tau}(\mathbf{x}) \in [-0.75, 0.75]$ estimator optimized directly via L2-regularized Joint Binary Log-Loss with chronological validation.
  - Enforced analytical output copulas (**Gumbel** for `OO` $\hat{\tau}>0$, **Clayton** for `UU` $\hat{\tau}>0$, exact polynomial **Frank** / **Gaussian** for negative/mixed concordance), strict Fréchet–Hoeffding bounds clamping, leg-swap feature symmetry (`build_pair_feature_row()`), and PSD covariance projection (`ensure_psd_covariance()`).
- **Walk-Forward Joint Ablation & Tests (`benchmarks/run_copula_ablation.py`, `benchmarks/copula_ablation_exp15_report.json`, `tests/test_conditional_copula.py`):**
  - Validated `conditional_copula` across `62,400` Fold 1 (`2024-25`) and `62,448` Fold 2 (`2025-26`) unseen pairs with `1,000`-resample `game_id` cluster bootstrap significance testing.
  - Added deterministic unit + API suite covering Fréchet–Hoeffding bounds, negative/positive concordance mapping, PSD covariance eigenvalues, temporal PIT invariance, and `/v1/allocate` conditional/static switching.

### Changed
- **Allocation Serving & Registry Calibration (`src/serving/predictor.py`, `src/serving/schemas.py`, `models/registry/`):**
  - Extended `AllocateCandidateOutcome` / `AllocateRequest` with backward-compatible optional context fields (`position`, `ewma_minutes`, `vacated_usage`, `opp_pace_factor`, `team_strength`, `projected_pace_diff`, `abs_team_strength_diff`, `use_conditional_copula`) and `AllocateResponse.copula_mode`.
  - Enabled conditional copula by default inside `InferencePredictor.allocate()` after clearing Fold 1 + Fold 2 Experiment 15 gates (`p50 = 2.143 ms` on 32-leg `/v1/allocate`), and populated Fold 2 out-of-sample marginal calibration (`brier_score = 0.243877`) in `models/registry/champion.json`.

---

## [2026-11-25] Phase 4: Sandbox-First Kalshi OMS, ZINB Ladder Pricing & Closed-Loop Reconciliation

### Added
- **Authenticated Kalshi Trade API v2 Client (`src/oms/kalshi_client.py`):**
  - Implemented async `httpx.AsyncClient` REST wrapper supporting dynamic **Ed25519** and **RSA-PSS (SHA-256)** PEM private key signing (`KALSHI-ACCESS-KEY`, `KALSHI-ACCESS-SIGNATURE`, `KALSHI-ACCESS-TIMESTAMP`) with automatic query-string stripping and `/trade-api/v2` path normalization.
  - Enforced strict Sandbox-first safety defaults (`env="demo"` -> `https://external-api.demo.kalshi.co/trade-api/v2`); `env="prod"` raises `RuntimeError` unless BOTH `env="prod"` and `live_armed=True` are passed explicitly.
  - Added deterministic UUID `client_order_id` idempotent `409 Conflict` handling and jittered exponential backoff for `429` / `5xx` responses.
- **ZINB Multi-Threshold Ladder Pricer (`src/oms/ladder_pricer.py`):**
  - Added exact ceiling-rounded Kalshi fee calculator `calculate_kalshi_fee()` (`ceil(100 * rate * multiplier * C * P * (1 - P)) / 100`), ZINB survival calculator `zinb_survival_at_least()`, dual YES (`bid`) / NO (`ask` at `1 - no_price`) evaluation across taker (`immediate_or_cancel`) and passive maker (`good_till_canceled`, `post_only=True`) modes, non-dominated rung filtering, and `/v1/allocate` Kelly sizing integration.
- **Deterministic Non-ML Risk Guardrails & Failsafe (`src/oms/risk_guardrails.py`):**
  - Enforced hard pre-trade checks outside the ML stack: `.oms_kill_switch` / `OMS_KILL_SWITCH` kill switch + cancel-all sweep, `DriftMonitor` `fallback_active` circuit-breaker halt, stale quote rejection (`> 5000 ms`), price (`[0.05, 0.95]`) & spread (`<= 0.08`) bounds, per-order/market/total exposure and daily loss caps, and inference-outage cancel-all failsafe.
- **Closed-Loop Settlement Reconciler & Runner (`src/oms/reconciler.py`, `src/oms/runner.py`):**
  - Added SQLAlchemy `oms_orders` and `oms_fills` tables sharing `DriftMonitor`'s engine with `prediction_telemetry.prediction_id` foreign-key lineage, idempotent fill slippage accounting, realized P&L computation, and closed-loop `DriftMonitor.record_settlement()` updates.
  - Added CLI execution runner `src/oms/runner.py` defaulting safely to `--env demo` and `dry_run=True` unless `--no-dry-run` is explicitly passed.
- **Offline Phase 4 Test Suite (`tests/test_kalshi_oms.py`):**
  - Added 100% offline tests covering Ed25519 & RSA-PSS signatures, `prod` lock, `409` idempotency + `429` retry, Kalshi fee math & YES/NO taker/maker ladder pricing, all 6 risk guardrails + failsafe cancellation, and full FastAPI ASGI `/v1/predict` + `/v1/allocate` + `/v1/drift` -> Kalshi `MockTransport` -> settlement -> `DriftMonitor` calibration loop.

### Changed
- **Dependencies (`requirements.txt`):**
  - Pinned `cryptography==50.0.2` alongside `httpx==0.28.1` for Kalshi Trade API v2 asymmetric key signing.
