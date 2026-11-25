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
