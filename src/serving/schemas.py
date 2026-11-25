"""Strict Pydantic v2 schemas for the StatArb inference and portfolio allocation service."""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.feature_contract import FEATURE_SCHEMA_VERSION, STAT_COL_TO_TARGET


class LeagueEnum(str, Enum):
    NBA = "NBA"
    WNBA = "WNBA"


class StatKeyEnum(str, Enum):
    pts = "pts"
    reb = "reb"
    ast = "ast"
    fg3m = "fg3m"
    stl = "stl"
    blk = "blk"
    tov = "tov"


class SideEnum(str, Enum):
    over = "Over"
    under = "Under"


class CopulaTypeEnum(str, Enum):
    auto = "auto"
    clayton = "clayton"
    gumbel = "gumbel"
    frank = "frank"
    gaussian = "gaussian"


def _assert_finite_float(value: float, field_name: str = "value") -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be a numeric float")
    val = float(value)
    if not math.isfinite(val):
        raise ValueError(f"{field_name} must be finite (NaN/Inf rejected)")
    return val


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, str_strip_whitespace=True)


class FeaturePayload(StrictBaseModel):
    """Exact 28-feature contract payload matching FEATURE_COLUMNS."""

    roll_min: float = Field(..., ge=0.0, le=65.0)
    roll_stat: float = Field(..., ge=0.0, le=150.0)
    roll_usg: float = Field(..., ge=0.0, le=1.0)
    days_rest: float = Field(..., ge=0.0, le=365.0)
    is_back_to_back: float = Field(..., ge=0.0, le=1.0)
    opp_def_rating: float = Field(..., gt=0.0, le=250.0)
    opp_pace_factor: float = Field(..., gt=0.0, le=200.0)
    opp_pos_pts_allowed: float = Field(..., ge=0.0, le=100.0)
    vacated_usage: float = Field(..., ge=0.0, le=2.0)
    roll_potential_assists: float = Field(..., ge=0.0, le=100.0)
    roll_touches: float = Field(..., ge=0.0, le=400.0)
    roll_passes_made: float = Field(..., ge=0.0, le=400.0)
    roll_time_of_possession_sec: float = Field(..., ge=0.0, le=3600.0)
    roll_defender_distance: float = Field(..., ge=0.0, le=50.0)
    is_home_game: float = Field(..., ge=0.0, le=1.0)
    pace_differential: float = Field(..., ge=-100.0, le=100.0)
    roll_stat_var: float = Field(..., ge=0.0, le=5000.0)
    roll_residual: float = Field(..., ge=-150.0, le=150.0)
    roll_ts_pct: float = Field(..., ge=0.0, le=3.0)
    roll_reb_rate: float = Field(..., ge=0.0, le=80.0)
    roll_ast_rate: float = Field(..., ge=0.0, le=80.0)
    roll_shorthanded_points: float = Field(..., ge=0.0, le=150.0)
    roll_plus_minus: float = Field(..., ge=-100.0, le=100.0)
    roll_usg_efficiency: float = Field(..., ge=0.0, le=50.0)
    player_id: Union[str, int]
    team_id: int = Field(..., ge=0)
    opponent_team_id: int = Field(..., ge=0)
    player_tier: int = Field(..., ge=0, le=3)

    bayesian_roll_stat: Optional[float] = Field(default=None, ge=0.0, le=150.0)
    alpha_disp_override: Optional[float] = Field(default=None, gt=0.0, le=20.0)
    zero_inflation_prob: Optional[float] = Field(default=None, ge=0.0, lt=1.0)

    @field_validator("player_id")
    @classmethod
    def validate_player_id(cls, v: Union[str, int]) -> str:
        s = str(v).strip()
        if not s:
            raise ValueError("player_id cannot be empty")
        return s

    @field_validator(
        "roll_min",
        "roll_stat",
        "roll_usg",
        "days_rest",
        "is_back_to_back",
        "opp_def_rating",
        "opp_pace_factor",
        "opp_pos_pts_allowed",
        "vacated_usage",
        "roll_potential_assists",
        "roll_touches",
        "roll_passes_made",
        "roll_time_of_possession_sec",
        "roll_defender_distance",
        "is_home_game",
        "pace_differential",
        "roll_stat_var",
        "roll_residual",
        "roll_ts_pct",
        "roll_reb_rate",
        "roll_ast_rate",
        "roll_shorthanded_points",
        "roll_plus_minus",
        "roll_usg_efficiency",
        "bayesian_roll_stat",
        "alpha_disp_override",
        "zero_inflation_prob",
    )
    @classmethod
    def check_finite_floats(cls, v: Optional[float], info) -> Optional[float]:
        if v is None:
            return None
        return _assert_finite_float(v, info.field_name)


class GameContextPayload(StrictBaseModel):
    """Pre-game context used when hydrating player history from Redis or inline logs."""

    game_id: int = Field(default=0, ge=0)
    start_time: Optional[str] = None
    season: Optional[str] = None
    home_team_id: int = Field(default=0, ge=0)
    away_team_id: int = Field(default=0, ge=0)
    team_id: int = Field(default=0, ge=0)
    opponent_team_id: int = Field(default=0, ge=0)
    days_rest: Optional[float] = Field(default=None, ge=0.0, le=365.0)
    is_back_to_back: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    is_home_game: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    opp_def_rating: float = Field(default=115.0, gt=0.0, le=250.0)
    opp_pace_factor: float = Field(default=99.0, gt=0.0, le=200.0)
    opp_pos_pts_allowed: float = Field(default=22.0, ge=0.0, le=100.0)
    pace_differential: float = Field(default=0.0, ge=-100.0, le=100.0)
    vacated_usage: float = Field(default=0.0, ge=0.0, le=2.0)
    position: str = Field(default="SG", min_length=1, max_length=10)


class GameLogEntry(StrictBaseModel):
    """Typed single historical box-score row for inline/Redis feature computation."""

    game_id: int = Field(default=0, ge=0)
    start_time: str
    season: Optional[str] = None
    home_team_id: int = Field(default=0, ge=0)
    away_team_id: int = Field(default=0, ge=0)
    team_id: Optional[int] = Field(default=None, ge=0)
    opponent_team_id: Optional[int] = Field(default=None, ge=0)
    minutes_played: float = Field(..., ge=0.0, le=65.0)
    points: float = Field(default=0.0, ge=0.0, le=150.0)
    rebounds: float = Field(default=0.0, ge=0.0, le=60.0)
    assists: float = Field(default=0.0, ge=0.0, le=50.0)
    turnovers: float = Field(default=0.0, ge=0.0, le=30.0)
    steals: float = Field(default=0.0, ge=0.0, le=20.0)
    blocks: float = Field(default=0.0, ge=0.0, le=20.0)
    three_pointers_made: float = Field(default=0.0, ge=0.0, le=30.0)
    field_goals_attempted: float = Field(default=0.0, ge=0.0, le=80.0)
    free_throws_attempted: float = Field(default=0.0, ge=0.0, le=60.0)
    plus_minus: float = Field(default=0.0, ge=-100.0, le=100.0)
    usage_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    potential_assists: float = Field(default=0.0, ge=0.0, le=80.0)
    touches: float = Field(default=0.0, ge=0.0, le=400.0)
    passes_made: float = Field(default=0.0, ge=0.0, le=400.0)
    time_of_possession_sec: float = Field(default=0.0, ge=0.0, le=3600.0)
    avg_defender_distance_ft: float = Field(default=4.5, ge=0.0, le=50.0)
    shorthanded_points: float = Field(default=0.0, ge=0.0, le=150.0)


class PredictionItemRequest(StrictBaseModel):
    """Single prediction request via pre-computed features OR player_id (+ Redis/inline history)."""

    schema_version: str = Field(default=FEATURE_SCHEMA_VERSION)
    league: LeagueEnum = LeagueEnum.NBA
    stat: StatKeyEnum = StatKeyEnum.pts
    player_id: Optional[Union[str, int]] = None
    features: Optional[FeaturePayload] = None
    game_context: Optional[GameContextPayload] = None
    history_logs: Optional[List[GameLogEntry]] = None
    thresholds: List[float] = Field(default_factory=list, max_length=32)

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, v: str) -> str:
        if v != FEATURE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version '{v}'; expected '{FEATURE_SCHEMA_VERSION}'"
            )
        return v

    @field_validator("stat", mode="before")
    @classmethod
    def normalize_stat(cls, v: Any) -> Any:
        if isinstance(v, str):
            low = v.strip().lower()
            if low in STAT_COL_TO_TARGET:
                return STAT_COL_TO_TARGET[low]
            return low
        return v

    @field_validator("thresholds")
    @classmethod
    def validate_thresholds(cls, vals: List[float]) -> List[float]:
        cleaned = []
        for idx, val in enumerate(vals):
            fval = _assert_finite_float(val, f"thresholds[{idx}]")
            if fval < 0.0 or fval > 200.0:
                raise ValueError(f"thresholds[{idx}] must be in [0.0, 200.0]")
            cleaned.append(fval)
        return cleaned

    @model_validator(mode="after")
    def require_features_or_player(self) -> "PredictionItemRequest":
        if self.features is None and (self.player_id is None or str(self.player_id).strip() == ""):
            raise ValueError("Either 'features' or non-empty 'player_id' must be provided")
        return self


class BatchPredictRequest(StrictBaseModel):
    schema_version: str = Field(default=FEATURE_SCHEMA_VERSION)
    requests: List[PredictionItemRequest] = Field(..., min_length=1, max_length=512)

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, v: str) -> str:
        if v != FEATURE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version '{v}'; expected '{FEATURE_SCHEMA_VERSION}'"
            )
        return v


class ZINBParams(StrictBaseModel):
    mu: float = Field(..., ge=0.0)
    alpha_disp: float = Field(..., gt=0.0)
    zero_inflation_prob: float = Field(..., ge=0.0, le=1.0)
    n_nb: float = Field(..., gt=0.0)


class ThresholdSurvival(StrictBaseModel):
    threshold: float
    prob_over: float = Field(..., ge=0.0, le=1.0)
    prob_under: float = Field(..., ge=0.0, le=1.0)


class PredictionItemResponse(StrictBaseModel):
    prediction_id: Optional[str] = None
    player_id: str
    stat: StatKeyEnum
    league: LeagueEnum
    schema_version: str
    predicted_mean: float = Field(..., ge=0.0)
    residual: float
    xgb_residual: float
    lgb_residual: float
    ewma_baseline: float = Field(..., ge=0.0)
    zinb: ZINBParams
    survival_probabilities: List[ThresholdSurvival] = Field(default_factory=list)
    cold_start: bool = False
    fallback_active: bool = False


class BatchPredictResponse(StrictBaseModel):
    schema_version: str
    model_version: str
    predictions: List[PredictionItemResponse]
    latency_ms: float = Field(..., ge=0.0)
    fallback_active: bool = False


class AllocateCandidateOutcome(StrictBaseModel):
    leg_id: Optional[str] = None
    player_id: str = Field(..., min_length=1)
    team_id: Optional[int] = Field(default=None, ge=0)
    game_id: Optional[int] = Field(default=None, ge=0)
    stat: str = Field(default="points", min_length=1)
    side: SideEnum = SideEnum.over
    marginal_prob: float = Field(..., gt=0.0, lt=1.0)
    payout_odds: float = Field(default=1.9090909, gt=1.0, le=1000.0)
    expected_edge: Optional[float] = Field(default=None, ge=-1.0, le=10.0)
    line: Optional[float] = Field(default=None, ge=0.0, le=200.0)
    zinb_mu: Optional[float] = Field(default=None, ge=0.0, le=200.0)
    zinb_pi: Optional[float] = Field(default=None, ge=0.0, lt=1.0)
    zinb_n: Optional[float] = Field(default=None, gt=0.0, le=1e7)

    @field_validator("side", mode="before")
    @classmethod
    def normalize_side(cls, v: Any) -> Any:
        if isinstance(v, str):
            low = v.strip().lower()
            if low == "over":
                return "Over"
            if low == "under":
                return "Under"
        return v

    @field_validator("marginal_prob", "payout_odds", "expected_edge", "line", "zinb_mu", "zinb_pi", "zinb_n")
    @classmethod
    def check_finite(cls, v: Optional[float], info) -> Optional[float]:
        if v is None:
            return None
        return _assert_finite_float(v, info.field_name)


class AllocateRequest(StrictBaseModel):
    schema_version: str = Field(default=FEATURE_SCHEMA_VERSION)
    league: LeagueEnum = LeagueEnum.NBA
    candidates: List[AllocateCandidateOutcome] = Field(..., min_length=1, max_length=100)
    covariance_matrix: Optional[List[List[float]]] = None
    copula_type: CopulaTypeEnum = CopulaTypeEnum.auto
    shrinkage_delta: float = Field(default=0.20, ge=0.0, le=1.0)
    max_bankroll_fraction: float = Field(default=0.05, gt=0.0, le=1.0)
    kelly_fraction: float = Field(default=0.25, gt=0.0, le=1.0)
    bankroll_usd: float = Field(default=50000.0, gt=0.0, le=1e9)

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, v: str) -> str:
        if v != FEATURE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version '{v}'; expected '{FEATURE_SCHEMA_VERSION}'"
            )
        return v

    @field_validator("shrinkage_delta", "max_bankroll_fraction", "kelly_fraction", "bankroll_usd")
    @classmethod
    def check_finite_scalar(cls, v: float, info) -> float:
        return _assert_finite_float(v, info.field_name)

    @model_validator(mode="after")
    def validate_covariance_shape(self) -> "AllocateRequest":
        if self.covariance_matrix is not None:
            n = len(self.candidates)
            if len(self.covariance_matrix) != n:
                raise ValueError(
                    f"covariance_matrix row count ({len(self.covariance_matrix)}) must match candidates ({n})"
                )
            for i, row in enumerate(self.covariance_matrix):
                if len(row) != n:
                    raise ValueError(f"covariance_matrix row {i} length ({len(row)}) must be {n}")
                for j, val in enumerate(row):
                    _assert_finite_float(val, f"covariance_matrix[{i}][{j}]")
                if self.covariance_matrix[i][i] < 0.0:
                    raise ValueError(f"covariance_matrix diagonal [{i}][{i}] must be non-negative")
        return self


class JointPairPricing(StrictBaseModel):
    leg_i: int
    leg_j: int
    relationship: Literal["teammate", "opponent", "independent"]
    copula_used: str
    rho: float
    joint_prob: float = Field(..., ge=0.0, le=1.0)
    independent_prob: float = Field(..., ge=0.0, le=1.0)


class AllocateResponse(StrictBaseModel):
    schema_version: str
    league: LeagueEnum
    allocations: List[float]
    dollar_stakes: List[float]
    masked_stakes: List[int]
    expected_edges: List[float]
    joint_pairs: List[JointPairPricing] = Field(default_factory=list)
    condition_number_raw: float = Field(..., ge=0.0)
    condition_number_shrunk: float = Field(..., ge=0.0)
    residual_norm: float = Field(..., ge=0.0)
    solve_residual_norm: float = Field(..., ge=0.0)
    solver_backend: str
    is_rank_deficient: bool
    numerical_rank: int = Field(..., ge=0)
    latency_ms: float = Field(..., ge=0.0)


class HealthResponse(StrictBaseModel):
    status: Literal["healthy", "degraded"]
    model_version: str
    git_sha: str
    feature_schema_version: str
    loaded_targets: List[str]
    redis_connected: bool
    redis_status: str
    cpp_eigen_available: bool
    drift_status: str = "HEALTHY"
    fallback_active: bool = False
    registry_champion_id: Optional[str] = None


class DriftReportResponse(BaseModel):
    schema_version: str = FEATURE_SCHEMA_VERSION
    model_version: str
    git_sha: str
    timestamp: str
    window_days: int
    total_predictions: int
    settled_predictions: int
    action: str
    fallback_active: bool
    reasons: List[str] = Field(default_factory=list)
    calibration: Dict[str, Any] = Field(default_factory=dict)
    covariate_drift: Dict[str, Any] = Field(default_factory=dict)
    thresholds: Dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: str
    detail: Any
    schema_version: str = FEATURE_SCHEMA_VERSION

