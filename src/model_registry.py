"""Lightweight JSON-backed Champion/Challenger model registry with atomic promotion and rollback."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.feature_contract import FEATURE_SCHEMA_VERSION, TARGET_METRICS

DEFAULT_REGISTRY_DIR = os.getenv("MODEL_REGISTRY_DIR", "models/registry")


class PromotionRejectedError(ValueError):
    """Raised when a challenger fails schema, integrity, or metric promotion gates."""


class RegistryError(RuntimeError):
    """Raised on invalid registry operations such as rollback without a prior champion."""


def compute_sha256(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_git_short_sha() -> str:
    env_sha = os.getenv("GIT_SHA")
    if env_sha:
        return env_sha.strip()[:12]
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return out.decode("utf-8").strip() or "unknown"
    except Exception:
        return "unknown"


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


class ModelRegistry:
    """Filesystem Champion/Challenger registry stored under ``models/registry/``."""

    def __init__(self, registry_dir: str = DEFAULT_REGISTRY_DIR):
        self.registry_dir = registry_dir
        self.releases_dir = os.path.join(self.registry_dir, "releases")
        self.registry_path = os.path.join(self.registry_dir, "registry.json")
        self.champion_path = os.path.join(self.registry_dir, "champion.json")
        os.makedirs(self.releases_dir, exist_ok=True)
        if not os.path.exists(self.registry_path):
            self._save_index(
                {
                    "schema_version": 1,
                    "active_champion_id": None,
                    "champion_history": [],
                    "versions": {},
                    "events": [],
                }
            )

    def _load_index(self) -> Dict[str, Any]:
        data = _read_json(self.registry_path)
        if not isinstance(data, dict):
            return {
                "schema_version": 1,
                "active_champion_id": None,
                "champion_history": [],
                "versions": {},
                "events": [],
            }
        data.setdefault("champion_history", [])
        data.setdefault("versions", {})
        data.setdefault("events", [])
        return data

    def _save_index(self, data: Dict[str, Any]) -> None:
        _atomic_write_json(self.registry_path, data)

    def get_champion(self) -> Optional[Dict[str, Any]]:
        return _read_json(self.champion_path)

    def get_version(self, version_id: str) -> Optional[Dict[str, Any]]:
        manifest = _read_json(os.path.join(self.releases_dir, f"{version_id}.json"))
        if manifest is not None:
            return manifest
        index = self._load_index()
        return index.get("versions", {}).get(version_id)

    def list_versions(self) -> List[Dict[str, Any]]:
        index = self._load_index()
        return list(index.get("versions", {}).values())

    @staticmethod
    def scan_model_artifacts(model_dir: str) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """Collects SHA-256 checksums and metadata across trained ensemble artifacts."""
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        checksums: Dict[str, str] = {}
        meta_summary: Dict[str, Any] = {
            "feature_schema_version": None,
            "train_seasons": None,
            "training_samples": None,
            "targets": [],
        }

        for target_key in TARGET_METRICS:
            found_for_target = False
            for fname in (
                f"global_x{target_key}_xgb.json",
                f"global_x{target_key}_lgb.txt",
                f"global_x{target_key}_meta.json",
                f"global_x{target_key}.json",
            ):
                fpath = os.path.join(model_dir, fname)
                if os.path.isfile(fpath):
                    checksums[fname] = compute_sha256(fpath)
                    found_for_target = True
                    if fname.endswith("_meta.json"):
                        try:
                            with open(fpath, "r") as f:
                                meta = json.load(f)
                            if meta_summary["feature_schema_version"] is None:
                                meta_summary["feature_schema_version"] = meta.get(
                                    "feature_schema_version"
                                )
                            if meta_summary["train_seasons"] is None:
                                meta_summary["train_seasons"] = meta.get("train_seasons")
                            if meta_summary["training_samples"] is None:
                                meta_summary["training_samples"] = meta.get("training_samples")
                        except Exception:
                            pass
            if found_for_target:
                meta_summary["targets"].append(target_key)

        if not checksums:
            raise FileNotFoundError(f"No model artifacts discovered under: {model_dir}")

        return checksums, meta_summary

    def generate_version_id(
        self,
        git_sha: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> str:
        dt = timestamp or datetime.now(timezone.utc)
        short_sha = (git_sha or resolve_git_short_sha())[:12]
        base = f"v{dt.strftime('%Y.%m.%d')}-{short_sha}"
        index = self._load_index()
        versions = index.get("versions", {})
        if base not in versions:
            return base
        suffix = 2
        while f"{base}-{suffix}" in versions:
            suffix += 1
        return f"{base}-{suffix}"

    def verify_checksums(self, manifest_or_version: Any) -> Tuple[bool, List[str]]:
        if isinstance(manifest_or_version, str):
            manifest = self.get_version(manifest_or_version)
            if manifest is None:
                return False, [f"Unknown version_id: {manifest_or_version}"]
        else:
            manifest = manifest_or_version

        model_dir = manifest.get("model_dir", "")
        checksums = manifest.get("artifact_checksums", {})
        if not checksums:
            return False, ["No artifact_checksums recorded in manifest"]

        mismatches: List[str] = []
        for rel_file, expected_sha in checksums.items():
            full_path = os.path.join(model_dir, rel_file)
            if not os.path.isfile(full_path):
                mismatches.append(f"Missing artifact: {full_path}")
                continue
            actual_sha = compute_sha256(full_path)
            if actual_sha != expected_sha:
                mismatches.append(
                    f"Checksum mismatch for {rel_file}: expected {expected_sha[:12]}, got {actual_sha[:12]}"
                )
        return len(mismatches) == 0, mismatches

    def register_version(
        self,
        *,
        model_dir: str,
        metrics: Dict[str, Any],
        git_sha: Optional[str] = None,
        version_id: Optional[str] = None,
        feature_schema_version: Optional[str] = None,
        train_seasons: Optional[List[str]] = None,
        training_samples: Optional[int] = None,
        promote_if_empty: bool = False,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        checksums, meta_summary = self.scan_model_artifacts(model_dir)
        resolved_sha = git_sha or resolve_git_short_sha()
        resolved_version = version_id or self.generate_version_id(git_sha=resolved_sha)

        schema_ver = (
            feature_schema_version
            or meta_summary.get("feature_schema_version")
            or FEATURE_SCHEMA_VERSION
        )
        seasons = (
            list(train_seasons)
            if train_seasons is not None
            else (meta_summary.get("train_seasons") or [])
        )
        samples = (
            int(training_samples)
            if training_samples is not None
            else int(meta_summary.get("training_samples") or 0)
        )

        now_iso = datetime.now(timezone.utc).isoformat()
        index = self._load_index()
        make_champion = promote_if_empty and index.get("active_champion_id") is None

        normalized_metrics = {
            "overall_mae": float(metrics["overall_mae"]),
            "overall_crps": float(metrics["overall_crps"]),
            "ewma_mae": float(metrics["ewma_mae"]),
            "brier_score": (
                float(metrics["brier_score"])
                if metrics.get("brier_score") is not None
                else None
            ),
            "by_stat": metrics.get("by_stat", {}),
        }

        manifest: Dict[str, Any] = {
            "version_id": resolved_version,
            "git_sha": resolved_sha,
            "registered_at": now_iso,
            "promoted_at": now_iso if make_champion else None,
            "status": "champion" if make_champion else "challenger",
            "model_dir": model_dir,
            "feature_schema_version": schema_ver,
            "train_seasons": seasons,
            "training_samples": samples,
            "targets": meta_summary.get("targets", []),
            "artifact_checksums": checksums,
            "metrics": normalized_metrics,
            "notes": notes,
        }

        release_path = os.path.join(self.releases_dir, f"{resolved_version}.json")
        _atomic_write_json(release_path, manifest)

        index["versions"][resolved_version] = manifest
        index["events"].append(
            {
                "event": "register_and_promote" if make_champion else "register",
                "version_id": resolved_version,
                "timestamp": now_iso,
            }
        )
        if make_champion:
            index["active_champion_id"] = resolved_version
            _atomic_write_json(self.champion_path, manifest)

        self._save_index(index)
        return manifest

    def promote_challenger(
        self,
        version_id: str,
        *,
        max_mae_tolerance: float = 0.0,
        min_mae_improvement: float = 0.0,
        verify_files: bool = True,
        require_beat_ewma: bool = True,
        require_beat_champion: bool = True,
    ) -> Dict[str, Any]:
        """Validates promotion gates and atomically promotes challenger to champion."""
        index = self._load_index()
        challenger = self.get_version(version_id)
        if challenger is None:
            raise PromotionRejectedError(f"Unknown challenger version: {version_id}")

        # Gate 1: Schema compatibility
        if challenger.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
            raise PromotionRejectedError(
                f"Schema mismatch for {version_id}: {challenger.get('feature_schema_version')} != {FEATURE_SCHEMA_VERSION}"
            )

        # Gate 2: Artifact checksum integrity
        if verify_files:
            ok, mismatches = self.verify_checksums(challenger)
            if not ok:
                raise PromotionRejectedError(
                    f"Artifact integrity check failed for {version_id}: {'; '.join(mismatches)}"
                )

        c_metrics = challenger.get("metrics", {})
        c_mae = float(c_metrics["overall_mae"])
        c_crps = float(c_metrics["overall_crps"])
        c_ewma = float(c_metrics["ewma_mae"])
        c_brier = c_metrics.get("brier_score")

        # Gate 3: Must beat EWMA baseline
        if require_beat_ewma and not (c_mae < c_ewma):
            raise PromotionRejectedError(
                f"Challenger {version_id} failed EWMA baseline gate: overall_mae={c_mae:.4f} >= ewma_mae={c_ewma:.4f}"
            )

        # Gate 4: Compare against incumbent Champion
        current_id = index.get("active_champion_id")
        current_champ = self.get_champion() if current_id else None
        if current_champ and require_beat_champion:
            if current_champ.get("version_id") == version_id:
                return current_champ

            p_metrics = current_champ.get("metrics", {})
            p_mae = float(p_metrics["overall_mae"])
            p_crps = float(p_metrics["overall_crps"])
            p_brier = p_metrics.get("brier_score")

            if c_mae > (p_mae - min_mae_improvement + max_mae_tolerance):
                raise PromotionRejectedError(
                    f"Challenger {version_id} overall_mae={c_mae:.4f} did not beat champion {current_id} ({p_mae:.4f})"
                )
            if c_crps > (p_crps + max_mae_tolerance):
                raise PromotionRejectedError(
                    f"Challenger {version_id} overall_crps={c_crps:.4f} regressed vs champion {current_id} ({p_crps:.4f})"
                )
            if c_brier is not None and p_brier is not None:
                if float(c_brier) > float(p_brier) + max_mae_tolerance:
                    raise PromotionRejectedError(
                        f"Challenger {version_id} brier_score={float(c_brier):.4f} regressed vs champion {current_id} ({float(p_brier):.4f})"
                    )

        now_iso = datetime.now(timezone.utc).isoformat()
        if current_champ and current_id and current_id != version_id:
            index["champion_history"].append(current_id)
            prev_manifest = self.get_version(current_id) or current_champ
            prev_manifest["status"] = "archived"
            _atomic_write_json(os.path.join(self.releases_dir, f"{current_id}.json"), prev_manifest)
            index["versions"][current_id] = prev_manifest

        challenger["status"] = "champion"
        challenger["promoted_at"] = now_iso
        _atomic_write_json(os.path.join(self.releases_dir, f"{version_id}.json"), challenger)
        _atomic_write_json(self.champion_path, challenger)

        index["active_champion_id"] = version_id
        index["versions"][version_id] = challenger
        index["events"].append(
            {
                "event": "promote",
                "version_id": version_id,
                "previous_champion_id": current_id,
                "timestamp": now_iso,
            }
        )
        self._save_index(index)
        return challenger

    def rollback_to_previous(self, reason: str = "manual_rollback") -> Dict[str, Any]:
        """Atomically restores the prior champion manifest."""
        index = self._load_index()
        history = index.get("champion_history", [])
        if not history:
            raise RegistryError("Cannot rollback: no previous champion exists in history.")

        prev_id = history.pop()
        prev_manifest = self.get_version(prev_id)
        if prev_manifest is None:
            raise RegistryError(f"Previous champion manifest missing for version_id={prev_id}")

        current_id = index.get("active_champion_id")
        now_iso = datetime.now(timezone.utc).isoformat()

        if current_id and current_id in index.get("versions", {}):
            curr_manifest = self.get_version(current_id) or index["versions"][current_id]
            curr_manifest["status"] = "rolled_back"
            curr_manifest["rolled_back_at"] = now_iso
            curr_manifest["rollback_reason"] = reason
            _atomic_write_json(
                os.path.join(self.releases_dir, f"{current_id}.json"), curr_manifest
            )
            index["versions"][current_id] = curr_manifest

        prev_manifest["status"] = "champion"
        prev_manifest["restored_at"] = now_iso
        _atomic_write_json(os.path.join(self.releases_dir, f"{prev_id}.json"), prev_manifest)
        _atomic_write_json(self.champion_path, prev_manifest)

        index["active_champion_id"] = prev_id
        index["versions"][prev_id] = prev_manifest
        index["events"].append(
            {
                "event": "rollback",
                "restored_version_id": prev_id,
                "demoted_version_id": current_id,
                "reason": reason,
                "timestamp": now_iso,
            }
        )
        self._save_index(index)
        return prev_manifest
