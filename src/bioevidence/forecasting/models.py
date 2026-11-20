"""Forecasting models for BioEvidence."""

import json
import pickle
import structlog
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from bioevidence.config import get_config
from bioevidence.db.models import (
    Chunk,
    DocType,
    SplitType,
    Trial,
    TrialCondition,
    TrialIntervention,
    TrialOutcome,
)
from bioevidence.db.session import get_session

logger = structlog.get_logger(__name__)


@dataclass
class ForecastFeatures:
    """Feature configuration for forecasting."""
    # Trial features
    use_phase: bool = True
    use_enrollment: bool = True
    use_conditions: bool = True
    use_interventions: bool = True
    use_sponsor: bool = True
    use_start_year: bool = True
    use_min_max_age: bool = True
    use_sex: bool = True

    # Target constraints
    min_duration_months: int = 1
    max_duration_months: int = 180


class TrialFeatureExtractor:
    """Extract features from trials for forecasting."""

    def __init__(self, config: ForecastFeatures | None = None):
        self.config = config or ForecastFeatures()
        self._label_encoders = {}
        self._scaler = StandardScaler()
        self._fitted = False

    def extract_features(self, trials: list[Trial]) -> pd.DataFrame:
        """Extract features from trial objects."""
        rows = []

        for trial in trials:
            row = {"nct_id": trial.nct_id}

            # Basic features
            if self.config.use_phase:
                row["phase"] = self._encode_phase(trial.phase)

            if self.config.use_enrollment:
                row["enrollment"] = trial.enrollment or 0

            if self.config.use_start_year:
                row["start_year"] = trial.start_date.year if trial.start_date else 0

            if self.config.use_sponsor:
                row["sponsor_class"] = trial.sponsor_class or "unknown"

            if self.config.use_min_max_age:
                row["min_age"] = trial.min_age_years or 0
                row["max_age"] = trial.max_age_years or 0

            if self.config.use_sex:
                row["sex"] = self._encode_sex(trial.sex)

            # Condition features (load from DB)
            # Intervention features (load from DB)
            # These would be loaded in a separate query for efficiency

            rows.append(row)

        df = pd.DataFrame(rows)
        return df

    def extract_features_with_relations(self, trial_ids: list[str]) -> pd.DataFrame:
        """Extract features including conditions and interventions."""
        with get_session() as session:
            trials = session.query(Trial).filter(Trial.nct_id.in_(trial_ids)).all()

            # Get conditions
            conditions = session.query(TrialCondition).filter(
                TrialCondition.nct_id.in_(trial_ids)
            ).all()
            cond_map = {}
            for c in conditions:
                cond_map.setdefault(c.nct_id, []).append(c.condition_mesh or c.condition)

            # Get interventions
            interventions = session.query(TrialIntervention).filter(
                TrialIntervention.nct_id.in_(trial_ids)
            ).all()
            int_map = {}
            for i in interventions:
                int_map.setdefault(i.nct_id, []).append(i.name)

            rows = []
            for trial in trials:
                row = {"nct_id": trial.nct_id}

                if self.config.use_phase:
                    row["phase"] = self._encode_phase(trial.phase)

                if self.config.use_enrollment:
                    row["enrollment"] = trial.enrollment or 0

                if self.config.use_start_year:
                    row["start_year"] = trial.start_date.year if trial.start_date else 0

                if self.config.use_sponsor:
                    row["sponsor_class"] = trial.sponsor_class or "unknown"

                if self.config.use_min_max_age:
                    row["min_age"] = trial.min_age_years or 0
                    row["max_age"] = trial.max_age_years or 0

                if self.config.use_sex:
                    row["sex"] = self._encode_sex(trial.sex)

                # Conditions
                if self.config.use_conditions:
                    conds = cond_map.get(trial.nct_id, [])
                    row["num_conditions"] = len(conds)
                    row["has_cancer"] = int(any("neoplasm" in c.lower() for c in conds))
                    row["has_inflammatory"] = int(any(k in " ".join(conds).lower() for k in ["arthritis", "psoriasis", "crohn", "colitis"]))
                    row["has_cardiovascular"] = int(any(k in " ".join(conds).lower() for k in ["heart", "cardio", "hypertension"]))

                # Interventions
                if self.config.use_interventions:
                    ints = int_map.get(trial.nct_id, [])
                    row["num_interventions"] = len(ints)
                    row["has_mab"] = int(any("mab" in i.lower() or "monoclonal" in i.lower() for i in ints))
                    row["has_small_molecule"] = int(any(k in " ".join(ints).lower() for k in ["inhibitor", "tablet", "oral", "capsule"]))

                rows.append(row)

            df = pd.DataFrame(rows)
            return df

    def _encode_phase(self, phase: str | None) -> int:
        """Encode phase as ordinal."""
        if not phase:
            return 0
        phase_lower = phase.lower()
        if "phase 1" in phase_lower and "phase 2" not in phase_lower:
            return 1
        elif "phase 2" in phase_lower and "phase 3" not in phase_lower:
            return 2
        elif "phase 3" in phase_lower and "phase 4" not in phase_lower:
            return 3
        elif "phase 4" in phase_lower:
            return 4
        return 0

    def _encode_sex(self, sex: str | None) -> int:
        """Encode sex."""
        if not sex:
            return 0
        sex_lower = sex.lower()
        if "female" in sex_lower and "male" not in sex_lower:
            return 1
        elif "male" in sex_lower and "female" not in sex_lower:
            return 2
        return 3  # both or unknown


def prepare_duration_targets(trial_ids: list[str]) -> dict[str, float]:
    """Prepare duration targets (months from start to completion)."""
    with get_session() as session:
        trials = session.query(Trial).filter(Trial.nct_id.in_(trial_ids)).all()

        targets = {}
        for trial in trials:
            if trial.start_date and trial.completion_date:
                delta = trial.completion_date - trial.start_date
                months = delta.days / 30.44
                targets[trial.nct_id] = months
            else:
                targets[trial.nct_id] = np.nan

        return targets


def prepare_success_targets(trial_ids: list[str]) -> dict[str, int]:
    """Prepare success targets (1 if completed with results, 0 otherwise)."""
    with get_session() as session:
        trials = session.query(Trial).filter(Trial.nct_id.in_(trial_ids)).all()

        targets = {}
        for trial in trials:
            # Success = Completed with results posted
            if trial.overall_status == "Completed":
                targets[trial.nct_id] = 1
            elif trial.overall_status in ["Terminated", "Withdrawn", "Suspended"]:
                targets[trial.nct_id] = 0
            else:
                targets[trial.nct_id] = np.nan  # Unknown/ongoing

        return targets


class DurationModel:
    """LightGBM model for trial duration prediction."""

    def __init__(self, params: dict | None = None):
        self.params = params or {
            "objective": "regression",
            "metric": "mae",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "seed": 42,
        }
        self.model = None
        self.feature_names = None

    def train(self, X: pd.DataFrame, y: pd.Series, val_X: pd.DataFrame | None = None, val_y: pd.Series | None = None):
        """Train the model."""
        self.feature_names = X.columns.tolist()

        train_data = lgb.Dataset(X, label=y)
        valid_data = lgb.Dataset(val_X, label=val_y) if val_X is not None else None

        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=[valid_data] if valid_data else None,
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)] if valid_data else [lgb.log_evaluation(100)],
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict durations."""
        if self.model is None:
            raise ValueError("Model not trained")
        return self.model.predict(X[self.feature_names])

    def save(self, path: Path):
        """Save model."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))

    @classmethod
    def load(cls, path: Path) -> "DurationModel":
        """Load model."""
        model = cls()
        model.model = lgb.Booster(model_file=str(path))
        model.feature_names = model.model.feature_name()
        return model


class SuccessModel:
    """XGBoost model for trial success probability."""

    def __init__(self, params: dict | None = None):
        self.params = params or {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": 42,
            "verbosity": 0,
        }
        self.model = None
        self.feature_names = None

    def train(self, X: pd.DataFrame, y: pd.Series, val_X: pd.DataFrame | None = None, val_y: pd.Series | None = None):
        """Train the model."""
        self.feature_names = X.columns.tolist()

        dtrain = xgb.DMatrix(X, label=y)
        dval = xgb.DMatrix(val_X, label=val_y) if val_X is not None else None

        evals = [(dval, "val")] if dval else None

        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=500,
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=100,
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict success probabilities."""
        if self.model is None:
            raise ValueError("Model not trained")
        dtest = xgb.DMatrix(X[self.feature_names])
        return self.model.predict(dtest)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict binary success."""
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)

    def save(self, path: Path):
        """Save model."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))

    @classmethod
    def load(cls, path: Path) -> "SuccessModel":
        """Load model."""
        model = cls()
        model.model = xgb.Booster()
        model.model.load_model(str(path))
        model.feature_names = model.model.feature_names
        return model


def train_forecasting_models(
    test_start_year: int = 2020,
    min_duration_months: int = 1,
    max_duration_months: int = 180,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Train both duration and success forecasting models."""
    config = get_config()
    output_dir = output_dir or Path(config.llm.cache_dir).parent.parent / "models" / "forecasting"
    output_dir.mkdir(parents=True, exist_ok=True)

    with get_session() as session:
        # Get all trial IDs from TRAIN and DEV splits (exclude TEST)
        from bioevidence.db.models import Chunk

        trial_ids = session.query(Chunk.doc_id).filter(
            Chunk.doc_type == DocType.TRIAL,
            Chunk.split.in_([SplitType.TRAIN, SplitType.DEV]),
        ).distinct().all()
        trial_ids = [t[0] for t in trial_ids]

        logger.info("forecasting_trial_count", count=len(trial_ids))

    # Extract features
    extractor = TrialFeatureExtractor()
    features_df = extractor.extract_features_with_relations(trial_ids)

    # Prepare targets
    duration_targets = prepare_duration_targets(trial_ids)
    success_targets = prepare_success_targets(trial_ids)

    # Merge features with targets
    features_df["duration"] = features_df["nct_id"].map(duration_targets)
    features_df["success"] = features_df["nct_id"].map(success_targets)

    # Filter by test_start_year for temporal split
    # (In production, use actual temporal split by start_date)
    train_df = features_df.dropna(subset=["duration"]).copy()
    train_success_df = features_df.dropna(subset=["success"]).copy()

    logger.info("training_data", duration=len(train_df), success=len(train_success_df))

    # Train duration model
    logger.info("training_duration_model")
    duration_features = train_df.drop(columns=["nct_id", "duration", "success"], errors="ignore")
    duration_target = train_df["duration"]

    X_train, X_val, y_train, y_val = train_test_split(
        duration_features, duration_target, test_size=0.2, random_state=42
    )

    duration_model = DurationModel()
    duration_model.train(X_train, y_train, X_val, y_val)

    # Evaluate duration
    val_preds = duration_model.predict(X_val)
    duration_mae = mean_absolute_error(y_val, val_preds)
    duration_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    logger.info("duration_metrics", mae=duration_mae, rmse=duration_rmse)

    # Train success model
    logger.info("training_success_model")
    success_features = train_success_df.drop(columns=["nct_id", "duration", "success"], errors="ignore")
    success_target = train_success_df["success"]

    X_train_s, X_val_s, y_train_s, y_val_s = train_test_split(
        success_features, success_target, test_size=0.2, random_state=42, stratify=success_target
    )

    success_model = SuccessModel()
    success_model.train(X_train_s, y_train_s, X_val_s, y_val_s)

    # Evaluate success
    val_probas = success_model.predict_proba(X_val_s)
    val_preds = success_model.predict(X_val_s)
    success_auc = roc_auc_score(y_val_s, val_probas)
    success_acc = accuracy_score(y_val_s, val_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_val_s, val_preds, average="binary")
    logger.info("success_metrics", auc=success_auc, accuracy=success_acc, f1=f1)

    # Save models
    duration_model.save(output_dir / "duration_model.txt")
    success_model.save(output_dir / "success_model.json")

    # Save feature names
    (output_dir / "feature_names.json").write_text(json.dumps({
        "duration": duration_model.feature_names,
        "success": success_model.feature_names,
    }))

    results = {
        "duration_mae": duration_mae,
        "duration_rmse": duration_rmse,
        "success_auc": success_auc,
        "success_accuracy": success_acc,
        "success_f1": f1,
        "n_train_duration": len(X_train),
        "n_train_success": len(X_train_s),
    }

    logger.info("forecasting_training_complete", **results)
    return results