from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import skfuzzy as fuzzy
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

from src.config import FCM_FUZZINESS, OPTIONAL_SUBJECTS


MANFIS_FEATURE_COLUMNS = [
    "natural_score",
    "social_score",
    "english_score",
    *[f"{subject}_avg" for subject in OPTIONAL_SUBJECTS],
    *[f"{subject}_trend" for subject in OPTIONAL_SUBJECTS],
]
PSEUDO_LABEL_COLUMNS = [
    "student_id",
    "top1_subject",
    "top1_score",
    "top2_subject",
    "top2_score",
]


def build_manfis_input(
    subject_averages: dict[str, float],
    period_scores: dict[str, float],
) -> pd.DataFrame:
    optional_subjects = [
        "physics", "chemistry", "biology", "informatics",
        "history", "geography", "english",
    ]
    features = {
        f"{subject}_avg": float(subject_averages.get(subject, 0.0))
        for subject in optional_subjects
    }
    for subject in optional_subjects:
        previous_scores = [
            float(period_scores[f"{subject}_{period}"])
            for period in ("10", "11")
            if float(period_scores[f"{subject}_{period}"]) > 0
        ]
        previous_average = float(np.mean(previous_scores)) if previous_scores else np.nan
        current_score = float(period_scores[f"{subject}_12_hk1"])
        features[f"{subject}_trend"] = (
            current_score - previous_average
            if np.isfinite(previous_average)
            else 0.0
        )

    features["natural_score"] = float(np.mean([
        float(subject_averages.get(subject, 0.0))
        for subject in ["math", "physics", "chemistry", "biology", "informatics"]
    ]))
    features["social_score"] = float(np.mean([
        float(subject_averages.get(subject, 0.0))
        for subject in ["literature", "history", "geography"]
    ]))
    features["english_score"] = float(subject_averages.get("english", 0.0))
    return pd.DataFrame([features], columns=MANFIS_FEATURE_COLUMNS)


def _membership_from_centers(
    samples: np.ndarray,
    centers: np.ndarray,
    fuzziness: float,
) -> np.ndarray:
    distances = np.linalg.norm(samples[:, None, :] - centers[None, :, :], axis=2)
    memberships = np.zeros_like(distances)
    exact_matches = distances <= 1e-12

    for row_index, row in enumerate(distances):
        matches = np.flatnonzero(exact_matches[row_index])
        if len(matches):
            memberships[row_index, matches] = 1.0 / len(matches)
            continue
        ratios = (row[:, None] / row[None, :]) ** (2.0 / (fuzziness - 1.0))
        memberships[row_index] = 1.0 / ratios.sum(axis=1)

    return memberships


@dataclass
class FCMSugenoMANFIS:
    class_names: list[str]
    n_rules: int = 3
    fuzziness: float = FCM_FUZZINESS
    ridge: float = 1e-3
    seed: int = 42

    def fit(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame,
    ) -> "FCMSugenoMANFIS":
        missing_features = set(MANFIS_FEATURE_COLUMNS) - set(features.columns)
        missing_labels = set(PSEUDO_LABEL_COLUMNS) - set(labels.columns)
        if missing_features:
            raise ValueError(f"Missing MANFIS features: {sorted(missing_features)}")
        if missing_labels:
            raise ValueError(f"Missing pseudo-label columns: {sorted(missing_labels)}")
        if self.fuzziness <= 1.0:
            raise ValueError("FCM fuzziness must be greater than 1.")
        if self.n_rules < 2 or self.n_rules >= len(features):
            raise ValueError("n_rules must be at least 2 and smaller than the sample count.")

        merged = features[["student_id", *MANFIS_FEATURE_COLUMNS]].merge(
            labels[PSEUDO_LABEL_COLUMNS],
            on="student_id",
            how="inner",
            validate="one_to_one",
        )
        if len(merged) != len(features) or merged.empty:
            raise ValueError("Features and pseudo-labels must have matching unique student IDs.")
        if not merged["top1_subject"].isin(self.class_names).all():
            raise ValueError("top1_subject contains an unknown optional subject.")
        if not merged["top2_subject"].isin(self.class_names).all():
            raise ValueError("top2_subject contains an unknown optional subject.")
        if merged["top1_subject"].eq(merged["top2_subject"]).any():
            raise ValueError("Top 1 and Top 2 pseudo-labels must be different.")

        raw_features = merged[MANFIS_FEATURE_COLUMNS].to_numpy(dtype=float)
        if not np.isfinite(raw_features).all():
            raise ValueError("MANFIS features must contain only finite numbers.")

        self.feature_mean_ = raw_features.mean(axis=0)
        self.feature_scale_ = raw_features.std(axis=0)
        self.feature_scale_[self.feature_scale_ <= 1e-12] = 1.0
        scaled_features = (raw_features - self.feature_mean_) / self.feature_scale_

        centers, memberships, _, _, _, _, _ = fuzzy.cluster.cmeans(
            data=scaled_features.T,
            c=self.n_rules,
            m=self.fuzziness,
            error=0.005,
            maxiter=500,
            seed=self.seed,
        )
        self.centers_ = centers
        self.training_memberships_ = memberships.T

        class_indexes = {name: index for index, name in enumerate(self.class_names)}
        top1_targets = np.eye(len(self.class_names))[
            merged["top1_subject"].map(class_indexes).to_numpy(dtype=int)
        ]
        top2_targets = np.eye(len(self.class_names))[
            merged["top2_subject"].map(class_indexes).to_numpy(dtype=int)
        ]
        self.top1_consequents_ = self._fit_consequents(
            scaled_features, self.training_memberships_, top1_targets
        )
        self.top2_consequents_ = self._fit_consequents(
            scaled_features, self.training_memberships_, top2_targets
        )
        self.top1_score_consequents_ = self._fit_consequents(
            scaled_features,
            self.training_memberships_,
            merged[["top1_score"]].to_numpy(dtype=float),
        )
        self.top2_score_consequents_ = self._fit_consequents(
            scaled_features,
            self.training_memberships_,
            merged[["top2_score"]].to_numpy(dtype=float),
        )
        return self

    def _fit_consequents(
        self,
        features: np.ndarray,
        memberships: np.ndarray,
        targets: np.ndarray,
    ) -> np.ndarray:
        design = np.column_stack([np.ones(len(features)), features])
        regularizer = np.eye(design.shape[1]) * self.ridge
        regularizer[0, 0] = 0.0
        consequents = []

        for rule_index in range(self.n_rules):
            weights = memberships[:, rule_index] ** self.fuzziness
            weighted_design = design * weights[:, None]
            normal_matrix = design.T @ weighted_design + regularizer
            right_hand_side = design.T @ (weights[:, None] * targets)
            consequents.append(np.linalg.solve(normal_matrix, right_hand_side))

        return np.stack(consequents)

    def _predict_outputs(self, features: pd.DataFrame, consequents: np.ndarray) -> np.ndarray:
        missing_features = set(MANFIS_FEATURE_COLUMNS) - set(features.columns)
        if missing_features:
            raise ValueError(f"Missing MANFIS features: {sorted(missing_features)}")

        raw_features = features[MANFIS_FEATURE_COLUMNS].to_numpy(dtype=float)
        if not np.isfinite(raw_features).all():
            raise ValueError("MANFIS features must contain only finite numbers.")
        scaled_features = (raw_features - self.feature_mean_) / self.feature_scale_
        memberships = _membership_from_centers(
            scaled_features, self.centers_, self.fuzziness
        )
        design = np.column_stack([np.ones(len(scaled_features)), scaled_features])
        local_outputs = np.einsum("nf,kfo->nko", design, consequents)
        return np.einsum("nk,nko->no", memberships, local_outputs)

    def predict_proba(self, features: pd.DataFrame, target: str = "top1") -> np.ndarray:
        if target == "top1":
            consequents = self.top1_consequents_
        elif target == "top2":
            consequents = self.top2_consequents_
        else:
            raise ValueError("target must be 'top1' or 'top2'.")

        logits = self._predict_outputs(features, consequents)
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        top1_probabilities = self.predict_proba(features, "top1")
        top2_probabilities = self.predict_proba(features, "top2")
        top1_indexes = top1_probabilities.argmax(axis=1)
        top2_probabilities[
            np.arange(len(top1_indexes)), top1_indexes
        ] = -np.inf
        top2_indexes = top2_probabilities.argmax(axis=1)
        top1_scores = self._predict_outputs(features, self.top1_score_consequents_).ravel()
        top2_scores = self._predict_outputs(features, self.top2_score_consequents_).ravel()
        top1_scores = np.clip(top1_scores, 0.0, 1.0)
        top2_scores = np.minimum(np.clip(top2_scores, 0.0, 1.0), top1_scores)

        predictions = pd.DataFrame({
            "top1_subject": [self.class_names[index] for index in top1_indexes],
            "top1_score": top1_scores,
            "top2_subject": [self.class_names[index] for index in top2_indexes],
            "top2_score": top2_scores,
        }, index=features.index)
        if "student_id" in features.columns:
            predictions.insert(0, "student_id", features["student_id"].astype(str).values)
        if "student_name" in features.columns:
            predictions.insert(1, "student_name", features["student_name"].values)
        if "class" in features.columns:
            insert_at = 2 if "student_name" in features.columns else 1
            predictions.insert(insert_at, "class", features["class"].values)
        return predictions


def split_pseudo_labels(
    labels: pd.DataFrame,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_indexes: list[int] = []
    validation_indexes: list[int] = []
    test_indexes: list[int] = []

    for _, group in labels.groupby("top1_subject", sort=True):
        indexes = group.index.to_numpy().copy()
        rng.shuffle(indexes)
        count = len(indexes)
        if count == 1:
            train_count, validation_count = 1, 0
        elif count == 2:
            train_count, validation_count = 1, 0
        else:
            train_count = max(1, int(count * 0.5))
            validation_count = max(1, int(count * 0.2))
            if train_count + validation_count >= count:
                validation_count = count - train_count - 1

        train_indexes.extend(indexes[:train_count].tolist())
        validation_indexes.extend(
            indexes[train_count:train_count + validation_count].tolist()
        )
        test_indexes.extend(indexes[train_count + validation_count:].tolist())

    return (
        np.asarray(sorted(train_indexes), dtype=int),
        np.asarray(sorted(validation_indexes), dtype=int),
        np.asarray(sorted(test_indexes), dtype=int),
    )


def _classification_metrics(
    truth: pd.Series,
    predictions: pd.Series,
) -> tuple[float, float]:
    return (
        float(accuracy_score(truth, predictions)),
        float(f1_score(truth, predictions, average="macro", zero_division=0)),
    )


def _selection_loss(
    truth: pd.DataFrame,
    predictions: pd.DataFrame,
) -> float:
    top1_accuracy, _ = _classification_metrics(
        truth["top1_subject"], predictions["top1_subject"]
    )
    top2_accuracy, _ = _classification_metrics(
        truth["top2_subject"], predictions["top2_subject"]
    )
    top1_mae = mean_absolute_error(truth["top1_score"], predictions["top1_score"])
    top2_mae = mean_absolute_error(truth["top2_score"], predictions["top2_score"])
    return float(
        0.35 * (1.0 - top1_accuracy)
        + 0.35 * (1.0 - top2_accuracy)
        + 0.15 * top1_mae
        + 0.15 * top2_mae
    )


def train_and_evaluate_manfis(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    seed: int = 42,
) -> tuple[FCMSugenoMANFIS, dict[str, float | int | str]]:
    labels = labels.reset_index(drop=True)
    feature_rows = features.reset_index(drop=True)
    if len(labels) != len(feature_rows):
        raise ValueError("Features and labels must have the same number of rows.")
    labels = feature_rows[["student_id"]].merge(
        labels[PSEUDO_LABEL_COLUMNS],
        on="student_id",
        how="inner",
        validate="one_to_one",
    )
    if len(labels) != len(feature_rows):
        raise ValueError("Features and pseudo-labels must have matching student IDs.")

    train_indexes, validation_indexes, test_indexes = split_pseudo_labels(labels, seed)
    class_names = list(OPTIONAL_SUBJECTS)
    candidate_rules = [rule_count for rule_count in (2, 3, 4) if rule_count < len(train_indexes)]
    if not candidate_rules:
        raise ValueError("Not enough training rows to initialize FCM rules.")

    validation_scores = []
    for rule_count in candidate_rules:
        candidate = FCMSugenoMANFIS(
            class_names=class_names,
            n_rules=rule_count,
            fuzziness=FCM_FUZZINESS,
            seed=seed,
        ).fit(feature_rows.iloc[train_indexes], labels.iloc[train_indexes])
        if len(validation_indexes):
            validation_predictions = candidate.predict(
                feature_rows.iloc[validation_indexes]
            )
            validation_scores.append((
                _selection_loss(
                    labels.iloc[validation_indexes].reset_index(drop=True),
                    validation_predictions.reset_index(drop=True),
                ),
                rule_count,
            ))
        else:
            validation_scores.append((0.0, rule_count))

    _, selected_rule_count = min(validation_scores)
    heldout_model = FCMSugenoMANFIS(
        class_names=class_names,
        n_rules=selected_rule_count,
        fuzziness=FCM_FUZZINESS,
        seed=seed,
    ).fit(feature_rows.iloc[train_indexes], labels.iloc[train_indexes])
    test_predictions = heldout_model.predict(feature_rows.iloc[test_indexes])
    test_truth = labels.iloc[test_indexes].reset_index(drop=True)
    test_predictions = test_predictions.reset_index(drop=True)
    top1_accuracy, top1_macro_f1 = _classification_metrics(
        test_truth["top1_subject"], test_predictions["top1_subject"]
    )
    top2_accuracy, top2_macro_f1 = _classification_metrics(
        test_truth["top2_subject"], test_predictions["top2_subject"]
    )

    metrics: dict[str, float | int | str] = {
        "target_kind": "pseudo_label_from_current_recommender",
        "train_samples": int(len(train_indexes)),
        "validation_samples": int(len(validation_indexes)),
        "test_samples": int(len(test_indexes)),
        "selected_rules": int(selected_rule_count),
        "validation_selection_loss": float(min(validation_scores)[0]),
        "test_top1_accuracy": top1_accuracy,
        "test_top1_macro_f1": top1_macro_f1,
        "test_top1_score_mae": float(mean_absolute_error(
            test_truth["top1_score"], test_predictions["top1_score"]
        )),
        "test_top2_accuracy": top2_accuracy,
        "test_top2_macro_f1": top2_macro_f1,
        "test_top2_score_mae": float(mean_absolute_error(
            test_truth["top2_score"], test_predictions["top2_score"]
        )),
    }

    final_model = FCMSugenoMANFIS(
        class_names=class_names,
        n_rules=selected_rule_count,
        fuzziness=FCM_FUZZINESS,
        seed=seed,
    ).fit(feature_rows, labels)
    return final_model, metrics