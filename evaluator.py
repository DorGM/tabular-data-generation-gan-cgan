"""Original-domain metrics for Adult synthetic rows.

Hard validity is separate from distributional fidelity. Importing this module
does not read the ARFF or train a model. Tolerances below were fixed from the
inverse-transform error of real fit and selection rows, measured before any
generator was scored with these metrics.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

INTEGER_COLUMNS = ("age", "education-num", "hours-per-week")
NONNEGATIVE_COLUMNS = ("capital-gain", "capital-loss")
SPARSE_COLUMNS = ("capital-gain", "capital-loss")

# Maximum absolute inverse-transform error observed on real fit and selection
# rows, seeds 42, 68, and 124, both scalers, on 2026-10-14:
# age 2.1e-6, fnlwgt 0.050, education-num 4.2e-7, capital-gain 0.0031,
# capital-loss 1.9e-4, hours 2.9e-6. Capital-gain reached -2.6e-5 from scaler
# dust. The allowances below sit above that noise and still reject a
# one-unit age error or a capital value of -1.
COLUMN_DOMAIN_ATOL = {
    "age": 1e-4,
    "fnlwgt": 1.0,
    "education-num": 1e-4,
    "capital-gain": 1e-2,
    "capital-loss": 1e-2,
    "hours-per-week": 1e-4,
}
INTEGER_ATOL = 1e-4
NONNEGATIVE_ATOL = 1e-2
RARE_FREQUENCY = 0.01
UPPER_QUANTILE = 0.99
NN_SUBSAMPLE = 400
NN_SEED = 0

SELECTION_DETECTOR = {
    "protocol": "A_selection_cv",
    "n_splits": 4,
    "n_estimators": 80,
    "max_depth": 16,
    "random_state": 42,
}
FINAL_DETECTOR = {
    "protocol": "B_development_fit_heldout_score",
    "n_estimators": 200,
    "random_state": 124,
}
UTILITY_FOREST = {"n_estimators": 100, "random_state": 42}

RESEARCH_TARGETS = {
    "hard_validity_violations": 0,
    "protocol_b_detection_auc": 0.75,
    "utility_ratio": 0.95,
    "note": (
        "Planning goals, not a claim that a model met them. "
        "A success statement needs all three. An unconditional generator has no utility ratio."
    ),
}


def schema_from_frame(frame, numeric_columns, categorical_columns):
    """Domain and vocabulary of a reference table.

    Pass the supplied table, not a synthetic sample. Every reference row is
    inside the resulting domain by construction, apart from inverse-transform
    noise covered by COLUMN_DOMAIN_ATOL.
    """
    missing = [column for column in list(numeric_columns) + list(categorical_columns) if column not in frame.columns]
    if missing:
        raise ValueError(f"Reference frame is missing columns: {missing}")
    domain = {}
    for column in numeric_columns:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        domain[column] = {"min": float(np.min(values)), "max": float(np.max(values))}
    categories = {}
    for column in categorical_columns:
        categories[column] = sorted(set(frame[column].map(lambda value: "?" if value is None else str(value))))
    return {
        "numeric_columns": list(numeric_columns),
        "categorical_columns": list(categorical_columns),
        "integer_columns": [column for column in INTEGER_COLUMNS if column in numeric_columns],
        "nonnegative_columns": [column for column in NONNEGATIVE_COLUMNS if column in numeric_columns],
        "sparse_columns": [column for column in SPARSE_COLUMNS if column in numeric_columns],
        "domain": domain,
        "categories": categories,
        "integer_atol": INTEGER_ATOL,
        "nonnegative_atol": NONNEGATIVE_ATOL,
        "domain_atol": dict(COLUMN_DOMAIN_ATOL),
        "rare_frequency": RARE_FREQUENCY,
        "upper_quantile": UPPER_QUANTILE,
    }


def frame_from_legacy_decoded(decoded, bundle, income=None):
    """Original-domain rows from decode_rows. All-zero categorical groups stay missing."""
    numeric = np.asarray(decoded["numeric"], dtype=float)
    data = {
        column: numeric[:, index]
        for index, column in enumerate(bundle["numeric_columns"])
    }
    hard = np.asarray(decoded["hard"], dtype=float)
    start = 0
    for group in bundle["categorical_groups"]:
        width = len(group["indexes"])
        block = hard[:, start:start + width]
        start += width
        labels = []
        for row in block:
            if float(row.sum()) != 1.0:
                labels.append(None)
            else:
                labels.append(group["labels"][int(np.argmax(row))])
        data[group["name"]] = labels
    frame = pd.DataFrame(data)
    if income is not None:
        frame["income"] = np.asarray(income).reshape(-1)
    return frame


def _as_frame(values):
    if isinstance(values, pd.DataFrame):
        return values.reset_index(drop=True)
    raise TypeError("Metrics expect an original-domain DataFrame.")


def hard_validity(frame, schema):
    """Structural checks. A zero violation count is the only passing result."""
    frame = _as_frame(frame)
    n_rows = len(frame)
    violations = {}
    missing_columns = [
        column for column in schema["numeric_columns"] + schema["categorical_columns"]
        if column not in frame.columns
    ]
    if missing_columns or n_rows == 0:
        return {
            "n_rows": n_rows,
            "missing_columns": missing_columns,
            "violation_counts": {},
            "violation_total": n_rows if n_rows else 1,
            "row_violation_fraction": 1.0,
            "valid": False,
        }

    numeric = {column: pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float) for column in schema["numeric_columns"]}
    finite_fail = np.zeros(n_rows, dtype=bool)
    for values in numeric.values():
        finite_fail |= ~np.isfinite(values)
    violations["nonfinite_rows"] = int(finite_fail.sum())

    domain_fail = np.zeros(n_rows, dtype=bool)
    for column, values in numeric.items():
        bounds = schema["domain"][column]
        atol = float(schema["domain_atol"].get(column, 0.0))
        domain_fail |= (values < bounds["min"] - atol) | (values > bounds["max"] + atol)
    violations["outside_reference_domain_rows"] = int(domain_fail.sum())

    integer_fail = np.zeros(n_rows, dtype=bool)
    for column in schema["integer_columns"]:
        values = numeric[column]
        integer_fail |= np.abs(values - np.round(values)) > float(schema["integer_atol"])
    violations["noninteger_discrete_rows"] = int(integer_fail.sum())

    negative_fail = np.zeros(n_rows, dtype=bool)
    for column in schema["nonnegative_columns"]:
        negative_fail |= numeric[column] < -float(schema["nonnegative_atol"])
    violations["negative_capital_rows"] = int(negative_fail.sum())

    category_fail = np.zeros(n_rows, dtype=bool)
    invalid_categories = {}
    for column in schema["categorical_columns"]:
        allowed = set(schema["categories"][column])
        observed = frame[column].map(lambda value: None if value is None or (isinstance(value, float) and np.isnan(value)) else str(value))
        bad = ~observed.isin(allowed) | observed.isna()
        category_fail |= bad.to_numpy()
        invalid_categories[column] = int(bad.sum())
    violations["invalid_category_rows"] = int(category_fail.sum())

    row_fail = finite_fail | domain_fail | integer_fail | negative_fail | category_fail
    total = int(row_fail.sum())
    return {
        "n_rows": n_rows,
        "missing_columns": [],
        "violation_counts": violations,
        "invalid_category_counts": invalid_categories,
        "violation_total": total,
        "row_violation_fraction": float(total / n_rows),
        "valid": total == 0,
    }


def _scale(values):
    return float(np.std(values)) + 1e-6


def _categorical_probabilities(frame, column, levels):
    observed = frame[column].map(
        lambda value: None if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)
    )
    counts = observed.value_counts(dropna=True)
    n_rows = max(len(observed), 1)
    probs = np.array([float(counts.get(level, 0.0)) for level in levels], dtype=float) / n_rows
    missing = float(observed.isna().sum()) / n_rows
    return probs, missing


def fidelity_report(real, synthetic, schema, nn_subsample=NN_SUBSAMPLE, nn_seed=NN_SEED):
    """Distribution distances on original-domain rows. Lower distances are closer."""
    real = _as_frame(real)
    synthetic = _as_frame(synthetic)
    numeric_report = {}
    for column in schema["numeric_columns"]:
        real_col = pd.to_numeric(real[column], errors="coerce").to_numpy(dtype=float)
        synth_col = pd.to_numeric(synthetic[column], errors="coerce").to_numpy(dtype=float)
        real_ok = real_col[np.isfinite(real_col)]
        synth_ok = synth_col[np.isfinite(synth_col)]
        if len(real_ok) == 0 or len(synth_ok) == 0:
            numeric_report[column] = {"ks": None, "wasserstein": None, "wasserstein_over_std": None}
            continue
        ks = float(stats.ks_2samp(real_ok, synth_ok, method="asymp").statistic)
        wasserstein = float(stats.wasserstein_distance(real_ok, synth_ok))
        scale = _scale(real_ok)
        entry = {
            "ks": ks,
            "wasserstein": wasserstein,
            "wasserstein_over_std": wasserstein / scale,
            "mean_abs_diff_over_std": abs(float(synth_ok.mean() - real_ok.mean())) / scale,
        }
        real_zero = float(np.mean(np.abs(real_ok) <= NONNEGATIVE_ATOL))
        synth_zero = float(np.mean(np.abs(synth_ok) <= NONNEGATIVE_ATOL))
        if column in schema["sparse_columns"] or real_zero >= 0.5:
            entry["zero_rate_real"] = real_zero
            entry["zero_rate_synthetic"] = synth_zero
            entry["zero_rate_abs_diff"] = abs(synth_zero - real_zero)
        quantile = float(schema["upper_quantile"])
        real_q = float(np.quantile(real_ok, quantile))
        synth_q = float(np.quantile(synth_ok, quantile))
        entry["upper_quantile"] = quantile
        entry["upper_quantile_abs_diff"] = abs(synth_q - real_q)
        entry["upper_quantile_abs_diff_over_std"] = abs(synth_q - real_q) / scale
        numeric_report[column] = entry

    categorical_report = {}
    tvs = []
    coverages = []
    rare_hits = 0
    rare_total = 0
    for column in schema["categorical_columns"]:
        levels = list(schema["categories"][column])
        real_p, real_missing = _categorical_probabilities(real, column, levels)
        synth_p, synth_missing = _categorical_probabilities(synthetic, column, levels)
        tv = float(0.5 * (np.abs(real_p - synth_p).sum() + abs(real_missing - synth_missing)))
        present_real = real_p > 0
        covered = float(np.mean(synth_p[present_real] > 0)) if present_real.any() else None
        rare = (real_p < float(schema["rare_frequency"])) & present_real
        rare_total += int(rare.sum())
        if rare.any():
            rare_hits += int(np.sum(synth_p[rare] > 0))
        categorical_report[column] = {
            "total_variation": tv,
            "coverage": covered,
            "rare_category_count": int(rare.sum()),
            "rare_category_recall": None if not rare.any() else float(np.mean(synth_p[rare] > 0)),
            "synthetic_missing_rate": float(synth_missing),
        }
        tvs.append(tv)
        if covered is not None:
            coverages.append(covered)

    association = _association_tv(real, synthetic, schema)
    correlation = _correlation_mae(real, synthetic, schema["numeric_columns"])
    diversity = _diversity(synthetic, schema)
    neighbors = _neighbor_distances(real, synthetic, schema, nn_subsample, nn_seed)
    conditional = None
    if "income" in real.columns and "income" in synthetic.columns:
        conditional = _conditional_profiles(real, synthetic, schema)
    return {
        "n_real": len(real),
        "n_synthetic": len(synthetic),
        "numeric": numeric_report,
        "numeric_ks_mean": _mean_key(numeric_report, "ks"),
        "numeric_wasserstein_over_std_mean": _mean_key(numeric_report, "wasserstein_over_std"),
        "categorical": categorical_report,
        "categorical_tv_mean": float(np.mean(tvs)) if tvs else None,
        "category_coverage_mean": float(np.mean(coverages)) if coverages else None,
        "rare_category_recall": None if rare_total == 0 else float(rare_hits / rare_total),
        "rare_category_count": rare_total,
        "correlation_mae": correlation,
        "pairwise_categorical_tv_mean": association,
        "synthetic_unique_fraction": diversity["unique_fraction"],
        "synthetic_duplicate_rate": diversity["duplicate_rate"],
        "nearest_neighbor": neighbors,
        "label_conditional": conditional,
    }


def _mean_key(report, key):
    values = [entry[key] for entry in report.values() if entry.get(key) is not None]
    if not values:
        return None
    return float(np.mean(values))


def _correlation_mae(real, synthetic, columns):
    if len(columns) < 2 or len(real) < 2 or len(synthetic) < 2:
        return None
    real_matrix = real.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    synth_matrix = synthetic.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        real_corr = np.corrcoef(real_matrix, rowvar=False)
        synth_corr = np.corrcoef(synth_matrix, rowvar=False)
    return float(np.nanmean(np.abs(real_corr - synth_corr)))


def _association_tv(real, synthetic, schema):
    columns = list(schema["categorical_columns"])
    if len(columns) < 2:
        return None
    distances = []
    for left_index, left in enumerate(columns):
        for right in columns[left_index + 1:]:
            levels_left = schema["categories"][left]
            levels_right = schema["categories"][right]
            real_joint = _joint(real, left, right, levels_left, levels_right)
            synth_joint = _joint(synthetic, left, right, levels_left, levels_right)
            distances.append(float(0.5 * np.abs(real_joint - synth_joint).sum()))
    return float(np.mean(distances)) if distances else None


def _joint(frame, left, right, levels_left, levels_right):
    left_values = frame[left].map(lambda value: None if value is None else str(value)).to_numpy()
    right_values = frame[right].map(lambda value: None if value is None else str(value)).to_numpy()
    counts = np.zeros((len(levels_left), len(levels_right)), dtype=float)
    left_index = {level: index for index, level in enumerate(levels_left)}
    right_index = {level: index for index, level in enumerate(levels_right)}
    for left_value, right_value in zip(left_values, right_values):
        if left_value in left_index and right_value in right_index:
            counts[left_index[left_value], right_index[right_value]] += 1.0
    total = counts.sum()
    if total <= 0:
        return counts
    return counts / total


def _rounded_keys(frame, schema):
    pieces = []
    for column in schema["numeric_columns"]:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if column in schema["integer_columns"] or column in ("fnlwgt", "capital-gain", "capital-loss"):
            pieces.append(np.round(values).astype(np.int64))
        else:
            pieces.append(np.round(values, 4))
    for column in schema["categorical_columns"]:
        pieces.append(frame[column].map(lambda value: "" if value is None else str(value)).to_numpy())
    keys = list(zip(*pieces)) if pieces else []
    return keys


def _diversity(frame, schema):
    keys = _rounded_keys(frame, schema)
    unique = len(set(keys))
    n_rows = max(len(keys), 1)
    fraction = unique / n_rows
    return {"unique_fraction": float(fraction), "duplicate_rate": float(1.0 - fraction)}


def privacy_report(synthetic, reference, schema):
    """Proxy checks only. Matching these numbers does not establish privacy."""
    synth_keys = _rounded_keys(synthetic, schema)
    reference_keys = set(_rounded_keys(reference, schema))
    copies = sum(1 for key in synth_keys if key in reference_keys)
    n_rows = max(len(synth_keys), 1)
    return {
        "n_synthetic": len(synth_keys),
        "n_reference": len(reference_keys),
        "exact_duplicate_fraction_vs_reference": float(copies / n_rows),
        "exact_duplicate_count": int(copies),
        "rounding": "integer fields, fnlwgt, and capital rounded to integers; other numerics to 4 decimals",
        "does_not_guarantee_privacy": True,
    }


def _design_matrix(frame, schema):
    blocks = []
    for column in schema["numeric_columns"]:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        values = np.where(np.isfinite(values), values, 0.0)
        blocks.append(values.reshape(-1, 1))
    for column in schema["categorical_columns"]:
        observed = frame[column].map(lambda value: None if value is None else str(value)).to_numpy()
        for level in schema["categories"][column]:
            blocks.append((observed == level).astype(float).reshape(-1, 1))
    if not blocks:
        raise ValueError("Schema has no columns.")
    return np.hstack(blocks)


def _neighbor_distances(real, synthetic, schema, subsample, seed):
    real_matrix = _design_matrix(real, schema)
    synth_matrix = _design_matrix(synthetic, schema)
    center = real_matrix.mean(axis=0)
    spread = real_matrix.std(axis=0)
    spread = np.where(spread < 1e-6, 1.0, spread)
    real_matrix = (real_matrix - center) / spread
    synth_matrix = (synth_matrix - center) / spread
    generator = np.random.default_rng(seed)
    real_index = np.arange(len(real_matrix))
    synth_index = np.arange(len(synth_matrix))
    if len(real_index) > subsample:
        real_index = generator.choice(real_index, size=subsample, replace=False)
    if len(synth_index) > subsample:
        synth_index = generator.choice(synth_index, size=subsample, replace=False)
    real_sample = real_matrix[real_index]
    synth_sample = synth_matrix[synth_index]
    if len(real_sample) < 2 or len(synth_sample) < 1:
        return {"available": False, "reason": "Not enough rows for nearest neighbors."}
    real_neighbors = NearestNeighbors(n_neighbors=2, algorithm="brute").fit(real_sample)
    real_distances, _ = real_neighbors.kneighbors(real_sample)
    synth_neighbors = NearestNeighbors(n_neighbors=1, algorithm="brute").fit(real_sample)
    synth_distances, _ = synth_neighbors.kneighbors(synth_sample)
    real_median = float(np.median(real_distances[:, 1]))
    synth_median = float(np.median(synth_distances[:, 0]))
    return {
        "available": True,
        "subsample": int(len(real_sample)),
        "seed": int(seed),
        "median_real_to_real": real_median,
        "median_synthetic_to_real": synth_median,
        "ratio_synthetic_over_real": None if real_median == 0 else float(synth_median / real_median),
        "does_not_guarantee_privacy": True,
    }


def _conditional_profiles(real, synthetic, schema):
    profiles = {}
    labels = sorted(set(real["income"].tolist()) | set(synthetic["income"].tolist()))
    for label in labels:
        real_part = real[real["income"] == label]
        synth_part = synthetic[synthetic["income"] == label]
        if len(real_part) == 0 or len(synth_part) == 0:
            profiles[str(label)] = {
                "n_real": int(len(real_part)),
                "n_synthetic": int(len(synth_part)),
                "available": False,
            }
            continue
        numeric = {}
        for column in schema["numeric_columns"]:
            real_col = pd.to_numeric(real_part[column], errors="coerce").to_numpy(dtype=float)
            synth_col = pd.to_numeric(synth_part[column], errors="coerce").to_numpy(dtype=float)
            numeric[column] = abs(float(np.nanmean(synth_col) - np.nanmean(real_col))) / _scale(real_col)
        tvs = []
        for column in schema["categorical_columns"]:
            levels = schema["categories"][column]
            real_p, real_missing = _categorical_probabilities(real_part, column, levels)
            synth_p, synth_missing = _categorical_probabilities(synth_part, column, levels)
            tvs.append(float(0.5 * (np.abs(real_p - synth_p).sum() + abs(real_missing - synth_missing))))
        profiles[str(label)] = {
            "available": True,
            "n_real": int(len(real_part)),
            "n_synthetic": int(len(synth_part)),
            "numeric_mean_abs_diff_over_std": numeric,
            "categorical_tv_mean": float(np.mean(tvs)) if tvs else None,
        }
    return profiles


def _oriented_auc(auc):
    return float(max(auc, 1.0 - auc))


def selection_detection(real, synthetic, schema, n_splits=4, n_estimators=80, max_depth=16, random_state=42):
    """Protocol A. Cross-validated forest on paired selection rows. For model selection only."""
    real_matrix = _design_matrix(real, schema)
    synth_matrix = _design_matrix(synthetic, schema)
    if len(real_matrix) != len(synth_matrix):
        raise ValueError("Protocol A compares equal numbers of real and synthetic selection rows.")
    features = np.vstack([real_matrix, synth_matrix])
    labels = np.concatenate([np.zeros(len(real_matrix)), np.ones(len(synth_matrix))])
    folder = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []
    for train_index, test_index in folder.split(features, labels):
        if set(train_index.tolist()) & set(test_index.tolist()):
            raise RuntimeError("A selection-detection fold reused rows.")
        forest = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=1,
        )
        forest.fit(features[train_index], labels[train_index])
        proba = forest.predict_proba(features[test_index])[:, 1]
        scores.append(float(roc_auc_score(labels[test_index], proba)))
    mean_auc = float(np.mean(scores))
    return {
        "protocol": "A_selection_cv",
        "uses_final_test": False,
        "n_real": int(len(real_matrix)),
        "n_synthetic": int(len(synth_matrix)),
        "fold_auc": scores,
        "mean_auc": mean_auc,
        "oriented_auc": _oriented_auc(mean_auc),
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "random_state": random_state,
    }


def final_detection(development_real, development_synthetic, test_real, test_synthetic, schema, n_estimators=200, random_state=124):
    """Protocol B. Fit only on development rows. Score a disjoint real test set and a fresh synthetic draw.

    This function does not look at which array is which beyond the arguments.
    Callers must not pass the final test into the development arguments, and
    must not use the returned AUC to choose a checkpoint, seed, or architecture.
    """
    dev_real = _design_matrix(development_real, schema)
    dev_synth = _design_matrix(development_synthetic, schema)
    hold_real = _design_matrix(test_real, schema)
    hold_synth = _design_matrix(test_synthetic, schema)
    train_x = np.vstack([dev_real, dev_synth])
    train_y = np.concatenate([np.zeros(len(dev_real)), np.ones(len(dev_synth))])
    test_x = np.vstack([hold_real, hold_synth])
    test_y = np.concatenate([np.zeros(len(hold_real)), np.ones(len(hold_synth))])
    forest = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=1)
    forest.fit(train_x, train_y)
    proba = forest.predict_proba(test_x)[:, 1]
    auc = float(roc_auc_score(test_y, proba))
    return {
        "protocol": "B_development_fit_heldout_score",
        "uses_final_test_for_fitting": False,
        "n_development_real": int(len(dev_real)),
        "n_development_synthetic": int(len(dev_synth)),
        "n_test_real": int(len(hold_real)),
        "n_test_synthetic": int(len(hold_synth)),
        "mean_auc": auc,
        "oriented_auc": _oriented_auc(auc),
        "n_estimators": n_estimators,
        "random_state": random_state,
    }


def utility_report(x_real, y_real, x_synthetic, y_synthetic, x_test, y_test, n_estimators=100, random_state=42):
    """Equal-sized real and synthetic training forests, both scored on the same real test rows."""
    if y_synthetic is None:
        return {
            "available": False,
            "reason": "The unconditional generator does not emit income labels, so no utility ratio is assigned.",
        }
    x_real = _design_matrix_or_array(x_real)
    x_synthetic = _design_matrix_or_array(x_synthetic)
    x_test = _design_matrix_or_array(x_test)
    y_real = np.asarray(y_real).reshape(-1)
    y_synthetic = np.asarray(y_synthetic).reshape(-1)
    y_test = np.asarray(y_test).reshape(-1)
    if len(x_real) != len(y_real) or len(x_synthetic) != len(y_synthetic) or len(x_test) != len(y_test):
        raise ValueError("Utility features and labels must have the same number of rows.")
    if len(x_real) != len(x_synthetic):
        raise ValueError("Utility training sets must be the same size.")
    if x_real.shape[1] != x_synthetic.shape[1] or x_real.shape[1] != x_test.shape[1]:
        raise ValueError("Utility matrices must share a column layout.")
    forest_real = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=1)
    forest_synth = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=1)
    forest_real.fit(x_real, y_real)
    forest_synth.fit(x_synthetic, y_synthetic)
    auc_real = float(roc_auc_score(y_test, forest_real.predict_proba(x_test)[:, 1]))
    auc_synth = float(roc_auc_score(y_test, forest_synth.predict_proba(x_test)[:, 1]))
    return {
        "available": True,
        "auc_real": auc_real,
        "auc_synthetic": auc_synth,
        "ratio": auc_synth / auc_real,
        "prior_baseline_auc": 0.5,
        "n_train": int(len(x_real)),
        "n_test": int(len(x_test)),
        "n_estimators": n_estimators,
        "random_state": random_state,
        "label_source": "caller-declared synthetic labels; this function does not invent them",
    }


def _design_matrix_or_array(values):
    if isinstance(values, pd.DataFrame):
        raise TypeError("Pass a design matrix to utility_report, not a frame.")
    return np.asarray(values, dtype=float)


def research_success(hard_valid, protocol_b_auc, utility_ratio):
    """All three planning goals. A missing utility ratio cannot pass."""
    validity_ok = bool(hard_valid)
    detection_ok = protocol_b_auc is not None and float(protocol_b_auc) <= RESEARCH_TARGETS["protocol_b_detection_auc"]
    utility_ok = utility_ratio is not None and float(utility_ratio) >= RESEARCH_TARGETS["utility_ratio"]
    return {
        "hard_validity": validity_ok,
        "detection": detection_ok,
        "utility": utility_ok,
        "all_three": bool(validity_ok and detection_ok and utility_ok),
    }


def metrics_schema_document():
    return {
        "version": 1,
        "domain": "decoded original-domain Adult rows; scaled matrices are not compared",
        "tolerances": {
            "integer_atol": INTEGER_ATOL,
            "nonnegative_atol": NONNEGATIVE_ATOL,
            "domain_atol": COLUMN_DOMAIN_ATOL,
            "provenance": "Fixed from real-row inverse-transform error before generator scoring.",
        },
        "research_targets": RESEARCH_TARGETS,
        "metrics": {
            "hard_validity.violation_total": {"direction": "lower", "target": 0, "split": "the rows being judged"},
            "numeric.ks": {"direction": "lower", "range": [0, 1]},
            "numeric.wasserstein_over_std": {"direction": "lower"},
            "numeric.zero_rate_abs_diff": {"direction": "lower", "columns": list(SPARSE_COLUMNS)},
            "numeric.upper_quantile_abs_diff": {"direction": "lower", "quantile": UPPER_QUANTILE},
            "categorical.total_variation": {"direction": "lower", "range": [0, 1]},
            "category_coverage_mean": {"direction": "higher", "range": [0, 1]},
            "rare_category_recall": {"direction": "higher", "rare_below": RARE_FREQUENCY},
            "correlation_mae": {"direction": "lower"},
            "pairwise_categorical_tv_mean": {"direction": "lower"},
            "synthetic_duplicate_rate": {"direction": "lower"},
            "exact_duplicate_fraction_vs_reference": {
                "direction": "lower_is_less_copying",
                "privacy": "proxy only; does not guarantee privacy",
            },
            "nearest_neighbor.ratio_synthetic_over_real": {
                "direction": "near_1_less_suspicious_than_much_below_1",
                "subsample": NN_SUBSAMPLE,
                "seed": NN_SEED,
                "privacy": "proxy only; does not guarantee privacy",
            },
            "selection_detection.mean_auc": {
                "direction": "lower_is_harder_to_distinguish",
                "chance": 0.5,
                "protocol": SELECTION_DETECTOR,
                "split": "paired real selection rows and synthetic rows; not the final test",
            },
            "final_detection.mean_auc": {
                "direction": "lower_is_harder_to_distinguish",
                "chance": 0.5,
                "target": RESEARCH_TARGETS["protocol_b_detection_auc"],
                "protocol": FINAL_DETECTOR,
                "split": "fit on development real+synthetic only; score untouched real test plus a fresh synthetic draw",
                "selection_use": "forbidden",
            },
            "utility.auc_real": {"direction": "higher", "split": "train on real fit; score on real test"},
            "utility.auc_synthetic": {"direction": "higher", "split": "train on an equal-sized synthetic draw; score on the same real test"},
            "utility.ratio": {"direction": "higher", "target": RESEARCH_TARGETS["utility_ratio"]},
            "oriented_auc": {
                "definition": "max(auc, 1-auc)",
                "use": "report beside the raw AUC so a flipped label cannot look like indistinguishability",
            },
        },
    }
