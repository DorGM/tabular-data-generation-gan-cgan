"""ACS California 2018 (ACSIncome task): load, filter and split by household. See PROTOCOL.md.

Source: 2018 ACS 1-year PUMS person file for California (psam_p06.csv, from csv_pca.zip at
www2.census.gov/programs-surveys/acs/data/pums/2018/1-Year/). Filter and features follow the
folktables ACSIncome task (Ding et al., 2021), which mimics the Adult extraction rules.

Importing this module does not read data.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "data" / "acs" / "psam_p06.csv"
SPLIT_PATH = ROOT / "data" / "acs" / "splits.json"

NUMERIC = ["AGEP", "WKHP"]
CATEGORICAL = ["COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "SEX", "RAC1P"]
INTEGER = {"AGEP", "WKHP"}
TARGET_THRESHOLD = 50_000
SPLIT_SEED = 2018
# Persons per part. Households are added in shuffled order until the part reaches its size.
PART_SIZES = (("train", 30_000), ("validation", 8_000), ("test", 30_000), ("reference", 38_000))
# Everything left after these parts is the reserve: never read by any script in this round.

# Domain rules (PROTOCOL.md). Each follows from what the fields mean and was checked on the
# training part only: a spouse of the reference person is married; an unmarried partner is not
# married; a member of the armed forces (OCCP 9800-9830) works for the federal government (COW 5).
# Training-part exceptions: 0, 0 and 1 of 30,000 rows.
MILITARY = ("9800", "9810", "9825", "9830")
SPOUSE, PARTNER, MARRIED, FEDERAL = "1", "13", "1", "5"


def rule_violations(frame):
    """Boolean mask of rows that break a domain rule, plus counts per rule."""
    rel, mar = frame["RELP"].astype(str), frame["MAR"].astype(str)
    occ, cow = frame["OCCP"].astype(str), frame["COW"].astype(str)
    rules = {
        "spouse_married": (rel == SPOUSE) & (mar != MARRIED),
        "partner_unmarried": (rel == PARTNER) & (mar == MARRIED),
        "military_federal": occ.isin(MILITARY) & (cow != FEDERAL),
    }
    mask = np.zeros(len(frame), dtype=bool)
    for value in rules.values():
        mask |= value.to_numpy()
    return mask, {name: int(value.sum()) for name, value in rules.items()}


def expected_violations(probs, levels):
    """Expected number of broken rules per row from per-column category probabilities
    (same form as the Adult version in ladder.py)."""
    def level(column, value):
        return probs[column][:, levels[column].index(value)] if value in levels[column] else 0.0
    spouse, partner, married = level("RELP", SPOUSE), level("RELP", PARTNER), level("MAR", MARRIED)
    military = sum(level("OCCP", code) for code in MILITARY)
    return spouse * (1 - married) + partner * married + military * (1 - level("COW", FEDERAL))


def load_acs(path=RAW_PATH):
    """ACSIncome rows in file order, with SERIALNO kept for household grouping."""
    usecols = ["SERIALNO", "PWGTP", "PINCP"] + NUMERIC + CATEGORICAL
    df = pd.read_csv(path, usecols=usecols, dtype={"SERIALNO": str})
    df = df[(df["AGEP"] > 16) & (df["PINCP"] > 100) & (df["WKHP"] > 0) & (df["PWGTP"] >= 1)]
    out = pd.DataFrame({"SERIALNO": df["SERIALNO"].to_numpy()})
    for column in NUMERIC:
        out[column] = df[column].to_numpy(dtype=float)
    for column in CATEGORICAL:
        out[column] = df[column].astype(int).astype(str).to_numpy()
    out["income"] = (df["PINCP"].to_numpy() > TARGET_THRESHOLD).astype(int)
    return out


def make_splits(frame, seed=SPLIT_SEED):
    """Row positions of each part. All persons of a household land in the same part."""
    households = frame["SERIALNO"].unique()  # file order, then shuffled
    order = np.random.default_rng(seed).permutation(len(households))
    sizes = frame.groupby("SERIALNO", sort=False).size()
    rows_of = frame.groupby("SERIALNO", sort=False).indices
    parts, cursor = {}, 0
    for name, target in PART_SIZES:
        chosen, count = [], 0
        while count < target:
            household = households[order[cursor]]
            chosen.append(household)
            count += int(sizes[household])
            cursor += 1
        parts[name] = np.sort(np.concatenate([rows_of[h] for h in chosen]))
    rest = [households[i] for i in order[cursor:]]
    parts["reserve"] = np.sort(np.concatenate([rows_of[h] for h in rest]))
    return parts


def split_digest(parts):
    blob = json.dumps({k: v.tolist() for k, v in sorted(parts.items())}).encode()
    return hashlib.sha256(blob).hexdigest()


def save_splits(frame, parts, path=SPLIT_PATH):
    payload = {
        "seed": SPLIT_SEED, "part_sizes": dict(PART_SIZES), "n_rows": int(len(frame)),
        "sha256": split_digest(parts),
        "rows": {k: v.tolist() for k, v in parts.items()},
        "households": {k: sorted(frame["SERIALNO"].iloc[v].unique().tolist()) for k, v in parts.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def load_splits(frame, path=SPLIT_PATH):
    """The saved parts, checked against a fresh rebuild so nobody can quietly re-split."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    parts = {k: np.asarray(v, dtype=int) for k, v in payload["rows"].items()}
    if not split_digest(parts) == payload["sha256"] == split_digest(make_splits(frame)):
        raise RuntimeError("split changed")
    return parts


def part_frames(frame, parts, names=("train", "validation")):
    """Frames for the named parts, without SERIALNO. The test part is only read by acs.py evaluate."""
    data = frame.drop(columns=["SERIALNO"])
    return [data.iloc[parts[name]].reset_index(drop=True) for name in names]


if __name__ == "__main__":
    frame = load_acs()
    parts = make_splits(frame)
    payload = save_splits(frame, parts)
    print({k: len(v) for k, v in parts.items()}, payload["sha256"])
