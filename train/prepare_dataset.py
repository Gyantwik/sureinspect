"""
train/prepare_dataset.py
Build fixed, hashed, leakage-free splits for one VisA category.

Input layout (made by the publisher's prepare_data.py, split type 1cls):
    <root>/<category>/train/good/*.JPG
    <root>/<category>/test/good/*.JPG
    <root>/<category>/test/bad/*.JPG
    <root>/<category>/ground_truth/bad/*.png      (evaluation-only masks)

Splits written (all seeded, deterministic):
    bank_normals          training normals used to build the memory bank
    calib_normals         training normals used ONLY for calibration thresholds
    dev_gate_fit_normals  development normals used to fit the quality gate
    dev_eval_normals      development normals used for development AUROC / tuning
    dev_anomalies         30% of the publisher's test anomalies (non-standard development split)
    demo_pool             a few dev images kept for demos
    final_test            publisher test normals + remaining 70% anomalies (untouched until the end)
    excluded              images dropped because they duplicate a test image

Rules enforced:
    * No exact or near-duplicate group crosses a split.
    * Counts are checked before allocating; failures give an actionable message.
    * Ground-truth masks are evaluation-only.

Usage:
    python train/prepare_dataset.py --root ~/datasets/VisA_pytorch/1cls --category pcb1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_SEED = 42
DEV_ANOMALY_FRACTION = 0.30
MIN_DEV_NORMALS = 30
MIN_TEST_ANOMALIES = 10
NEAR_DUP_HAMMING = 2
DEMO_NORMALS = 6
DEMO_ANOMALIES = 4
GATE_FIT_FRACTION = 0.25
MIN_DEV_FOR_SEPARATE_GATE = 80


class SplitError(RuntimeError):
    """Raised with an actionable message when splits cannot be built safely."""


# ----------------------------------------------------------------- hashing

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(path: Path) -> int:
    """64-bit difference hash (9x8 grayscale). Used only to find near-duplicates."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SplitError(f"Cannot read image (corrupt or unsupported): {path}")
    small = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


# -------------------------------------------------------------- collection

def _list_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise SplitError(
            f"Missing folder: {directory}\n"
            "Run the publisher's utils/prepare_data.py (split type 1cls) first, and check --root "
            "points at the folder that contains the category folders."
        )
    files = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT]
    return sorted(files, key=lambda p: p.name)


def collect_entries(root: Path, category: str) -> list[dict]:
    base = root / category
    entries: list[dict] = []
    for pool, sub in (("train_good", "train/good"), ("test_good", "test/good"), ("test_bad", "test/bad")):
        for img in _list_images(base / sub):
            e = {
                "path": img.relative_to(root).as_posix(),
                "pool": pool,
                "sha256": sha256_file(img),
                "dhash": dhash(img),
                "mask": None,
                "mask_sha256": None,
                "group": None,
            }
            if pool == "test_bad":
                mask_dir = base / "ground_truth" / "bad"
                candidates = [mask_dir / f"{img.stem}{ext}" for ext in (".png", ".PNG", ".bmp", ".jpg")]
                mask = next((m for m in candidates if m.is_file()), None)
                if mask is None:
                    raise SplitError(f"No ground-truth mask for anomalous image {img.name} in {mask_dir}")
                e["mask"] = mask.relative_to(root).as_posix()
                e["mask_sha256"] = sha256_file(mask)
            entries.append(e)
    return entries


# ----------------------------------------------------------------- grouping

def assign_groups(entries: list[dict], near_dup_hamming: int) -> dict[str, list[dict]]:
    """Union exact duplicates (same sha256) and near-duplicates (dHash within threshold)."""
    n = len(entries)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    by_hash: dict[str, int] = {}
    for i, e in enumerate(entries):
        if e["sha256"] in by_hash:
            union(i, by_hash[e["sha256"]])
        else:
            by_hash[e["sha256"]] = i
    for i in range(n):
        for j in range(i + 1, n):
            if hamming(entries[i]["dhash"], entries[j]["dhash"]) <= near_dup_hamming:
                union(i, j)

    members: dict[int, list[dict]] = {}
    for i, e in enumerate(entries):
        members.setdefault(find(i), []).append(e)
    ordered = sorted(members.values(), key=lambda g: min(m["path"] for m in g))
    groups: dict[str, list[dict]] = {}
    for idx, g in enumerate(ordered):
        gid = f"g{idx:04d}"
        for m in g:
            m["group"] = gid
        groups[gid] = sorted(g, key=lambda m: m["path"])
    return groups


# --------------------------------------------------------------- allocation

def _group_list(entries: list[dict]) -> list[list[dict]]:
    by: dict[str, list[dict]] = {}
    for e in entries:
        by.setdefault(e["group"], []).append(e)
    return [by[k] for k in sorted(by)]


def allocate(groups: list[list[dict]], targets: dict[str, int], overflow: str, seed: str) -> dict[str, list[dict]]:
    """Whole-group greedy fill in seeded random order. Targets are filled in dict order."""
    order = list(groups)
    random.Random(seed).shuffle(order)
    remaining = dict(targets)
    out: dict[str, list[dict]] = {k: [] for k in targets}
    out[overflow] = []
    for g in order:
        for name in targets:
            if len(g) <= remaining[name]:
                out[name].extend(g)
                remaining[name] -= len(g)
                break
        else:
            out[overflow].extend(g)
    unmet = {k: v for k, v in remaining.items() if v > 0}
    if unmet:
        raise SplitError(
            f"Could not fill split targets {unmet} without splitting a duplicate group. "
            "Lower the counts in config/categories.yaml or lower --near-dup-hamming."
        )
    return out


def public(e: dict) -> dict:
    d = {"path": e["path"], "sha256": e["sha256"], "dhash": f"{e['dhash']:016x}", "group": e["group"]}
    if e.get("mask"):
        d["mask"] = e["mask"]
        d["mask_sha256"] = e["mask_sha256"]
    return d


def _sorted(entries: list[dict]) -> list[dict]:
    return [public(e) for e in sorted(entries, key=lambda x: x["path"])]


# ------------------------------------------------------------------- build

def build_splits(
    root: str | Path,
    category: str,
    bank_images: int,
    calib_images: int,
    seed: int = DEFAULT_SEED,
    near_dup_hamming: int = NEAR_DUP_HAMMING,
    dev_anomaly_fraction: float = DEV_ANOMALY_FRACTION,
) -> dict:
    root = Path(root)
    entries = collect_entries(root, category)
    groups = assign_groups(entries, near_dup_hamming)

    # --- cross-pool rules -------------------------------------------------
    excluded: list[dict] = []
    forced_final: set[str] = set()
    for members in groups.values():
        pools = {m["pool"] for m in members}
        if "train_good" in pools and pools & {"test_good", "test_bad"}:
            for m in members:
                if m["pool"] == "train_good":
                    m["excluded"] = True
                    excluded.append({**public(m), "reason": "duplicate_or_near_duplicate_of_test_image"})
        if "test_good" in pools and "test_bad" in pools:
            for m in members:
                if m["pool"] == "test_bad":
                    forced_final.add(m["path"])

    train = [e for e in entries if e["pool"] == "train_good" and not e.get("excluded")]
    test_good = [e for e in entries if e["pool"] == "test_good"]
    test_bad = [e for e in entries if e["pool"] == "test_bad"]

    # --- count checks BEFORE allocating ------------------------------------
    need_train = bank_images + calib_images + MIN_DEV_NORMALS
    if len(train) < need_train:
        raise SplitError(
            f"Category '{category}' has {len(train)} usable training normals but needs at least "
            f"{need_train} (bank {bank_images} + calibration {calib_images} + {MIN_DEV_NORMALS} development). "
            "Lower bank_images/calib_images in config/categories.yaml, or check the dataset download."
        )
    if len(test_bad) < MIN_TEST_ANOMALIES:
        raise SplitError(
            f"Category '{category}' has only {len(test_bad)} test anomalies; need at least "
            f"{MIN_TEST_ANOMALIES} for a development/final split. Check the dataset download."
        )
    if not test_good:
        raise SplitError(f"Category '{category}' has no test normals in test/good.")

    # --- training normals: bank / calibration / development ----------------
    a = allocate(_group_list(train), {"bank": bank_images, "calib": calib_images}, "dev", f"{seed}:train")
    dev_all = a["dev"]
    if len(dev_all) < MIN_DEV_NORMALS:
        raise SplitError(
            f"Only {len(dev_all)} development normals remain after allocating the bank and calibration "
            f"sets (need {MIN_DEV_NORMALS}). Lower the counts in config/categories.yaml."
        )

    # development normals: demo / gate-fit / eval (group-safe)
    n_demo = DEMO_NORMALS if len(dev_all) >= MIN_DEV_NORMALS + DEMO_NORMALS else 0
    left = len(dev_all) - n_demo
    shared = left < MIN_DEV_FOR_SEPARATE_GATE
    n_gate = 0 if shared else int(round(left * GATE_FIT_FRACTION))
    d = allocate(_group_list(dev_all), {"demo": n_demo, "gate": n_gate}, "eval", f"{seed}:dev")
    dev_eval = d["eval"]
    dev_gate = dev_eval if shared else d["gate"]
    demo_normals = d["demo"]

    # --- anomalies: development (30%) / final (70%) ------------------------
    dev_target = int(round(dev_anomaly_fraction * len(test_bad)))
    free_bad = [e for e in test_bad if e["path"] not in forced_final]
    b = allocate(_group_list(free_bad), {"dev": dev_target}, "final", f"{seed}:anom")
    dev_anoms = b["dev"]
    final_anoms = b["final"] + [e for e in test_bad if e["path"] in forced_final]

    n_demo_a = DEMO_ANOMALIES if len(dev_anoms) >= DEMO_ANOMALIES + 5 else 0
    da = allocate(_group_list(dev_anoms), {"demo": n_demo_a}, "dev", f"{seed}:demo_anom")
    dev_anoms, demo_anoms = da["dev"], da["demo"]

    splits = {
        "bank_normals": _sorted(a["bank"]),
        "calib_normals": _sorted(a["calib"]),
        "dev_gate_fit_normals": _sorted(dev_gate),
        "dev_eval_normals": _sorted(dev_eval),
        "dev_anomalies": _sorted(dev_anoms),
        "demo_pool": {"normals": _sorted(demo_normals), "anomalies": _sorted(demo_anoms)},
        "final_test": {"normals": _sorted(test_good), "anomalies": _sorted(final_anoms)},
        "excluded": sorted(excluded, key=lambda x: x["path"]),
    }

    verify_no_leakage(splits, shared_dev_normals=shared)

    multi = [g for g in groups.values() if len(g) > 1]
    largest = max((len(g) for g in groups.values()), default=1)
    n_fn, n_fa = len(final_anoms), len(test_good)
    result = {
        "meta": {
            "category": category,
            "seed": seed,
            "near_dup_hamming": near_dup_hamming,
            "dev_anomaly_fraction": dev_anomaly_fraction,
            "bank_images": bank_images,
            "calib_images": calib_images,
            "dev_normals_shared_between_gate_and_eval": shared,
            "note": "dev_anomalies are a NON-STANDARD development split (30% of the publisher's test anomalies). "
                    "Ground-truth masks are evaluation-only.",
        },
        "counts": {
            "bank_normals": len(a["bank"]),
            "calib_normals": len(a["calib"]),
            "dev_gate_fit_normals": len(dev_gate),
            "dev_eval_normals": len(dev_eval),
            "dev_anomalies": len(dev_anoms),
            "demo_normals": len(demo_normals),
            "demo_anomalies": len(demo_anoms),
            "final_test_normals": n_fa,
            "final_test_anomalies": n_fn,
            "excluded": len(excluded),
        },
        "near_duplicates": {
            "groups_with_multiple_images": len(multi),
            "largest_group": largest,
            "groups": [
                {"group": g[0]["group"], "paths": [m["path"] for m in g]} for g in multi
            ],
        },
        "expected_95pct_interval_width_at_p_0_5": {
            "final_normals": round(float(np.diff(wilson_interval(n_fa // 2, n_fa))[0]), 3),
            "final_anomalies": round(float(np.diff(wilson_interval(n_fn // 2, n_fn))[0]), 3),
        },
        "splits": splits,
    }
    return result


# ------------------------------------------------------- leakage verification

def _iter_named(splits: dict, shared_dev_normals: bool):
    named = {
        "bank_normals": splits["bank_normals"],
        "calib_normals": splits["calib_normals"],
        "dev_eval_normals": splits["dev_eval_normals"],
        "dev_anomalies": splits["dev_anomalies"],
        "demo_normals": splits["demo_pool"]["normals"],
        "demo_anomalies": splits["demo_pool"]["anomalies"],
        "final_normals": splits["final_test"]["normals"],
        "final_anomalies": splits["final_test"]["anomalies"],
    }
    if not shared_dev_normals:
        named["dev_gate_fit_normals"] = splits["dev_gate_fit_normals"]
    return named


def verify_no_leakage(splits: dict, shared_dev_normals: bool = False) -> None:
    """Fail loudly if any path, file hash or duplicate group appears in two splits."""
    named = _iter_named(splits, shared_dev_normals)
    # Final-test normals and anomalies are ONE split. A duplicate group that holds both
    # (a test normal and a test anomaly that look alike) is kept together on purpose.
    family = {"final_normals": "final", "final_anomalies": "final"}
    for key in ("path", "sha256", "group"):
        seen: dict[str, str] = {}
        for name, items in named.items():
            fam = family.get(name, name)
            for item in items:
                value = item[key]
                if value in seen and seen[value] != fam:
                    raise SplitError(
                        f"Leakage: {key} '{value}' appears in both '{seen[value]}' and '{fam}'"
                    )
                seen[value] = fam


# --------------------------------------------------------------------- CLI

def load_category_config(config_path: Path, category: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for c in cfg.get("categories", []):
        if c.get("id") == category:
            return c
    ids = [c.get("id") for c in cfg.get("categories", [])]
    raise SplitError(f"Category '{category}' not found in {config_path}. Available: {ids}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build hashed, leakage-free dataset splits.")
    ap.add_argument("--root", required=True, help="folder containing the category folders (e.g. .../VisA_pytorch/1cls)")
    ap.add_argument("--category", default="pcb1")
    ap.add_argument("--config", default="config/categories.yaml")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--near-dup-hamming", type=int, default=NEAR_DUP_HAMMING)
    ap.add_argument("--out", default=None, help="output JSON (default artifacts/splits/<category>.json)")
    args = ap.parse_args(argv)

    try:
        cat = load_category_config(Path(args.config), args.category)
        result = build_splits(
            args.root, args.category, int(cat["bank_images"]), int(cat["calib_images"]),
            seed=args.seed, near_dup_hamming=args.near_dup_hamming,
        )
    except SplitError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    out = Path(args.out or f"artifacts/splits/{args.category}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Wrote {out}")
    for k, v in result["counts"].items():
        print(f"  {k:24s} {v}")
    nd = result["near_duplicates"]
    print(f"  near-duplicate groups    {nd['groups_with_multiple_images']} (largest {nd['largest_group']})")
    if nd["largest_group"] > 20:
        print("WARNING: a very large near-duplicate group exists; consider --near-dup-hamming 1 or 0.")
    w = result["expected_95pct_interval_width_at_p_0_5"]
    print(f"  expected 95% interval width at p=0.5: normals {w['final_normals']}, anomalies {w['final_anomalies']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
