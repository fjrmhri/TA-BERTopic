#!/usr/bin/env python
"""Deterministic six-paragraph smoke test without starting FastAPI server."""

from __future__ import annotations

import csv
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT_DIR / "dataset"
MODEL_DIR = ROOT_DIR / "indobert_hoax_model_v1"

TEXT_PRIORITY = ("summary", "Clean Narasi", "Narasi", "isi_berita", "judul")
EXPECTED_FILES = (
    "data_nonhoaks_cnn.csv",
    "data_nonhoaks_detik.csv",
    "data_nonhoaks_kompas.csv",
    "data_hoaks_turnbackhoaks.csv",
)



def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()



def _parse_label(raw: object) -> Optional[int]:
    try:
        value = int(float(str(raw).strip()))
    except Exception:
        return None
    return value if value in (0, 1) else None



def _pick_text(row: Dict[str, str]) -> tuple[str, str]:
    for column in TEXT_PRIORITY:
        value = row.get(column, "")
        if isinstance(value, str) and value.strip():
            return _normalize_text(value), column
    return "", "UNKNOWN"



def _collect_candidates() -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    missing = [name for name in EXPECTED_FILES if not (DATASET_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Dataset tidak lengkap di {DATASET_DIR}. Missing: {', '.join(missing)}"
        )

    fakta_rows: List[Dict[str, str]] = []
    hoaks_rows: List[Dict[str, str]] = []

    for filename in EXPECTED_FILES:
        csv_path = DATASET_DIR / filename
        with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                label = _parse_label(row.get("hoax"))
                if label is None:
                    continue
                text, text_source = _pick_text(row)
                if not text:
                    continue
                payload = {
                    "source_file": filename,
                    "text_source": text_source,
                    "text": text,
                    "expected_label": "Hoaks" if label == 1 else "Fakta",
                }
                if label == 1:
                    hoaks_rows.append(payload)
                else:
                    fakta_rows.append(payload)

    return fakta_rows, hoaks_rows



def _select_six_samples(seed: int = 42) -> List[Dict[str, str]]:
    fakta_rows, hoaks_rows = _collect_candidates()
    if len(fakta_rows) < 3 or len(hoaks_rows) < 3:
        raise RuntimeError(
            f"Sampel tidak cukup. fakta={len(fakta_rows)} hoaks={len(hoaks_rows)}"
        )

    rng = random.Random(seed)
    rng.shuffle(fakta_rows)
    rng.shuffle(hoaks_rows)

    selected = fakta_rows[:3] + hoaks_rows[:3]
    selected.sort(key=lambda item: item["expected_label"])  # Fakta dulu, lalu Hoaks
    return selected



def _preview(text: str, max_chars: int = 160) -> str:
    compact = _normalize_text(text)
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars].rstrip() + "..."



def main() -> int:
    os.environ.setdefault("MODEL_DIR", str(MODEL_DIR))
    os.environ.setdefault("DECISION_MODE", "argmax")

    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))

    try:
        from backend import app as backend_app
    except Exception as exc:
        print(f"[ERROR] Gagal import backend.app: {exc}")
        return 1

    try:
        backend_app._load_classifier()
        backend_app._run_startup_sanity()
    except Exception as exc:
        print(f"[ERROR] Gagal inisialisasi classifier backend: {exc}")
        return 1

    try:
        samples = _select_six_samples(seed=42)
    except Exception as exc:
        print(f"[ERROR] Gagal menyiapkan 6 sampel: {exc}")
        return 1

    tp = fp = fn = tn = 0
    correct = 0

    print("=== SIX PARAGRAPH TEST (deterministic from dataset) ===")
    print(f"decision_mode_effective={backend_app.DECISION_MODE_EFFECTIVE}")
    print(f"hoax_threshold={backend_app.HOAX_THRESHOLD:.3f} source={backend_app.HOAX_THRESHOLD_SOURCE}")

    for idx, sample in enumerate(samples, start=1):
        sentences = backend_app._split_sentences(sample["text"])
        predictions = backend_app._predict_batch(sentences)

        paragraph_label = "Hoaks" if any(row["label"] == "Hoaks" for row in predictions) else "Fakta"
        expected = sample["expected_label"]

        if paragraph_label == expected:
            correct += 1

        if expected == "Hoaks" and paragraph_label == "Hoaks":
            tp += 1
        elif expected == "Fakta" and paragraph_label == "Hoaks":
            fp += 1
        elif expected == "Hoaks" and paragraph_label == "Fakta":
            fn += 1
        else:
            tn += 1

        top_rows = sorted(predictions, key=lambda row: float(row["prob_hoax"]), reverse=True)[:2]

        print(
            f"\n[{idx}] expected={expected} predicted={paragraph_label} "
            f"source={sample['source_file']}:{sample['text_source']}"
        )
        print(f"preview: {_preview(sample['text'])}")
        for row in top_rows:
            print(
                "  - "
                f"prob_hoax={float(row['prob_hoax']):.4f} "
                f"prob_fakta={float(row['prob_fakta']):.4f} "
                f"label={row['label']} :: {_preview(row['text'], 120)}"
            )

    total = len(samples)
    accuracy = (correct / total) if total else 0.0

    print("\n=== CONFUSION SUMMARY (paragraph-level) ===")
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"ACCURACY={accuracy:.4f} ({correct}/{total})")

    return 0



if __name__ == "__main__":
    sys.exit(main())

