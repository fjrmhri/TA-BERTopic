#!/usr/bin/env python3
"""Smoke test backend classifier on 5 hoaks + 5 fakta dataset samples."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

TEXT_PRIORITY = ("summary", "Clean Narasi", "Narasi", "isi_berita", "judul")

HOAX_FILE = "data_hoaks_turnbackhoaks.csv"
FAKTA_FILES = (
    "data_nonhoaks_detik.csv",
    "data_nonhoaks_kompas.csv",
    "data_nonhoaks_cnn.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test endpoint /analyze")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:7860",
        help="Backend base URL, misal http://127.0.0.1:7860 atau https://<space>.hf.space",
    )
    parser.add_argument(
        "--dataset-dir",
        default=str(Path(__file__).resolve().parents[1] / "dataset"),
        help="Path folder dataset CSV",
    )
    parser.add_argument("--limit-per-class", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=45)
    return parser.parse_args()


def pick_text(row: Dict[str, str]) -> Tuple[str, str]:
    for column in TEXT_PRIORITY:
        value = row.get(column, "")
        if isinstance(value, str) and value.strip():
            return value.strip(), column
    return "", "UNKNOWN"


def parse_label(raw: str) -> Optional[int]:
    try:
        value = int(float(str(raw).strip()))
    except Exception:
        return None
    return value if value in (0, 1) else None


def collect_samples(csv_path: Path, expected_label: int, limit: int) -> List[Dict[str, str]]:
    if not csv_path.exists():
        return []

    rows: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if parse_label(row.get("hoax")) != expected_label:
                continue
            text, text_source = pick_text(row)
            if not text:
                continue
            rows.append(
                {
                    "text": text,
                    "text_source": text_source,
                    "source_file": csv_path.name,
                    "gold_label": "Hoaks" if expected_label == 1 else "Fakta",
                }
            )
            if len(rows) >= limit:
                break
    return rows


def analyze_text(base_url: str, text: str, timeout: int) -> Dict[str, object]:
    url = base_url.rstrip("/") + "/analyze"
    response = requests.post(url, json={"text": text}, timeout=timeout)
    response.raise_for_status()
    return response.json()


def extract_first_sentence(payload: Dict[str, object]) -> Dict[str, object]:
    paragraphs = payload.get("paragraphs", [])
    if not paragraphs:
        return {}
    sentences = paragraphs[0].get("sentences", [])
    if not sentences:
        return {}
    return sentences[0]


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)

    if not dataset_dir.exists():
        print(f"[ERROR] Dataset folder tidak ditemukan: {dataset_dir}")
        return 1

    hoax_samples = collect_samples(dataset_dir / HOAX_FILE, expected_label=1, limit=args.limit_per_class)

    fakta_samples: List[Dict[str, str]] = []
    for filename in FAKTA_FILES:
        needed = args.limit_per_class - len(fakta_samples)
        if needed <= 0:
            break
        fakta_samples.extend(collect_samples(dataset_dir / filename, expected_label=0, limit=needed))

    samples = fakta_samples + hoax_samples
    if len(fakta_samples) < args.limit_per_class or len(hoax_samples) < args.limit_per_class:
        print("[WARN] Sampel kurang dari target.")
        print(f"  Fakta: {len(fakta_samples)} | Hoaks: {len(hoax_samples)}")

    print(f"[INFO] Menjalankan smoke test ke: {args.base_url}")
    print(f"[INFO] Total sampel: {len(samples)}")

    results = []
    for idx, sample in enumerate(samples, start=1):
        try:
            payload = analyze_text(args.base_url, sample["text"], timeout=args.timeout)
            sentence = extract_first_sentence(payload)
            model_meta = payload.get("model", {})
            result = {
                "idx": idx,
                "gold": sample["gold_label"],
                "pred": sentence.get("label", "UNKNOWN"),
                "prob_hoax": sentence.get("prob_hoax", None),
                "prob_fakta": sentence.get("prob_fakta", None),
                "confidence": sentence.get("confidence", None),
                "color": sentence.get("color", None),
                "model_source": model_meta.get("source"),
                "threshold": model_meta.get("hoax_threshold"),
                "source_file": sample["source_file"],
                "text_preview": sample["text"][:120].replace("\n", " "),
            }
            results.append(result)
        except Exception as exc:
            results.append(
                {
                    "idx": idx,
                    "gold": sample["gold_label"],
                    "pred": f"ERROR: {exc}",
                    "prob_hoax": None,
                    "prob_fakta": None,
                    "confidence": None,
                    "color": None,
                    "model_source": None,
                    "threshold": None,
                    "source_file": sample["source_file"],
                    "text_preview": sample["text"][:120].replace("\n", " "),
                }
            )

    print("\n=== HASIL DETAIL ===")
    for row in results:
        print(
            f"#{row['idx']:02d} gold={row['gold']:<5} pred={row['pred']:<12} "
            f"hoax={row['prob_hoax']} fakta={row['prob_fakta']} conf={row['confidence']} "
            f"source={row['model_source']} th={row['threshold']} file={row['source_file']}"
        )

    summary = {
        "gold_fakta": 0,
        "gold_hoaks": 0,
        "pred_fakta": 0,
        "pred_hoaks": 0,
        "errors": 0,
    }
    for row in results:
        if row["gold"] == "Fakta":
            summary["gold_fakta"] += 1
        elif row["gold"] == "Hoaks":
            summary["gold_hoaks"] += 1

        if row["pred"] == "Fakta":
            summary["pred_fakta"] += 1
        elif row["pred"] == "Hoaks":
            summary["pred_hoaks"] += 1
        elif str(row["pred"]).startswith("ERROR"):
            summary["errors"] += 1

    print("\n=== RINGKASAN ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
