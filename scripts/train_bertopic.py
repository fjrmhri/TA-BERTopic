#!/usr/bin/env python
"""Train BERTopic artifact from local dataset CSV files."""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from pathlib import Path
from typing import Iterable, List

TEXT_PRIORITY = ("summary", "Clean Narasi", "Narasi", "isi_berita", "judul")
EXPECTED_FILES = (
    "data_nonhoaks_cnn.csv",
    "data_nonhoaks_detik.csv",
    "data_nonhoaks_kompas.csv",
    "data_hoaks_turnbackhoaks.csv",
)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _pick_text(row: dict) -> str:
    for column in TEXT_PRIORITY:
        value = row.get(column, "")
        if isinstance(value, str) and value.strip():
            return _normalize_text(value)
    return ""


def _iter_rows(csv_path: Path) -> Iterable[dict]:
    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def load_documents(dataset_dir: Path, max_docs: int, seed: int) -> List[str]:
    missing = [name for name in EXPECTED_FILES if not (dataset_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Dataset tidak lengkap di {dataset_dir}. Missing: {', '.join(missing)}"
        )

    docs: List[str] = []
    for name in EXPECTED_FILES:
        csv_path = dataset_dir / name
        for row in _iter_rows(csv_path):
            text = _pick_text(row)
            if text:
                docs.append(text)

    if not docs:
        raise RuntimeError("Tidak ada dokumen valid untuk training BERTopic.")

    docs = list(dict.fromkeys(docs))
    if max_docs > 0 and len(docs) > max_docs:
        rng = random.Random(seed)
        docs = rng.sample(docs, max_docs)

    return docs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train BERTopic artifact dari dataset lokal")
    parser.add_argument("--dataset-dir", default="dataset", help="Folder dataset CSV")
    parser.add_argument("--output-dir", default="bertopic_model", help="Folder output artifact")
    parser.add_argument("--max-docs", type=int, default=12000, help="Maksimum jumlah dokumen")
    parser.add_argument("--seed", type=int, default=42, help="Seed sampling")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Model embedding untuk BERTopic",
    )
    parser.add_argument("--min-topic-size", type=int, default=15, help="Minimum topic size")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    try:
        from bertopic import BERTopic
    except Exception as exc:
        print(f"[ERROR] BERTopic belum tersedia: {exc}")
        return 1

    if not dataset_dir.exists() or not dataset_dir.is_dir():
        print(f"[ERROR] Dataset folder tidak ditemukan: {dataset_dir}")
        return 1

    try:
        documents = load_documents(dataset_dir, max_docs=args.max_docs, seed=args.seed)
    except Exception as exc:
        print(f"[ERROR] Gagal memuat dokumen: {exc}")
        return 1

    print(f"[INFO] Documents siap: {len(documents)}")
    print(f"[INFO] Embedding model: {args.embedding_model}")

    topic_model = BERTopic(
        language="multilingual",
        embedding_model=args.embedding_model,
        min_topic_size=max(2, int(args.min_topic_size)),
        calculate_probabilities=True,
        low_memory=True,
        verbose=True,
    )

    topic_model.fit_transform(documents)

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        topic_model.save(
            str(output_dir),
            serialization="safetensors",
            save_ctfidf=True,
            save_embedding_model=True,
        )
    except TypeError:
        topic_model.save(str(output_dir))

    print(f"[OK] BERTopic artifact tersimpan di: {output_dir}")
    print("[NEXT] Untuk backend lokal/folder deployment:")
    print(f"       set TOPIC_MODEL_DIR={output_dir}")
    print("[NEXT] Untuk Hugging Face Hub: upload folder artifact, lalu set TOPIC_MODEL_ID=<repo_id>")
    return 0


if __name__ == "__main__":
    sys.exit(main())

