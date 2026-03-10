import csv
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("hoax-backend")

MODEL_ID = os.getenv("MODEL_ID", "fjrmhri/Deteksi_Hoax_IndoBERT_BERTopic")
MODEL_DIR_ENV = os.getenv("MODEL_DIR")
LOCAL_MODEL_PATH_ENV = os.getenv("LOCAL_MODEL_PATH")  # legacy env
CALIBRATION_PATH_ENV = os.getenv("CALIBRATION_PATH")
HF_TOKEN = os.getenv("HF_TOKEN")
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN")
ORANGE_THRESHOLD = float(os.getenv("ORANGE_THRESHOLD", "0.65"))
DEFAULT_HOAX_THRESHOLD = float(os.getenv("HOAX_THRESHOLD", "0.5"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "256"))
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "50000"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))

DEFAULT_LOCAL_MODEL_DIRNAME = "indobert_hoax_model_v1"

ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = Path(__file__).resolve().parent

PARAGRAPH_SPLIT_RE = re.compile(r"(?:\r?\n){2,}")
SENTENCE_RE = re.compile(r"[^.!?]+(?:[.!?]+(?:[\"”’\)\]]+)?)|[^.!?]+$")
HOAX_LABEL_TOKENS = ("hoaks", "hoax", "fake", "false", "disinfo", "misinfo")
FAKTA_LABEL_TOKENS = ("fakta", "fact", "true", "valid", "nonhoax", "non-hoax")
INFERENCE_CLEAN_PATTERNS = [
    (re.compile(r"(?i)\buncategorized\b"), " "),
    (re.compile(r"(?i)\b(?:facebook|twitter|x\.com|tiktok|youtube|instagram|whatsapp)\b"), " "),
    (re.compile(r"(?i)\bakun\b[^.!?\n]{0,140}\bunggah\b[^.!?\n]*"), " "),
    (re.compile(r"(?i)\bbaca juga:\s*[^.!?\n]*"), " "),
    (re.compile(r"(?i)\blihat juga:\s*[^.!?\n]*"), " "),
    (re.compile(r"(?i)\badvertisement\b\s*scroll to continue with content"), " "),
    (re.compile(r"(?i)\bturnbackhoax(?:s)?\b"), " "),
    (re.compile(r"(?i)\bcnn indonesia\b"), " "),
    (re.compile(r"(?i)\bkompas\.com\b"), " "),
    (re.compile(r"(?i)\bdetik(?:com)?\b"), " "),
    (re.compile(r"(?i)\bmafindo\b"), " "),
    (re.compile(r"(?i)\b\d{1,2}\s*[/-]\s*\d{1,2}\s*[/-]\s*\d{2,4}\b"), " "),
    (re.compile(r"(?i)\b\d{1,2}\s+\d{1,2}\s+\d{4}\b"), " "),
    (re.compile(r"(?i)\b\d{1,2}:\d{2}\s*wib\b"), " "),
]

REQUIRED_LOCAL_MODEL_FILES = ["config.json", "tokenizer_config.json"]
TOKENIZER_ARTIFACT_CANDIDATES = ("tokenizer.json", "vocab.txt")
OPTIONAL_LOCAL_MODEL_FILES = ["special_tokens_map.json"]
WEIGHT_CANDIDATES = ("model.safetensors", "pytorch_model.bin")
ARTIFACT_VALIDATION_MODE = "local-or-hub"

DEBUG_TEXT_PRIORITY = ("summary", "Clean Narasi", "Narasi", "isi_berita", "judul")
DEBUG_HOAX_FILE = "data_hoaks_turnbackhoaks.csv"
DEBUG_FAKTA_FILES = (
    "data_nonhoaks_detik.csv",
    "data_nonhoaks_kompas.csv",
    "data_nonhoaks_cnn.csv",
)

STARTUP_SANITY_SENTENCES = [
    "Beredar unggahan yang mengklaim ada rekrutmen CPNS fiktif dan masyarakat diminta transfer biaya pendaftaran.",
    "PT Transjakarta melakukan modifikasi layanan pada empat rute untuk meningkatkan kenyamanan penumpang.",
    "Video lama diklaim sebagai kericuhan terbaru dan narasi itu ramai dibagikan di media sosial.",
    "Pemerintah daerah akan membahas penertiban izin fasilitas olahraga di area permukiman.",
]


class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="Teks input multi paragraf.")


app = FastAPI(title="Hoax Sentence Analyzer API", version="3.1.0")

allowed_origins = [FRONTEND_ORIGIN] if FRONTEND_ORIGIN else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASSIFIER_TOKENIZER = None
CLASSIFIER_MODEL = None
MODEL_SOURCE = "unknown"
MODEL_LOAD_REASON = "not_loaded"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ID2LABEL: Dict[int, str] = {0: "Fakta", 1: "Hoaks"}
LABEL2ID: Dict[str, int] = {"Fakta": 0, "Hoaks": 1}
NUM_LABELS = 2
FAKTA_CLASS_ID = 0
HOAX_CLASS_ID = 1
LABEL_MAPPING_WARNINGS: List[str] = []

HOAX_THRESHOLD = DEFAULT_HOAX_THRESHOLD
HOAX_THRESHOLD_SOURCE = "default"
CALIBRATION_LOADED = False

LOCAL_MODEL_PATH = Path("/app") / DEFAULT_LOCAL_MODEL_DIRNAME
LOCAL_MODEL_PATH_SOURCE = "unresolved"
LOCAL_MODEL_CANDIDATES: List[str] = []
LOCAL_MODEL_VALID = False
CALIBRATION_PATH = LOCAL_MODEL_PATH / "calibration.json"

MISSING_REQUIRED_LOCAL_ARTIFACTS: List[str] = []
MISSING_OPTIONAL_LOCAL_ARTIFACTS: List[str] = []
MISSING_LOCAL_ARTIFACTS: List[str] = []

STARTUP_SANITY: Dict[str, object] = {
    "checked": False,
    "status": "not_run",
    "message": "startup sanity belum dijalankan",
}

TOPICS_PAYLOAD = {"enabled": False, "items": []}


def _float(value: float) -> float:
    return round(float(value), 6)


def _hf_auth_kwargs() -> Dict[str, str]:
    return {"token": HF_TOKEN} if HF_TOKEN else {}


def _normalize_label(name: str) -> str:
    return re.sub(r"[^a-z0-9\-]+", "", str(name).strip().lower())


def _normalize_unit_text(text: str) -> str:
    cleaned = str(text)
    for pattern, replacement in INFERENCE_CLEAN_PATTERNS:
        cleaned = pattern.sub(replacement, cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.strip(" -:;,.\")\n\t")


def _preview_text(text: str, max_chars: int = 240) -> str:
    compact = re.sub(r"\s+", " ", str(text)).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars].rstrip() + "..."


def _parse_binary_label(raw: Any) -> Optional[int]:
    try:
        parsed = int(float(str(raw).strip()))
    except Exception:
        return None
    return parsed if parsed in (0, 1) else None


def _build_local_model_candidates() -> List[Tuple[str, Path]]:
    specs: List[Tuple[str, Path]] = []

    if MODEL_DIR_ENV:
        specs.append(("MODEL_DIR", Path(MODEL_DIR_ENV).expanduser()))

    if LOCAL_MODEL_PATH_ENV:
        specs.append(("LOCAL_MODEL_PATH", Path(LOCAL_MODEL_PATH_ENV).expanduser()))

    specs.extend(
        [
            ("auto:/app", Path("/app") / DEFAULT_LOCAL_MODEL_DIRNAME),
            ("auto:backend_dir", BACKEND_DIR / DEFAULT_LOCAL_MODEL_DIRNAME),
            ("auto:root_dir", ROOT_DIR / DEFAULT_LOCAL_MODEL_DIRNAME),
            ("auto:cwd", Path.cwd() / DEFAULT_LOCAL_MODEL_DIRNAME),
        ]
    )

    deduped: List[Tuple[str, Path]] = []
    seen = set()
    for source, path in specs:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((source, path))

    return deduped


def _resolve_local_model_path() -> Tuple[Path, str, List[str], bool]:
    specs = _build_local_model_candidates()
    candidate_lines = [f"{source} => {path}" for source, path in specs]

    for source, path in specs:
        if path.exists() and path.is_dir():
            return path.resolve(), source, candidate_lines, True

    if specs:
        fallback_path, fallback_source = specs[0][1], specs[0][0]
    else:
        fallback_path = Path("/app") / DEFAULT_LOCAL_MODEL_DIRNAME
        fallback_source = "auto:fallback"

    return fallback_path, fallback_source, candidate_lines, False


def _resolve_calibration_path(local_model_path: Path) -> Path:
    if CALIBRATION_PATH_ENV:
        return Path(CALIBRATION_PATH_ENV).expanduser()
    return local_model_path / "calibration.json"


def _missing_local_model_artifacts(
    local_model_path: Path, calibration_path: Path
) -> Tuple[List[str], List[str]]:
    missing_required: List[str] = []
    missing_optional: List[str] = []

    if not local_model_path.exists():
        missing_required.append(str(local_model_path))
        return missing_required, missing_optional

    for rel in REQUIRED_LOCAL_MODEL_FILES:
        target = local_model_path / rel
        if not target.exists():
            missing_required.append(str(target))

    if not any((local_model_path / rel).exists() for rel in TOKENIZER_ARTIFACT_CANDIDATES):
        missing_required.append(
            " or ".join(str(local_model_path / rel) for rel in TOKENIZER_ARTIFACT_CANDIDATES)
        )

    if not any((local_model_path / rel).exists() for rel in WEIGHT_CANDIDATES):
        missing_required.append(
            " or ".join(str(local_model_path / rel) for rel in WEIGHT_CANDIDATES)
        )

    for rel in OPTIONAL_LOCAL_MODEL_FILES:
        target = local_model_path / rel
        if not target.exists():
            missing_optional.append(str(target))

    if not calibration_path.exists():
        missing_optional.append(str(calibration_path))
    else:
        try:
            payload = json.loads(calibration_path.read_text(encoding="utf-8"))
            if payload.get("best_threshold", payload.get("threshold")) is None:
                missing_optional.append(f"{calibration_path} (missing best_threshold/threshold)")
        except Exception as exc:
            missing_optional.append(f"{calibration_path} (invalid json: {exc})")

    return missing_required, missing_optional


def _resolve_label_maps(model_config: Any) -> None:
    global ID2LABEL, LABEL2ID, NUM_LABELS, FAKTA_CLASS_ID, HOAX_CLASS_ID, LABEL_MAPPING_WARNINGS

    LABEL_MAPPING_WARNINGS = []

    raw_id2label = getattr(model_config, "id2label", None)
    parsed: Dict[int, str] = {}
    if isinstance(raw_id2label, dict) and raw_id2label:
        for key, value in raw_id2label.items():
            try:
                parsed[int(key)] = str(value)
            except Exception:
                continue

    if not parsed:
        parsed = {0: "Fakta", 1: "Hoaks"}
        LABEL_MAPPING_WARNINGS.append("id2label tidak valid; fallback ke default {0:Fakta,1:Hoaks}.")

    ID2LABEL = dict(sorted(parsed.items(), key=lambda item: item[0]))
    LABEL2ID = {name: idx for idx, name in ID2LABEL.items()}
    NUM_LABELS = len(ID2LABEL)

    hoax_candidates: List[int] = []
    fakta_candidates: List[int] = []

    for idx, label_name in ID2LABEL.items():
        normalized = _normalize_label(label_name)
        if any(token in normalized for token in HOAX_LABEL_TOKENS):
            hoax_candidates.append(idx)
        if any(token in normalized for token in FAKTA_LABEL_TOKENS):
            fakta_candidates.append(idx)

    if hoax_candidates:
        HOAX_CLASS_ID = hoax_candidates[0]
    else:
        HOAX_CLASS_ID = 1 if NUM_LABELS > 1 else 0
        LABEL_MAPPING_WARNINGS.append(
            f"Label Hoaks tidak terdeteksi dari config; fallback HOAX_CLASS_ID={HOAX_CLASS_ID}."
        )

    if fakta_candidates:
        FAKTA_CLASS_ID = fakta_candidates[0]
    else:
        FAKTA_CLASS_ID = 0 if HOAX_CLASS_ID != 0 else (1 if NUM_LABELS > 1 else 0)
        LABEL_MAPPING_WARNINGS.append(
            f"Label Fakta tidak terdeteksi dari config; fallback FAKTA_CLASS_ID={FAKTA_CLASS_ID}."
        )

    if FAKTA_CLASS_ID == HOAX_CLASS_ID and NUM_LABELS > 1:
        fallback_fakta = 0 if HOAX_CLASS_ID != 0 else 1
        LABEL_MAPPING_WARNINGS.append(
            "FAKTA_CLASS_ID dan HOAX_CLASS_ID sama; memaksa FAKTA_CLASS_ID ke kelas lain."
        )
        FAKTA_CLASS_ID = fallback_fakta

    if LABEL_MAPPING_WARNINGS:
        LOGGER.warning("Label mapping warnings: %s", LABEL_MAPPING_WARNINGS)


def _load_calibration(calibration_path: Path) -> None:
    global HOAX_THRESHOLD, CALIBRATION_LOADED, HOAX_THRESHOLD_SOURCE

    HOAX_THRESHOLD = DEFAULT_HOAX_THRESHOLD
    CALIBRATION_LOADED = False
    HOAX_THRESHOLD_SOURCE = "default"

    if not calibration_path.exists():
        HOAX_THRESHOLD_SOURCE = "default_no_calibration"
        LOGGER.info(
            "Calibration file tidak ditemukan: %s. Menggunakan threshold default %.3f",
            calibration_path,
            HOAX_THRESHOLD,
        )
        return

    try:
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
        candidate = payload.get("best_threshold", payload.get("threshold"))
        if candidate is None:
            raise ValueError("key best_threshold/threshold tidak ditemukan")

        value = float(candidate)
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"threshold out of range: {value}")

        HOAX_THRESHOLD = value
        CALIBRATION_LOADED = True
        HOAX_THRESHOLD_SOURCE = f"calibration:{calibration_path}"
        LOGGER.info("Calibration loaded | path=%s | hoax_threshold=%.3f", calibration_path, HOAX_THRESHOLD)
    except Exception as exc:
        HOAX_THRESHOLD_SOURCE = "default_invalid_calibration"
        LOGGER.warning(
            "Gagal membaca calibration file %s (%s). Menggunakan threshold default %.3f",
            calibration_path,
            exc,
            HOAX_THRESHOLD,
        )


def _load_classifier() -> None:
    global CLASSIFIER_MODEL, CLASSIFIER_TOKENIZER
    global MODEL_SOURCE, MODEL_LOAD_REASON
    global LOCAL_MODEL_PATH, LOCAL_MODEL_PATH_SOURCE, LOCAL_MODEL_CANDIDATES, LOCAL_MODEL_VALID
    global CALIBRATION_PATH
    global MISSING_REQUIRED_LOCAL_ARTIFACTS, MISSING_OPTIONAL_LOCAL_ARTIFACTS, MISSING_LOCAL_ARTIFACTS

    auth_kwargs = _hf_auth_kwargs()

    resolved_path, resolved_source, candidates, path_exists = _resolve_local_model_path()
    LOCAL_MODEL_PATH = resolved_path
    LOCAL_MODEL_PATH_SOURCE = resolved_source
    LOCAL_MODEL_CANDIDATES = candidates
    CALIBRATION_PATH = _resolve_calibration_path(LOCAL_MODEL_PATH)

    missing_required, missing_optional = _missing_local_model_artifacts(LOCAL_MODEL_PATH, CALIBRATION_PATH)
    MISSING_REQUIRED_LOCAL_ARTIFACTS = missing_required
    MISSING_OPTIONAL_LOCAL_ARTIFACTS = missing_optional
    MISSING_LOCAL_ARTIFACTS = list(missing_required)
    LOCAL_MODEL_VALID = len(missing_required) == 0

    local_exc: Optional[Exception] = None
    hub_exc: Optional[Exception] = None

    CLASSIFIER_MODEL = None
    CLASSIFIER_TOKENIZER = None

    if not path_exists:
        LOGGER.warning("Local model path tidak ditemukan dari kandidat: %s", LOCAL_MODEL_CANDIDATES)

    if not missing_required:
        try:
            LOGGER.info("Loading classifier from local path (%s): %s", LOCAL_MODEL_PATH_SOURCE, LOCAL_MODEL_PATH)
            CLASSIFIER_TOKENIZER = AutoTokenizer.from_pretrained(
                str(LOCAL_MODEL_PATH),
                local_files_only=True,
            )
            CLASSIFIER_MODEL = AutoModelForSequenceClassification.from_pretrained(
                str(LOCAL_MODEL_PATH),
                local_files_only=True,
                use_safetensors=(LOCAL_MODEL_PATH / "model.safetensors").exists(),
                low_cpu_mem_usage=True,
            )
            MODEL_SOURCE = "local"
            MODEL_LOAD_REASON = f"local load success from {LOCAL_MODEL_PATH}"
        except Exception as exc:
            local_exc = exc
            CLASSIFIER_MODEL = None
            CLASSIFIER_TOKENIZER = None
            LOGGER.warning("Local model load failed (%s): %s", LOCAL_MODEL_PATH, exc)
    else:
        MODEL_LOAD_REASON = "local artifacts incomplete"
        LOGGER.warning("Local model required artifacts tidak valid: %s", missing_required)

    if missing_optional:
        LOGGER.info("Local model optional artifacts tidak lengkap: %s", missing_optional)

    if CLASSIFIER_MODEL is None or CLASSIFIER_TOKENIZER is None:
        try:
            LOGGER.info("Loading classifier from Hub fallback: %s", MODEL_ID)
            CLASSIFIER_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID, **auth_kwargs)
            CLASSIFIER_MODEL = AutoModelForSequenceClassification.from_pretrained(
                MODEL_ID,
                use_safetensors=True,
                low_cpu_mem_usage=True,
                **auth_kwargs,
            )
            MODEL_SOURCE = "hub"
            details = []
            if missing_required:
                details.append("missing required local artifacts")
            if local_exc is not None:
                details.append(f"local load error: {local_exc}")
            MODEL_LOAD_REASON = "hub fallback due to " + ("; ".join(details) if details else "local unavailable")
        except Exception as exc:
            hub_exc = exc

    if CLASSIFIER_MODEL is None or CLASSIFIER_TOKENIZER is None:
        detail = []
        if missing_required:
            detail.append("missing required local artifacts: " + "; ".join(missing_required))
        if local_exc is not None:
            detail.append(f"local load failed: {local_exc}")
        if hub_exc is not None:
            detail.append(f"hub load failed: {hub_exc}")
        raise RuntimeError("Tidak dapat memuat model classifier. " + " | ".join(detail))

    CLASSIFIER_MODEL.to(DEVICE)
    CLASSIFIER_MODEL.eval()
    _resolve_label_maps(CLASSIFIER_MODEL.config)
    _load_calibration(CALIBRATION_PATH)

    LOGGER.info(
        (
            "Classifier ready | source=%s | reason=%s | device=%s | num_labels=%s | "
            "id2label=%s | label2id=%s | fakta_class_id=%s | hoax_class_id=%s | "
            "hoax_threshold=%.3f | threshold_source=%s | calibration_loaded=%s"
        ),
        MODEL_SOURCE,
        MODEL_LOAD_REASON,
        DEVICE,
        NUM_LABELS,
        ID2LABEL,
        LABEL2ID,
        FAKTA_CLASS_ID,
        HOAX_CLASS_ID,
        HOAX_THRESHOLD,
        HOAX_THRESHOLD_SOURCE,
        CALIBRATION_LOADED,
    )


def _predict_batch(sentences: List[str]) -> List[Dict[str, object]]:
    if not sentences:
        return []

    rows: List[Dict[str, object]] = []
    with torch.inference_mode():
        for start_idx in range(0, len(sentences), BATCH_SIZE):
            batch = sentences[start_idx : start_idx + BATCH_SIZE]
            encoded = CLASSIFIER_TOKENIZER(
                batch,
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
                pad_to_multiple_of=8 if DEVICE.type == "cuda" else None,
                return_tensors="pt",
            )
            encoded = {key: value.to(DEVICE) for key, value in encoded.items()}
            logits = CLASSIFIER_MODEL(**encoded).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu()
            argmax_ids = probs.argmax(dim=-1).tolist()

            for text, argmax_id, prob_tensor in zip(batch, argmax_ids, probs):
                values = prob_tensor.tolist()

                prob_hoax = values[HOAX_CLASS_ID] if HOAX_CLASS_ID < len(values) else 0.0
                prob_fakta = values[FAKTA_CLASS_ID] if FAKTA_CLASS_ID < len(values) else 0.0

                is_hoax = prob_hoax >= HOAX_THRESHOLD
                pred_id = HOAX_CLASS_ID if is_hoax else FAKTA_CLASS_ID
                label = "Hoaks" if is_hoax else "Fakta"
                confidence = max(prob_hoax, prob_fakta)
                color = "orange" if confidence < ORANGE_THRESHOLD else ("red" if label == "Hoaks" else "green")

                rows.append(
                    {
                        "text": text,
                        "label": label,
                        "pred_id": int(pred_id),
                        "argmax_id": int(argmax_id),
                        "prob_hoax": _float(prob_hoax),
                        "prob_fakta": _float(prob_fakta),
                        "confidence": _float(confidence),
                        "color": color,
                    }
                )
    return rows


def _split_paragraphs(text: str) -> List[str]:
    paragraphs = [p.strip() for p in PARAGRAPH_SPLIT_RE.split(text.strip()) if p.strip()]
    if paragraphs:
        return paragraphs
    stripped = text.strip()
    return [stripped] if stripped else []


def _split_sentences(paragraph: str) -> List[str]:
    normalized = re.sub(r"\s+", " ", paragraph).strip()
    if not normalized:
        return []
    sentences = [_normalize_unit_text(match.group(0).strip()) for match in SENTENCE_RE.finditer(normalized)]
    sentences = [sentence for sentence in sentences if sentence]
    return sentences or [_normalize_unit_text(normalized)]


def _find_dataset_dir_for_debug() -> Optional[Path]:
    candidates = [
        Path("/app/dataset"),
        BACKEND_DIR / "dataset",
        ROOT_DIR / "dataset",
        Path.cwd() / "dataset",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def _extract_text_from_row(row: Dict[str, str]) -> Tuple[str, str]:
    for column in DEBUG_TEXT_PRIORITY:
        value = row.get(column, "")
        if isinstance(value, str) and value.strip():
            return value.strip(), column
    return "", "UNKNOWN"


def _read_debug_sample(csv_path: Path, expected_label: Optional[int]) -> Optional[Dict[str, Any]]:
    if not csv_path.exists():
        return None

    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if expected_label is not None:
                label = _parse_binary_label(row.get("hoax"))
                if label != expected_label:
                    continue

            text, text_source = _extract_text_from_row(row)
            if not text:
                continue

            return {
                "source_file": csv_path.name,
                "text_source": text_source,
                "gold_label": expected_label,
                "text": text,
            }

    return None


def _build_debug_dataset_samples() -> Dict[str, Any]:
    dataset_dir = _find_dataset_dir_for_debug()
    if dataset_dir is None:
        return {
            "status": "dataset_not_found",
            "dataset_dir": None,
            "message": "Folder dataset tidak ditemukan di runtime backend.",
            "samples": [],
        }

    hoax_sample = _read_debug_sample(dataset_dir / DEBUG_HOAX_FILE, expected_label=1)

    fakta_sample = None
    for filename in DEBUG_FAKTA_FILES:
        fakta_sample = _read_debug_sample(dataset_dir / filename, expected_label=0)
        if fakta_sample is not None:
            break

    available_samples = []
    ordered_pairs: List[Tuple[str, Dict[str, Any]]] = []

    if fakta_sample is not None:
        ordered_pairs.append(("fakta", fakta_sample))
    if hoax_sample is not None:
        ordered_pairs.append(("hoaks", hoax_sample))

    predicted_rows: List[Dict[str, Any]] = []
    if ordered_pairs and CLASSIFIER_MODEL is not None and CLASSIFIER_TOKENIZER is not None:
        texts = [item[1]["text"] for item in ordered_pairs]
        predicted_rows = _predict_batch(texts)

    for idx, (kind, sample) in enumerate(ordered_pairs):
        pred = predicted_rows[idx] if idx < len(predicted_rows) else None
        available_samples.append(
            {
                "kind": kind,
                "gold_label": sample["gold_label"],
                "source_file": sample["source_file"],
                "text_source": sample["text_source"],
                "text_preview": _preview_text(sample["text"]),
                "prediction": pred,
            }
        )

    return {
        "status": "ok" if available_samples else "samples_not_found",
        "dataset_dir": str(dataset_dir),
        "samples": available_samples,
    }


def _run_startup_sanity() -> None:
    global STARTUP_SANITY

    try:
        predictions = _predict_batch(STARTUP_SANITY_SENTENCES)
        pred_ids = sorted({int(row["pred_id"]) for row in predictions})
        if len(predictions) != len(STARTUP_SANITY_SENTENCES):
            STARTUP_SANITY = {
                "checked": True,
                "status": "warning",
                "message": "startup sanity: jumlah output tidak sesuai input.",
            }
        elif len(pred_ids) < 2:
            STARTUP_SANITY = {
                "checked": True,
                "status": "warning",
                "message": "startup sanity: prediksi sampel hanya satu kelas.",
                "pred_ids": pred_ids,
                "num_samples_checked": len(predictions),
            }
        else:
            STARTUP_SANITY = {
                "checked": True,
                "status": "ok",
                "message": "startup sanity: kedua kelas muncul pada sampel uji statis.",
                "pred_ids": pred_ids,
                "num_samples_checked": len(predictions),
            }
    except Exception as exc:
        STARTUP_SANITY = {
            "checked": True,
            "status": "warning",
            "message": f"startup sanity gagal dijalankan: {exc}",
        }

    LOGGER.info("Startup sanity: %s", STARTUP_SANITY)


def _model_meta_payload() -> Dict[str, Any]:
    return {
        "source": MODEL_SOURCE,
        "model_id": MODEL_ID,
        "model_path": str(LOCAL_MODEL_PATH),
        "model_path_source": LOCAL_MODEL_PATH_SOURCE,
        "analysis_mode": "sentence_split_doc_model",
        "max_length": MAX_LENGTH,
        "num_labels": NUM_LABELS,
        "fakta_class_id": int(FAKTA_CLASS_ID),
        "hoax_class_id": int(HOAX_CLASS_ID),
        "hoax_threshold": float(HOAX_THRESHOLD),
        "hoax_threshold_source": HOAX_THRESHOLD_SOURCE,
        "calibration_loaded": bool(CALIBRATION_LOADED),
        "id2label": {str(k): v for k, v in ID2LABEL.items()},
        "label2id": LABEL2ID,
        "label_mapping_warnings": LABEL_MAPPING_WARNINGS,
    }


@app.on_event("startup")
def startup_event() -> None:
    _load_classifier()
    _run_startup_sanity()


@app.get("/")
def root() -> Dict[str, object]:
    return {
        "status": "ok",
        "message": "Hoax backend is running.",
        "endpoints": {
            "health": "/health",
            "debug": "/debug",
            "analyze": "/analyze (POST)",
            "docs": "/docs",
        },
    }


@app.get("/health")
def health() -> Dict[str, object]:
    return {
        "status": "ok",
        "artifact_validation_mode": ARTIFACT_VALIDATION_MODE,
        "model_source": MODEL_SOURCE,
        "model_load_reason": MODEL_LOAD_REASON,
        "model_id": MODEL_ID,
        "local_model_path": str(LOCAL_MODEL_PATH),
        "local_model_path_source": LOCAL_MODEL_PATH_SOURCE,
        "local_model_candidates": LOCAL_MODEL_CANDIDATES,
        "local_model_valid": bool(LOCAL_MODEL_VALID),
        "missing_required_artifacts": MISSING_REQUIRED_LOCAL_ARTIFACTS,
        "missing_optional_artifacts": MISSING_OPTIONAL_LOCAL_ARTIFACTS,
        "missing_local_artifacts": MISSING_LOCAL_ARTIFACTS,
        "calibration_path": str(CALIBRATION_PATH),
        "hoax_threshold": float(HOAX_THRESHOLD),
        "hoax_threshold_source": HOAX_THRESHOLD_SOURCE,
        "calibration_loaded": bool(CALIBRATION_LOADED),
        "num_labels": NUM_LABELS,
        "id2label": {str(k): v for k, v in ID2LABEL.items()},
        "label2id": LABEL2ID,
        "fakta_class_id": int(FAKTA_CLASS_ID),
        "hoax_class_id": int(HOAX_CLASS_ID),
        "label_mapping_warnings": LABEL_MAPPING_WARNINGS,
        "startup_sanity": STARTUP_SANITY,
        "topics_enabled": bool(TOPICS_PAYLOAD.get("enabled", False)),
        "topics": TOPICS_PAYLOAD,
    }


@app.get("/debug")
def debug() -> Dict[str, Any]:
    debug_samples = _build_debug_dataset_samples()
    return {
        "status": "ok",
        "model": _model_meta_payload(),
        "model_source": MODEL_SOURCE,
        "model_load_reason": MODEL_LOAD_REASON,
        "topics": TOPICS_PAYLOAD,
        "startup_sanity": STARTUP_SANITY,
        "dataset_debug": debug_samples,
    }


@app.post("/analyze")
def analyze(payload: AnalyzeRequest) -> Dict[str, object]:
    if CLASSIFIER_MODEL is None or CLASSIFIER_TOKENIZER is None:
        raise HTTPException(status_code=503, detail="Model classifier belum siap.")

    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Field 'text' tidak boleh kosong.")
    if len(text) > MAX_INPUT_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"Input terlalu panjang ({len(text)} chars). Maksimum {MAX_INPUT_CHARS} chars.",
        )

    paragraphs_raw = _split_paragraphs(text)
    sentence_counts: List[int] = []
    flat_sentences: List[str] = []
    for paragraph_text in paragraphs_raw:
        sentences = _split_sentences(paragraph_text)
        sentence_counts.append(len(sentences))
        flat_sentences.extend(sentences)

    classified = _predict_batch(flat_sentences)

    paragraph_responses = []
    total_sentences = 0
    total_hoax = 0
    total_fakta = 0
    total_low_conf = 0
    cursor = 0

    for paragraph_idx, sentence_count in enumerate(sentence_counts):
        sentence_rows = classified[cursor : cursor + sentence_count]
        cursor += sentence_count

        sentence_items = []
        paragraph_hoax = 0
        paragraph_fakta = 0
        paragraph_low = 0
        conf_values: List[float] = []
        hoax_probs: List[float] = []

        for sentence_idx, row in enumerate(sentence_rows):
            if row["label"] == "Hoaks":
                paragraph_hoax += 1
            else:
                paragraph_fakta += 1
            if row["confidence"] < ORANGE_THRESHOLD:
                paragraph_low += 1

            conf_values.append(float(row["confidence"]))
            hoax_probs.append(float(row["prob_hoax"]))
            sentence_items.append(
                {
                    "sentence_index": sentence_idx,
                    "text": row["text"],
                    "label": row["label"],
                    "pred_id": row["pred_id"],
                    "argmax_id": row["argmax_id"],
                    "prob_hoax": row["prob_hoax"],
                    "prob_fakta": row["prob_fakta"],
                    "confidence": row["confidence"],
                    "color": row["color"],
                }
            )

        paragraph_responses.append(
            {
                "paragraph_index": paragraph_idx,
                "sentences": sentence_items,
                "paragraph_summary": {
                    "hoax_sentences": paragraph_hoax,
                    "fakta_sentences": paragraph_fakta,
                    "avg_confidence": _float(sum(conf_values) / len(conf_values)) if conf_values else 0.0,
                    "max_hoax_prob": _float(max(hoax_probs)) if hoax_probs else 0.0,
                },
            }
        )

        total_sentences += len(sentence_items)
        total_hoax += paragraph_hoax
        total_fakta += paragraph_fakta
        total_low_conf += paragraph_low

    return {
        "model": _model_meta_payload(),
        "summary": {
            "num_paragraphs": len(paragraph_responses),
            "num_sentences": total_sentences,
            "hoax_sentences": total_hoax,
            "fakta_sentences": total_fakta,
            "low_conf_sentences": total_low_conf,
        },
        "paragraphs": paragraph_responses,
        "topics": TOPICS_PAYLOAD,
    }
