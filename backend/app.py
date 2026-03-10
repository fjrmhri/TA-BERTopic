import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("hoax-backend")

MODEL_ID = os.getenv("MODEL_ID", "fjrmhri/Deteksi_Hoax_IndoBERT_BERTopic")
HF_TOKEN = os.getenv("HF_TOKEN")
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN")
ORANGE_THRESHOLD = float(os.getenv("ORANGE_THRESHOLD", "0.65"))
DEFAULT_HOAX_THRESHOLD = float(os.getenv("HOAX_THRESHOLD", "0.5"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "256"))
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "50000"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))

ROOT_DIR = Path(__file__).resolve().parents[1]
LOCAL_MODEL_PATH = Path(
    os.getenv("LOCAL_MODEL_PATH", str(ROOT_DIR / "indobert_hoax_model_v1"))
)
CALIBRATION_PATH = Path(
    os.getenv("CALIBRATION_PATH", str(LOCAL_MODEL_PATH / "calibration.json"))
)

PARAGRAPH_SPLIT_RE = re.compile(r"(?:\r?\n){2,}")
SENTENCE_RE = re.compile(r"[^.!?]+(?:[.!?]+(?:[\"”’)\]]+)?)|[^.!?]+$")
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
REQUIRED_LOCAL_MODEL_FILES = [
    "config.json",
    "tokenizer_config.json",
]
TOKENIZER_ARTIFACT_CANDIDATES = ("tokenizer.json", "vocab.txt")
OPTIONAL_LOCAL_MODEL_FILES = [
    "special_tokens_map.json",
]
WEIGHT_CANDIDATES = ("model.safetensors", "pytorch_model.bin")
ARTIFACT_VALIDATION_MODE = "local-or-hub"
STARTUP_SANITY_SENTENCES = [
    "Beredar unggahan yang mengklaim ada rekrutmen CPNS fiktif dan masyarakat diminta transfer biaya pendaftaran.",
    "PT Transjakarta melakukan modifikasi layanan pada empat rute untuk meningkatkan kenyamanan penumpang.",
    "Video lama diklaim sebagai kericuhan terbaru dan narasi itu ramai dibagikan di media sosial.",
    "Pemerintah daerah akan membahas penertiban izin fasilitas olahraga di area permukiman.",
]


class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="Teks input multi paragraf.")


app = FastAPI(title="Hoax Sentence Analyzer API", version="3.0.0")

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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ID2LABEL: Dict[int, str] = {0: "Fakta", 1: "Hoaks"}
LABEL2ID: Dict[str, int] = {"Fakta": 0, "Hoaks": 1}
NUM_LABELS = 2
FAKTA_CLASS_ID = 0
HOAX_CLASS_ID = 1
HOAX_THRESHOLD = DEFAULT_HOAX_THRESHOLD
CALIBRATION_LOADED = False
LOCAL_MODEL_VALID = False
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
    return cleaned.strip(" -:;,.")


def _missing_local_model_artifacts() -> tuple[List[str], List[str]]:
    missing_required: List[str] = []
    missing_optional: List[str] = []

    if not LOCAL_MODEL_PATH.exists():
        missing_required.append(str(LOCAL_MODEL_PATH))
        return missing_required, missing_optional

    for rel in REQUIRED_LOCAL_MODEL_FILES:
        target = LOCAL_MODEL_PATH / rel
        if not target.exists():
            missing_required.append(str(target))

    if not any((LOCAL_MODEL_PATH / rel).exists() for rel in TOKENIZER_ARTIFACT_CANDIDATES):
        missing_required.append(
            " or ".join(str(LOCAL_MODEL_PATH / rel) for rel in TOKENIZER_ARTIFACT_CANDIDATES)
        )

    if not any((LOCAL_MODEL_PATH / rel).exists() for rel in WEIGHT_CANDIDATES):
        missing_required.append(
            " or ".join(str(LOCAL_MODEL_PATH / rel) for rel in WEIGHT_CANDIDATES)
        )

    for rel in OPTIONAL_LOCAL_MODEL_FILES:
        target = LOCAL_MODEL_PATH / rel
        if not target.exists():
            missing_optional.append(str(target))

    if not CALIBRATION_PATH.exists():
        missing_optional.append(str(CALIBRATION_PATH))
    else:
        try:
            payload = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
            if payload.get("best_threshold", payload.get("threshold")) is None:
                missing_optional.append(f"{CALIBRATION_PATH} (missing best_threshold/threshold)")
        except Exception as exc:
            missing_optional.append(f"{CALIBRATION_PATH} (invalid json: {exc})")

    return missing_required, missing_optional


def _resolve_label_maps(model_config) -> None:
    global ID2LABEL, LABEL2ID, NUM_LABELS, FAKTA_CLASS_ID, HOAX_CLASS_ID

    raw_id2label = getattr(model_config, "id2label", None)
    if isinstance(raw_id2label, dict) and raw_id2label:
        parsed = {}
        for key, value in raw_id2label.items():
            try:
                parsed[int(key)] = str(value)
            except Exception:
                continue
        if parsed:
            ID2LABEL = dict(sorted(parsed.items(), key=lambda item: item[0]))
        else:
            ID2LABEL = {0: "Fakta", 1: "Hoaks"}
    else:
        ID2LABEL = {0: "Fakta", 1: "Hoaks"}

    LABEL2ID = {name: idx for idx, name in ID2LABEL.items()}
    NUM_LABELS = len(ID2LABEL)

    hoax_candidates = []
    fakta_candidates = []
    for idx, label_name in ID2LABEL.items():
        normalized = _normalize_label(label_name)
        if any(token in normalized for token in HOAX_LABEL_TOKENS):
            hoax_candidates.append(idx)
        if any(token in normalized for token in FAKTA_LABEL_TOKENS):
            fakta_candidates.append(idx)

    HOAX_CLASS_ID = hoax_candidates[0] if hoax_candidates else (1 if NUM_LABELS > 1 else 0)
    if fakta_candidates:
        FAKTA_CLASS_ID = fakta_candidates[0]
    else:
        FAKTA_CLASS_ID = 0 if HOAX_CLASS_ID != 0 else (1 if NUM_LABELS > 1 else 0)


def _load_calibration() -> None:
    global HOAX_THRESHOLD, CALIBRATION_LOADED

    HOAX_THRESHOLD = DEFAULT_HOAX_THRESHOLD
    CALIBRATION_LOADED = False
    if not CALIBRATION_PATH.exists():
        LOGGER.info(
            "Calibration file tidak ditemukan: %s. Menggunakan threshold default %.3f",
            CALIBRATION_PATH,
            HOAX_THRESHOLD,
        )
        return

    try:
        payload = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
        candidate = payload.get("best_threshold", payload.get("threshold"))
        if candidate is None:
            raise ValueError("key best_threshold/threshold tidak ditemukan")
        value = float(candidate)
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"threshold out of range: {value}")
        HOAX_THRESHOLD = value
        CALIBRATION_LOADED = True
        LOGGER.info("Calibration loaded | path=%s | hoax_threshold=%.3f", CALIBRATION_PATH, HOAX_THRESHOLD)
    except Exception as exc:
        LOGGER.warning(
            "Gagal membaca calibration file %s (%s). Menggunakan threshold default %.3f",
            CALIBRATION_PATH,
            exc,
            HOAX_THRESHOLD,
        )


def _load_classifier() -> None:
    global CLASSIFIER_MODEL, CLASSIFIER_TOKENIZER, MODEL_SOURCE, LOCAL_MODEL_VALID
    global MISSING_REQUIRED_LOCAL_ARTIFACTS, MISSING_OPTIONAL_LOCAL_ARTIFACTS, MISSING_LOCAL_ARTIFACTS

    auth_kwargs = _hf_auth_kwargs()
    missing_required, missing_optional = _missing_local_model_artifacts()
    MISSING_REQUIRED_LOCAL_ARTIFACTS = missing_required
    MISSING_OPTIONAL_LOCAL_ARTIFACTS = missing_optional
    MISSING_LOCAL_ARTIFACTS = list(missing_required)
    LOCAL_MODEL_VALID = len(missing_required) == 0

    local_exc = None
    hub_exc = None

    if not missing_required:
        try:
            LOGGER.info("Loading classifier from local primary: %s", LOCAL_MODEL_PATH)
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
        except Exception as exc:
            local_exc = exc
            CLASSIFIER_TOKENIZER = None
            CLASSIFIER_MODEL = None
            LOGGER.warning("Local primary load failed: %s", exc)
    else:
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
    _load_calibration()
    LOGGER.info(
        (
            "Classifier ready | source=%s | device=%s | num_labels=%s | "
            "id2label=%s | label2id=%s | fakta_class_id=%s | hoax_class_id=%s | "
            "hoax_threshold=%.3f | calibration_loaded=%s"
        ),
        MODEL_SOURCE,
        DEVICE,
        NUM_LABELS,
        ID2LABEL,
        LABEL2ID,
        FAKTA_CLASS_ID,
        HOAX_CLASS_ID,
        HOAX_THRESHOLD,
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
            pred_ids = probs.argmax(dim=-1).tolist()

            for text, pred_id, prob_tensor in zip(batch, pred_ids, probs):
                values = prob_tensor.tolist()
                prob_hoax = values[HOAX_CLASS_ID] if HOAX_CLASS_ID < len(values) else 0.0
                prob_fakta = values[FAKTA_CLASS_ID] if FAKTA_CLASS_ID < len(values) else 0.0
                threshold_pred_id = 1 if prob_hoax >= HOAX_THRESHOLD else 0
                label = "Hoaks" if threshold_pred_id == 1 else "Fakta"
                confidence = max(prob_hoax, prob_fakta)
                color = "orange" if confidence < ORANGE_THRESHOLD else ("red" if label == "Hoaks" else "green")
                rows.append(
                    {
                        "text": text,
                        "label": label,
                        "pred_id": int(threshold_pred_id),
                        "argmax_id": int(pred_id),
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
        "local_model_valid": bool(LOCAL_MODEL_VALID),
        "missing_required_artifacts": MISSING_REQUIRED_LOCAL_ARTIFACTS,
        "missing_optional_artifacts": MISSING_OPTIONAL_LOCAL_ARTIFACTS,
        "missing_local_artifacts": MISSING_LOCAL_ARTIFACTS,
        "model_id": MODEL_ID,
        "num_labels": NUM_LABELS,
        "id2label": {str(k): v for k, v in ID2LABEL.items()},
        "label2id": LABEL2ID,
        "fakta_class_id": int(FAKTA_CLASS_ID),
        "hoax_class_id": int(HOAX_CLASS_ID),
        "hoax_threshold": float(HOAX_THRESHOLD),
        "calibration_loaded": bool(CALIBRATION_LOADED),
        "startup_sanity": STARTUP_SANITY,
        "topics": TOPICS_PAYLOAD,
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
        "model": {
            "source": MODEL_SOURCE,
            "model_id": MODEL_ID,
            "analysis_mode": "sentence_split_doc_model",
            "max_length": MAX_LENGTH,
            "num_labels": NUM_LABELS,
            "fakta_class_id": int(FAKTA_CLASS_ID),
            "hoax_class_id": int(HOAX_CLASS_ID),
            "hoax_threshold": float(HOAX_THRESHOLD),
            "calibration_loaded": bool(CALIBRATION_LOADED),
            "id2label": {str(k): v for k, v in ID2LABEL.items()},
            "label2id": LABEL2ID,
        },
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
