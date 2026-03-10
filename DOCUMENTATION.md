# Documentation

## Ringkasan Sistem
Repositori ini sekarang diposisikan sebagai pipeline klasifikasi hoaks berbasis `indolem/indobert-base-uncased` dengan:

- training notebook utama di [Deteksi_Hoax_V1.ipynb](/d:/TA/code/Berita_Hoax_BERTopic/Deteksi_Hoax_V1.ipynb)
- notebook legacy di [Deteksi_Hoax.ipynb](/d:/TA/code/Berita_Hoax_BERTopic/Deteksi_Hoax.ipynb)
- backend FastAPI untuk Hugging Face Spaces di [backend/app.py](/d:/TA/code/Berita_Hoax_BERTopic/backend/app.py)
- frontend statis untuk Vercel di [frontend/index.html](/d:/TA/code/Berita_Hoax_BERTopic/frontend/index.html), [frontend/app.js](/d:/TA/code/Berita_Hoax_BERTopic/frontend/app.js), dan [frontend/styles.css](/d:/TA/code/Berita_Hoax_BERTopic/frontend/styles.css)

Keputusan final patch:

- notebook utama memakai `dataset/` lokal
- notebook utama diperbarui ke `Deteksi_Hoax_V1.ipynb`
- backbone tetap IndoBERT
- NER dihapus penuh dari backend dan frontend
- sentence-level dipakai pada inferensi saja, bukan sebagai retraining sentence-level
- BERTopic dinilai feasible sebagai modul terpisah, tetapi tidak diaktifkan pada patch ini agar deployment tetap ringan

## Dataset
### Skema Konsisten
Keempat CSV di `dataset/` memakai skema yang sama: `url`, `judul`, `tanggal`, `isi_berita`, `Narasi`, `Clean Narasi`, `hoax`, `summary`, `source`.

Semua kolom penting memiliki missing rate `0.0` pada seluruh file:

- `url`
- `judul`
- `tanggal`
- `isi_berita`
- `Narasi`
- `Clean Narasi`
- `summary`
- `hoax`
- `source`

### Jumlah Baris dan Label
| File | Rows | Label 0 | Label 1 |
| --- | ---: | ---: | ---: |
| `dataset/data_nonhoaks_cnn.csv` | 807 | 807 | 0 |
| `dataset/data_nonhoaks_detik.csv` | 18,711 | 18,711 | 0 |
| `dataset/data_nonhoaks_kompas.csv` | 5,136 | 5,136 | 0 |
| `dataset/data_hoaks_turnbackhoaks.csv` | 12,353 | 0 | 12,353 |
| Total | 37,007 | 24,654 | 12,353 |

### Statistik Panjang Teks per File
Format: `min / mean / p95` dalam jumlah karakter.

| File | `judul` | `summary` | `Clean Narasi` | `Narasi` | `isi_berita` |
| --- | --- | --- | --- | --- | --- |
| `data_nonhoaks_cnn.csv` | `19 / 59.77 / 70` | `135 / 388.18 / 576.7` | `405 / 5814.16 / 10449.1` | `515 / 8146.49 / 15095.6` | `515 / 8146.49 / 15095.6` |
| `data_nonhoaks_detik.csv` | `3 / 62.51 / 77` | `10 / 249.35 / 477.5` | `123 / 2732.61 / 5592.5` | `201 / 3840.56 / 8113.5` | `201 / 3840.56 / 8113.5` |
| `data_nonhoaks_kompas.csv` | `29 / 73.21 / 96` | `50 / 276.24 / 404` | `1027 / 3860.81 / 6036` | `1515 / 5405.41 / 8022.25` | `1515 / 5405.41 / 8022.25` |
| `data_hoaks_turnbackhoaks.csv` | `20 / 68.5 / 110` | `15 / 463.77 / 802` | `11 / 354.52 / 592` | `15 / 463.77 / 802` | `2126 / 3966.48 / 5655.6` |

### Implikasi ke `max_length`
Statistik token dihitung dengan tokenizer `indolem/indobert-base-uncased` pada seluruh 37,007 baris:

| Kolom | Mean Token | P95 Token | Share `>256` | Implikasi |
| --- | ---: | ---: | ---: | --- |
| `summary` | 70.0 | 146.0 | 0.42% | Paling aman untuk `max_length=256` |
| `judul` | 15.76 | 23.0 | 0.00% | Terlalu pendek bila dipakai sendiri |
| `Clean Narasi` | 411.65 | 1035.0 | 62.09% | Truncation berat |
| `Narasi` | 601.61 | 1525.0 | 64.19% | Truncation sangat berat |
| `isi_berita` | 905.22 | 1601.7 | 97.38% | Hampir pasti terpotong |

Kesimpulan dataset:

- kolom yang paling konsisten untuk training adalah `summary`
- `Clean Narasi` dan `Narasi` tetap berguna sebagai fallback, tetapi terlalu panjang bila dijadikan prioritas utama
- `isi_berita` tidak cocok untuk setup `max_length=256` yang ingin aman di Colab T4

## Analisis Notebook
### Deteksi_Hoax_V1.ipynb
Notebook utama saat ini. Struktur cell disusun ulang untuk Colab T4:

- install/import dependensi inti tanpa KaggleHub
- resolusi dataset lokal + kandidat Google Drive
- training aman T4 (`fp16`, `auto_find_batch_size`, `gradient_checkpointing`, dynamic padding, grad accumulation)
- evaluasi ringkas validation/test
- kalibrasi `best_threshold` ke `calibration.json`
- demo inferensi multi-paragraf per kalimat
- upload folder artefak ke repo HF `fjrmhri/Deteksi_Hoax_IndoBERT_BERTopic`

### Deteksi_Hoax.ipynb (legacy)
Pra-patch yang dibaca saat analisis:

- Cell 1 memakai path `Summarized_CNN.csv`, `Summarized_Detik.csv`, `Summarized_Kompas.csv`, `Summarized_TurnBackHoax.csv`, dan `Summarized_2020+.csv`
- Cell 2 mengunduh dataset dari KaggleHub
- Cell 4 memilih teks dengan urutan `Clean Narasi -> Narasi -> isi_berita -> judul`
- Cell 5 memakai split stratified `70/15/15` dan oversampling minoritas hanya pada train
- Cell 8 memakai `max_length=256`, `train_batch_size=96`, `eval_batch_size=384`, `gradient_accumulation_steps=2`, `fp16` bila CUDA tersedia, tanpa `evaluation_strategy`, tanpa `gradient_checkpointing`, tanpa `auto_find_batch_size`
- Notebook tidak punya cell inferensi multi-paragraf per kalimat

Setelah patch:

- Cell 1 sekarang mendefinisikan kandidat folder `dataset/` lokal dan setelan training aman untuk T4
- Cell 2 memvalidasi empat CSV lokal
- Cell 3 memuat empat file lokal tanpa KaggleHub
- Cell 4 memprioritaskan `summary`
- Cell 8 menambahkan `auto_find_batch_size`, `gradient_checkpointing`, `early stopping`, `save/eval per epoch`, `eval_accumulation_steps`, dan dynamic padding via `DataCollatorWithPadding`
- Cell 12 menambahkan demo input multi-paragraf -> output label per kalimat

### Deteksi_Hoax_NER_Optimized.ipynb
Bukti yang relevan:

- Cell 4 tetap memakai backbone `indolem/indobert-base-uncased`
- Cell 5 membaca `data/processed/train.csv`, `val.csv`, `test.csv`, plus `leakage_audit.json`
- Cell 5 mengharuskan kolom `text`, `label`, `source`, `url_hash`, `title_hash`, `unit_type`
- Cell 7 menunjukkan optimasi training yang lebih aman untuk T4: `max_length=192`, batch lebih kecil, `auto_find_batch_size`, `gradient_checkpointing`, `EarlyStoppingCallback`
- Cell 8 sudah punya pola inferensi `paragraph -> sentence -> classify`

Batas bukti:

- `scripts/build_dataset.py` dan folder `data/processed/` tidak ada di workspace saat analisis
- karena itu, cara builder menghasilkan `unit_type` dan label sentence-level adalah `UNKNOWN`

## Jawaban Q1-Q6
### Q1. Migrasi dataset
Kesimpulan:
Ya, `Deteksi_Hoax_V1.ipynb` memakai `dataset/` lokal tanpa mengganti backbone IndoBERT atau arsitektur klasifikasi.

Bukti:

- skema keempat CSV lokal sudah konsisten
- pra-patch notebook sudah memuat struktur kolom yang sama di cell 3 dan melakukan mapping label di cell 4
- patch final di [Deteksi_Hoax_V1.ipynb](/d:/TA/code/Berita_Hoax_BERTopic/Deteksi_Hoax_V1.ipynb) memuat alur dataset lokal + prioritas kolom teks

Risiko:

- jika tetap memakai prioritas lama `Clean Narasi`, lebih dari 62% sampel akan melebihi 256 token
- file `Summarized_2020+.csv` lama tidak tersedia di workspace ini

Rekomendasi:

- pakai empat file lokal saja
- map label langsung dari kolom `hoax`
- gunakan prioritas `summary -> Clean Narasi -> Narasi -> isi_berita -> judul`

Perubahan minimal yang dipakai:

- path:
  `dataset/data_nonhoaks_cnn.csv`
  `dataset/data_nonhoaks_detik.csv`
  `dataset/data_nonhoaks_kompas.csv`
  `dataset/data_hoaks_turnbackhoaks.csv`
- label:
  `hoax=0` untuk non-hoax
  `hoax=1` untuk hoax
- text:
  `summary` sebagai sumber utama

### Q2. Integrasi BERTopic
Kesimpulan:
Feasible, tetapi paling aman dijadikan post-processing CPU-only yang opsional. Patch ini sengaja tidak mengaktifkannya.

Bukti:

- backend saat ini hanya membutuhkan classifier di [backend/app.py](/d:/TA/code/Berita_Hoax_BERTopic/backend/app.py)
- dependensi backend aktif di [backend/requirements.txt](/d:/TA/code/Berita_Hoax_BERTopic/backend/requirements.txt) belum mencakup BERTopic
- kontrak response final sudah menyisakan `topics` placeholder di [backend/app.py](/d:/TA/code/Berita_Hoax_BERTopic/backend/app.py) dan render opsional di [frontend/app.js](/d:/TA/code/Berita_Hoax_BERTopic/frontend/app.js)

Risiko:

- `bertopic`, `sentence-transformers`, `umap-learn`, dan `hdbscan` menambah ukuran image serta cold start
- inferensi topik online akan menambah latency bila embedding dihitung per request

Rekomendasi:

- fit BERTopic offline pada corpus train
- simpan artifact topik terpisah dari classifier
- jalankan inferensi topik di CPU
- cache berdasarkan hash teks/paragraf
- default `disabled`, aktifkan hanya bila artefak tersedia

### Q3. Deteksi tingkat kalimat
Kesimpulan:
Feasible. Pilihan paling aman saat ini adalah sentence-level pada inferensi, bukan retraining sentence-level.

Bukti:

- `Deteksi_Hoax_NER_Optimized.ipynb` cell 8 sudah menunjukkan pola `split_paragraphs` dan `split_sentences`
- patch final di [Deteksi_Hoax.ipynb](/d:/TA/code/Berita_Hoax_BERTopic/Deteksi_Hoax.ipynb) cell 12 menambahkan demo inferensi multi-paragraf per kalimat
- backend final di [backend/app.py](/d:/TA/code/Berita_Hoax_BERTopic/backend/app.py) line 469 mengembalikan hasil per kalimat

Risiko:

- opsi minimal memakai model doc-level untuk kalimat individual, sehingga ada mismatch konteks
- opsi optimal membutuhkan builder sentence-level yang tidak bisa diverifikasi penuh karena `scripts/build_dataset.py` tidak ada

Rekomendasi:

- gunakan inferensi per kalimat sebagai heuristic highlight
- pertahankan training doc-level agar akurasi baseline tidak terganggu
- bila ingin sentence-level training sungguhan, buat fase terpisah setelah builder dan label sentence-level dapat diverifikasi

Perbandingan opsi:

- Opsi minimal:
  split kalimat saat inferensi, jalankan classifier doc-level per kalimat
- Opsi optimal:
  buat processed dataset sentence-level dan fine-tune lagi tanpa NER

Pilihan final:

- Opsi minimal untuk implementasi sekarang

### Q4. Perubahan backend + frontend
Kesimpulan:
Feasible dan sudah dipatch tanpa NER.

Bukti:

- request backend sekarang hanya `text` di [backend/app.py](/d:/TA/code/Berita_Hoax_BERTopic/backend/app.py#L73)
- inferensi dibatch di [backend/app.py](/d:/TA/code/Berita_Hoax_BERTopic/backend/app.py#L327)
- endpoint `/analyze` membangun response `summary -> paragraphs -> sentences` di [backend/app.py](/d:/TA/code/Berita_Hoax_BERTopic/backend/app.py#L469)
- frontend hanya merender summary, highlight, confidence, dan topik opsional di [frontend/app.js](/d:/TA/code/Berita_Hoax_BERTopic/frontend/app.js), [frontend/index.html](/d:/TA/code/Berita_Hoax_BERTopic/frontend/index.html), [frontend/styles.css](/d:/TA/code/Berita_Hoax_BERTopic/frontend/styles.css)

Risiko:

- jika model lokal tidak tersedia, backend fallback ke Hub dan cold start tetap ada
- topic panel saat ini hanyalah hook opsional, bukan inference BERTopic aktif

Rekomendasi:

- muat model sekali saat startup
- pertahankan `torch.inference_mode()` dan batching
- kirim response tanpa NER
- hide panel topik ketika `topics.enabled=false`

Kontrak API final:

Request:

```json
{
  "text": "teks multi paragraf"
}
```

Response:

```json
{
  "model": {
    "source": "local|hub",
    "model_id": "fjrmhri/Deteksi_Hoax_IndoBERT_BERTopic",
    "analysis_mode": "sentence_split_doc_model",
    "max_length": 256,
    "num_labels": 2,
    "fakta_class_id": 0,
    "hoax_class_id": 1,
    "hoax_threshold": 0.5,
    "calibration_loaded": false,
    "id2label": {"0": "Fakta", "1": "Hoaks"},
    "label2id": {"Fakta": 0, "Hoaks": 1}
  },
  "summary": {
    "num_paragraphs": 2,
    "num_sentences": 4,
    "hoax_sentences": 1,
    "fakta_sentences": 3,
    "low_conf_sentences": 1
  },
  "paragraphs": [
    {
      "paragraph_index": 0,
      "sentences": [
        {
          "sentence_index": 0,
          "text": "....",
          "label": "Hoaks",
          "pred_id": 1,
          "argmax_id": 1,
          "prob_hoax": 0.87,
          "prob_fakta": 0.13,
          "confidence": 0.87,
          "color": "red"
        }
      ],
      "paragraph_summary": {
        "hoax_sentences": 1,
        "fakta_sentences": 0,
        "avg_confidence": 0.87,
        "max_hoax_prob": 0.87
      }
    }
  ],
  "topics": {
    "enabled": false,
    "items": []
  }
}
```

### Q5. Optimasi `Deteksi_Hoax_V1.ipynb` untuk Colab T4
Kesimpulan:
Aman dilakukan, dan patch notebook sudah mengarah ke konfigurasi yang lebih realistis untuk T4 15GB.

Bukti:

- pra-patch cell 1 dan 8 memakai batch `96/384`, yang berisiko tinggi OOM
- patch final cell 1 dan 8 menurunkan batch ke `16/32`, menaikkan `grad_accumulation=4`, tetap `fp16`, menambah `auto_find_batch_size`, `gradient_checkpointing`, `EarlyStoppingCallback`, dan dynamic padding
- pemilihan `summary` menurunkan p95 token dari `1035` pada `Clean Narasi` menjadi `146`

Risiko:

- batch lebih kecil bisa menambah wall-clock time per epoch
- `gradient_checkpointing` menghemat memori tetapi sedikit menambah compute

Rekomendasi:

- tetap `max_length=256`
- pakai `summary` sebagai input utama
- `per_device_train_batch_size=16`
- `per_device_eval_batch_size=32`
- `gradient_accumulation_steps=4`
- `fp16=True` saat CUDA tersedia
- `auto_find_batch_size=True`
- `gradient_checkpointing=True`
- `eval/save` per epoch
- `eval_accumulation_steps=8`
- `report_to=\"none\"`
- dynamic padding dengan `pad_to_multiple_of=8`

### Q6. Pro dan kontra jika Q1-Q5 diterapkan
Kesimpulan:
Perubahan yang dipilih menekan risiko OOM dan menjaga akurasi baseline lebih baik daripada refactor besar ke sentence-level training atau joint model topik.

Bukti:

- dataset lokal tetap sejalan dengan schema notebook
- `summary` jauh lebih pendek daripada `Clean Narasi`
- backend baru tetap load-once, eval-mode, no-grad, dan batching
- sentence-level sekarang hanya heuristic inference, bukan retraining noisy

Risiko:

- highlight per kalimat bukan sentence-level model murni
- BERTopic belum aktif, jadi belum ada topik nyata di response
- tanpa kalibrasi baru, threshold bisa tetap 0.5 jika `calibration.json` tidak tersedia

Rekomendasi:

- gunakan patch ini sebagai baseline stabil
- validasi ulang akurasi setelah retraining notebook pada dataset lokal
- aktifkan BERTopic hanya bila ada kebutuhan produk yang jelas dan artifact offline siap

## Arsitektur Final
### Model
- Backbone: `indolem/indobert-base-uncased`
- Tugas utama: binary sequence classification
- Input training final: `summary`
- Output inferensi: label per kalimat hasil sentence splitting pada sisi inferensi

### Mengapa bukan sentence-level training
- builder sentence-level dari notebook NER tidak tersedia di workspace
- memaksa label artikel ke semua kalimat berpotensi menambah noise
- tujuan utama pengguna adalah migrasi dataset lokal, hapus NER, dan jaga stabilitas

### Mengapa BERTopic belum diaktifkan
- aman secara desain hanya bila diposisikan sebagai modul terpisah
- belum ada artifact topic model di workspace
- deployment HF Spaces lebih aman tanpa dependency BERTopic tambahan

## Frontend Structure
- [frontend/index.html](/d:/TA/code/Berita_Hoax_BERTopic/frontend/index.html): layout panel input, ringkasan, highlight, confidence, dan topik opsional
- [frontend/app.js](/d:/TA/code/Berita_Hoax_BERTopic/frontend/app.js): request `/analyze`, render summary/highlight/confidence/topics, copy hasil
- [frontend/styles.css](/d:/TA/code/Berita_Hoax_BERTopic/frontend/styles.css): styling kartu, legend, accordion, confidence block, dan topic card

## Cara Menjalankan
### Backend
Deploy folder [backend](/d:/TA/code/Berita_Hoax_BERTopic/backend) ke Hugging Face Spaces dengan runtime Docker. `Dockerfile` sudah mengatur service FastAPI pada port `7860`.

Env penting:

- `MODEL_DIR` (prioritas path model lokal untuk runtime Docker)
- `MODEL_ID`
- `LOCAL_MODEL_PATH`
- `CALIBRATION_PATH`
- `FRONTEND_ORIGIN`
- `HOAX_THRESHOLD`
- `MAX_LENGTH`
- `BATCH_SIZE`

Default fallback backend saat ini:

- `MODEL_ID=fjrmhri/Deteksi_Hoax_IndoBERT_BERTopic`
- `LOCAL_MODEL_PATH=indobert_hoax_model_v1`
- `MODEL_DIR=/app/indobert_hoax_model_v1` (direkomendasikan untuk HF Spaces backend-only)

### Frontend
Di folder [frontend](/d:/TA/code/Berita_Hoax_BERTopic/frontend):

- deploy sebagai static files ke Vercel

Override API bisa memakai query string:

```text
?api=https://your-backend-host
```

## Verifikasi di HF Spaces
### 1) Cek status model dan threshold
- Buka `https://<space-host>/health`
- Pastikan minimal:
  - `model_source` = `local` (jika folder model lokal tersedia di Space)
  - `local_model_valid` = `true`
  - `hoax_threshold` sesuai `calibration.json` (contoh: `0.1`)
  - `calibration_loaded` = `true`

### 2) Cek endpoint debug forensik
- Buka `https://<space-host>/debug`
- Endpoint ini menampilkan:
  - sumber model efektif (`local`/`hub`) + alasan fallback
  - mapping label (`id2label`/`label2id`)
  - threshold + sumber threshold
  - contoh prediksi 1 sampel fakta + 1 sampel hoaks dari `dataset/*.csv` (jika dataset tersedia di runtime)

### 3) Cek inferensi utama
- Kirim `POST https://<space-host>/analyze` dengan body:
```json
{
  "text": "Beredar klaim lowongan CPNS fiktif yang meminta transfer biaya pendaftaran."
}
```
- Verifikasi field `model.source`, `model.hoax_threshold`, `paragraphs[].sentences[]`.

### Notebook
Gunakan [Deteksi_Hoax_V1.ipynb](/d:/TA/code/Berita_Hoax_BERTopic/Deteksi_Hoax_V1.ipynb) sebagai notebook utama di Google Colab. Notebook legacy [Deteksi_Hoax.ipynb](/d:/TA/code/Berita_Hoax_BERTopic/Deteksi_Hoax.ipynb) dipertahankan untuk referensi historis.

## Status BERTopic
- BERTopic belum diaktifkan di jalur inferensi utama backend.
- Backend mengirim `topics = {"enabled": false, "items": []}` secara eksplisit.
- Frontend hanya menampilkan panel topik jika `topics.enabled === true` dan ada item, sehingga panel topik disembunyikan pada mode saat ini.

## Smoke Test Maintainer
- Script: [scripts/smoke_backend.py](/d:/TA/code/Berita_Hoax_BERTopic/scripts/smoke_backend.py)
- Fungsi:
  - ambil 5 sampel fakta + 5 sampel hoaks dari `dataset/*.csv`
  - panggil endpoint `/analyze`
  - cetak distribusi prediksi dan metadata model source/threshold untuk sanity check cepat

## Verifikasi yang Dilakukan
- `python -m py_compile backend/app.py`
- pencarian `include_ner|NER|ner_pipeline|nerPanel|includeNer|ner_entities|entity_group` pada `backend/` dan `frontend/` tidak menemukan sisa NER
- semua code cell notebook selain cell magic install berhasil di-compile sebagai Python source

## Perubahan File
- [Deteksi_Hoax_V1.ipynb](/d:/TA/code/Berita_Hoax_BERTopic/Deteksi_Hoax_V1.ipynb): notebook utama baru untuk Colab T4, kalibrasi threshold, dan upload HF Hub
- [Deteksi_Hoax.ipynb](/d:/TA/code/Berita_Hoax_BERTopic/Deteksi_Hoax.ipynb): notebook legacy (tidak jadi target utama)
- [backend/app.py](/d:/TA/code/Berita_Hoax_BERTopic/backend/app.py): hapus NER, sederhanakan artifact handling, batching inferensi, kontrak API baru
- [backend/requirements.txt](/d:/TA/code/Berita_Hoax_BERTopic/backend/requirements.txt): buang dependency runtime yang tidak lagi dipakai
- [frontend/index.html](/d:/TA/code/Berita_Hoax_BERTopic/frontend/index.html): hapus panel dan kontrol NER
- [frontend/app.js](/d:/TA/code/Berita_Hoax_BERTopic/frontend/app.js): sesuaikan request/response tanpa NER
- [frontend/styles.css](/d:/TA/code/Berita_Hoax_BERTopic/frontend/styles.css): hapus styling NER, tambah styling topic card opsional
