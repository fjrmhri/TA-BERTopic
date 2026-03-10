# Plan

## Scope yang Dipilih
- Q1: feasible, diimplementasikan
- Q2: feasible sebagai modul terpisah, ditunda
- Q3: feasible dengan sentence-level inference, diimplementasikan
- Q4: feasible, diimplementasikan
- Q5: feasible, diimplementasikan pada notebook

## Roadmap Implementasi
### 1. Audit dataset dan notebook
File:

- `dataset/*.csv`
- `Deteksi_Hoax.ipynb`
- `Deteksi_Hoax_NER_Optimized.ipynb`

Tujuan:

- pastikan schema konsisten
- ukur distribusi label, missing rate, dan panjang teks
- identifikasi perubahan minimal agar notebook utama bisa memakai dataset lokal

Status:

- selesai

### 2. Patch notebook utama agar memakai dataset lokal
File:

- `Deteksi_Hoax.ipynb`

Perubahan:

- hapus ketergantungan KaggleHub
- validasi `dataset/` lokal
- ganti sumber file ke empat CSV yang ada
- ubah prioritas kolom teks ke `summary`
- turunkan batch size dan tambah guardrail T4
- tambahkan demo inferensi multi-paragraf per kalimat

Status:

- selesai

### 3. Sederhanakan backend menjadi classifier-only
File:

- `backend/app.py`
- `backend/requirements.txt`

Perubahan:

- hapus `include_ner`
- hapus pipeline dan agregasi NER
- pertahankan load-once model, `eval()`, `torch.inference_mode()`, batching
- sederhanakan validasi artefak lokal agar cocok dengan output notebook classifier
- pertahankan placeholder `topics` sebagai hook opsional tanpa menambah dependency BERTopic

Status:

- selesai

### 4. Sinkronkan frontend dengan kontrak API baru
File:

- `frontend/index.html`
- `frontend/app.js`
- `frontend/styles.css`

Perubahan:

- hapus checkbox, panel, render, tooltip, dan copy-output NER
- render highlight per kalimat
- render confidence per paragraf
- render topic panel hanya bila backend mengirim `topics.enabled=true`

Status:

- selesai

### 5. BERTopic sebagai fase opsional berikutnya
File yang akan terdampak bila diaktifkan nanti:

- `backend/app.py`
- `backend/requirements.txt`
- artifact topic model terpisah
- frontend panel topik yang sudah disiapkan

Langkah:

1. fit BERTopic offline pada corpus train
2. simpan artifact model topik di CPU
3. tambahkan cache inferensi berbasis hash teks
4. aktifkan field `topics.items` hanya bila artifact tersedia

Status:

- ditunda sengaja

## File yang Diubah
| File | Alasan |
| --- | --- |
| `Deteksi_Hoax.ipynb` | migrasi dataset lokal dan optimasi T4 |
| `backend/app.py` | hapus NER, batching, kontrak API baru |
| `backend/requirements.txt` | buang dependency backend yang tidak lagi dipakai |
| `frontend/index.html` | hapus panel NER |
| `frontend/app.js` | hapus logic NER, sinkronkan request/response |
| `frontend/styles.css` | hapus styling NER |

## Risiko
### 1. Heuristic sentence-level inference
Risiko:

- model dilatih doc-level, tetapi dipakai untuk kalimat individual saat inferensi

Mitigasi:

- jangan ubah training menjadi sentence-level tanpa builder yang tervalidasi
- tampilkan confidence dan warna low-confidence

### 2. OOM di Colab T4
Risiko:

- dataset cukup besar dan `indobert-base` sensitif terhadap batch dan panjang input

Mitigasi:

- gunakan `summary`
- `train_batch_size=16`
- `eval_batch_size=32`
- `grad_accumulation=4`
- `fp16`
- `auto_find_batch_size`
- `gradient_checkpointing`

### 3. Cold start backend
Risiko:

- fallback ke Hub tetap membutuhkan waktu muat model

Mitigasi:

- siapkan `LOCAL_MODEL_PATH`
- pertahankan startup load sekali
- simpan `calibration.json` bila tersedia

### 4. BERTopic menambah kompleksitas
Risiko:

- image lebih berat, latency naik, cold start makin lama

Mitigasi:

- jadikan modul opsional
- fit offline
- inferensi CPU + caching
- jangan campur ke classifier training

## Acceptance Criteria
### Wajib
- [x] `Deteksi_Hoax.ipynb` memakai `dataset/` lokal
- [x] backbone tetap `indolem/indobert-base-uncased`
- [x] backend tidak lagi memiliki `include_ner`
- [x] frontend tidak lagi memiliki panel atau kontrol NER
- [x] backend mengembalikan `summary` dan `paragraphs[].sentences[]`
- [x] notebook memiliki cell demo multi-paragraf -> label per kalimat

### Verifikasi ringan yang sudah dilakukan
- [x] `backend/app.py` lolos `py_compile`
- [x] code cell notebook lolos compile check, kecuali cell magic install
- [x] pencarian teks pada `backend/` dan `frontend/` tidak menemukan sisa NER

### Verifikasi yang masih perlu dijalankan oleh user
- [ ] retraining notebook penuh di Google Colab T4 untuk memastikan tidak OOM
- [ ] end-to-end backend dengan model nyata
- [ ] frontend terhadap backend aktif
- [ ] evaluasi ulang akurasi sesudah retraining pada dataset lokal

## Exit Criteria
Implementasi dianggap selesai bila:

1. notebook dapat dijalankan dengan dataset lokal tanpa mengganti backbone
2. backend dan frontend tidak lagi memiliki NER
3. inferensi multi-paragraf menghasilkan label per kalimat
4. tidak ada refactor besar yang mengubah arsitektur inti classifier
5. BERTopic tetap opsional dan tidak mengganggu jalur classifier utama
