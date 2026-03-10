# DIAGNOSIS

## Gejala
- Dari frontend, hasil analisis cenderung selalu `Fakta` dengan confidence tinggi.
- Panel BERTopic tidak muncul.

## Bukti Forensik Utama
1. Artefak model lokal valid:
- `indobert_hoax_model_v1/config.json`: `id2label={"0":"Fakta","1":"Hoaks"}`, `label2id={"Fakta":0,"Hoaks":1}`, `num_labels=2`.
- `indobert_hoax_model_v1/calibration.json`: `best_threshold=0.1`.

2. Runtime backend yang aktif sebelumnya tidak memakai model lokal:
- `GET /health` pada Space mengembalikan:
  - `model_source: "hub"`
  - `local_model_valid: false`
  - `missing_required_artifacts: ["/indobert_hoax_model_v1"]`
  - `hoax_threshold: 0.5`
  - `calibration_loaded: false`
- Ini menunjukkan model lokal dan `calibration.json` tidak termuat pada runtime lama.

3. `backend/Dockerfile` lama hanya menyalin `app.py`:
- File: `backend/Dockerfile`
- Sebelumnya: `COPY app.py /app/app.py`
- Akibat: folder model lokal tidak ikut image jika deploy backend-only.

4. BERTopic memang belum diaktifkan by design:
- `backend/app.py` mengirim `TOPICS_PAYLOAD = {"enabled": False, "items": []}`.
- `frontend/app.js` hanya render topik jika `topics.enabled === true` dan `items` ada.

## Hipotesis Penyebab (A-F) + Bukti
### A) Backend meload model yang salah (hub fallback)
Status: **Sangat mungkin (utama)**.

Bukti:
- Health runtime lama menunjukkan `model_source=hub` + local model invalid.
- Docker lama tidak membawa folder model lokal ke image.

### B) Label mapping terbalik
Status: **Kecil kemungkinan**.

Bukti:
- `config.json` lokal konsisten: `0=Fakta`, `1=Hoaks`.
- Kode backend membaca `id2label/label2id` dan menentukan ID kelas berbasis token label.

### C) Threshold tidak dipakai / salah diterapkan
Status: **Sangat mungkin (utama, efek turunan A)**.

Bukti:
- Kalibrasi lokal `best_threshold=0.1`, tetapi runtime lama menunjukkan `hoax_threshold=0.5` dan `calibration_loaded=false`.
- Kondisi ini terjadi saat backend fallback ke hub tanpa `calibration.json` lokal.

### D) `prob_hoax` dihitung salah
Status: **Kecil kemungkinan**.

Bukti:
- Backend menghitung `softmax(logits)` lalu mengambil indeks kelas hoaks/fakta terpisah.
- Label ditentukan dari `prob_hoax >= threshold`.

### E) Preprocessing inference mismatch
Status: **Kemungkinan sekunder**.

Bukti:
- Ada pembersihan teks inferensi (`INFERENCE_CLEAN_PATTERNS`) yang menghapus beberapa token sumber.
- Namun bukti utama kolaps prediksi lebih kuat menunjuk ke model/threshold runtime, bukan semata preprocessing.

### F) Frontend salah interpretasi response
Status: **Kecil kemungkinan**.

Bukti:
- Frontend menampilkan nilai `label/prob_hoax/prob_fakta/confidence/color` langsung dari backend.
- Tidak ada perhitungan ulang label di frontend.

## Keputusan Akar Masalah
Penyebab paling mungkin adalah kombinasi:
1. **A**: runtime backend memuat model hub (fallback), bukan model lokal `indobert_hoax_model_v1`.
2. **C**: karena local calibration tidak termuat, threshold kembali ke default `0.5` (bukan `0.1`).

Dampak kombinasi A+C adalah sistem condong memprediksi `Fakta` pada banyak input, terlihat sebagai gejala "selalu Fakta".

## Status BERTopic
- Tidak muncul karena memang belum diimplementasikan pada pipeline aktif.
- Behavior saat ini konsisten: backend kirim `topics.enabled=false`, frontend menyembunyikan panel topik.
