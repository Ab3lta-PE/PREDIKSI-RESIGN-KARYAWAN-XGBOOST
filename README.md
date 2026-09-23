# Employee Attrition Predictor

Website sederhana untuk memprediksi risiko resign karyawan, menggunakan model
XGBoost (`model_attrition.pkl`) + StandardScaler (`scaler_attrition.pkl`) yang
sudah kamu latih sebelumnya.

## Struktur project
```
attrition-predictor/
├── app.py                  # Flask backend (routing + inference)
├── requirements.txt
├── model_attrition.pkl     # model XGBoost yang sudah dilatih
├── scaler_attrition.pkl    # StandardScaler untuk 23 kolom numerik
├── feature_defaults.pkl    # nilai default untuk kolom yang tidak ada di form
└── templates/
    └── index.html          # UI (HTML + Tailwind CDN + vanilla JS)
```

## Cara menjalankan

1. (Opsional tapi disarankan) buat virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Jalankan server:
   ```bash
   python app.py
   ```

4. Buka browser ke **http://127.0.0.1:5000**

## Cara kerja alurnya

- `templates/index.html` menampilkan form (slider untuk fitur numerik, dropdown
  untuk fitur kategorikal seperti Department, JobRole, OverTime, dll).
- Saat tombol "Jalankan Prediksi" ditekan, JavaScript mengumpulkan semua nilai
  form jadi JSON dan mengirimnya ke endpoint `/predict` lewat `fetch()`
  (tanpa reload halaman).
- Di `app.py`, fungsi `build_feature_row()`:
  1. Mengambil nilai default (`feature_defaults.pkl`) untuk semua 44 kolom
     yang dipakai saat training.
  2. Menimpa nilai numerik dari form.
  3. Meng-encode input kategorikal jadi kolom one-hot yang sesuai
     (mis. `Department` = "Sales" -> `Department_Sales = 1`).
  4. Menerapkan `scaler_attrition.pkl` hanya ke 23 kolom numerik (kolom
     one-hot tidak di-scale, sesuai cara scaler ini dilatih).
- Hasil `model.predict()` dan `model.predict_proba()` dikembalikan sebagai
  JSON (`prediction`, `attrition_risk_pct`, `stay_pct`) dan ditampilkan di
  halaman sebagai progress bar.

## Catatan
- Form ini hanya menampilkan sebagian fitur (yang paling relevan secara
  bisnis). Fitur lain yang tidak ditampilkan di form otomatis memakai nilai
  rata-rata/median dari `feature_defaults.pkl`. Tambahkan field baru di
  `NUMERIC_FORM_FIELDS` (app.py) dan di `index.html` kalau ingin
  mengekspos fitur tersebut ke user.
- Jangan jalankan `app.run(debug=True)` di production — gunakan server
  seperti `gunicorn` saat deploy.
