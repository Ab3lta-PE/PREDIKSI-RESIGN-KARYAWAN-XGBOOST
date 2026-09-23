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


