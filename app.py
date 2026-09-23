from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# ------------------------------------------------------------------
# Load model artifacts ONCE at startup (not on every request)
# ------------------------------------------------------------------
model = joblib.load("model_attrition.pkl")
scaler = joblib.load("scaler_attrition.pkl")
feature_defaults = joblib.load("feature_defaults.pkl")  # pandas Series

NUMERIC_COLS = list(scaler.feature_names_in_)   # 23 numeric cols the scaler expects
ALL_COLS = list(feature_defaults.index)          # full 44-col training column order

# One-hot groups -> the exact dummy columns the model was trained on.
# The "missing" option in each group is the baseline (all dummies = 0).
ONEHOT_GROUPS = {
    "BusinessTravel": {
        "options": ["Non-Travel", "Travel_Rarely", "Travel_Frequently"],
        "columns": ["BusinessTravel_Travel_Rarely", "BusinessTravel_Travel_Frequently"],
    },
    "Department": {
        "options": ["Human Resources", "Research & Development", "Sales"],
        "columns": ["Department_Research & Development", "Department_Sales"],
    },
    "EducationField": {
        "options": ["Human Resources", "Life Sciences", "Marketing", "Medical", "Other", "Technical Degree"],
        "columns": [
            "EducationField_Life Sciences", "EducationField_Marketing",
            "EducationField_Medical", "EducationField_Other",
            "EducationField_Technical Degree",
        ],
    },
    "Gender": {
        "options": ["Female", "Male"],
        "columns": ["Gender_Male"],
    },
    "JobRole": {
        "options": [
            "Healthcare Representative", "Human Resources", "Laboratory Technician",
            "Manager", "Manufacturing Director", "Research Director",
            "Research Scientist", "Sales Executive", "Sales Representative",
        ],
        "columns": [
            "JobRole_Human Resources", "JobRole_Laboratory Technician", "JobRole_Manager",
            "JobRole_Manufacturing Director", "JobRole_Research Director",
            "JobRole_Research Scientist", "JobRole_Sales Executive",
            "JobRole_Sales Representative",
        ],
    },
    "MaritalStatus": {
        "options": ["Divorced", "Married", "Single"],
        "columns": ["MaritalStatus_Married", "MaritalStatus_Single"],
    },
    "OverTime": {
        "options": ["No", "Yes"],
        "columns": ["OverTime_Yes"],
    },
}

# Numeric fields exposed directly on the form (rest fall back to feature_defaults)
NUMERIC_FORM_FIELDS = [
    "Age", "MonthlyIncome", "DistanceFromHome", "NumCompaniesWorked",
    "TotalWorkingYears", "YearsAtCompany", "YearsInCurrentRole",
    "JobSatisfaction", "EnvironmentSatisfaction", "WorkLifeBalance",
    "StockOptionLevel", "PercentSalaryHike",
]


def build_feature_row(payload: dict) -> pd.DataFrame:
    """Merge form input with defaults, one-hot encode, return a 1-row DataFrame
    with columns in the exact order the model was trained on."""
    row = feature_defaults.copy()

    for col in NUMERIC_FORM_FIELDS:
        if payload.get(col) not in (None, ""):
            row[col] = float(payload[col])

    for group, cfg in ONEHOT_GROUPS.items():
        for col in cfg["columns"]:
            row[col] = 0.0
        selected = payload.get(group)
        if selected:
            col_name = f"{group}_{selected}"
            if col_name in row.index:
                row[col_name] = 1.0
            # if `selected` is the baseline option, all dummies stay 0 - correct.

    X = pd.DataFrame([row])[ALL_COLS]
    X[NUMERIC_COLS] = scaler.transform(X[NUMERIC_COLS])
    return X


@app.route("/")
def home():
    return render_template("index.html", groups=ONEHOT_GROUPS)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
        X = build_feature_row(payload)

        pred = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0]

        return jsonify({
            "prediction": "Yes" if pred == 1 else "No",
            "attrition_risk_pct": round(float(proba[1]) * 100, 2),
            "stay_pct": round(float(proba[0]) * 100, 2),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
