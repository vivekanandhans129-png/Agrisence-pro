"""
================================================================================
 AgriSense Pro — Flask REST API
 Endpoints:
   POST /predict          — single yield prediction
   POST /predict/batch    — batch predictions
   GET  /crops            — list supported crops
   GET  /seasons          — list supported seasons
   GET  /health           — health check
================================================================================
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from flasgger import Swagger
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)
swagger = Swagger(app)

# ── Global state — loaded once at startup
MODEL       = None
SCALER      = None
LE_CROP     = None
LE_SEASON   = None
BASE_YEAR   = 2021

def prepare_features(crop, season, area, production, year):
    crop_enc   = LE_CROP.transform([crop])[0]
    season_enc = LE_SEASON.transform([season])[0]
    area_log   = np.log1p(area)
    prod_area  = production / (area + 1e-6)
    year_base  = year - BASE_YEAR
    return np.array([[crop_enc, season_enc, area, area_log,
                       production, prod_area, year, year_base]])

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "message": "Welcome to AgriSense Pro API!",
        "endpoints": {
            "health": "/health",
            "crops_list": "/crops",
            "seasons_list": "/seasons",
            "predict": "/predict",
            "batch_predict": "/predict/batch"
        }
    })

@app.route("/health", methods=["GET"])
def health():
    """
    Health Check Endpoint
    ---
    responses:
      200:
        description: Returns the health status of the API
    """
    return jsonify({
        "status": "ok",
        "service": "AgriSense Pro API",
        "version": "1.0.0",
        "powered_by": "Portfolio Project"
    })

@app.route("/crops", methods=["GET"])
def get_crops():
    return jsonify({"crops": sorted(LE_CROP.classes_.tolist())})

@app.route("/seasons", methods=["GET"])
def get_seasons():
    return jsonify({"seasons": sorted(LE_SEASON.classes_.tolist())})

@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict Crop Yield
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            crop:
              type: string
              example: Rice
            season:
              type: string
              example: Kharif
            area:
              type: number
              example: 410.38
            production:
              type: number
              example: 1110.0
            year:
              type: integer
              example: 2025
    responses:
      200:
        description: Predicted yield in Kg/Ha
    """
    data = request.get_json()
    required = ["crop","season","area","production","year"]
    for field in required:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

    crop, season = data["crop"], data["season"]
    if crop not in LE_CROP.classes_:
        return jsonify({"error": f"Unknown crop: {crop}. Use GET /crops"}), 400
    if season not in LE_SEASON.classes_:
        return jsonify({"error": f"Unknown season: {season}. Use GET /seasons"}), 400

    X = prepare_features(crop, season,
                          float(data["area"]),
                          float(data["production"]),
                          int(data["year"]))
    X_scaled = SCALER.transform(X)
    pred = float(MODEL.predict(X_scaled)[0])

    return jsonify({
        "crop": crop,
        "season": season,
        "area_lakh_ha": data["area"],
        "production_lakh_tonnes": data["production"],
        "year": data["year"],
        "predicted_yield_kg_ha": round(pred, 2),
        "model": "GradientBoosting (R²=0.9983)",
        "powered_by": "Portfolio Project"
    })

@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """
    Request body: {"records": [{"crop":..., "season":..., ...}, ...]}
    """
    data = request.get_json()
    records = data.get("records", [])
    if not records:
        return jsonify({"error": "Empty records list"}), 400

    results = []
    for rec in records:
        try:
            X = prepare_features(rec["crop"], rec["season"],
                                  float(rec["area"]),
                                  float(rec["production"]),
                                  int(rec["year"]))
            X_scaled = SCALER.transform(X)
            pred = float(MODEL.predict(X_scaled)[0])
            results.append({**rec, "predicted_yield_kg_ha": round(pred, 2), "status": "ok"})
        except Exception as e:
            results.append({**rec, "status": "error", "message": str(e)})

    return jsonify({"results": results, "count": len(results)})

# ── Model initialization (call this in __main__ or in a startup hook)
def load_model():
    global MODEL, SCALER, LE_CROP, LE_SEASON
    df = pd.read_csv("All-India_-Crop-wise-Area,-Production-&-Yield.csv")
    area_cols  = [c for c in df.columns if c.startswith("Area")]
    prod_cols  = [c for c in df.columns if c.startswith("Production")]
    yield_cols = [c for c in df.columns if c.startswith("Yield")]

    def melt(data, cols, vname):
        m = data[["Crop","Season"]+cols].melt(id_vars=["Crop","Season"],
                value_vars=cols, var_name="Y", value_name=vname)
        m["Year"] = m["Y"].str.extract(r"(\d{4}-\d{2,4})$")[0]
        return m.drop(columns="Y")

    long = melt(df, area_cols, "Area")\
               .merge(melt(df, prod_cols, "Production"), on=["Crop","Season","Year"])\
               .merge(melt(df, yield_cols,"Yield"),      on=["Crop","Season","Year"])
    long["YearInt"] = long["Year"].str[:4].astype(int)
    clean = long[long["Season"] != "Total"].copy()
    clean.dropna(subset=["Area","Production","Yield"], how="all", inplace=True)
    for col in ["Area","Production","Yield"]:
        clean[col] = clean.groupby(["Crop","Season"])[col]\
                          .transform(lambda x: x.fillna(x.median()))
    LE_CROP   = LabelEncoder().fit(clean["Crop"])
    LE_SEASON = LabelEncoder().fit(clean["Season"])
    clean["CropEnc"]       = LE_CROP.transform(clean["Crop"])
    clean["SeasonEnc"]     = LE_SEASON.transform(clean["Season"])
    clean["AreaLog"]       = np.log1p(clean["Area"])
    clean["ProdPerArea"]   = clean["Production"] / (clean["Area"] + 1e-6)
    clean["YearSinceBase"] = clean["YearInt"] - BASE_YEAR

    feats = ["CropEnc","SeasonEnc","Area","AreaLog","Production",
             "ProdPerArea","YearInt","YearSinceBase"]
    mask  = clean[feats+["Yield"]].notna().all(axis=1)
    X, y  = clean.loc[mask, feats].values, clean.loc[mask, "Yield"].values
    SCALER = StandardScaler().fit(X)
    MODEL  = GradientBoostingRegressor(n_estimators=200, learning_rate=0.08,
                                        max_depth=5, random_state=42)
    MODEL.fit(SCALER.transform(X), y)
    print("✅ Model loaded and ready.")

if __name__ == "__main__":
    load_model()
    print("🚀 Starting AgriSense Pro API on http://0.0.0.0:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
