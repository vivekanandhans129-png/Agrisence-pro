"""
================================================================================
 AgriSense Pro: Intelligent Crop Yield Analytics & Forecasting System
 Author   : Final Year CS Project — Built for AI Startup Deployment
 Dataset  : All India Crop-wise Area, Production & Yield (2021–2026)
================================================================================
"""

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import json

tf.get_logger().setLevel("ERROR")

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
DATA_PATH   = "All-India_-Crop-wise-Area,-Production-&-Yield.csv"
OUTPUT_DIR  = "outputs"
FIGURES_DIR = "images"
os.makedirs(FIGURES_DIR, exist_ok=True)

PALETTE = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B",
           "#44BBA4", "#E94F37", "#393E41", "#F5A623", "#7B2D8B"]
sns.set_theme(style="whitegrid", palette=PALETTE)
plt.rcParams.update({"figure.dpi": 140, "font.family": "DejaVu Sans"})

results_store = {}   # collects all model metrics for final comparison

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1 ▸ DATA LOADING & UNDERSTANDING
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  SECTION 1 ▸ DATA LOADING & UNDERSTANDING")
print("="*70)

raw = pd.read_csv(DATA_PATH)
print(f"Dataset shape : {raw.shape}  ({raw.shape[0]} rows × {raw.shape[1]} columns)")
print(f"\nColumns:\n  {list(raw.columns)}")
print(f"\nUnique Crops   : {raw['Crop'].nunique()}")
print(f"Unique Seasons : {raw['Season'].nunique()} → {raw['Season'].unique().tolist()}")
print(f"\nMissing values per column:\n{raw.isnull().sum().to_string()}")

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2 ▸ DATA CLEANING & FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  SECTION 2 ▸ DATA CLEANING & FEATURE ENGINEERING")
print("="*70)

# ── Wide → Long (melt)
area_cols  = [c for c in raw.columns if c.startswith("Area")]
prod_cols  = [c for c in raw.columns if c.startswith("Production")]
yield_cols = [c for c in raw.columns if c.startswith("Yield")]

def melt_metric(df, cols, value_name):
    m = df[["Crop","Season"] + cols].melt(
        id_vars=["Crop","Season"],
        value_vars=cols,
        var_name="YearTag",
        value_name=value_name
    )
    m["Year"] = m["YearTag"].str.extract(r"(\d{4}-\d{2,4})$")[0]
    return m.drop(columns="YearTag")

df_area  = melt_metric(raw, area_cols,  "Area")
df_prod  = melt_metric(raw, prod_cols,  "Production")
df_yield = melt_metric(raw, yield_cols, "Yield")

df = df_area.merge(df_prod,  on=["Crop","Season","Year"]) \
            .merge(df_yield, on=["Crop","Season","Year"])

# ── Remove 'Total' season rows (aggregates, not raw observations)
df_raw = df[df["Season"] != "Total"].copy()

# ── Drop rows missing ALL three metrics
df_raw.dropna(subset=["Area","Production","Yield"], how="all", inplace=True)

# ── Fill remaining NaNs with median per Crop+Season group
for col in ["Area","Production","Yield"]:
    df_raw[col] = df_raw.groupby(["Crop","Season"])[col]\
                        .transform(lambda x: x.fillna(x.median()))

# ── Encode Year as integer
df_raw["YearInt"] = df_raw["Year"].str[:4].astype(int)

# ── Label encode categoricals
le_crop   = LabelEncoder()
le_season = LabelEncoder()
df_raw["CropEnc"]   = le_crop.fit_transform(df_raw["Crop"])
df_raw["SeasonEnc"] = le_season.fit_transform(df_raw["Season"])

# ── Feature engineering
df_raw["AreaSq"]           = df_raw["Area"] ** 2
df_raw["AreaLog"]          = np.log1p(df_raw["Area"])
df_raw["ProdPerArea"]      = df_raw["Production"] / (df_raw["Area"] + 1e-6)
df_raw["YearSinceBase"]    = df_raw["YearInt"] - df_raw["YearInt"].min()

print(f"Cleaned dataset shape: {df_raw.shape}")
print(f"\nSample rows:\n{df_raw[['Crop','Season','Year','Area','Production','Yield']].head(8).to_string(index=False)}")
print(f"\nFeature columns: {list(df_raw.columns)}")

# ── Also keep 'Total' season rows for trend analysis
df_total = df[df["Season"] == "Total"].copy()
df_total["YearInt"] = df_total["Year"].str[:4].astype(int)

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3 ▸ EXPLORATORY DATA ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  SECTION 3 ▸ EXPLORATORY DATA ANALYSIS")
print("="*70)

# ── Figure 1: Distribution of Yield
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df_raw["Yield"].dropna(), bins=35, color=PALETTE[0], edgecolor="white", alpha=0.85)
axes[0].set_title("Yield Distribution (Kg/Ha)", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Yield (Kg/Ha)"); axes[0].set_ylabel("Frequency")

axes[1].hist(np.log1p(df_raw["Yield"].dropna()), bins=35, color=PALETTE[1], edgecolor="white", alpha=0.85)
axes[1].set_title("Log-Transformed Yield Distribution", fontsize=13, fontweight="bold")
axes[1].set_xlabel("log(1 + Yield)"); axes[1].set_ylabel("Frequency")
plt.suptitle("Yield Distribution Analysis", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig1_yield_distribution.png", bbox_inches="tight")
plt.close()

# ── Figure 2: Top 10 Crops by Average Yield
top_yield = (df_raw[df_raw["Yield"] > 0]
             .groupby("Crop")["Yield"].mean()
             .sort_values(ascending=False).head(10))
fig, ax = plt.subplots(figsize=(13, 5))
bars = ax.barh(top_yield.index[::-1], top_yield.values[::-1],
               color=PALETTE[:10], edgecolor="white", height=0.65)
for bar, val in zip(bars, top_yield.values[::-1]):
    ax.text(bar.get_width() + 400, bar.get_y() + bar.get_height()/2,
            f"{val:,.0f}", va="center", fontsize=9)
ax.set_xlabel("Average Yield (Kg/Ha)", fontsize=11)
ax.set_title("Top 10 Crops by Average Yield (All India)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig2_top_crops_yield.png", bbox_inches="tight")
plt.close()

# ── Figure 3: Season-wise Yield Boxplot
fig, ax = plt.subplots(figsize=(10, 5))
season_data = [df_raw[df_raw["Season"]==s]["Yield"].dropna().values
               for s in df_raw["Season"].unique()]
bp = ax.boxplot(season_data, labels=df_raw["Season"].unique(),
                patch_artist=True, notch=False,
                boxprops=dict(facecolor=PALETTE[0], alpha=0.7),
                medianprops=dict(color="red", linewidth=2))
for patch, color in zip(bp["boxes"], PALETTE):
    patch.set_facecolor(color)
ax.set_title("Season-wise Yield Distribution", fontsize=13, fontweight="bold")
ax.set_ylabel("Yield (Kg/Ha)"); ax.set_xlabel("Season")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig3_season_yield_box.png", bbox_inches="tight")
plt.close()

# ── Figure 4: Yield Trend for Key Crops over Years
key_crops = ["Rice","Wheat","Maize","Sugarcane","Groundnut"]
fig, ax = plt.subplots(figsize=(13, 6))
for i, crop in enumerate(key_crops):
    sub = df_total[df_total["Crop"]==crop].sort_values("YearInt")
    if len(sub) > 0:
        ax.plot(sub["YearInt"], sub["Yield"], marker="o", linewidth=2.5,
                color=PALETTE[i], label=crop, markersize=7)
ax.set_title("Yield Trend for Key Crops (2021–2026)", fontsize=13, fontweight="bold")
ax.set_xlabel("Year"); ax.set_ylabel("Yield (Kg/Ha)")
ax.legend(loc="upper left"); ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig4_yield_trend.png", bbox_inches="tight")
plt.close()

# ── Figure 5: Correlation Heatmap
fig, ax = plt.subplots(figsize=(9, 7))
corr_cols = ["Area","Production","Yield","YearInt","CropEnc","SeasonEnc","AreaLog","ProdPerArea"]
corr = df_raw[corr_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, ax=ax, annot_kws={"size": 9})
ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig5_correlation_heatmap.png", bbox_inches="tight")
plt.close()

# ── Figure 6: Production by Crop Category
category_map = {
    "Rice":"Cereals","Wheat":"Cereals","Maize":"Cereals","Barley":"Cereals",
    "Jowar":"Cereals","Bajra":"Cereals","Ragi":"Cereals",
    "Tur":"Pulses","Gram":"Pulses","Urad":"Pulses","Moong":"Pulses","Lentil":"Pulses",
    "Groundnut":"Oilseeds","Soybean":"Oilseeds","Rapeseed & Mustard":"Oilseeds",
    "Sunflower":"Oilseeds","Castorseed":"Oilseeds","Sesamum":"Oilseeds",
    "Sugarcane":"Cash Crops","Cotton":"Cash Crops","Jute":"Cash Crops","Tobacco":"Cash Crops"
}
df_raw["Category"] = df_raw["Crop"].map(category_map).fillna("Other")
cat_prod = df_raw.groupby("Category")["Production"].sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(9, 5))
wedges, texts, autotexts = ax.pie(cat_prod.values, labels=cat_prod.index,
                                   autopct="%1.1f%%", colors=PALETTE[:len(cat_prod)],
                                   startangle=140, pctdistance=0.82)
for at in autotexts:
    at.set_fontsize(9)
ax.set_title("Production Share by Crop Category", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig6_production_by_category.png", bbox_inches="tight")
plt.close()

print("✅ EDA figures saved (6 charts).")

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4 ▸ FEATURE SELECTION & DATASET PREP
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  SECTION 4 ▸ FEATURE SELECTION & ML DATASET PREPARATION")
print("="*70)

FEATURES = ["CropEnc","SeasonEnc","Area","AreaLog","Production","ProdPerArea",
            "YearInt","YearSinceBase"]
TARGET   = "Yield"

ml_df = df_raw[FEATURES + [TARGET]].dropna()
X = ml_df[FEATURES].values
y = ml_df[TARGET].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

print(f"Training samples : {X_train.shape[0]}")
print(f"Testing  samples : {X_test.shape[0]}")
print(f"Features used    : {FEATURES}")

def evaluate(name, model, yt, yp):
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mae  = mean_absolute_error(yt, yp)
    r2   = r2_score(yt, yp)
    results_store[name] = {"R2": round(r2,4), "RMSE": round(rmse,2), "MAE": round(mae,2)}
    print(f"  {name:<30} R²={r2:.4f}  RMSE={rmse:,.1f}  MAE={mae:,.1f}")
    return yp

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 5 ▸ MACHINE LEARNING MODELS
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  SECTION 5 ▸ MACHINE LEARNING MODELS")
print("="*70)

# ── 5A: Linear Regression (baseline)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = evaluate("Linear Regression", lr, y_test, lr.predict(X_test))

# ── 5B: Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = evaluate("Ridge Regression", ridge, y_test, ridge.predict(X_test))

# ── 5C: Random Forest
rf = RandomForestRegressor(n_estimators=200, max_depth=12,
                           min_samples_leaf=2, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = evaluate("Random Forest", rf, y_test, rf.predict(X_test))

# ── 5D: Gradient Boosting
gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.08,
                               max_depth=5, random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = evaluate("Gradient Boosting", gb, y_test, gb.predict(X_test))

# ── 5E: XGBoost
xgbm = xgb.XGBRegressor(n_estimators=300, learning_rate=0.07,
                         max_depth=6, subsample=0.8,
                         colsample_bytree=0.8, random_state=42,
                         verbosity=0)
xgbm.fit(X_train, y_train)
y_pred_xgb = evaluate("XGBoost", xgbm, y_test, xgbm.predict(X_test))

# ── Feature Importance Plot (Random Forest)
fig, ax = plt.subplots(figsize=(9, 5))
imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=True)
imp.plot.barh(ax=ax, color=PALETTE[0], edgecolor="white")
ax.set_title("Random Forest — Feature Importance", fontsize=13, fontweight="bold")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig7_feature_importance.png", bbox_inches="tight")
plt.close()

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 6 ▸ DEEP LEARNING MODEL (ANN)
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  SECTION 6 ▸ DEEP LEARNING — ARTIFICIAL NEURAL NETWORK (ANN)")
print("="*70)
print("  Rationale: Non-linear crop relationships + multi-feature interaction")
print("  cannot be fully captured by tree/linear models; ANN learns latent")
print("  representations across crop-season-area-production feature space.\n")

tf.random.set_seed(42)

def build_ann(input_dim):
    model = Sequential([
        Dense(256, activation="relu", input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.25),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss="mse", metrics=["mae"])
    return model

ann = build_ann(X_train.shape[1])
ann.summary()

callbacks = [
    EarlyStopping(patience=20, restore_best_weights=True, monitor="val_loss"),
    ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6)
]

history = ann.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=150,
    batch_size=16,
    callbacks=callbacks,
    verbose=0
)

y_pred_ann = ann.predict(X_test, verbose=0).flatten()
evaluate("ANN (Deep Learning)", None, y_test, y_pred_ann)

# ── ANN Training History Plot
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(history.history["loss"],     label="Train Loss",  color=PALETTE[0])
axes[0].plot(history.history["val_loss"], label="Val Loss",    color=PALETTE[1])
axes[0].set_title("ANN — Loss Curve", fontweight="bold"); axes[0].legend()
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE Loss")

axes[1].plot(history.history["mae"],     label="Train MAE",  color=PALETTE[2])
axes[1].plot(history.history["val_mae"], label="Val MAE",    color=PALETTE[3])
axes[1].set_title("ANN — MAE Curve", fontweight="bold"); axes[1].legend()
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("MAE")
plt.suptitle("ANN Training History", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig8_ann_training.png", bbox_inches="tight")
plt.close()

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 7 ▸ MODEL COMPARISON & EVALUATION
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  SECTION 7 ▸ MODEL COMPARISON & EVALUATION")
print("="*70)

comp = pd.DataFrame(results_store).T.sort_values("R2", ascending=False)
print("\n" + comp.to_string())

# ── Comparison bar charts
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
colors = [PALETTE[i] for i in range(len(comp))]
for ax, metric, title in zip(axes, ["R2","RMSE","MAE"],
                               ["R² Score (Higher=Better)",
                                "RMSE (Lower=Better)",
                                "MAE (Lower=Better)"]):
    vals = comp[metric].values
    bars = ax.bar(comp.index, vals, color=colors, edgecolor="white", width=0.6)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(vals)*0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=35)
plt.suptitle("ML vs DL Model Comparison — AgriSense Pro", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig9_model_comparison.png", bbox_inches="tight")
plt.close()

# ── Actual vs Predicted — Best Model
best_name = comp.index[0]
best_preds = {"Linear Regression": y_pred_lr, "Ridge Regression": y_pred_ridge,
              "Random Forest": y_pred_rf, "Gradient Boosting": y_pred_gb,
              "XGBoost": y_pred_xgb, "ANN (Deep Learning)": y_pred_ann}
y_pred_best = best_preds[best_name]

fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(y_test, y_pred_best, alpha=0.55, color=PALETTE[0], edgecolors="white", s=40)
lims = [min(y_test.min(), y_pred_best.min())*0.9,
        max(y_test.max(), y_pred_best.max())*1.05]
ax.plot(lims, lims, "r--", linewidth=2, label="Perfect Prediction")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("Actual Yield (Kg/Ha)", fontsize=11)
ax.set_ylabel("Predicted Yield (Kg/Ha)", fontsize=11)
ax.set_title(f"Actual vs Predicted — {best_name}", fontsize=12, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig10_actual_vs_predicted.png", bbox_inches="tight")
plt.close()

print("\n✅ All 10 figures saved.")

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 8 ▸ SAVE RESULTS JSON
# ──────────────────────────────────────────────────────────────────────────────
with open(f"{OUTPUT_DIR}/model_results.json", "w") as f:
    json.dump(results_store, f, indent=2)

print("\n✅ Model results saved to model_results.json")
print("\n" + "="*70)
print("  ALL PROCESSING COMPLETE — AgriSense Pro")
print("="*70)
