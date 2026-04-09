# ============================================================
#  RESTAURANT RATING PREDICTION — ML PROJECT
#  Author  : BTech Internship Project
#  Target  : Predict Aggregate Rating of a Restaurant
# ============================================================

# ── 1. IMPORTS ───────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


df = pd.read_csv('Dataset.csv', encoding='latin-1')

drop_cols = [
    'Restaurant ID', 'Restaurant Name', 'Address',
    'Locality', 'Locality Verbose', 'Currency',
    'Rating color', 'Rating text',
    'Switch to order menu', 'Is delivering now'
]
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
print("Dataset loaded! Shape:", df.shape)

# ── 3. INITIAL DATA EXPLORATION ──────────────────────────────
print("=" * 60)
print("STEP 1 — DATA EXPLORATION")
print("=" * 60)
print(f"Shape          : {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Columns        : {list(df.columns)}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nBasic Statistics:\n{df.describe()}")
print(f"\nMissing Values:\n{df.isnull().sum()}")

# ── 4. PREPROCESSING ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — PREPROCESSING")
print("=" * 60)

# 4a. Remove unrated restaurants (Aggregate rating = 0)
before = len(df)
df = df[df['Aggregate rating'] > 0].copy()
print(f"Removed {before - len(df)} unrated rows. Remaining: {len(df)}")

# 4b. Fill missing values
df['Cuisines'].fillna('Unknown', inplace=True)
for col in df.select_dtypes(include='number').columns:
    df[col].fillna(df[col].median(), inplace=True)
print("Missing values handled.")

# 4c. Encode Yes/No columns to 1/0
for col in ['Has Table booking', 'Has Online delivery']:
    df[col] = df[col].map({'Yes': 1, 'No': 0})
print("Binary columns encoded: Has Table booking, Has Online delivery")

# 4d. Label encode City and Cuisines
le = LabelEncoder()
for col in ['City', 'Cuisines']:
    df[col] = le.fit_transform(df[col].astype(str))
print("Label encoding done for: City, Cuisines")

print(f"\nPreprocessed sample:\n{df.head()}")

# ── 5. FEATURE & TARGET SELECTION ────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — FEATURE & TARGET SELECTION")
print("=" * 60)

FEATURES = [
    'Country Code', 'City', 'Cuisines',
    'Average Cost for two', 'Has Table booking',
    'Has Online delivery', 'Price range',
    'Votes', 'Longitude', 'Latitude'
]
TARGET = 'Aggregate rating'

X = df[FEATURES]
y = df[TARGET]

print(f"Features used  : {FEATURES}")
print(f"Target column  : {TARGET}")
print(f"Rating range   : {y.min()} — {y.max()}")

# ── 6. TRAIN-TEST SPLIT ──────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain samples  : {len(X_train)}")
print(f"Test samples   : {len(X_test)}")

# Scale features for Linear Regression
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── 7. MODEL TRAINING & EVALUATION ───────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — MODEL TRAINING & EVALUATION")
print("=" * 60)

models = {
    'Linear Regression' : (LinearRegression(),                                           True),
    'Decision Tree'     : (DecisionTreeRegressor(max_depth=6, random_state=42),          False),
    'Random Forest'     : (RandomForestRegressor(n_estimators=100, random_state=42),     False),
    'Gradient Boosting' : (GradientBoostingRegressor(n_estimators=100, random_state=42), False),
}

results = {}
for name, (model, use_scaled) in models.items():
    Xtr = X_train_sc if use_scaled else X_train
    Xte = X_test_sc  if use_scaled else X_test

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)

    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    results[name] = {
        'model': model, 'y_pred': y_pred,
        'MSE': round(mse,4), 'RMSE': round(rmse,4),
        'MAE': round(mae,4), 'R2': round(r2,4)
    }
    print(f"\n{name}")
    print(f"   MSE  = {mse:.4f}")
    print(f"   RMSE = {rmse:.4f}")
    print(f"   MAE  = {mae:.4f}")
    print(f"   R²   = {r2:.4f}")

best_name = max(results, key=lambda k: results[k]['R2'])
print(f"\n✅ Best Model : {best_name}  (R² = {results[best_name]['R2']})")

# ── 8. FEATURE IMPORTANCE ────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 — FEATURE IMPORTANCE (Random Forest)")
print("=" * 60)

rf = results['Random Forest']['model']
importance_df = pd.DataFrame({
    'Feature'   : FEATURES,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print(importance_df.to_string(index=False))

# ── 9. VISUALIZATIONS ────────────────────────────────────────
print("\nGenerating plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Restaurant Rating Prediction — Model Analysis', fontsize=16, fontweight='bold')

# Plot 1 — Model Comparison (R²)
ax1 = axes[0, 0]
names  = list(results.keys())
r2s    = [results[m]['R2'] for m in names]
colors = ['#e74c3c' if m == best_name else '#3498db' for m in names]
bars   = ax1.barh(names, r2s, color=colors, edgecolor='white')
ax1.set_xlabel('R² Score')
ax1.set_title('Model Comparison (R²)', fontweight='bold')
ax1.set_xlim(0, 1.15)
for bar, val in zip(bars, r2s):
    ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=9)
ax1.legend(handles=[
    plt.Rectangle((0,0),1,1, color='#e74c3c', label='Best Model'),
    plt.Rectangle((0,0),1,1, color='#3498db', label='Other Models')
], fontsize=8)

# Plot 2 — Feature Importance
ax2 = axes[0, 1]
ax2.barh(importance_df['Feature'], importance_df['Importance'], color='#2ecc71', edgecolor='white')
ax2.set_xlabel('Importance Score')
ax2.set_title('Feature Importance (Random Forest)', fontweight='bold')

# Plot 3 — Actual vs Predicted
ax3 = axes[1, 0]
y_pred_best = results[best_name]['y_pred']
ax3.scatter(y_test, y_pred_best, alpha=0.8, color='#9b59b6', s=80, edgecolors='white', linewidth=0.5)
ax3.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction')
ax3.set_xlabel('Actual Rating')
ax3.set_ylabel('Predicted Rating')
ax3.set_title(f'Actual vs Predicted ({best_name})', fontweight='bold')
ax3.legend(fontsize=8)

# Plot 4 — Rating Distribution
ax4 = axes[1, 1]
ax4.hist(y, bins=15, color='#e67e22', edgecolor='white', alpha=0.85)
ax4.axvline(y.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean = {y.mean():.2f}')
ax4.set_xlabel('Aggregate Rating')
ax4.set_ylabel('Frequency')
ax4.set_title('Distribution of Ratings', fontweight='bold')
ax4.legend(fontsize=8)

plt.tight_layout()
plt.savefig('restaurant_rating_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved as 'restaurant_rating_analysis.png'")

# ── 10. FINAL SUMMARY ────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL RESULTS SUMMARY")
print("=" * 60)
summary = pd.DataFrame({
    'Model' : list(results.keys()),
    'MSE'   : [results[m]['MSE']  for m in results],
    'RMSE'  : [results[m]['RMSE'] for m in results],
    'MAE'   : [results[m]['MAE']  for m in results],
    'R2'    : [results[m]['R2']   for m in results],
})
print(summary.to_string(index=False))
print(f"\n  Best Model : {best_name}")
print(f"  R² Score   : {results[best_name]['R2']}")
print(f"  RMSE       : {results[best_name]['RMSE']}")
print("\nProject Complete!")
