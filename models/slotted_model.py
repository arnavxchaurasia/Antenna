"""
slotted_model.py
----------------
Research-grade ML training for slotted microstrip patch antenna S11 prediction.

Problem:
    Given (frequency, L, W, Ls, Ws) → predict S11(dB)

Evaluation strategy:
    Configuration-Level Leave-One-Out Cross Validation (LOOCV).
    Each fold holds out ONE entire antenna geometry.
    This gives honest "unseen geometry" generalization metrics.

Models:
    - MLPRegressor (primary — dense network)
    - GradientBoostingRegressor (strong baseline)

Saved artifacts:
    models/best_model.pkl      : Best LOOCV model
    models/best_scaler_X.pkl   : Input scaler
    models/best_scaler_y.pkl   : Output scaler
    models/loocv_results.csv   : Per-fold metrics
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_loader import load_slotted_dataset, loocv_splits, FEATURE_COLS, TARGET_COL, GEO_COLS

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))


# ─────────────────────────────────────────────────────────────────────────────
# Model Definitions
# ─────────────────────────────────────────────────────────────────────────────

def build_mlp():
    return MLPRegressor(
        hidden_layer_sizes=(256, 128, 64, 32),
        activation='relu',
        solver='adam',
        max_iter=3000,
        learning_rate_init=0.001,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=40,
        random_state=42,
        tol=1e-5,
    )


def build_gb():
    return GradientBoostingRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.8,
        min_samples_leaf=3,
        random_state=42,
    )


# ─────────────────────────────────────────────────────────────────────────────
# LOOCV Training & Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_loocv(df: pd.DataFrame, model_name: str = 'MLP'):
    """
    Run configuration-level Leave-One-Out Cross Validation.

    Returns
    -------
    fold_results : list of dicts — per-fold metrics
    all_preds    : DataFrame — collected predictions across all folds
    """
    fold_results = []
    all_preds    = []

    print(f"\n{'='*65}")
    print(f"  Configuration-Level LOOCV — {model_name}")
    print(f"  (Each fold holds out one complete antenna geometry)")
    print(f"{'='*65}")
    print(f"  {'Fold':<6} {'Hold-out (L,W,Ls,Ws)':<28} {'R²':>7} {'MAE':>8} {'MSE':>8}")
    print(f"  {'-'*58}")

    for fold_id, held_out, train_df, test_df in loocv_splits(df):
        # Scale features
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train = train_df[FEATURE_COLS].values
        y_train = train_df[TARGET_COL].values
        X_test  = test_df[FEATURE_COLS].values
        y_test  = test_df[TARGET_COL].values

        X_train_s = scaler_X.fit_transform(X_train)
        X_test_s  = scaler_X.transform(X_test)
        y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

        # Build & train
        model = build_mlp() if model_name == 'MLP' else build_gb()

        if model_name == 'MLP':
            model.fit(X_train_s, y_train_s)
            y_pred_s = model.predict(X_test_s)
            y_pred   = scaler_y.inverse_transform(
                           y_pred_s.reshape(-1, 1)).ravel()
        else:
            model.fit(X_train_s, y_train)   # GB on raw y
            y_pred = model.predict(X_test_s)

        fold_r2  = r2_score(y_test, y_pred)
        fold_mae = mean_absolute_error(y_test, y_pred)
        fold_mse = mean_squared_error(y_test, y_pred)

        L, W, Ls, Ws = held_out
        tag = f"L={L},W={W},Ls={Ls},Ws={Ws}"
        print(f"  {fold_id+1:<6} {tag:<28} {fold_r2:>7.4f} {fold_mae:>8.4f} {fold_mse:>8.4f}")

        fold_results.append({
            'fold':       fold_id + 1,
            'L_mm':       L, 'W_mm': W, 'Ls_mm': Ls, 'Ws_mm': Ws,
            'R2':         fold_r2,
            'MAE_dB':     fold_mae,
            'MSE':        fold_mse,
        })

        # Collect predictions for plotting
        preds_df = test_df[['freq_GHz', 'L_mm', 'W_mm', 'Ls_mm', 'Ws_mm',
                             TARGET_COL]].copy()
        preds_df['S11_pred'] = y_pred
        preds_df['fold']     = fold_id + 1
        all_preds.append(preds_df)

    r2_vals  = [r['R2']     for r in fold_results]
    mae_vals = [r['MAE_dB'] for r in fold_results]

    print(f"  {'-'*58}")
    print(f"  {'MEAN':<6} {'':<28} {np.mean(r2_vals):>7.4f} {np.mean(mae_vals):>8.4f}")
    print(f"  {'STD':<6} {'':<28} {np.std(r2_vals):>7.4f} {np.std(mae_vals):>8.4f}")
    print(f"  {'MIN':<6} {'':<28} {np.min(r2_vals):>7.4f} {np.min(mae_vals):>8.4f}")
    print(f"{'='*65}")

    return fold_results, pd.concat(all_preds, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Final Model Training (on ALL data)
# ─────────────────────────────────────────────────────────────────────────────

def train_final_model(df: pd.DataFrame, model_name: str = 'MLP'):
    """
    After LOOCV evaluation, train the chosen model on the FULL dataset.
    This is the model saved for deployment (forward prediction + inverse design).

    Note: Final model is trained on all data — LOOCV metrics represent
    generalization performance, not this model's training error.
    """
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_s = scaler_X.fit_transform(X)
    y_s = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    model = build_mlp() if model_name == 'MLP' else build_gb()

    if model_name == 'MLP':
        model.fit(X_s, y_s)
    else:
        model.fit(X_s, y)

    return model, scaler_X, scaler_y


# ─────────────────────────────────────────────────────────────────────────────
# Compare MLP vs GB via LOOCV
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate():
    """
    Full pipeline:
    1. Load slotted dataset
    2. Run LOOCV for MLP
    3. Run LOOCV for GB
    4. Choose best model by mean LOOCV R²
    5. Train final model on ALL data
    6. Save to disk
    """
    print("\n📡 Slotted Microstrip Patch Antenna — ML Training Pipeline")
    print("=" * 65)

    df, df_configs = load_slotted_dataset()

    print(f"\n✔  Dataset loaded: {len(df):,} frequency-domain samples")
    print(f"   Slotted configurations: {df.groupby(GEO_COLS).ngroups}")
    print(f"\n   Per-configuration summary:")
    print(df_configs.to_string(index=False))

    # ── LOOCV for both models ────────────────────────────────────────────────
    mlp_folds, mlp_preds = run_loocv(df, 'MLP')
    gb_folds,  gb_preds  = run_loocv(df, 'GB')

    mlp_mean_r2 = np.mean([f['R2'] for f in mlp_folds])
    gb_mean_r2  = np.mean([f['R2'] for f in gb_folds])

    best_name  = 'MLP' if mlp_mean_r2 >= gb_mean_r2 else 'GB'
    best_folds = mlp_folds if best_name == 'MLP' else gb_folds

    print(f"\n  MLP mean LOOCV R² = {mlp_mean_r2:.4f}")
    print(f"  GB  mean LOOCV R² = {gb_mean_r2:.4f}")
    print(f"  → Best model: {best_name}")

    # ── LOOCV results summary ────────────────────────────────────────────────
    comparison = pd.DataFrame([
        {'Model': 'MLP Neural Network',
         'Mean R²': round(mlp_mean_r2, 4),
         'Std R²':  round(np.std([f['R2'] for f in mlp_folds]), 4),
         'Mean MAE (dB)': round(np.mean([f['MAE_dB'] for f in mlp_folds]), 4),
         'Min R²': round(min(f['R2'] for f in mlp_folds), 4)},
        {'Model': 'Gradient Boosting',
         'Mean R²': round(gb_mean_r2, 4),
         'Std R²':  round(np.std([f['R2'] for f in gb_folds]), 4),
         'Mean MAE (dB)': round(np.mean([f['MAE_dB'] for f in gb_folds]), 4),
         'Min R²': round(min(f['R2'] for f in gb_folds), 4)},
    ])
    print(f"\n  LOOCV Comparison Table:")
    print(comparison.to_string(index=False))

    # ── Train final model on ALL data ────────────────────────────────────────
    print(f"\n  Training final {best_name} model on all {len(df)} samples...")
    final_model, scaler_X, scaler_y = train_final_model(df, best_name)

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(os.path.join(MODEL_DIR, 'best_model.pkl'),    'wb') as f:
        pickle.dump(final_model, f)
    with open(os.path.join(MODEL_DIR, 'best_scaler_X.pkl'), 'wb') as f:
        pickle.dump(scaler_X, f)
    with open(os.path.join(MODEL_DIR, 'best_scaler_y.pkl'), 'wb') as f:
        pickle.dump(scaler_y, f)
    with open(os.path.join(MODEL_DIR, 'best_model_name.txt'), 'w') as f:
        f.write(best_name)

    # Save LOOCV fold-level results
    loocv_df = pd.DataFrame(best_folds)
    loocv_df.to_csv(os.path.join(MODEL_DIR, 'loocv_results.csv'), index=False)

    # Save predictions for plotting
    best_preds = mlp_preds if best_name == 'MLP' else gb_preds
    best_preds.to_csv(os.path.join(MODEL_DIR, 'loocv_predictions.csv'), index=False)

    # Save comparison
    comparison.to_csv(os.path.join(MODEL_DIR, 'model_comparison.csv'), index=False)

    print(f"\n✔  Models saved to models/")
    print(f"✔  LOOCV results saved to models/loocv_results.csv")
    print("=" * 65)

    return final_model, scaler_X, scaler_y, best_name, comparison, best_folds


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def load_best_model():
    """Load the saved best model and scalers from disk."""
    with open(os.path.join(MODEL_DIR, 'best_model.pkl'),    'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'best_scaler_X.pkl'), 'rb') as f:
        scaler_X = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'best_scaler_y.pkl'), 'rb') as f:
        scaler_y = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'best_model_name.txt'), 'r') as f:
        model_name = f.read().strip()
    return model, scaler_X, scaler_y, model_name


def predict_s11_curve(L_mm, W_mm, Ls_mm, Ws_mm,
                      freq_start=1.0, freq_end=3.5, n_points=300,
                      model=None, scaler_X=None, scaler_y=None, model_name=None):
    """
    Predict S11 vs Frequency for a slotted patch configuration.
    Loads model from disk if not provided.
    """
    if model is None:
        model, scaler_X, scaler_y, model_name = load_best_model()

    freqs  = np.linspace(freq_start, freq_end, n_points)
    X_pred = np.column_stack([
        freqs,
        np.full(n_points, L_mm),
        np.full(n_points, W_mm),
        np.full(n_points, Ls_mm),
        np.full(n_points, Ws_mm),
    ])
    X_s  = scaler_X.transform(X_pred)
    y_s  = model.predict(X_s)

    if model_name == 'MLP':
        s11 = scaler_y.inverse_transform(y_s.reshape(-1, 1)).ravel()
    else:
        s11 = y_s  # GB trained on raw y

    return freqs, s11


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    train_and_evaluate()
