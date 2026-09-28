"""
inverse_optimizer.py
--------------------
Inverse design for slotted microstrip patch antennas using
Differential Evolution (DE) + the trained ML surrogate.

Problem:
    Given a target resonant frequency (and optional S11 target),
    find the geometric dimensions (L, W, Ls, Ws) that achieve it.

Approach:
    - DE explores the continuous 4D design space
    - At each candidate, the ML surrogate predicts the S11 curve
    - Resonant frequency is extracted as the frequency of minimum S11
    - A cost function penalizes frequency deviation and S11 violation
    - DE minimizes cost over many generations

Bounds are derived from the dataset parameter ranges:
    L_mm  : 29.4 – 87.0 mm
    W_mm  : 26.0 – 130.0 mm
    Ls_mm : 14.0 – 85.0 mm
    Ws_mm : 12.0 – 115.0 mm
"""

import os
import numpy as np
import warnings
from scipy.optimize import differential_evolution

warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.slotted_model import load_best_model, predict_s11_curve


# ─────────────────────────────────────────────────────────────────────────────
# Design Space Bounds (from dataset)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_BOUNDS = [
    (29.0, 90.0),    # L_mm
    (26.0, 132.0),   # W_mm
    (14.0, 86.0),    # Ls_mm
    (12.0, 116.0),   # Ws_mm
]


# ─────────────────────────────────────────────────────────────────────────────
# Cost Function
# ─────────────────────────────────────────────────────────────────────────────

def _cost(x, target_fr, target_s11, model, scaler_X, scaler_y,
          model_name, w_fr=1.0, w_s11=0.4, sweep_pts=30):
    """
    DE objective function.

    x           : [L_mm, W_mm, Ls_mm, Ws_mm]
    target_fr   : Target resonant frequency (GHz)
    target_s11  : Target S11 minimum (dB) — the model must meet or exceed this
    sweep_pts   : Number of frequency points in evaluation window
    """
    L_mm, W_mm, Ls_mm, Ws_mm = x

    # Dense sweep within ±0.6 GHz of target
    f_lo = max(0.5, target_fr - 0.6)
    f_hi = target_fr + 0.6
    freqs = np.linspace(f_lo, f_hi, sweep_pts)

    X_pred = np.column_stack([
        freqs,
        np.full(sweep_pts, L_mm),
        np.full(sweep_pts, W_mm),
        np.full(sweep_pts, Ls_mm),
        np.full(sweep_pts, Ws_mm),
    ])
    X_s = scaler_X.transform(X_pred)
    y_s = model.predict(X_s)
    if model_name == 'MLP':
        s11 = scaler_y.inverse_transform(y_s.reshape(-1, 1)).ravel()
    else:
        s11 = y_s

    # Extract resonant frequency
    idx_min  = np.argmin(s11)
    pred_fr  = freqs[idx_min]
    pred_s11 = s11[idx_min]

    # Frequency error (normalized)
    error_fr = abs(pred_fr - target_fr) / max(target_fr, 1e-6)

    # S11 penalty: only penalize if pred_s11 > target_s11 (worse matching)
    error_s11 = max(0.0, (pred_s11 - target_s11) / abs(target_s11 + 1e-6))

    return w_fr * error_fr + w_s11 * error_s11


# ─────────────────────────────────────────────────────────────────────────────
# Inverse Design
# ─────────────────────────────────────────────────────────────────────────────

def inverse_design_slotted(
    target_fr: float,
    target_s11: float = -15.0,
    bounds=None,
    maxiter: int = 500,
    popsize: int = 15,
    tol: float = 1e-5,
    verbose: bool = False,
):
    """
    Inverse design for slotted microstrip patch antenna.

    Parameters
    ----------
    target_fr   : Desired resonant frequency in GHz
    target_s11  : Required minimum S11 (dB), e.g. -15 dB
    bounds      : Optional list of (min, max) for [L, W, Ls, Ws]
    maxiter     : DE maximum generations
    popsize     : DE population size multiplier
    tol         : DE convergence tolerance

    Returns
    -------
    dict with keys:
        optimal_dimensions   : Dict of L_mm, W_mm, Ls_mm, Ws_mm
        predicted_performance: Dict of fr_GHz, S11_min_dB, BW_MHz
        freq_curve           : List of frequency points
        s11_curve            : List of predicted S11 values
        de_cost              : Final optimizer cost
        converged            : Whether DE converged within maxiter
    """
    model, scaler_X, scaler_y, model_name = load_best_model()

    if bounds is None:
        bounds = DEFAULT_BOUNDS

    result = differential_evolution(
        func=_cost,
        bounds=bounds,
        args=(target_fr, target_s11, model, scaler_X, scaler_y, model_name),
        strategy='best1bin',
        maxiter=maxiter,
        popsize=popsize,
        tol=tol,
        seed=42,
        disp=verbose,
        polish=True,   # L-BFGS-B refinement after global search
    )

    opt_L, opt_W, opt_Ls, opt_Ws = result.x

    # Full S11 curve for the found configuration
    freqs, s11 = predict_s11_curve(
        opt_L, opt_W, opt_Ls, opt_Ws,
        freq_start=max(0.5, target_fr - 1.0),
        freq_end=target_fr + 1.0,
        n_points=200,
        model=model, scaler_X=scaler_X,
        scaler_y=scaler_y, model_name=model_name,
    )

    idx_min  = np.argmin(s11)
    pred_fr  = float(freqs[idx_min])
    pred_s11 = float(s11[idx_min])

    below   = freqs[s11 <= -10.0]
    bw_mhz  = float((below[-1] - below[0]) * 1000) \
               if len(below) >= 2 else 0.0

    return {
        'optimal_dimensions': {
            'L_mm':  round(float(opt_L),  2),
            'W_mm':  round(float(opt_W),  2),
            'Ls_mm': round(float(opt_Ls), 2),
            'Ws_mm': round(float(opt_Ws), 2),
        },
        'predicted_performance': {
            'fr_GHz':     round(pred_fr, 4),
            'S11_min_dB': round(pred_s11, 2),
            'BW_MHz':     round(bw_mhz, 1),
        },
        'freq_curve':  freqs.tolist(),
        's11_curve':   s11.tolist(),
        'de_cost':     float(result.fun),
        'converged':   bool(result.success),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI Test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    targets = [(2.0, -15), (2.4, -20), (2.8, -15)]
    for fr, s11_t in targets:
        print(f"\n🎯 Target: fr={fr} GHz, S11≤{s11_t} dB")
        try:
            res = inverse_design_slotted(fr, s11_t, verbose=False)
            d   = res['optimal_dimensions']
            p   = res['predicted_performance']
            err = abs(p['fr_GHz'] - fr) / fr * 100
            print(f"   Dims  : L={d['L_mm']} W={d['W_mm']} Ls={d['Ls_mm']} Ws={d['Ws_mm']} mm")
            print(f"   Result: fr={p['fr_GHz']} GHz  S11={p['S11_min_dB']} dB  BW={p['BW_MHz']} MHz")
            print(f"   Error : {err:.2f}%  (cost={res['de_cost']:.5f})")
        except FileNotFoundError:
            print("   ⚠  Model not found — run models/slotted_model.py first")
