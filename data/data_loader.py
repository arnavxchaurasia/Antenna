"""
data_loader.py
--------------
Single source of truth for loading, cleaning, and preparing the
slotted microstrip patch antenna dataset.

Data source: Kaggle — shreyasinha/dataset-containing-antenna-parameters
             (or local copy at data/dataset_antenna.csv)

Schema:
  Freq(GHz)             : Frequency sweep point
  length of patch in mm : Patch length L
  width of patch in mm  : Patch width W
  Slot length in mm     : Slot length Ls
  slot width in mm      : Slot width Ws
  s11(dB)               : Return loss at that frequency

Only slotted configurations (Ls > 0 OR Ws > 0) are kept.
"""

import os
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

LOCAL_CSV   = os.path.join(os.path.dirname(__file__), 'dataset_antenna.csv')
KAGGLE_SLUG = 'shreyasinha/dataset-containing-antenna-parameters'

COL_RENAME = {
    'Freq(GHz)':             'freq_GHz',
    'length of patch in mm': 'L_mm',
    'width of patch in mm':  'W_mm',
    'Slot length in mm':     'Ls_mm',
    'slot width in mm':      'Ws_mm',
    's11(dB)':               'S11_dB',
}

FEATURE_COLS = ['freq_GHz', 'L_mm', 'W_mm', 'Ls_mm', 'Ws_mm']
TARGET_COL   = 'S11_dB'
GEO_COLS     = ['L_mm', 'W_mm', 'Ls_mm', 'Ws_mm']
S11_THRESHOLD = -10.0   # dB — standard impedance-match criterion


# ─────────────────────────────────────────────────────────────────────────────
# 1. Raw Load
# ─────────────────────────────────────────────────────────────────────────────

def load_raw(prefer_local: bool = True) -> pd.DataFrame:
    """
    Load the raw CSV.  Tries local path first; falls back to Kaggle download.
    """
    if prefer_local and os.path.exists(LOCAL_CSV):
        df = pd.read_csv(LOCAL_CSV)
    else:
        try:
            import kagglehub
            path = kagglehub.dataset_download(KAGGLE_SLUG)
            df = pd.read_csv(os.path.join(path, 'dataset_antenna.csv'))
        except Exception as e:
            raise FileNotFoundError(
                f"Local CSV not found at {LOCAL_CSV} and Kaggle download "
                f"failed: {e}"
            )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Clean & Filter
# ─────────────────────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns, drop NaNs, keep ONLY slotted configurations.
    Returns a clean DataFrame.
    """
    df = df.rename(columns=COL_RENAME).dropna()

    # Keep slotted patches only (Ls > 0 OR Ws > 0)
    slotted_mask = (df['Ls_mm'] > 0) | (df['Ws_mm'] > 0)
    df = df[slotted_mask].copy().reset_index(drop=True)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Per-Configuration Performance Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_config_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each unique antenna configuration (L, W, Ls, Ws) compute:
      fr_GHz     : Resonant frequency (freq at minimum S11)
      S11_min_dB : Deepest return loss (best impedance match)
      BW_MHz     : -10 dB bandwidth (0 if S11 never reaches -10 dB)

    Returns a summary DataFrame with one row per configuration.
    """
    records = []
    for key, grp in df.groupby(GEO_COLS):
        L_mm, W_mm, Ls_mm, Ws_mm = key
        grp = grp.sort_values('freq_GHz')
        freqs = grp['freq_GHz'].values
        s11   = grp['S11_dB'].values

        idx_min  = np.argmin(s11)
        fr       = freqs[idx_min]
        s11_min  = s11[idx_min]

        below = freqs[s11 <= S11_THRESHOLD]
        bw_mhz = round(float((below[-1] - below[0]) * 1000), 1) \
                 if len(below) >= 2 else 0.0

        records.append({
            'L_mm':      L_mm,
            'W_mm':      W_mm,
            'Ls_mm':     Ls_mm,
            'Ws_mm':     Ws_mm,
            'fr_GHz':    round(fr, 4),
            'S11_min_dB': round(s11_min, 4),
            'BW_MHz':    bw_mhz,
            'n_pts':     len(grp),
        })

    return pd.DataFrame(records).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Configuration-Level LOOCV Splits
# ─────────────────────────────────────────────────────────────────────────────

def loocv_splits(df: pd.DataFrame):
    """
    Yields (fold_id, train_df, test_df) for Leave-One-Out
    Cross-Validation at the CONFIGURATION level.

    At each fold, one unique (L, W, Ls, Ws) configuration is held out
    as the test set.  The remaining configurations form the training set.

    This is the only honest evaluation strategy when the number of
    unique geometries is small (< 20), because a random row-level split
    would leak configuration identity into the test set.
    """
    configs = df.groupby(GEO_COLS).groups.keys()
    configs = sorted(configs)

    for fold_id, held_out in enumerate(configs):
        L, W, Ls, Ws = held_out
        mask = (
            (df['L_mm']  == L)  &
            (df['W_mm']  == W)  &
            (df['Ls_mm'] == Ls) &
            (df['Ws_mm'] == Ws)
        )
        test_df  = df[mask].copy()
        train_df = df[~mask].copy()
        yield fold_id, held_out, train_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
# 5. Convenience: Load Everything
# ─────────────────────────────────────────────────────────────────────────────

def load_slotted_dataset():
    """
    Top-level convenience function.
    Returns:
        df        : Cleaned slotted-only DataFrame
        df_configs: Per-configuration performance summary
    """
    raw = load_raw()
    df  = clean(raw)
    df_configs = extract_config_features(df)
    return df, df_configs


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    df, df_cfg = load_slotted_dataset()

    print("=" * 65)
    print("  Slotted Patch Dataset Summary")
    print("=" * 65)
    print(f"  Total rows       : {len(df):,}")
    print(f"  Unique configs   : {df.groupby(GEO_COLS).ngroups}")
    print(f"  Freq range       : {df.freq_GHz.min():.3f} – {df.freq_GHz.max():.3f} GHz")
    print(f"  S11 range        : {df.S11_dB.min():.2f} – {df.S11_dB.max():.2f} dB")
    print()
    print("  Per-Configuration Performance:")
    print(df_cfg.to_string(index=False))
    print()
    print("  LOOCV folds:")
    for fid, key, tr, te in loocv_splits(df):
        print(f"    Fold {fid+1}: hold-out={key}  train={len(tr)} pts  test={len(te)} pts")
