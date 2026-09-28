"""
process_hfss_export.py
----------------------
Converts HFSS exported S11 CSV files into the unified dataset format
used by the ML pipeline.

Usage:
    python data/process_hfss_export.py --input_dir data/hfss_exports/ --output data/hfss_combined.csv

Expected input file naming convention:
    hfss_L{L}_W{W}_Ls{Ls}_Ws{Ws}.csv
    e.g.: hfss_L30_W40_Ls14_Ws4.csv

Expected CSV format from HFSS export:
    Freq [GHz],dB(S(1,1)) []
    1.0000,  -0.2341
    1.0174,  -0.3102
    ...

Output format (matches dataset_antenna.csv schema):
    Freq(GHz),length of patch in mm,width of patch in mm,Slot length in mm,slot width in mm,s11(dB)
"""

import os
import re
import argparse
from typing import Optional, Tuple
import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_COLUMNS = [
    'Freq(GHz)',
    'length of patch in mm',
    'width of patch in mm',
    'Slot length in mm',
    'slot width in mm',
    's11(dB)',
]

# Regex to parse dimensions from filename
FILENAME_PATTERN = re.compile(
    r'hfss_L([\d.]+)_W([\d.]+)_Ls([\d.]+)_Ws([\d.]+)\.csv',
    re.IGNORECASE
)


# ─────────────────────────────────────────────────────────────────────────────
# Parse Single HFSS Export File
# ─────────────────────────────────────────────────────────────────────────────

def parse_hfss_file(filepath: str) -> Optional[pd.DataFrame]:
    """
    Parse one HFSS-exported S11 CSV file.

    HFSS exports have a header row like:
        Freq [GHz],dB(S(1,1)) []
    or sometimes:
        "Freq [GHz]","dB(S(LumpPort1,LumpPort1)) []"

    Returns a DataFrame with columns matching OUTPUT_COLUMNS, or None on error.
    """
    filename = os.path.basename(filepath)

    # Parse (L, W, Ls, Ws) from filename
    match = FILENAME_PATTERN.match(filename)
    if not match:
        print(f"  ⚠️  Skipping (filename format unrecognized): {filename}")
        print(f"      Expected: hfss_L30_W40_Ls14_Ws4.csv")
        return None

    L_mm  = float(match.group(1))
    W_mm  = float(match.group(2))
    Ls_mm = float(match.group(3))
    Ws_mm = float(match.group(4))

    # Read the CSV — try multiple header formats
    try:
        raw = pd.read_csv(filepath, comment='!', skip_blank_lines=True)
    except Exception as e:
        print(f"  ❌ Error reading {filename}: {e}")
        return None

    # Find frequency column and S11 column (flexible matching)
    freq_col = None
    s11_col  = None
    for col in raw.columns:
        col_lower = col.lower().strip('"')
        if 'freq' in col_lower:
            freq_col = col
        if 'db' in col_lower or 's11' in col_lower or 's(1' in col_lower:
            s11_col = col

    if freq_col is None or s11_col is None:
        # Try positional (some HFSS versions export without proper header)
        if len(raw.columns) >= 2:
            freq_col = raw.columns[0]
            s11_col  = raw.columns[1]
        else:
            print(f"  ❌ Cannot identify frequency/S11 columns in {filename}")
            print(f"      Columns found: {list(raw.columns)}")
            return None

    try:
        freqs = pd.to_numeric(raw[freq_col], errors='coerce').dropna()
        s11   = pd.to_numeric(raw[s11_col],  errors='coerce').dropna()
        min_len = min(len(freqs), len(s11))
        freqs = freqs.iloc[:min_len].values
        s11   = s11.iloc[:min_len].values
    except Exception as e:
        print(f"  ❌ Error parsing numeric data in {filename}: {e}")
        return None

    if len(freqs) < 10:
        print(f"  ⚠️  Only {len(freqs)} data points in {filename} — skipping")
        return None

    df = pd.DataFrame({
        'Freq(GHz)':             freqs,
        'length of patch in mm': L_mm,
        'width of patch in mm':  W_mm,
        'Slot length in mm':     Ls_mm,
        'slot width in mm':      Ws_mm,
        's11(dB)':               s11,
    })[OUTPUT_COLUMNS]

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Validate Physical Constraints
# ─────────────────────────────────────────────────────────────────────────────

def validate_config(L, W, Ls, Ws) -> Tuple[bool, str]:
    """
    Check whether the geometric dimensions are physically sensible.
    Returns (is_valid, reason_if_invalid).
    """
    if Ls >= 0.6 * L:
        return False, f"Slot too long: Ls={Ls} ≥ 0.6*L={0.6*L:.1f}"
    if Ws >= 0.5 * W:
        return False, f"Slot too wide: Ws={Ws} ≥ 0.5*W={0.5*W:.1f}"
    if Ws >= Ls:
        return False, f"Slot not rectangular: Ws={Ws} ≥ Ls={Ls}"
    if Ls <= 0 or Ws <= 0:
        return False, "Slot dimensions must be positive (slotted only)"
    return True, ""


# ─────────────────────────────────────────────────────────────────────────────
# Process All Files in Directory
# ─────────────────────────────────────────────────────────────────────────────

def process_directory(input_dir: str, output_path: str,
                      merge_existing: bool = True) -> Optional[pd.DataFrame]:
    """
    Process all hfss_*.csv files in input_dir and write the combined dataset.

    Parameters
    ----------
    input_dir       : Folder containing HFSS-exported CSV files
    output_path     : Output combined CSV path
    merge_existing  : If True, append to existing dataset_antenna.csv
    """
    print(f"\n{'='*65}")
    print(f"  HFSS Export Processor")
    print(f"  Input directory : {input_dir}")
    print(f"  Output path     : {output_path}")
    print(f"{'='*65}\n")

    if not os.path.isdir(input_dir):
        raise ValueError(f"Input directory not found: {input_dir}")

    csv_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().startswith('hfss_') and f.lower().endswith('.csv')
    ])

    if not csv_files:
        print(f"  ❌ No hfss_*.csv files found in {input_dir}")
        print(f"     Rename your files to: hfss_L30_W40_Ls14_Ws4.csv")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    print(f"  Found {len(csv_files)} HFSS export files\n")

    frames = []
    stats  = {'ok': 0, 'skipped': 0, 'invalid': 0}

    for fname in csv_files:
        fpath  = os.path.join(input_dir, fname)
        match  = FILENAME_PATTERN.match(fname)

        if match:
            L, W, Ls, Ws = (float(match.group(i)) for i in range(1, 5))
            ok, reason = validate_config(L, W, Ls, Ws)
            if not ok:
                print(f"  ⚠️  {fname} — INVALID GEOMETRY: {reason}")
                stats['invalid'] += 1
                continue

        df = parse_hfss_file(fpath)
        if df is not None:
            n = len(df)
            freq_lo = df['Freq(GHz)'].min()
            freq_hi = df['Freq(GHz)'].max()
            s11_min = df['s11(dB)'].min()
            print(f"  ✅ {fname:42s}  {n:3d} pts  "
                  f"{freq_lo:.2f}–{freq_hi:.2f} GHz  S11_min={s11_min:.1f} dB")
            frames.append(df)
            stats['ok'] += 1
        else:
            stats['skipped'] += 1

    if not frames:
        print("\n  ❌ No valid data to combine.")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)

    # Optionally merge with existing Kaggle data
    existing_path = os.path.join(os.path.dirname(output_path),
                                 'dataset_antenna.csv')
    if merge_existing and os.path.exists(existing_path):
        existing = pd.read_csv(existing_path)
        existing.columns = OUTPUT_COLUMNS
        # Keep only slotted from existing
        existing = existing[(existing['Slot length in mm'] > 0) |
                            (existing['slot width in mm'] > 0)]
        combined = pd.concat([existing, combined], ignore_index=True)
        print(f"\n  Merged with existing {existing_path}: "
              f"{len(existing)} existing + {len(combined)-len(existing)} new rows")

    # Remove duplicates (same L,W,Ls,Ws,Freq rounded to 3 decimal places)
    combined = combined.round({'Freq(GHz)': 3})
    before   = len(combined)
    combined = combined.drop_duplicates(
        subset=['Freq(GHz)', 'length of patch in mm',
                'width of patch in mm', 'Slot length in mm', 'slot width in mm'])
    after    = len(combined)
    if before != after:
        print(f"  Removed {before-after} duplicate rows")

    combined = combined.sort_values(
        ['length of patch in mm', 'width of patch in mm',
         'Slot length in mm', 'slot width in mm', 'Freq(GHz)']
    ).reset_index(drop=True)

    combined.to_csv(output_path, index=False)

    # Summary
    n_configs = combined.groupby(
        ['length of patch in mm','width of patch in mm',
         'Slot length in mm','slot width in mm']
    ).ngroups

    print(f"\n{'='*65}")
    print(f"  SUMMARY")
    print(f"  Files processed successfully : {stats['ok']}")
    print(f"  Files skipped (bad format)  : {stats['skipped']}")
    print(f"  Files skipped (bad geometry): {stats['invalid']}")
    print(f"  Total rows in output        : {len(combined):,}")
    print(f"  Unique antenna configs      : {n_configs}")
    print(f"  Output saved to             : {output_path}")
    print(f"{'='*65}\n")

    if n_configs >= 60:
        print("  ✅ Dataset size adequate for conference paper submission!")
    elif n_configs >= 30:
        print("  ⚠️  Adequate for workshop/short paper. Aim for 60+ for conference.")
    else:
        print("  ❌ Not enough configurations yet. Run more HFSS simulations.")

    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Process HFSS S11 export CSV files into ML pipeline format'
    )
    parser.add_argument(
        '--input_dir', '-i',
        default='data/hfss_exports',
        help='Directory containing hfss_L*_W*_Ls*_Ws*.csv files (default: data/hfss_exports)'
    )
    parser.add_argument(
        '--output', '-o',
        default='data/hfss_combined.csv',
        help='Output combined CSV path (default: data/hfss_combined.csv)'
    )
    parser.add_argument(
        '--no_merge', action='store_true',
        help='Do NOT merge with existing dataset_antenna.csv'
    )
    args = parser.parse_args()

    process_directory(
        input_dir=args.input_dir,
        output_path=args.output,
        merge_existing=not args.no_merge
    )

    print("Next step:")
    print(f"  python models/slotted_model.py --data {args.output}")


if __name__ == '__main__':
    main()
