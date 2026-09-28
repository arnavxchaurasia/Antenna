"""
app.py — AI Antenna Designer (Slotted Microstrip Patch)
Research-grade Streamlit application.

Three tabs:
  1. Data Analysis   — S11 curves, LOOCV results, config table
  2. Forward Design  — predict S11(freq) for given (L, W, Ls, Ws)
  3. Inverse Design  — find (L, W, Ls, Ws) from target fr
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Antenna Designer — Slotted Patch",
    layout="wide",
    page_icon="📡",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero { background: linear-gradient(135deg,#1a1a2e 0%,#16213e 60%,#0f3460 100%);
        padding: 2rem 2.5rem 1.5rem; border-radius:16px; margin-bottom:1.5rem; }
.hero h1 { color:#e2e8f0; font-size:2.1rem; font-weight:700; margin:0 0 .4rem; }
.hero p  { color:#94a3b8; font-size:.95rem; margin:0; }
.hero .badge { display:inline-block; background:#0f3460; border:1px solid #334155;
               border-radius:6px; padding:2px 10px; color:#7dd3fc;
               font-size:.75rem; font-weight:600; margin-right:6px; }

.section-hdr { font-size:1.05rem; font-weight:700; color:#3b82f6;
               border-left:4px solid #3b82f6; padding-left:10px;
               margin:1.2rem 0 .6rem; }

[data-testid="stMetricValue"] { font-size:1.7rem!important; font-weight:700!important; color:#3b82f6!important; }
[data-testid="stMetricLabel"] { font-size:.75rem!important; color:#94a3b8!important;
                                 text-transform:uppercase; letter-spacing:.06em; }

.result-box { background:#0f172a; border:1px solid #1e293b; border-radius:12px;
              padding:1.2rem 1.4rem; margin-top:.8rem; }

[data-testid="stSidebar"] { background:linear-gradient(180deg,#0f172a 0%,#1e293b 100%); }
[data-testid="stSidebar"] * { color:#cbd5e1!important; }
[data-testid="stSidebar"] button { border-color:#334155!important; }

.stAlert { border-radius:10px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Hero Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>📡 AI Antenna Designer</h1>
  <p>Machine Learning Surrogate for Slot-Loaded Microstrip Patch Antennas</p>
  <br>
  <span class="badge">Real Simulation Data</span>
  <span class="badge">LOOCV Evaluation</span>
  <span class="badge">Differential Evolution</span>
  <span class="badge">S11 Prediction</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.divider()

    if st.button("🚀 Train ML Models (LOOCV)", use_container_width=True,
                 help="Runs configuration-level Leave-One-Out CV then saves the best model"):
        from models.slotted_model import train_and_evaluate
        with st.spinner("Running LOOCV training… this may take ~2–3 min"):
            try:
                *_, model_name, comparison, folds = train_and_evaluate()
                st.session_state['trained']  = True
                st.session_state['model_name'] = model_name
                st.session_state['comparison'] = comparison
                st.session_state['folds']    = folds
                st.success(f"✅ Best model: {model_name}")
            except Exception as e:
                st.error(f"Training failed: {e}")

    st.divider()

    # Show whether model is available
    model_ready = os.path.exists('models/best_model.pkl')
    if model_ready:
        with open('models/best_model_name.txt') as f:
            mname = f.read().strip()
        st.success(f"✔ Model ready ({mname})")
    else:
        st.warning("⚠ No model found yet. Click Train above.")

    st.divider()
    st.caption("📖 Data: CST Simulation · 7 Slotted Configs · 1,236 pts")
    st.caption("🔬 Evaluation: Configuration-Level LOOCV")

# ─────────────────────────────────────────────────────────────────────────────
# Load Data (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    from data.data_loader import load_slotted_dataset
    return load_slotted_dataset()

df, df_configs = get_data()

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊 Data Analysis",
    "🔮 Forward Design",
    "🎯 Inverse Design",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    # ── Top Stats ─────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Data Points", f"{len(df):,}")
    c2.metric("Slotted Configs",   str(df.groupby(['L_mm','W_mm','Ls_mm','Ws_mm']).ngroups))
    c3.metric("Freq Range",        f"{df.freq_GHz.min():.1f}–{df.freq_GHz.max():.1f} GHz")
    c4.metric("S11 Range",         f"{df.S11_dB.min():.1f} to {df.S11_dB.max():.1f} dB")

    st.divider()

    # ── S11 Curves ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">S11 vs Frequency — All Slotted Configurations</div>',
                unsafe_allow_html=True)

    palette = ['#3b82f6','#f59e0b','#10b981','#ef4444','#8b5cf6','#ec4899','#06b6d4']
    grp_dict = {k: g for k, g in df.groupby(['L_mm','W_mm','Ls_mm','Ws_mm'])}

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#111827')
    for i, (key, grp) in enumerate(grp_dict.items()):
        g = grp.sort_values('freq_GHz')
        L, W, Ls, Ws = key
        ax.plot(g.freq_GHz, g.S11_dB, color=palette[i % len(palette)],
                lw=2.0, label=f"L={L},W={W},Ls={Ls},Ws={Ws}", alpha=0.9)
    ax.axhline(-10, color='white', ls='--', lw=1.0, alpha=0.4, label='−10 dB threshold')
    ax.set_xlabel("Frequency (GHz)", color='#94a3b8', fontsize=11)
    ax.set_ylabel("S11 (dB)",        color='#94a3b8', fontsize=11)
    ax.set_title("CST Simulation: S11 vs Frequency (Slotted Configurations)",
                 color='#e2e8f0', fontsize=12)
    ax.legend(facecolor='#1e293b', labelcolor='white', fontsize=7,
              loc='lower right', framealpha=0.8)
    ax.tick_params(colors='#94a3b8')
    for sp in ['bottom','left']: ax.spines[sp].set_color('#334155')
    for sp in ['top','right']:   ax.spines[sp].set_visible(False)
    st.pyplot(fig); plt.close(fig)

    # ── Config Table ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Extracted Antenna Performance — Per Configuration</div>',
                unsafe_allow_html=True)

    st.info("**Resonant frequency** (fr) = frequency at minimum S11. "
            "**Bandwidth** = frequency range where S11 ≤ −10 dB.", icon="ℹ️")

    display_cfg = df_configs.copy()
    display_cfg.columns = ['L (mm)', 'W (mm)', 'Ls (mm)', 'Ws (mm)',
                            'fr (GHz)', 'S11 min (dB)', 'BW (MHz)', '# pts']
    st.dataframe(
        display_cfg.style.format({'fr (GHz)': '{:.4f}', 'S11 min (dB)': '{:.2f}',
                                   'BW (MHz)': '{:.1f}'}),
        use_container_width=True
    )

    # ── LOOCV Results ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">LOOCV Model Evaluation Results</div>',
                unsafe_allow_html=True)

    loocv_path = 'models/loocv_results.csv'
    comp_path  = 'models/model_comparison.csv'
    pred_path  = 'models/loocv_predictions.csv'

    if not os.path.exists(loocv_path):
        st.info("Train the models first (sidebar → 🚀 Train ML Models) to see LOOCV metrics here.")
    else:
        loocv_df = pd.read_csv(loocv_path)
        comp_df  = pd.read_csv(comp_path)
        pred_df  = pd.read_csv(pred_path)

        st.markdown("#### Model Comparison (LOOCV on unseen geometries)")
        st.table(comp_df.style.format({'Mean R²': '{:.4f}', 'Std R²': '{:.4f}',
                                        'Mean MAE (dB)': '{:.4f}',
                                        'Min R²': '{:.4f}'}))

        st.markdown("#### Per-Fold Results")
        fold_display = loocv_df.copy()
        fold_display['Config'] = fold_display.apply(
            lambda r: f"L={r.L_mm},W={r.W_mm},Ls={r.Ls_mm},Ws={r.Ws_mm}", axis=1)
        fold_display = fold_display[['fold','Config','R2','MAE_dB','MSE']]
        fold_display.columns = ['Fold','Hold-out Config','R²','MAE (dB)','MSE']
        st.dataframe(
            fold_display.style.format({'R²': '{:.4f}', 'MAE (dB)': '{:.4f}', 'MSE': '{:.4f}'}),
            use_container_width=True,
        )

        # Per-fold bar chart
        fig2, axes = plt.subplots(1, 2, figsize=(11, 4))
        fig2.patch.set_facecolor('#0f172a')
        for ax2 in axes: ax2.set_facecolor('#111827')

        folds_x = [f"Fold {r}" for r in loocv_df.fold]
        bars_r2  = axes[0].bar(folds_x, loocv_df.R2,     color=palette[:len(loocv_df)], alpha=0.85)
        bars_mae = axes[1].bar(folds_x, loocv_df.MAE_dB, color=palette[:len(loocv_df)], alpha=0.85)

        axes[0].set_title("R² per Fold (higher = better)", color='#e2e8f0')
        axes[1].set_title("MAE per Fold (lower = better)", color='#e2e8f0')
        axes[0].set_ylabel("R²",       color='#94a3b8')
        axes[1].set_ylabel("MAE (dB)", color='#94a3b8')
        for ax2 in axes:
            ax2.tick_params(colors='#94a3b8', rotation=30, axis='x', labelsize=7)
            ax2.tick_params(colors='#94a3b8', axis='y')
            for sp in ['top','right']: ax2.spines[sp].set_visible(False)
            for sp in ['bottom','left']: ax2.spines[sp].set_color('#334155')
        axes[0].axhline(loocv_df.R2.mean(),     color='white', ls='--', lw=1, alpha=0.5)
        axes[1].axhline(loocv_df.MAE_dB.mean(), color='white', ls='--', lw=1, alpha=0.5)

        fig2.tight_layout()
        st.pyplot(fig2); plt.close(fig2)

        # Predicted vs Actual per fold
        st.markdown('<div class="section-hdr">Predicted vs Actual S11 Curves (LOOCV Hold-outs)</div>',
                    unsafe_allow_html=True)

        n_folds = pred_df.fold.nunique()
        ncols   = 2
        nrows   = (n_folds + 1) // ncols
        fig3, axes3 = plt.subplots(nrows, ncols, figsize=(12, nrows * 3.4))
        fig3.patch.set_facecolor('#0f172a')
        axes3_flat = axes3.flatten() if n_folds > 1 else [axes3]

        for i, (fid, fgrp) in enumerate(pred_df.groupby('fold')):
            ax3 = axes3_flat[i]
            ax3.set_facecolor('#111827')
            fgrp = fgrp.sort_values('freq_GHz')
            ax3.plot(fgrp.freq_GHz, fgrp.S11_dB,   color='#3b82f6', lw=2.0, label='Actual (CST)')
            ax3.plot(fgrp.freq_GHz, fgrp.S11_pred,  color='#f59e0b', lw=1.5, ls='--', label='ML Predicted')
            ax3.axhline(-10, color='#64748b', ls=':', lw=0.8)
            L,W,Ls,Ws = fgrp[['L_mm','W_mm','Ls_mm','Ws_mm']].iloc[0]
            ax3.set_title(f"Hold-out: L={L},W={W},Ls={Ls},Ws={Ws}",
                          color='#e2e8f0', fontsize=8)
            ax3.set_xlabel("Freq (GHz)", color='#64748b', fontsize=7)
            ax3.set_ylabel("S11 (dB)",  color='#64748b', fontsize=7)
            ax3.tick_params(colors='#64748b', labelsize=6)
            for sp in ['top','right']: ax3.spines[sp].set_visible(False)
            for sp in ['bottom','left']: ax3.spines[sp].set_color('#334155')
            if i == 0: ax3.legend(facecolor='#1e293b', labelcolor='white', fontsize=6)

        for j in range(i + 1, len(axes3_flat)):
            axes3_flat[j].set_visible(False)

        fig3.suptitle("ML Surrogate: Predicted vs Actual S11 (LOOCV Hold-outs)",
                      color='#e2e8f0', fontsize=11)
        fig3.tight_layout()
        st.pyplot(fig3); plt.close(fig3)

    # ── Raw data ──────────────────────────────────────────────────────────────
    with st.expander("📄 View Raw Data"):
        st.dataframe(df.head(300), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — FORWARD DESIGN
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-hdr">Forward Design: Predict S11 Curve</div>',
                unsafe_allow_html=True)
    st.markdown(
        "Enter the **physical dimensions** of a slotted microstrip patch — "
        "the ML surrogate will predict its **S11 vs frequency** response."
    )

    # Parameter ranges from dataset
    col_L, col_W, col_Ls, col_Ws = st.columns(4)
    inp_L  = col_L.number_input("Patch Length L (mm)",  min_value=25.0,  max_value=95.0,  value=35.5, step=0.5)
    inp_W  = col_W.number_input("Patch Width W (mm)",   min_value=24.0,  max_value=135.0, value=26.0, step=0.5)
    inp_Ls = col_Ls.number_input("Slot Length Ls (mm)", min_value=0.0,   max_value=90.0,  value=19.5, step=0.5)
    inp_Ws = col_Ws.number_input("Slot Width Ws (mm)",  min_value=0.0,   max_value=120.0, value=12.0, step=1.0)

    col_fs, col_fe = st.columns(2)
    freq_s = col_fs.number_input("Sweep Start (GHz)", 0.5, 5.0, 1.0, step=0.1)
    freq_e = col_fe.number_input("Sweep End (GHz)",   0.5, 5.0, 3.5, step=0.1)

    if inp_Ls == 0 and inp_Ws == 0:
        st.warning("⚠️ Model trained on slotted patches only. "
                   "Set Ls > 0 or Ws > 0 for reliable predictions.")

    if st.button("⚡ Predict S11 Curve", key="btn_fwd"):
        if not os.path.exists('models/best_model.pkl'):
            st.error("No trained model found — click '🚀 Train ML Models' in the sidebar first.")
        else:
            from models.slotted_model import predict_s11_curve
            with st.spinner("Computing S11 curve…"):
                freqs, s11 = predict_s11_curve(
                    inp_L, inp_W, inp_Ls, inp_Ws,
                    freq_start=freq_s, freq_end=freq_e, n_points=400)

            # Extract key performance indicators
            idx_fr  = np.argmin(s11)
            fr_pred = freqs[idx_fr]
            s11_min = s11[idx_fr]
            below   = freqs[s11 <= -10.0]
            bw_pred = float((below[-1] - below[0]) * 1000) if len(below) >= 2 else 0.0

            ka, kb, kc = st.columns(3)
            ka.metric("Resonant Frequency fr", f"{fr_pred:.3f} GHz")
            kb.metric("S11 Minimum",           f"{s11_min:.2f} dB")
            kc.metric("−10 dB Bandwidth",      f"{bw_pred:.1f} MHz")

            if s11_min <= -10:
                st.success(f"✅ This configuration achieves impedance matching (S11 ≤ −10 dB at {fr_pred:.3f} GHz).")
            else:
                st.warning(f"⚠️ S11 minimum is {s11_min:.2f} dB — does not meet −10 dB matching criterion.")

            # Plot
            fig_fwd, ax_fwd = plt.subplots(figsize=(10, 4.5))
            fig_fwd.patch.set_facecolor('#0f172a')
            ax_fwd.set_facecolor('#111827')
            ax_fwd.plot(freqs, s11, color='#3b82f6', lw=2.5, label='Predicted S11')
            ax_fwd.axhline(-10, color='#f59e0b', ls='--', lw=1.2, alpha=0.7, label='−10 dB threshold')
            ax_fwd.axvline(fr_pred, color='#ef4444', ls=':', lw=1.5,
                           label=f'fr = {fr_pred:.3f} GHz')
            if bw_pred > 0:
                ax_fwd.fill_between(freqs, s11, -10,
                                    where=(s11 <= -10),
                                    alpha=0.15, color='#10b981', label='Matched Band')
            ax_fwd.set_xlabel("Frequency (GHz)", color='#94a3b8', fontsize=11)
            ax_fwd.set_ylabel("S11 (dB)",        color='#94a3b8', fontsize=11)
            ax_fwd.set_title(
                f"Predicted S11 — L={inp_L}, W={inp_W}, Ls={inp_Ls}, Ws={inp_Ws} mm",
                color='#e2e8f0', fontsize=11)
            ax_fwd.legend(facecolor='#1e293b', labelcolor='white', fontsize=9)
            ax_fwd.tick_params(colors='#94a3b8')
            for sp in ['top','right']: ax_fwd.spines[sp].set_visible(False)
            for sp in ['bottom','left']: ax_fwd.spines[sp].set_color('#334155')
            st.pyplot(fig_fwd); plt.close(fig_fwd)

    # ── Compare against training configs ──────────────────────────────────────
    st.divider()
    st.markdown('<div class="section-hdr">Compare Against Training Configurations</div>',
                unsafe_allow_html=True)
    if os.path.exists('models/best_model.pkl'):
        from models.slotted_model import predict_s11_curve as _pred_curve
        fig_cmp, ax_cmp = plt.subplots(figsize=(11, 4.5))
        fig_cmp.patch.set_facecolor('#0f172a')
        ax_cmp.set_facecolor('#111827')

        for i, (key, grp) in enumerate(df.groupby(['L_mm','W_mm','Ls_mm','Ws_mm'])):
            g = grp.sort_values('freq_GHz')
            ax_cmp.plot(g.freq_GHz, g.S11_dB, color=palette[i % len(palette)],
                        lw=1.2, alpha=0.45, label=f"CST: L={key[0]},Ls={key[2]}")

        # Overlay user input
        try:
            fr_u, s11_u = _pred_curve(inp_L, inp_W, inp_Ls, inp_Ws,
                                      freq_start=freq_s, freq_end=freq_e, n_points=300)
            ax_cmp.plot(fr_u, s11_u, color='white', lw=2.5, ls='-', label='Your Config (predicted)')
        except Exception:
            pass

        ax_cmp.axhline(-10, color='#475569', ls='--', lw=0.8)
        ax_cmp.set_xlabel("Frequency (GHz)", color='#94a3b8')
        ax_cmp.set_ylabel("S11 (dB)",        color='#94a3b8')
        ax_cmp.set_title("Your Config vs Training Data", color='#e2e8f0')
        ax_cmp.legend(facecolor='#1e293b', labelcolor='white', fontsize=7,
                      loc='lower right')
        ax_cmp.tick_params(colors='#94a3b8')
        for sp in ['top','right']: ax_cmp.spines[sp].set_visible(False)
        for sp in ['bottom','left']: ax_cmp.spines[sp].set_color('#334155')
        st.pyplot(fig_cmp); plt.close(fig_cmp)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — INVERSE DESIGN
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-hdr">Inverse Design: Find Optimal Slotted Patch Dimensions</div>',
                unsafe_allow_html=True)
    st.markdown(
        "Specify a **target resonant frequency** and **S11 requirement**. "
        "The Differential Evolution optimizer will search the design space "
        "using the ML surrogate to find dimensions that meet your specification."
    )

    col_t1, col_t2 = st.columns(2)
    target_fr  = col_t1.number_input("Target Resonant Frequency (GHz)",
                                     min_value=1.0, max_value=4.0,
                                     value=2.4, step=0.05)
    target_s11 = col_t2.number_input("Required S11 ≤ (dB)",
                                     min_value=-40.0, max_value=-5.0,
                                     value=-15.0, step=1.0)

    with st.expander("⚙️ Advanced Optimizer Settings"):
        maxiter  = st.slider("Max DE Iterations",    100, 1000, 500)
        popsize  = st.slider("Population Size (×dim)", 5,  25,  15)

    st.caption(f"🔍 Search bounds: L∈[29–90], W∈[26–132], Ls∈[14–86], Ws∈[12–116] mm (from dataset range)")

    if st.button("🚀 Find Optimal Dimensions", key="btn_inv"):
        if not os.path.exists('models/best_model.pkl'):
            st.error("No trained model — click '🚀 Train ML Models' in the sidebar first.")
        else:
            from models.inverse_optimizer import inverse_design_slotted
            with st.spinner(f"Differential Evolution searching for {target_fr} GHz design…"):
                try:
                    res  = inverse_design_slotted(
                        target_fr, target_s11,
                        maxiter=maxiter, popsize=popsize)
                    opt  = res['optimal_dimensions']
                    perf = res['predicted_performance']
                    err  = abs(perf['fr_GHz'] - target_fr) / target_fr * 100

                    status = "✅ Converged" if res['converged'] else "⚠️ Reached max iterations"
                    st.success(f"{status} — DE cost = {res['de_cost']:.5f}")

                    # Dimensions
                    st.markdown("#### 🔩 Recommended Slotted Patch Dimensions")
                    d1, d2, d3, d4 = st.columns(4)
                    d1.metric("Patch Length L",  f"{opt['L_mm']} mm")
                    d2.metric("Patch Width W",   f"{opt['W_mm']} mm")
                    d3.metric("Slot Length Ls",  f"{opt['Ls_mm']} mm")
                    d4.metric("Slot Width Ws",   f"{opt['Ws_mm']} mm")

                    # Performance
                    st.markdown("#### 📈 Predicted Performance")
                    p1, p2, p3 = st.columns(3)
                    p1.metric("Resonant Frequency",
                              f"{perf['fr_GHz']:.4f} GHz",
                              delta=f"{perf['fr_GHz'] - target_fr:+.4f} GHz from target")
                    p2.metric("S11 Minimum",   f"{perf['S11_min_dB']:.2f} dB")
                    p3.metric("−10 dB Bandwidth", f"{perf['BW_MHz']:.1f} MHz")

                    if err < 1.0:
                        st.success(f"🎯 Frequency error: **{err:.3f}%** — Excellent match!")
                    elif err < 3.0:
                        st.warning(f"⚠️ Frequency error: **{err:.2f}%** — Acceptable, but near edge of design space.")
                    else:
                        st.error(f"❌ Frequency error: **{err:.2f}%** — Poor match. Try adjusting bounds.")

                    # S11 curve
                    freqs_c = np.array(res['freq_curve'])
                    s11_c   = np.array(res['s11_curve'])

                    fig_inv, ax_inv = plt.subplots(figsize=(10, 4.5))
                    fig_inv.patch.set_facecolor('#0f172a')
                    ax_inv.set_facecolor('#111827')
                    ax_inv.plot(freqs_c, s11_c, color='#3b82f6', lw=2.5, label='Predicted S11')
                    ax_inv.axhline(-10, color='#f59e0b', ls='--', lw=1.2, alpha=0.7, label='−10 dB')
                    ax_inv.axhline(target_s11, color='#ef4444', ls=':', lw=1.2,
                                   label=f'Target S11 ({target_s11} dB)')
                    ax_inv.axvline(target_fr,       color='#64748b', ls=':', lw=1.2,
                                   label=f'Target fr ({target_fr} GHz)')
                    ax_inv.axvline(perf['fr_GHz'],  color='#10b981', ls='-', lw=1.5,
                                   label=f"Achieved fr ({perf['fr_GHz']:.3f} GHz)")
                    below_c = freqs_c[s11_c <= -10]
                    if len(below_c) >= 2:
                        ax_inv.fill_between(freqs_c, s11_c, -10,
                                            where=(s11_c <= -10),
                                            alpha=0.15, color='#10b981', label='Matched Band')

                    ax_inv.set_xlabel("Frequency (GHz)", color='#94a3b8', fontsize=11)
                    ax_inv.set_ylabel("S11 (dB)",        color='#94a3b8', fontsize=11)
                    ax_inv.set_title(
                        f"Optimized Slotted Patch — L={opt['L_mm']}, W={opt['W_mm']}, "
                        f"Ls={opt['Ls_mm']}, Ws={opt['Ws_mm']} mm",
                        color='#e2e8f0', fontsize=11)
                    ax_inv.legend(facecolor='#1e293b', labelcolor='white', fontsize=8)
                    ax_inv.tick_params(colors='#94a3b8')
                    for sp in ['top','right']: ax_inv.spines[sp].set_visible(False)
                    for sp in ['bottom','left']: ax_inv.spines[sp].set_color('#334155')
                    st.pyplot(fig_inv); plt.close(fig_inv)

                except Exception as e:
                    st.error(f"Optimization error: {e}")

    # ── Limitation note ───────────────────────────────────────────────────────
    st.divider()
    st.info(
        "**Research Note:** The ML surrogate was trained on 7 slotted configurations from CST simulation. "
        "Inverse design accuracy is highest when the target frequency falls within the training range "
        "(approximately 1.5–3.5 GHz). Results outside this range should be validated by EM simulation.",
        icon="🔬"
    )
