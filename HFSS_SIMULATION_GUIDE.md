# HFSS Simulation Guide — Slotted Microstrip Patch Antenna Dataset

## Goal
Generate **~100 unique slotted antenna configurations** to train a properly 
generalizing ML surrogate. With 100 configs you will have publishable LOOCV 
results and can legitimately claim a research contribution.

---

## 1. Antenna Geometry to Model in HFSS

```
┌─────────────────────────────┐
│   W (patch width)           │  ← top copper layer
│   ┌─────────────────────┐   │
│   │   ┌────────────┐    │   │  ← rectangular slot (slot removes copper)
│   │   │  Ls × Ws   │    │   │
│   │   └────────────┘    │   │
│   └─────────────────────┘   │
│   L (patch length)          │
├─────────────────────────────┤  ← substrate (FR4)
│   (ground plane, full size) │  ← bottom copper
└─────────────────────────────┘
```

### Fixed Substrate (FR4 — universally available)
| Parameter         | Value       |
|-------------------|-------------|
| Material          | FR4_epoxy   |
| Dielectric (εr)   | 4.4         |
| Loss tangent      | 0.02        |
| Height (h)        | 1.6 mm      |
| Ground plane      | Full (same XY as substrate) |

### Feed Type
Use **Lumped Port** or **Wave Port** at the edge, or a **coaxial probe**.  
Easiest for parametric sweep: **Lumped Port** at the edge center.

Recommended port impedance: **50 Ω**

---

## 2. Parametric Sweep Design

### Why These Ranges?
- L (patch length) determines resonant frequency: fr ≈ c / (2L√εr_eff)
- For FR4 (εr=4.4): fr ≈ 71.5/L GHz when L is in mm
- Range 20–50 mm gives fr ≈ 1.4–3.6 GHz

### Parameter Table — Enter as HFSS Optimetrics Variables

Set up these as project variables in HFSS:

| Variable | Symbol | Values (mm)                      | # Values |
|----------|--------|----------------------------------|----------|
| `L_patch`| L      | 22, 26, 30, 34, 38, 42, 46       | 7        |
| `W_patch`| W      | 28, 34, 40                       | 3        |
| `L_slot` | Ls     | 6, 10, 14, 18                    | 4        |
| `W_slot` | Ws     | 2, 4, 6                          | 3        |

**Total combos = 7 × 3 × 4 × 3 = 252**  

### Physical Validity Constraints (CRITICAL)
Not all combinations are physically valid. HFSS will still run them but results 
will be garbage. Apply these constraints in Optimetrics:

```
Ls < 0.55 * L_patch     (slot shorter than half the patch)
Ws < 0.45 * W_patch     (slot narrower than half the patch)
Ws < Ls                 (slot is always longer than wide — rectangular slot)
```

**After filtering: ~120–140 valid combinations.** 
Even 80 is enough to start — prioritize getting 80+ simulations done.

### Recommended Reduced Set (60 configs — Manageable Starting Point)
If time is limited, use this fixed reduced sweep:

| L (mm) | W (mm) | Ls (mm) | Ws (mm) | Valid? |
|--------|--------|---------|---------|--------|
| 26     | 34     | 6       | 2       | ✅     |
| 26     | 34     | 6       | 4       | ✅     |
| 26     | 34     | 10      | 2       | ✅     |
| 26     | 34     | 10      | 4       | ✅     |
| 30     | 34     | 6       | 2       | ✅     |
| 30     | 34     | 10      | 2       | ✅     |
| 30     | 34     | 10      | 4       | ✅     |
| 30     | 34     | 14      | 4       | ✅     |
| 30     | 40     | 6       | 2       | ✅     |
| 30     | 40     | 10      | 4       | ✅     |
| 30     | 40     | 14      | 4       | ✅     |
| 34     | 40     | 6       | 2       | ✅     |
| 34     | 40     | 10      | 4       | ✅     |
| 34     | 40     | 14      | 4       | ✅     |
| 34     | 40     | 14      | 6       | ✅     |
| 34     | 40     | 18      | 4       | ✅     |
| 38     | 40     | 10      | 4       | ✅     |
| 38     | 40     | 14      | 4       | ✅     |
| 38     | 40     | 14      | 6       | ✅     |
| 38     | 40     | 18      | 4       | ✅     |
| 38     | 40     | 18      | 6       | ✅     |
| 38     | 46     | 10      | 4       | ✅     |
| 38     | 46     | 14      | 6       | ✅     |
| 38     | 46     | 18      | 6       | ✅     |
| 42     | 46     | 10      | 4       | ✅     |
| 42     | 46     | 14      | 4       | ✅     |
| 42     | 46     | 14      | 6       | ✅     |
| 42     | 46     | 18      | 4       | ✅     |
| 42     | 46     | 18      | 6       | ✅     |
| 42     | 52     | 10      | 4       | ✅     |
| 42     | 52     | 14      | 6       | ✅     |
| 42     | 52     | 18      | 6       | ✅     |
| 46     | 52     | 10      | 4       | ✅     |
| 46     | 52     | 14      | 4       | ✅     |
| 46     | 52     | 14      | 6       | ✅     |
| 46     | 52     | 18      | 6       | ✅     |
| 46     | 52     | 22      | 6       | ✅     |
| 46     | 58     | 14      | 6       | ✅     |
| 46     | 58     | 18      | 6       | ✅     |
| 46     | 58     | 22      | 6       | ✅     |

*(Continue filling to 60+ by adding W=28 variants for L=26,30 and W=58 for L=46)*

---

## 3. HFSS Solution Setup

```
Solution type:   Modal Network Analysis (or Driven Modal)
Freq setup:      Interpolating Sweep
  Start:         1.0 GHz
  Stop:          4.5 GHz  
  Max passes:    15
  Delta S:       0.02

Or use: "Fast Sweep" 1–4.5 GHz, 201 points
```

---

## 4. How to Export S11 Data from HFSS

After solving, export as CSV:

```
1. Go to Results → Create Modal Solution Data Report → Rectangular Plot
2. Y-axis: dB(S(LumpPort1, LumpPort1))  [or S11 dB]
3. X-axis: Freq [GHz]
4. Click: Export → Export to CSV
5. File name:  hfss_L26_W34_Ls6_Ws2.csv
```

### Naming Convention for Export Files
Use this exact naming format so the processing script can auto-parse dimensions:
```
hfss_L{L}_W{W}_Ls{Ls}_Ws{Ws}.csv
```
Examples:
```
hfss_L30_W40_Ls14_Ws4.csv
hfss_L38_W46_Ls18_Ws6.csv
```

### Exported CSV Format (HFSS default)
```csv
Freq [GHz],dB(S(1,1)) []
1.0000,  -0.2341
1.0174,  -0.3102
...
4.5000,  -1.1240
```

### Batch Export (Parametric)
If you ran Optimetrics, you can batch export:
```
Optimetrics → right-click solution → Export Results
→ Choose parameter combinations → Export all to separate CSVs
```

---

## 5. Processing Script

Run this script after collecting your CSV files:
```
.\venv\Scripts\python.exe data\process_hfss_export.py 
  --input_dir data\hfss_exports\
  --output    data\hfss_combined.csv
```

Then retrain:
```
.\venv\Scripts\python.exe models\slotted_model.py --data data\hfss_combined.csv
```

---

## 6. Expected Publication Impact

| Dataset Size | Expected LOOCV R² | Publishable? | Target Venue |
|---|---|---|---|
| 7 configs (current) | −0.98 | ❌ No | — |
| 30 configs | ~0.6–0.7 | ⚠️ Maybe workshop | APMC workshop |
| **60 configs** | **~0.85+** | **✅ Conference** | **IEEE APMC / PIERS** |
| 100+ configs | ~0.92+ | ✅ Journal | IEEE AWPL |
| 100+ + Fabrication | ~0.92+ measured | ✅ Strong journal | IEEE TAP |

---

## 7. Estimated Time

| Task | Time |
|---|---|
| Build HFSS geometry once | 30–60 min |
| Set up Optimetrics parametric sweep | 20 min |
| Simulate 60 configs (2–3 min each) | **~2.5–3 hours** (run overnight) |
| Export all CSVs | 20 min |
| Run processing + training | 10 min |
| Write results section of paper | 2–3 hours |
