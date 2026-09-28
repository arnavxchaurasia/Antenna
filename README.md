# Antenna ML Framework: S11 Prediction and Inverse Design

A research-grade machine learning framework for predicting the S11 frequency-domain response of slot-loaded microstrip patch antennas and performing inverse design optimization.

## Overview

This project investigates the feasibility of machine learning surrogate models for predicting antenna performance, specifically focusing on honest generalization assessment using **configuration-level Leave-One-Out Cross-Validation (LOOCV)**.

Traditional ML antenna studies often use random row-level splits on frequency-sweep datasets, which falsely inflates performance by leaking configuration data into the test set. This framework implements strict configuration-level holding-out to realistically evaluate whether an ML model can predict the S11 curve of a *completely unseen* physical antenna geometry.

## Key Features

- **Forward ML Modeling**: Predict full frequency-domain S11 curves from geometric parameters (Patch Length `L`, Patch Width `W`, Slot Length `Ls`, Slot Width `Ws`) using Multi-Layer Perceptrons (MLP) and Gradient Boosting Regressors.
- **Inverse Design Optimizer**: Uses Differential Evolution (DE) coupled with the ML surrogate to find the required geometric dimensions that satisfy a target resonant frequency and S11 return loss.
- **Rigorous LOOCV Evaluation**: Ensures models are validated against completely unseen antenna geometries to test true generalization limits.
- **Streamlit Web Application**: An interactive dashboard (`app.py`) for live prediction, optimization, and dataset visualization.
- **Simulation Data Integration**: Includes scripts to process and ingest real-world full-wave electromagnetic simulation exports (e.g., from Ansys HFSS or CST Microwave Studio).

## Repository Structure

- `app.py`: Main Streamlit web application providing an intuitive UI for the framework.
- `models/`: Contains the ML model definitions and evaluation scripts (`slotted_model.py`, `inverse_optimizer.py`).
- `data/`: Contains dataset files (`dataset_antenna.csv`), data processing scripts (`data_loader.py`, `process_hfss_export.py`), and raw HFSS exports.
- `paper.md`: The complete research methodology, literature review, and LOOCV feasibility study results.
- `HFSS_SIMULATION_GUIDE.md`: Comprehensive guide for generating, validating, and exporting data from Ansys HFSS.

## Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:CaptJack05/Antenna.git
   cd Antenna
   ```

2. Create and activate a virtual environment (optional but highly recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Interactive Web Application

To launch the interactive Streamlit dashboard:

```bash
streamlit run app.py
```

This will start a local server and open the application in your default web browser, allowing you to explore the dataset, train surrogate models, run LOOCV evaluations, and perform inverse design.

### Command Line Tools

You can also run the models and data processing scripts directly:

```bash
# Process new HFSS exports into the dataset
python data/process_hfss_export.py

# Run the ML models and evaluations via CLI
python models/slotted_model.py
```

## Research Paper

For a detailed explanation of the problem formulation, methodology, empirical LOOCV results, and the limitations of this ML approach (specifically concerning dataset geometric diversity), please refer to the comprehensive [paper.md](paper.md) document included in this repository.

## Dataset Acknowledgement
The dataset structure is based on simulations of centrally slotted microstrip patch antennas covering key design variables: operating frequency (`f`), patch dimensions (`L`, `W`), slot dimensions (`Ls`, `Ws`), and return loss (`S11`).
