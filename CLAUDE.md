# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PI-DCON is the official implementation of "Physics-informed discretization-independent deep compositional operator network" - a neural operator approach for solving Partial Differential Equations (PDEs) with irregular domain geometries. The key innovation is handling boundary conditions as function value observations rather than discretized points, making the approach discretization-independent.

**Paper:** [Physics-informed discretization-independent deep compositional operator network](https://www.sciencedirect.com/science/article/abs/pii/S0045782524005309)

**Python version:** 3.8+

## Common Commands

### Installation
```bash
pip install -r requirements.txt
```

### Training (Darcy flow problem)
```bash
cd Main
python exp_pinn_darcy.py --model='DCON' --phase='train'
python exp_pinn_darcy.py --model='DON' --phase='train'   # DeepONet baseline
python exp_pinn_darcy.py --model='IDON' --phase='train'  # Improved DeepONet
```

### Training (2D plate stress problem)
```bash
cd Main
python exp_pinn_plate.py --model='DCON' --phase='train'
python exp_pinn_plate.py --model='DON' --phase='train'
python exp_pinn_plate.py --model='IDON' --phase='train'
```

### Testing
```bash
# Replace --phase='train' with --phase='test'
python exp_pinn_darcy.py --model='DCON' --phase='test'
python exp_pinn_plate.py --model='DCON' --phase='test'
```

### Command-line Arguments
- `--model`: Model type ('DCON', 'DON', 'IDON', 'self_defined')
- `--phase`: 'train' or 'test'
- `--data`: Dataset name (default: 'Darcy_star' for darcy, 'plate_dis_high' for plate)

## Code Architecture

### Entry Points
- `Main/exp_pinn_darcy.py` - Darcy flow experiments
- `Main/exp_pinn_plate.py` - 2D plate stress experiments

### Core Files
- `Main/models.py` - Neural network model implementations
  - `DeepONet_darcy` / `DeepONet_plate` - Baseline DeepONet
  - `Improved_DeepOnet_darcy` / `Improved_DeepONet_plate` - Improved DeepONet
  - `DCON_darcy` / `DCON_plate` - Proposed DCON model
  - `New_model_darcy` / `New_model_plate` - Template for custom models

- `Main/data.py` - Data loading and preprocessing (MATLAB .mat files)
- `Main/darcy_utils.py` - Training utilities for Darcy flow problem
- `Main/plate_utils.py` - Training utilities for plate stress problem

### Configuration
- `Main/configs/{MODEL}_{DATASET}.yaml` - Model architecture and training parameters
  - `model/fc_dim`: Hidden layer dimension (default: 512)
  - `model/N_layer`: Number of layers (default: 3)
  - `train/epochs`: Training epochs (default: 200)
  - `train/batchsize`: Batch size (default: 20)
  - `train/base_lr`: Learning rate (default: 0.0001)
  - `train/bc_weight`: Boundary condition loss weight (default: 300)

### Data and Results
- `data/` - Dataset files (.mat format) - download from [Google Drive](https://drive.google.com/drive/folders/10c5BWVvd-Oj13tMGhE07Tau07aTWfOhM)
- `res/saved_models/` - Trained model checkpoints
- `res/plots/` - Generated visualizations

## Model Types

### DCON (Proposed Model)
- Uses both coordinates (x,y) and function values (u) from boundary conditions
- Branch network processes full `(x, y, u)` tuples via max pooling
- Trunk network with gated modulation by branch encoding

### DON (DeepONet Baseline)
- Standard DeepONet architecture
- Branch network uses only function values `par[...,-1]`
- Trunk network processes coordinates, combined via inner product

### IDON (Improved DeepONet)
- Adds embedding-based gating mechanism
- Blends parameter and coordinate embeddings during forward pass

## Custom Model Development

To add a custom model architecture:

1. Edit `Main/models.py`:
   - Implement `New_model_darcy` class for Darcy flow
   - Implement `New_model_plate` class for plate stress
   - Both take `(config)` in `__init__` and `(x_coor, y_coor, par)` in `forward`

2. Edit `Main/configs/self_defined_Darcy_star.yaml` and `self_defined_plate_dis_high.yaml` for hyperparameters

3. Run with `--model='self_defined'`

## Model Input/Output Format

### Input
- `x_coor` (B, M): x-axis coordinates of collocation points
- `y_coor` (B, M): y-axis coordinates of collocation points
- `par` (B, N, 3): Boundary conditions where each row is `(x, y, u)`
  - N varies per sample (discretization-independent)
  - Last dimension contains function values

### Output
- Darcy: `u` (B, M) - Solution function values over domain
- Plate: `u, v` (B, M), (B, M) - Displacement field components
