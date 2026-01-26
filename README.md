# ChemSpaceA

ChemSpaceA is a molecular representation learning framework that leverages traditional chemical descriptors as
**prior-defined similarity geometries**. It consists of three stages:

1) **Chemical space construction** (descriptor-induced prior spaces)  
2) **Prior-alignment pre-training** (population-level alignment)  
3) **Downstream fine-tuning** for single-molecule prediction and pair-wise solute–solvent–temperature solubility regression

The graph encoder is CMPNN-style message passing adapted from Chemprop.

---

## Repository structure

```
CHEMSPACEA/
  __pycache__/
  chemprop/           # lightweight Chemprop/CMPNN components (reference implementation)
  data/               # single-molecule datasets (MoleculeNet-style CSVs)
  model/              # ChemSpaceA models (encoders, fusion heads, losses)
  pair_data/          # pair-wise solubility datasets (solute, solvent, temperature, logS CSVs)

  pretrain.py         # prior-alignment pre-training
  finetune_single.py  # MoleculeNet-style fine-tuning (classification/regression)
  finetune_pair.py    # pure-solvent pair-wise solubility fine-tuning (solute + solvent + T -> logS)
  finetune_mix.py     # mixed-solvent training / fine-tuning (if applicable)

  requirements.txt
  README.md
```

Notes:
- Output folders such as `ckpt/` and `results/` are created by scripts via `--save_dir` / `--out_dir`.
- Avoid committing large checkpoints and raw datasets to git unless you intend to release them.

---

## Data sources and formats

### 1) Single-molecule datasets (`data/`)
- CSVs converted from **MoleculeNet** (keeping original labels and standard splits when available).
- Typical format:
  - `smiles` column
  - one or more label columns (task-specific)
  - optional split columns/files depending on your dataset preparation

### 2) Pair-wise solubility datasets (`pair_data/`)
- Solute–solvent–temperature regression datasets (e.g., BigSolDB-derived train/val, Leeds, SolProp),
  processed following a fastsolv-style pipeline.
- Typical fields (names may vary by your pipeline; follow script arguments):
  - solute SMILES
  - solvent SMILES
  - temperature
  - logS label

> If you cannot redistribute some datasets, release **(i)** split files and **(ii)** preprocessing scripts so others can
reconstruct them from public sources.

---

## Installation

We recommend using a dedicated environment:

```bash
conda create -n chemspacea python=3.11
conda activate chemspacea
pip install -r requirements.txt
```

---

## Usage

All scripts support `--help`:

```bash
python pretrain.py --help
python finetune_single.py --help
python finetune_pair.py --help
python finetune_mix.py --help
```

### 1) Pre-train encoders (prior alignment)

Example: pre-train encoders with `tau=0.1` and `tau_phys_scale=0.25` on a VSA-augmented ZINC subset:

```bash
python pretrain.py   --vsa_csv ./data/vsa_zinc_25k.csv   --tau 0.1   --tau_phys_scale 0.25   --save_dir ./ckpt/tau_0.1__scale_0.25
```

Expected output:
- multiple encoder checkpoints under `--save_dir` (e.g., per-space subfolders such as `task_0/`, `task_1/`, ...),
  depending on your implementation.

---

### 2) Fine-tune for single-molecule tasks (MoleculeNet)

Example: fine-tune on BBBP classification using multiple encoders + weighted fusion:

```bash
python finetune_single.py   --dataset bbbp   --encoder_ckpts     ckpt/tau_0.1__scale_0.25/task_0/best_model.pth,    ckpt/tau_0.1__scale_0.25/task_1/best_model.pth,    ckpt/tau_0.1__scale_0.25/task_2/best_model.pth,    ckpt/tau_0.1__scale_0.25/task_none/best_model.pth   --space_indices 0,1,2,3   --seeds 42,43,44   --batch_size 64   --epochs 200   --learning_rate 1e-4   --dropout 0.3   --weight_decay 0.0   --out_dir results/single_bbbp
```

Outputs:
- printed validation/test metrics (ROC-AUC or RMSE depending on task)
- a CSV/JSON summary under `--out_dir` (depending on your implementation)

---

### 3) Fine-tune for pair-wise solubility regression (pure-solvent)

Example: train on a BigSolDB-style train set and evaluate on Leeds + SolProp:

```bash
python finetune_pair.py   --train_csv pair_data/bigsoldb_chemprop_nonaq.csv   --test_csvs pair_data/leeds_all_chemprop.csv,pair_data/solprop_chemprop_nonaq.csv   --encoder_ckpts     ckpt/tau_0.1__scale_0.25/task_0/best_model.pth,    ckpt/tau_0.1__scale_0.25/task_1/best_model.pth,    ckpt/tau_0.1__scale_0.25/task_2/best_model.pth,    ckpt/tau_0.1__scale_0.25/task_none/best_model.pth   --space_indices 0,1,2   --seeds 42,43,44   --batch_size 64   --epochs 200   --val_ratio 0.05   --learning_rate 1e-4   --weight_decay 0.0   --solute_dropout 0.3   --solvent_dropout 0.3   --mlp_hidden 1024,128,64   --mlp_dropout 0.2   --t_embed_dim 32   --out_dir results/pair_solvation
```

Outputs:
- validation/test RMSE and pct-in-1 (|pred - logS| ≤ 1) for each test dataset (as implemented)
- run logs and a summary file saved under `--out_dir`

---

### 4) Mixed-solvent training / fine-tuning (if used)

Use `finetune_mix.py` for mixed-solvent settings. Since mixed-solvent CSV formats differ across pipelines,
start with:

```bash
python finetune_mix.py --help
```

Typical usage mirrors `finetune_pair.py` but points to mixed-solvent CSV(s) and any mixture-specific arguments
(e.g., solvent ratios or multiple solvent columns).

---

## Reproducibility notes (recommended)

- Fix `--seeds` and report mean ± std over multiple runs (e.g., 3 seeds).
- Log the git commit hash + full config for each run.
- Keep dataset split files under version control (especially for leakage-controlled OOD splits).

---

## References

- Chemprop (CMPNN): https://github.com/chemprop/chemprop  
- fastsolv preprocessing pipeline: https://github.com/JacksonBurns/fastsolv
