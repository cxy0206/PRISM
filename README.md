# ChemSpaceAlign
<img width="1563" height="1503" alt="image" src="https://github.com/user-attachments/assets/2de02ae7-564b-4ab9-81ae-7073159047bf" />
ChemSpaceA is a new molecular representation learning structure that uses traditional chemical descriptors, which consists of three main components: (1)chemical space construction, (2) space alignment-based pre-training, and (3) single-molecular and pair-wise interaction fine-tuning.


The graph encoder used in this project is based on CMPNN, which was proposed by [Chemprop](https://github.com/chemprop/chemprop).

---

## Repository structure and data sources

This repository contains the reference implementation of ChemSpaceA, including
pre-training, single-molecule fine-tuning, and pair-wise solvation regression.

**Data sources**

- Single-molecule datasets in `data/` come from **MoleculeNet** and are
  converted to CSV format while keeping the original labels and standard
  splits (or commonly used random splits).  
  See: [MoleculeNet: A Benchmark for Molecular Machine Learning](https://moleculenet.org/).

- Pair-wise solute–solvent–temperature datasets in `pair_data/` (e.g.,
  BigSolDB-derived training set, Leeds, SolProp) are generated following the
  preprocessing pipeline provided by
  [JacksonBurns/fastsolv](https://github.com/JacksonBurns/fastsolv).

**Directory layout**

- `data/`  
  Single-molecule classification and regression datasets from MoleculeNet, stored
  as CSV files with a `smiles` column and task-specific label columns.

- `pair_data/`  
  Solvation datasets for pair-wise tasks (solute SMILES, solvent SMILES,
  temperature, logS label), processed via the fastsolv-style pipeline.

- `model/`  
  Core model definitions:
  - `mpn_vas_st.py`: CMPNN-based graph encoder with VSA-guided pre-training loss
    used as the ChemSpaceA encoder.
  - `fusion.py`: fusion modules (e.g., `WeightedFusion`) and fine-tuning heads
    for single-molecule and pair-wise tasks.

- `chemprop/`  
  A minimal copy of the original Chemprop implementation used as a reference
  for the CMPNN-style encoder design.

- `ckpt/`  
  Pre-trained encoder checkpoints and pair-wise models:
  - Subfolders such as  
    `tau_0.1__scale_0.25/`, `tau_0.1__scale_0.5/`, …, `tau_0.3__scale_0.75/`  
    store **encoder checkpoints** trained with different `(τ, τ_phys_scale)`
    configurations.  
    Inside each of these folders there are subdirectories
    `task_0/`, `task_1/`, `task_2/`, and `task_none/` corresponding to the four
    ChemSpaceA pre-training objectives, each containing a `best_model.pth`.
  - `leeds.pth` and `solprop.pth` are **pair-wise solvation models** trained on
    the BigSolDB-style training set and selected for best performance on the
    Leeds and SolProp benchmarks, respectively.  
    Both models use encoders from `tau_0.1__scale_0.25` and the configuration  
    `ep-200_fusion-weighted_mlp-1024-128-64_sd-0_t-16_td-0.0_vd-0_wd-0.0`.

- Top-level scripts:
  - `pretrain.py` – VSA-based pre-training of graph encoders in multiple
    chemical spaces.
  - `finetune_single.py` – fine-tuning for **single-molecule** tasks
    (classification / regression) on MoleculeNet-style datasets.
  - `finetune_pair.py` – fine-tuning for **pair-wise solubility regression**
    (solute + solvent + temperature → logS).
  - `test_pair.py` – evaluation of a trained pair-wise solvation model on
    external datasets (e.g., Leeds, SolProp) given the corresponding encoder
    checkpoints.

---

## Installation

It is recommended to use a dedicated virtual environment (e.g., `conda`) and
install dependencies via `pip`:

```bash
# create and activate environment
conda create -n chemspacea python=3.11
conda activate chemspacea

# install project dependencies
pip install -r requirements.txt
```

## Basic usage

1. Pre-train ChemSpaceA encoders
Example: pre-train encoders with τ = 0.1 and τ_phys_scale = 0.25 on a VSA
augmented ZINC subset:

```bash
python pretrain.py \
    --vsa_csv ./data/vsa_zinc_25k.csv \
    --tau 0.1 \
    --tau_phys_scale 0.25 \
    --save_dir ./ckpt/tau_0.1__scale_0.25
```
This produces four encoder checkpoints in


`ckpt/tau_0.1__scale_0.25/task_0/best_model.pth`
`ckpt/tau_0.1__scale_0.25/task_1/best_model.pth`
`ckpt/tau_0.1__scale_0.25/task_2/best_model.pth`
`ckpt/tau_0.1__scale_0.25/task_none/best_model.pth`

corresponding to the three VSA spaces and one “fingerprint” baseline space.

2. Fine-tune for single-molecule tasks
Example: fine-tune on the BBBP classification dataset using four encoders
and Weighted fusion:

```bash
python finetune_single.py \
    --dataset bbbp \
    --encoder_ckpts \
        ckpt/tau_0.1__scale_0.25/task_0/best_model.pth,\
        ckpt/tau_0.1__scale_0.25/task_1/best_model.pth,\
        ckpt/tau_0.1__scale_0.25/task_2/best_model.pth,\
        ckpt/tau_0.1__scale_0.25/task_none/best_model.pth \
    --space_indices 0,1,2,3 \
    --seeds 42,43,44 \
    --batch_size 64 \
    --epochs 200 \
    --val_split 0.1 \
    --test_split 0.1 \
    --weight_decay 0.0 \
    --dropout 0.3 \
    --learning_rate 1e-4 \
    --out_dir results/single_bbbp
```

The script reports the validation and test metrics (e.g., ROC-AUC / RMSE)
averaged over the given random seeds and writes a CSV summary in
results/single_bbbp/.

3. Fine-tune for pair-wise solubility regression

Example: train a solvation regression model on BigSolDB (non-aqueous) with
dual-branch fusion and a temperature-aware MLP head, and evaluate on both
Leeds and SolProp:

```bash
python finetune_pair.py \
    --train_csv pair_data/bigsoldb_chemprop_nonaq.csv \
    --test_csvs pair_data/leeds_all_chemprop.csv,pair_data/solprop_chemprop_nonaq.csv \
    --encoder_ckpts \
        ckpt/tau_0.1__scale_0.25/task_0/best_model.pth,\
        ckpt/tau_0.1__scale_0.25/task_1/best_model.pth,\
        ckpt/tau_0.1__scale_0.25/task_2/best_model.pth,\
        ckpt/tau_0.1__scale_0.25/task_none/best_model.pth \
    --space_indices 0,1,2 \
    --seeds 42,43,44 \
    --batch_size 64 \
    --epochs 200 \
    --val_ratio 0.05 \
    --weight_decay 0.0 \
    --solute_dropout 0.3 \
    --solvent_dropout 0.3 \
    --mlp_hidden 1024,128,64 \
    --mlp_dropout 0.2 \
    --t_embed_dim 32 \
    --learning_rate 1e-4 \
    --out_dir results/pair_solvation
```

The script prints validation and test RMSE as well as
%(|pred - logS| ≤ 1) for each test dataset and writes an ablation summary to
results/pair_solvation/.

4. Evaluate released pair-wise checkpoints (leeds.pth, solprop.pth)

The repository ships two ready-to-use pair-wise models in ckpt/:

`ckpt/leeds.pth – best model on the Leeds benchmark.`

`ckpt/solprop.pth – best model on the SolProp benchmark.`

Both models use encoders from `ckpt/tau_0.1__scale_0.25/` and the configuration
`ep-200_fusion-weighted_mlp-1024-128-64_sd-0_t-16_td-0.0_vd-0_wd-0.0`.

You can evaluate them with test_pair.py. Example for the Leeds model:

```bash
python test_pair.py \
    --train_csv pair_data/bigsoldb_chemprop_nonaq.csv \
    --test_csvs pair_data/leeds_all_chemprop.csv \
    --encoder_ckpts \
        ckpt/tau_0.1__scale_0.25/task_0/best_model.pth,\
        ckpt/tau_0.1__scale_0.25/task_1/best_model.pth,\
        ckpt/tau_0.1__scale_0.25/task_2/best_model.pth,\
        ckpt/tau_0.1__scale_0.25/task_none/best_model.pth \
    --space_indices 0,1,2 \
    --model_ckpt ckpt/leeds.pth \
    --batch_size 64 \
    --emb_dim 512 \
    --solute_dropout 0.3 \
    --solvent_dropout 0.3 \
    --mlp_hidden 1024,128,64 \
    --mlp_dropout 0.2 \
    --t_embed_dim 32 \
    --out_dir results/pair_leeds_test
```

Similarly, you can replace `ckpt/leeds.pth` with `ckpt/solprop.pth` and change
the test CSV to `pair_data/solprop_chemprop_nonaq.csv` to evaluate the SolProp
model.

Each run logs the MSE and pct(|pred - logS| ≤ 1) for every test set and saves
the results as a CSV file in out_dir for further analysis and plotting.
