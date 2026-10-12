# PRISM

PRISM is a molecular representation learning framework for molecular property prediction and solubility generalization. It uses chemical descriptors and molecular fingerprints to define complementary notions of molecular similarity, and pretrains separate graph encoders to preserve these relations through neighborhood-distribution alignment. The pretrained representations are combined during downstream learning.

The framework uses four prior spaces: **SMR_VSA**, **SlogP_VSA**, **PEOE_VSA**, and **Morgan fingerprints**.

This repository contains entry points for:

- Prior-specific pretraining on ZINC250k.
- Single-molecule classification and regression.
- Pure-solvent solubility prediction using solute, solvent, and temperature inputs.
- Mixed-solvent solubility prediction using solute, solvent identities, composition, and temperature.


## Repository structure

```text
PRISM/
├── chemprop/                 # Local Chemprop-derived graph components
├── model/                    # Featurization, prior encoders, alignment, and fusion
├── data/
│   ├── zinc15_250K.csv        # Pretraining molecules
│   ├── single_data/          # BACE, BBBP, ESOL, FreeSolv, HIV, and Lipophilicity
│   └── pair_data/            # BigSolDB, Leeds, SolProp, and mixed-solvent data
├── pretrain.py
├── finetune_single.py
├── finetune_pair.py
├── finetune_mix.py
├── requriements.txt          # Existing dependency-file name
└── README.md
```

## Environment setup

The dependency file records a CUDA 11.8 environment with PyTorch 2.4.0, PyTorch Geometric 2.6.1, and RDKit 2025.3.6. It includes Linux-specific packages, so the following setup targets Linux with an NVIDIA GPU. A fresh installation of the complete environment still needs validation.

```bash
conda create -n prism python=3.11
conda activate prism

python -m pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

The CUDA wheel command follows the [PyTorch installation instructions for version 2.4.0](https://pytorch.org/get-started/previous-versions/#v240). T

The local `chemprop/` directory is part of the implementation; installing a separate Chemprop release does not replace it.

## Data

See [data/README.md](data/README.md) for the file inventory, column definitions, data preparation, and source references.

### Pretraining

`data/zinc15_250K.csv` contains a `smiles` column. VSA descriptors and molecular fingerprints are computed from molecular structures during preprocessing and pretraining; no downstream property labels are required for this stage.

### Single-molecule tasks

The supplied CSVs contain `smiles` and a dataset-specific target column:

| Dataset | File | Target column | Task |
| --- | --- | --- | --- |
| BACE | `data/single_data/bace.csv` | `bace` | Classification |
| BBBP | `data/single_data/bbbp.csv` | `bbbp` | Classification |
| ESOL | `data/single_data/esol.csv` | `esol` | Regression |
| FreeSolv | `data/single_data/freesolv.csv` | `freesolv` | Regression |
| HIV | `data/single_data/hiv.csv` | `hiv` | Classification |
| Lipophilicity | `data/single_data/lipo.csv` | `lipo` | Regression |

Use `--label_col` to select the target column. The single-molecule script generates scaffold-based train/validation/test splits.

### Pure-solvent solubility

Files are stored in `data/pair_data/`:

- `bigsoldb_chemprop_nonaq.csv`: BigSolDB training data.
- `leeds_all_chemprop.csv`: Leeds external evaluation data.
- `solprop_chemprop_nonaq.csv`: SolProp external evaluation data.

Required columns are:

```text
solute_smiles,solvent_smiles,temperature,logS
```

Temperature is expressed in kelvin. The scripts use the supplied `logS` values directly without applying another logarithm. `finetune_pair.py` groups records by canonical solute SMILES when splitting the training CSV into training and validation sets. Leeds and SolProp remain separate external test sets; no internal test split is created from BigSolDB. External-solute exclusion is a separate data-preparation step described in [data/README.md](data/README.md).

### Mixed-solvent solubility

`data/pair_data/mix.csv` uses the following input and target columns:

```text
solute_smiles,solvent1_smiles,solvent2_smiles,Xsolvent1,Xsolvent2,temperature,logS
```

`Xsolvent1` and `Xsolvent2` specify solvent composition. The loader normalizes these two values to sum to one by default. Temperature is in kelvin, and the supplied mixture target is `logS = log10(s)`. Additional descriptive columns in the CSV are not required as model inputs.

## Optional global molecular features

Global molecular features are **disabled by default** in pretraining and downstream fusion. VSA descriptors still define the pretraining targets; they are not concatenated to the default encoder input or downstream representation.

To enable the optional eight global descriptors, pass `--use_global_features` to pretraining and downstream commands. Their 32-dimensional encoder output is included in the saved architecture, and downstream fusion uses the resulting embedding width automatically. Loading a global-feature checkpoint without this flag raises an error. The same flag also enables global features in downstream fusion where applicable.

Checkpoints include the encoder configuration and are loaded strictly. Do not bypass architecture mismatches with `strict=False`. `--emb_dim` is retained in downstream CLIs for compatibility, but the actual width is read from the checkpoints.

## 1. Prior-specific pretraining

Each invocation of `pretrain.py` trains **one** prior-specific encoder. The prior is selected using `--property_index`:

| Argument | Prior | Directory used in the examples |
| --- | --- | --- |
| `0` | SMR_VSA | `task_0` |
| `1` | SlogP_VSA | `task_1` |
| `2` | PEOE_VSA | `task_2` |
| `none` | Morgan fingerprints | `task_none` |

The following examples use 100 maximum epochs, an embedding-neighborhood temperature of 0.1, and a prior-distance scale of 0.25:

```bash
python pretrain.py \
  --data_csv data/zinc15_250K.csv \
  --property_index 0 --num_epochs 100 --hidden_dim 512 \
  --tau 0.1 --tau_phys_scale 0.25 \
  --save_dir ckpt/tau_0.1__scale_0.25/task_0

python pretrain.py \
  --data_csv data/zinc15_250K.csv \
  --property_index 1 --num_epochs 100 --hidden_dim 512 \
  --tau 0.1 --tau_phys_scale 0.25 \
  --save_dir ckpt/tau_0.1__scale_0.25/task_1

python pretrain.py \
  --data_csv data/zinc15_250K.csv \
  --property_index 2 --num_epochs 100 --hidden_dim 512 \
  --tau 0.1 --tau_phys_scale 0.25 \
  --save_dir ckpt/tau_0.1__scale_0.25/task_2

python pretrain.py \
  --data_csv data/zinc15_250K.csv \
  --property_index none --num_epochs 100 --hidden_dim 512 \
  --tau 0.1 --tau_phys_scale 0.25 \
  --save_dir ckpt/tau_0.1__scale_0.25/task_none
```

For the automatically scaled teacher temperature, a larger `--tau_phys_scale` produces a smoother prior neighborhood. The examples use one configuration to illustrate the interface; settings for a particular paper experiment must match that experiment's configuration.

Each completed run saves `best_model.pth`, `config.json`, `split.csv`, `history.csv`, temperature-scale statistics, and a training-curve plot under its own `--save_dir`. Use `--seed` to set the pretraining seed and `--device cpu` or `--device cuda` to select the execution device. Downstream examples refer to these locally generated checkpoints in the order SMR_VSA, SlogP_VSA, PEOE_VSA, and fingerprints.

## 2. Single-molecule fine-tuning

Example for BBBP classification with three runs:

```bash
python finetune_single.py \
  --task_type classification \
  --data_csv data/single_data/bbbp.csv --label_col bbbp \
  --encoder_ckpts "ckpt/tau_0.1__scale_0.25/task_0/best_model.pth,ckpt/tau_0.1__scale_0.25/task_1/best_model.pth,ckpt/tau_0.1__scale_0.25/task_2/best_model.pth,ckpt/tau_0.1__scale_0.25/task_none/best_model.pth" \
  --space_indices 0,1,2,3 \
  --seed 42 --runs 3 \
  --val_split 0.1 --test_split 0.1 \
  --batch_size 64 --epochs 200 --learning_rate 1e-4 \
  --out_dir results/single_bbbp
```

For ESOL regression, use the same checkpoint list and change the task and dataset arguments:

```bash
python finetune_single.py \
  --task_type regression \
  --data_csv data/single_data/esol.csv --label_col esol \
  --encoder_ckpts "ckpt/tau_0.1__scale_0.25/task_0/best_model.pth,ckpt/tau_0.1__scale_0.25/task_1/best_model.pth,ckpt/tau_0.1__scale_0.25/task_2/best_model.pth,ckpt/tau_0.1__scale_0.25/task_none/best_model.pth" \
  --space_indices 0,1,2,3 \
  --seed 42 --runs 3 \
  --val_split 0.1 --test_split 0.1 \
  --batch_size 64 --epochs 200 --learning_rate 1e-4 \
  --out_dir results/single_esol
```

The single-molecule entry point fine-tunes both encoders and fusion. Its default `--fusion_type auto` selects static fusion for classification and adaptive fusion for regression; `static` or `adaptive` can be specified explicitly. `--seed 42 --runs 3` selects seeds 42, 43, and 44. The script reports ROC-AUC for classification and RMSE for regression, and writes per-run and summary CSV files to `--out_dir`.

## 3. Pure-solvent solubility fine-tuning

The following example trains on BigSolDB and evaluates on Leeds and SolProp:

```bash
python finetune_pair.py \
  --train_csv data/pair_data/bigsoldb_chemprop_nonaq.csv \
  --test_csvs "data/pair_data/leeds_all_chemprop.csv,data/pair_data/solprop_chemprop_nonaq.csv" \
  --encoder_ckpts "ckpt/tau_0.1__scale_0.25/task_0/best_model.pth,ckpt/tau_0.1__scale_0.25/task_1/best_model.pth,ckpt/tau_0.1__scale_0.25/task_2/best_model.pth,ckpt/tau_0.1__scale_0.25/task_none/best_model.pth" \
  --space_indices 0,1,2,3 --seeds 42,43,44 \
  --batch_size 64 --epochs 200 --val_ratio 0.05 \
  --learning_rate 1e-4 --weight_decay 0.0 \
  --solute_dropout 0.3 --solvent_dropout 0.3 \
  --mlp_hidden 1024,128,64 --mlp_dropout 0.2 --t_embed_dim 32 \
  --out_dir results/pure_solvent
```

The script writes detailed and summary metric CSVs to `--out_dir`. Each seed also saves `seed{seed}/split.csv`, `config.json`, and `model_best.pth`. The split file records source row indices, canonical solute SMILES, and train/validation membership. `--val_ratio` specifies the fraction of solute groups reserved for validation, so the fraction of records may differ. Pure-solvent fusion defaults to `adaptive`, matching the mixed-solvent default. Solubility metrics include RMSE and the percentage of predictions satisfying `abs(predicted logS - observed logS) <= 1`.

## 4. Mixed-solvent solubility fine-tuning

The mixed-solvent script requires a subcommand. Use `run` for repeated experiments:

```bash
python finetune_mix.py run \
  --mix_csv data/pair_data/mix.csv \
  --pretrain_base_dir ckpt/tau_0.1__scale_0.25 \
  --task_dirs task_0,task_1,task_2,task_none \
  --encoder_ckpt_name best_model.pth \
  --split_strategy both \
  --train_frac 0.8 --val_frac 0.1 --test_frac 0.1 \
  --seed 42 --num_runs 5 \
  --batch_size 64 --epochs 100 --lr 1e-4 \
  --mlp_hidden 512,512 --mlp_dropout 0.1 --t_embed_dim 32 \
  --freeze_encoders \
  --experiment_name PRISM \
  --out_root results/mixed_solvent
```

`--split_strategy both` runs random and solvent-pair-disjoint splits. Use `random` or `solvent_pair_disjoint` to run one protocol. `--seed 42 --num_runs 5` selects seeds 42 through 46 for each protocol. Encoders are frozen by default; the fusion modules and prediction head remain trainable.

This example initializes the fusion modules without a pure-solvent warm start. For fusion-only warm-started runs, first train pure-solvent models with the same branch order, fusion type, and global-feature setting. Use `--seeds 42,43,44,45,46` in the pure-solvent command if using all five mixed-solvent seeds, then add:

```bash
--init_ckpt 'results/pure_solvent/seed{seed}/model_best.pth'
```

The loader substitutes the current seed and strictly loads the solute and solvent fusion modules. The molecular encoders come from the original pretraining checkpoints; the mixed-solvent regression head starts with new parameters. Frozen encoders remain in evaluation mode during mixed-solvent training. For branch-subset ablations, provide a pure-solvent checkpoint trained with exactly that subset, or omit warm start.

Mixed-solvent runs save split assignments, test predictions, metric files, training plots, and a local model checkpoint in the output directory. The `ablation` subcommand provides branch-subset experiments.

## Reproducibility and outputs

The scripts use different arguments for repeated runs:

| Script | Run controls |
| --- | --- |
| `pretrain.py` | One encoder per invocation; `--seed` |
| `finetune_single.py` | `--seed` and `--runs` |
| `finetune_pair.py` | `--seeds`, as a comma-separated list |
| `finetune_mix.py run` | `--seed` and `--num_runs` |

Summary standard deviations use `ddof=0`; RMSE is computed for each run before averaging across runs. Retain the data version, split assignments, exact command, checkpoint configuration, and per-run predictions for each reported experiment. Use the same test records and predictions when computing multiple metrics for a run. A seed alone does not specify a split independently of data ordering and preprocessing.

Generated checkpoints and training outputs can remain local and should be excluded from version control through `.gitignore`. The absence of distributed checkpoints does not remove the need to save and load local checkpoints between training stages.

To inspect all supported options:

```bash
python pretrain.py --help
python finetune_single.py --help
python finetune_pair.py --help
python finetune_mix.py run --help
python finetune_mix.py ablation --help
```

## Code checks

Run the focused regression checks from the repository root:

```bash
python -m unittest discover -s tests -v
```

The checks cover checkpoint reconstruction, opt-in global features, graph padding,
solute-group splits, and fusion-only warm starts. Small CPU runs have also exercised
all four pretraining branches, both single-molecule task types, pure-solvent training,
and both mixed-solvent split protocols, with and without global features. These are
implementation checks, not a full rerun of the paper experiments. CPU validation used
Python 3.12, PyTorch 2.4.1, PyTorch Geometric 2.6.1, and RDKit 2025.3.6.

## Acknowledgments

The implementation uses Chemprop-derived components and RDKit molecular features. Pure-solvent data preparation follows the fastsolv preprocessing workflow. We acknowledge the original dataset authors and maintainers of these projects:

- [Chemprop](https://github.com/chemprop/chemprop)
- [RDKit](https://www.rdkit.org/)
- [fastsolv](https://github.com/JacksonBurns/fastsolv)
- MoleculeNet, ZINC, BigSolDB, Leeds, SolProp, and CheMixHub.
