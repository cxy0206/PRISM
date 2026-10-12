# PRISM datasets

This directory contains the processed CSV files used by the PRISM training scripts. Paths below are relative to `data/`. Row counts refer to the supplied files, excluding headers; they are not counts of unique molecules or guarantees of the number retained after graph construction.

For installation and training commands, see the [repository README](../README.md).

## File inventory

### Molecular pretraining

| File | Rows | Required column | Use |
| --- | ---: | --- | --- |
| `zinc15_250K.csv` | 250,000 | `smiles` | Unlabeled molecular pretraining |

This is the ZINC250k subset supplied with PRISM. `pretrain.py` computes VSA descriptors and molecular fingerprints from the SMILES strings; a separate descriptor CSV is not required. The pretraining loader skips failed descriptor calculations and filters graphs with no atoms or bonds.

### Single-molecule benchmarks

Each file contains a `smiles` column and the target column listed below. Select the target with `--label_col` in `finetune_single.py`.

| File | Rows | Target column | Prediction task |
| --- | ---: | --- | --- |
| `single_data/bace.csv` | 1,513 | `bace` | Binary BACE activity classification |
| `single_data/bbbp.csv` | 2,039 | `bbbp` | Binary blood–brain barrier permeability classification |
| `single_data/esol.csv` | 1,128 | `esol` | Aqueous log-solubility regression |
| `single_data/freesolv.csv` | 642 | `freesolv` | Hydration free-energy regression |
| `single_data/hiv.csv` | 41,127 | `hiv` | Binary HIV activity classification |
| `single_data/lipo.csv` | 4,200 | `lipo` | Lipophilicity regression |

These are processed versions of the [MoleculeNet benchmarks](https://doi.org/10.1039/C7SC02664A). Targets retain their task-specific meaning; in particular, FreeSolv labels are hydration free energies, not solubilities. The original FreeSolv resource is maintained by [MobleyLab](https://github.com/MobleyLab/FreeSolv).

### Solubility datasets

| File | Rows | Use |
| --- | ---: | --- |
| `pair_data/bigsoldb_chemprop_nonaq.csv` | 41,724 | Pure-solvent training and validation source |
| `pair_data/leeds_all_chemprop.csv` | 1,294 | Leeds external evaluation |
| `pair_data/leeds_acetone_chemprop.csv` | 324 | Solvent-specific Leeds analysis |
| `pair_data/leeds_benzene_chemprop.csv` | 397 | Solvent-specific Leeds analysis |
| `pair_data/leeds_ethanol_chemprop.csv` | 573 | Solvent-specific Leeds analysis |
| `pair_data/solprop_chemprop_nonaq.csv` | 5,230 | SolProp external evaluation |
| `pair_data/mix.csv` | 26,730 | Mixed-solvent training, validation, and evaluation |

Use `leeds_all_chemprop.csv` for overall Leeds evaluation. The three solvent-specific files are provided for separate analyses and should not be appended to the overall file as additional test observations.

## Pure-solvent format and preparation

The pure-solvent tables were prepared using the [fastsolv preprocessing workflow](https://github.com/JacksonBurns/fastsolv/tree/main/paper), associated with [Attia et al., Data-driven organic solubility prediction at the limit of aleatoric uncertainty](https://www.nature.com/articles/s41467-025-62717-7).

The PRISM external-transfer protocol removes BigSolDB records whose solutes occur in Leeds or SolProp before supervised training. Molecular identity for this exclusion is defined using canonical solute SMILES. The supplied BigSolDB CSV is the processed input, rather than the complete upstream database. The current training entry point reads this prepared file and does not repeat the overlap-removal procedure.

Required fields are:

| Column | Meaning |
| --- | --- |
| `solute_smiles` | Solute molecular structure |
| `solvent_smiles` | Solvent molecular structure |
| `temperature` | Measurement temperature in kelvin |
| `logS` | Base-10 logarithm of solubility expressed in mol/L, following the fastsolv data convention |

Thus, the pure-solvent target is `log10(S / (1 mol/L))`. The scripts read `logS` directly and do not apply another logarithm or concentration-unit conversion.

Some files also contain a `source` field or an exported row-index column with an empty name or a name such as `Unnamed: 0`. These are metadata and are not prediction inputs. An exported row index is not a train/validation/test assignment.

## Mixed-solvent format

The mixed-solvent data originate from [Bao et al., Towards the prediction of drug solubility in binary solvent mixtures at various temperatures using machine learning](https://doi.org/10.1186/s13321-024-00911-3), also used for the drug-solubility task in [CheMixHub](https://github.com/chemcognition-lab/chemixhub). The supplied `mix.csv` is the processed PRISM working table, not the full upstream release.

The training script reads:

| Column | Meaning |
| --- | --- |
| `solute_smiles` | Solute molecular structure |
| `solvent1_smiles` | First solvent molecular structure |
| `solvent2_smiles` | Second solvent molecular structure |
| `Xsolvent1` | Mole fraction of the first solvent in the solvent mixture |
| `Xsolvent2` | Mole fraction of the second solvent in the solvent mixture |
| `temperature` | Temperature in kelvin |
| `logS` | Regression target, equal to `log10(s)` in the supplied file |

Additional columns are `solute`, `solvent1`, and `solvent2` (names), and `s` (the corresponding non-logarithmic solubility value). They are not used as prediction inputs.

In the supplied file, all 26,730 records satisfy `logS = log10(s)` and `Xsolvent1 + Xsolvent2 = 1` to numerical precision. Temperature ranges from 252.55 to 368.15 K. The mixed-solvent loader normalizes the two composition values to sum to one by default and uses the existing `logS` column without applying another logarithm.

**Solubility-scale note:** The numerical relationship `logS = log10(s)` establishes the logarithm base, but does not establish a physical concentration unit. The supplied CSV does not carry unit metadata for `s`. Retain the supplied scale for these experiments; do not assume it is mol/L or combine these targets with the pure-solvent targets without checking the original conversion. The Bao et al. publication reports a g/100 g convention, which must be traced through preprocessing before assigning that unit to this particular processed column.

## Data splits and evaluation

The CSV files contain observations rather than fixed split assignments. The current entry points handle splitting as follows:

| Entry point | Current splitting behavior |
| --- | --- |
| `pretrain.py` | Random 80/20 training/validation split after graph filtering |
| `finetune_single.py` | Scaffold split; default fractions are 80/10/10 |
| `finetune_pair.py` | Random split of canonical solute groups into training/validation; external tests are supplied separately |
| `finetune_mix.py run` | Random or solvent-pair-disjoint split; default fractions are 80/10/10 |

For mixed-solvent solvent-pair-disjoint evaluation, records are grouped by unordered solvent pair. The mixed-solvent script saves split assignments under each run's output directory. Keep these files together with the data version and run configuration.

`finetune_pair.py` assigns all records for a canonical solute to the same training or validation split and saves row indices, canonical solutes, and assignments in `seed{seed}/split.csv`. `--val_ratio` is a fraction of solute groups. No internal BigSolDB test set is created. The external-test solute exclusion described above is a separate preparation step. Pretraining also saves its filtered-graph split indices and SMILES in `split.csv`.

Leeds and SolProp are external evaluation sets and should remain separate from supervised training and validation. If regenerating data, apply overlap removal using consistent molecular canonicalization before constructing training splits. Equivalent molecules can have different SMILES strings, so raw string equality is insufficient for this check.

## Sources and attribution

The CSVs are processed research inputs derived from existing datasets. Please cite the original sources as well as PRISM when using them:

- ZINC: the source database for the supplied ZINC250k molecular subset.
- [MoleculeNet](https://doi.org/10.1039/C7SC02664A): the six single-molecule benchmarks.
- [BigSolDB](https://doi.org/10.26434/chemrxiv-2023-qqslt): pure-solvent training measurements.
- Vermeire, Chung, and Green, *Predicting solubility limits of organic solutes for a wide range of solvents and temperatures* (JACS, 2022): SolProp.
- Boobier, Hose, Blacker, and Nguyen, *Machine learning with physicochemical relationships: solubility prediction in organic solvents and water* (Nature Communications, 2020): Leeds.
- [Attia et al.](https://www.nature.com/articles/s41467-025-62717-7) and [fastsolv](https://github.com/JacksonBurns/fastsolv): pure-solvent preprocessing.
- [Bao et al.](https://doi.org/10.1186/s13321-024-00911-3) and [CheMixHub](https://github.com/chemcognition-lab/chemixhub): binary-solvent solubility data.

Original dataset attribution and reuse terms continue to apply. The upstream download and cleaning pipelines are not bundled in this directory; use the supplied CSVs to match this repository's data snapshot.
