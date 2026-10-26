# METABOLIC PROCESS MODELS

Code and data accompanying the manuscript *Integrating Metabolic Networks into Hybrid Bioprocess Models* by M. Gotsmy and G. Guillén-Gosálbez (ETH Zürich).

This repository implements **FBA-Hyb**, a hybrid bioprocess modeling framework that tightly integrates flux balance analysis (FBA) into a neural controlled ODE. Rather than solving an FBA linear program at every integration step, FBA-Hyb uses a fully differentiable surrogate model discovered a priori via symbolic regression, which allows end-to-end gradient-based training. FBA-Hyb is validated against a standard hybrid baseline (**Std-Hyb**) through leave-one-process-out cross validation on two *E. coli* fed-batch case studies: protein L production (**PROL**) and plasmid DNA production under sulfate limitation (**SLIM**).


## Table of Contents

- [Repository Structure](#repository-structure)
  - [Folders](#folders)
  - [Notebooks](#notebooks)
- [Paper Figures](#paper-figures)
- [Reproducing the LOPO Cross Validation](#reproducing-the-lopo-cross-validation)
- [Citation](#citation)

## Repository Structure

The two case studies are differentiated by two abbreviations: `PROL` for the protein L production study, and `SLIM` for the plasmid DNA production study with sulfate limitation. Scripts, notebooks, and folders are always labeled with one of the abbreviations respectively.

### Folders

|        Folder        | Description |
|----------------------|-------------|
| [`./MPMs`](./MPMs)                             | Core Python package. Contains the model architectures ([MPM_functions_07.py](MPMs/MPM_functions_07.py)), data preprocessing and scalers ([DATA_functions_01.py](MPMs/DATA_functions_01.py)), training/LOPO helpers ([HELPER_functions_02.py](MPMs/HELPER_functions_02.py)), evaluation and plotting ([ANA_functions_02.py](MPMs/ANA_functions_02.py)), and shared utilities ([UTIL_functions_01.py](MPMs/UTIL_functions_01.py)). |
| [`./prol_model_data`](./prol_model_data)       | Trained PROL Std-Hyb and FBA-Hyb model parameters (`.eqx`), training histories, and LOPO cross-validation outputs for the protein L case study. |
| [`./prol_model_scripts`](./prol_model_scripts) | Runnable PROL training scripts. The canonical FBA-Hyb entry point is [MPM_144B.py](prol_model_scripts/MPM_144B.py); the other `MPM_*.py` files are architecture and configuration variants of the structural sensitivity analysis (Sup. Sec. S1.2, Sup. Fig. S4). |
| [`./prol_fba_data`](./prol_fba_data)           | PROL GSMM, parsimonious-FBA training data for the symbolic-regression surrogate, and the fitted `W₁`, `W₂` parameter matrices of Eq. 12 (Sup. Tab. S1). |
| [`./prol_process_data`](./prol_process_data)   | PROL experimental fed-batch time series (*E. coli* BL21(DE3), glycerol, full-factorial `T × μ_f` design) and the 15× augmented training data (Sec. 2.5). |
| [`./slim_model_data`](./slim_model_data)       | Trained SLIM Std-Hyb and FBA-Hyb model parameters and LOPO cross-validation outputs for the plasmid DNA / sulfate-limitation case study. |
| [`./slim_model_scripts`](./slim_model_scripts) | Runnable SLIM training scripts. The canonical FBA-Hyb entry point is [SLIM_03C.py](slim_model_scripts/SLIM_03C.py). |
| [`./slim_fba_data`](./slim_fba_data)           | SLIM GSMM, parsimonious-FBA training data, and fitted surrogate FBA parameters (Sup. Tab. S2). |
| [`./slim_process_data`](./slim_process_data)   | SLIM experimental fed-batch data (*E. coli* JM108, constant glucose feed, sulfate-excess and sulfate-limited conditions) and the augmented training data. |

### Notebooks

| Notebook | Description |
|----------|-------------|
| [PROL_check_MPMs.ipynb](PROL_check_MPMs.ipynb)                 | Analysis of the trained PROL Std-Hyb and FBA-Hyb models: LOPO cross-validation metrics, architecture/configuration comparison, ensemble predictions, mass balance violations, and internal rates (Fig. 3, Sup. Figs. S4 and S5). |
| [PROL_validation.ipynb](PROL_validation.ipynb)                 | Validation of the PROL ensembles on three unseen test processes (`VAL_R1`–`R3`) and comparison with the LOPO cross validation and OptFed (Fig. 2, Sup. Fig. S3). |
| [SLIM_check_MPMs.ipynb](SLIM_check_MPMs.ipynb)                 | Same as above for the SLIM case study, including the sulfate-limitation analysis (Fig. 4, Sup. Fig. S6, graphical abstract). |
| [static_n_comparison.ipynb](static_n_comparison.ipynb)         | Constant-`n` variant study fixing the FBA objective weight vector `n` to a constant (dFBA-like) (Sup. Fig. S7). |
| [prol_fba_data/03_build_fba_surrogate.ipynb](prol_fba_data/03_build_fba_surrogate.ipynb) | Symbolic regression and parameter fitting of the PROL FBA surrogate model (Sup. Fig. S1, Sup. Tab. S1). Requires the [env_pysr.yml](env_pysr.yml) environment. |
| [slim_fba_data/03_build_fba_surrogate.ipynb](slim_fba_data/03_build_fba_surrogate.ipynb) | Same as above for the SLIM case study (Sup. Fig. S2, Sup. Tab. S2). Requires the [env_pysr.yml](env_pysr.yml) environment. |
| [PROL_generate_ANN_data.ipynb](PROL_generate_ANN_data.ipynb)   | FBA-Hyb-SR variant study: generates the PROL data for replacing ANN<sub>1</sub> with symbolic equations (Sup. Sec. S1.2). |
| [SLIM_generate_ANN_data.ipynb](SLIM_generate_ANN_data.ipynb)   | FBA-Hyb-SR variant study: generates the SLIM data for replacing ANN<sub>1</sub> with symbolic equations (Sup. Sec. S1.2). |


## Paper Figures

All figures of the manuscript that are based on data are generated in the notebooks below. In the notebooks in the repository root, each figure cell is preceded by a heading of the form `Figure <number>: <title>`; in the two surrogate notebooks, the figure is found under the heading *Figure for Manuscript*.

| Figure | Title | Notebook |
|--------|-------|----------|
| Fig. 2      | Validation results                                        | [PROL_validation.ipynb](PROL_validation.ipynb) |
| Fig. 3      | FBA-Hyb external state and internal rate predictions      | [PROL_check_MPMs.ipynb](PROL_check_MPMs.ipynb) |
| Fig. 4      | Sulfate limitation during process optimization            | [SLIM_check_MPMs.ipynb](SLIM_check_MPMs.ipynb) |
| Sup. Fig. S1 | Performance of the surrogate FBA model (PROL)            | [prol_fba_data/03_build_fba_surrogate.ipynb](prol_fba_data/03_build_fba_surrogate.ipynb) |
| Sup. Fig. S2 | Performance of the surrogate FBA model (SLIM)            | [slim_fba_data/03_build_fba_surrogate.ipynb](slim_fba_data/03_build_fba_surrogate.ipynb) |
| Sup. Fig. S3 | Test set predictions                                     | [PROL_validation.ipynb](PROL_validation.ipynb) |
| Sup. Fig. S4 | Model architecture and configuration comparison          | [PROL_check_MPMs.ipynb](PROL_check_MPMs.ipynb) |
| Sup. Fig. S5 | Std-Hyb internal rate predictions                        | [PROL_check_MPMs.ipynb](PROL_check_MPMs.ipynb) |
| Sup. Fig. S6 | SLIM optimization with Std-Hyb                           | [SLIM_check_MPMs.ipynb](SLIM_check_MPMs.ipynb) |
| Sup. Fig. S7 | LOPO CV results for static n                             | [static_n_comparison.ipynb](static_n_comparison.ipynb) |

Fig. 1 (study overview) is a schematic and not generated from data.


## Reproducing the LOPO Cross Validation

Install the conda environment from [env_cpu_jax.yml](env_cpu_jax.yml) (CPU only):

```bash
conda env create -f env_cpu_jax.yml
conda activate jax
```

Run the two case-study training scripts from the repository root:

```bash
python prol_model_scripts/MPM_144B.py
python slim_model_scripts/SLIM_03C.py
```

Each script executes the full leave-one-process-out cross validation and writes the trained model checkpoints, training histories, and metrics into the corresponding `*_model_data` folder.


## Citation

> Gotsmy, M. & Guillén-Gosálbez, G. *Integrating Metabolic Networks into Hybrid Bioprocess Models.* ETH Zürich, Department of Chemistry and Applied Biosciences.

