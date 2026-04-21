# METABOLIC PROCESS MODELS

Code and data accompanying the manuscript *Integrating Metabolic Networks into Hybrid Bioprocess Models* by M. Gotsmy and G. Guillén-Gosálbez (ETH Zürich).

This repository implements **FBA-Hyb**, a hybrid bioprocess modeling framework that tightly integrates flux balance analysis (FBA) into a neural controlled ODE. Rather than solving an FBA linear program at every integration step, FBA-Hyb uses a fully differentiable surrogate model discovered a priori via symbolic regression, which allows end-to-end gradient-based training. FBA-Hyb is validated against a standard hybrid baseline (**Std-Hyb**) through leave-one-process-out cross validation on two *E. coli* fed-batch case studies: protein L production (**PROL**) and plasmid DNA production under sulfate limitation (**SLIM**).


## Table of Contents

- [Repository Structure](#repository-structure)
  - [Folders](#folders)
  - [Notebooks](#notebooks)
- [Reproducing the LOPO Cross Validation](#reproducing-the-lopo-cross-validation)
- [Citation](#citation)

## Repository Structure

The two case studies are differentiated by two abbreviations: `PROL` for the protein L production study, and `SLIM` for the plasmid DNA production study with sulfate limitation. Scripts, notebooks, and folders are always labeled with one of the abbreviations respectively.

### Folders

|        Folder        | Description |
|----------------------|-------------|
| [`./MPMs`](./MPMs)                             | Core Python package. Contains the model architectures ([MPM_functions_07.py](MPMs/MPM_functions_07.py)), data preprocessing and scalers ([DATA_functions_01.py](MPMs/DATA_functions_01.py)), training/LOPO helpers ([HELPER_functions_02.py](MPMs/HELPER_functions_02.py)), evaluation and plotting ([ANA_functions_02.py](MPMs/ANA_functions_02.py)), and shared utilities ([UTIL_functions_01.py](MPMs/UTIL_functions_01.py)). |
| [`./prol_model_data`](./prol_model_data)       | Trained PROL Std-Hyb and FBA-Hyb model parameters (`.eqx`), training histories, and LOPO cross-validation outputs for the protein L case study. |
| [`./prol_model_scripts`](./prol_model_scripts) | Runnable PROL training scripts. The canonical FBA-Hyb entry point is [MPM_144B.py](prol_model_scripts/MPM_144B.py); the other `MPM_*.py` files are architecture-variant ablations from the structural sensitivity analysis (Sup. Fig. S3). |
| [`./prol_fba_data`](./prol_fba_data)           | PROL GSMM, parsimonious-FBA training data for the symbolic-regression surrogate, and the fitted `W₁`, `W₂` parameter matrices of Eq. 12 (Sup. Tab. S1). |
| [`./prol_process_data`](./prol_process_data)   | PROL experimental fed-batch time series (*E. coli* BL21(DE3), glycerol, full-factorial `T × μ_f` design) and the 15× augmented training data (Sec. 2.5). |
| [`./slim_model_data`](./slim_model_data)       | Trained SLIM Std-Hyb and FBA-Hyb model parameters and LOPO cross-validation outputs for the plasmid DNA / sulfate-limitation case study. |
| [`./slim_model_scripts`](./slim_model_scripts) | Runnable SLIM training scripts. The canonical FBA-Hyb entry point is [SLIM_03C.py](slim_model_scripts/SLIM_03C.py). |
| [`./slim_fba_data`](./slim_fba_data)           | SLIM GSMM, parsimonious-FBA training data, and fitted surrogate FBA parameters (Sup. Tab. S2). |
| [`./slim_process_data`](./slim_process_data)   | SLIM experimental fed-batch data (*E. coli* JM108, constant glucose feed, sulfate-excess and sulfate-limited conditions) and the augmented training data. |

### Notebooks

| Notebook | Description |
|----------|-------------|
| [PROL_check_MPMs.ipynb](PROL_check_MPMs.ipynb)                 | Interactive inspection of trained PROL Std-Hyb and FBA-Hyb models (ensemble predictions, internal fluxes, cross-validation metrics). |
| [SLIM_check_MPMs.ipynb](SLIM_check_MPMs.ipynb)                 | Same as above for the SLIM case study, including the sulfate-limitation analysis of Fig. 4. |
| [PROL_generate_ANN_data.ipynb](PROL_generate_ANN_data.ipynb)   | SRE Variant study: Generates the PROL data for replacing the ANN with an symbolic equation (Sup. Sec. 1.2). |
| [SLIM_generate_ANN_data.ipynb](SLIM_generate_ANN_data.ipynb)   | SRE Variant study: Generates the SLIM data for replacing the ANN with an symbolic equation (Sup. Sec. 1.2). |
| [static_n_comparison.ipynb](static_n_comparison.ipynb)         | Constant n Variant study fixing the FBA objective weight vector `n` to a constant (dFBA-like) (Sup. Fig. S6.). |


## Reproducing the LOPO Cross Validation

Install the conda environment from [env_cpu_jax.yml](env_cpu_jax.yml) (CPU only) or [env_cuda_jax.yml](env_cuda_jax.yml) (CUDA):

```bash
conda env create -f env_cpu_jax.yml   # CPU
# or
conda env create -f env_cuda_jax.yml  # CUDA
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

