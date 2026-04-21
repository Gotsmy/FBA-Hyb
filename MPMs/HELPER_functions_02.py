import numpy as np
import pandas as pd
import jax.numpy as jnp
import equinox as eqx
import pickle
import MPMs.MPM_functions_07 as MF
import MPMs.DATA_functions_01 as DF
import MPMs.UTIL_functions_01 as UF
import MPMs.ANA_functions_02 as AF

# --- HELPER FUNCTIONS --- #

def get_process_scalers(method):

    if method == "standard":
        COL_off = ["G","P","X","V_N"]
        COL_onl = ["t","T","Cf_N","G_Cf"]
        SCL_off = DF.SCL_standard(jnp.zeros((1,4)),COL_off)
        SCL_onl = DF.SCL_standard(jnp.zeros((1,4)),COL_onl)
        SCL_off = eqx.tree_deserialise_leaves("/mnt/y/code/250513_metabolic_process_models/prol_process_data/250804_exp.off_scl.standard.eqx", SCL_off)
        SCL_onl = eqx.tree_deserialise_leaves("/mnt/y/code/250513_metabolic_process_models/prol_process_data/250804_exp.onl_scl.standard.eqx", SCL_onl)
        return SCL_off, SCL_onl
    elif method == "mean":
        COL_off = ["G","P","X","V_N"]
        COL_onl = ["t","T","Cf_N","G_Cf"]
        SCL_off = DF.SCL_mean(jnp.zeros((1,4)),COL_off)
        SCL_onl = DF.SCL_mean(jnp.zeros((1,4)),COL_onl)
        SCL_off = eqx.tree_deserialise_leaves("/mnt/y/code/250513_metabolic_process_models/prol_process_data/250804_exp.off_scl.mean.eqx", SCL_off)
        SCL_onl = eqx.tree_deserialise_leaves("/mnt/y/code/250513_metabolic_process_models/prol_process_data/250804_exp.onl_scl.mean.eqx", SCL_onl)
        return SCL_off, SCL_onl
    else:
        raise NotImplementedError(f"Unknown method: {method}")
    
def SLIM_get_process_scalers(method):

    if method == "standard":
        raise NotImplementedError("SLIM does not support standard scaling for process data.")
        COL_off = ["G","P","X","V_N"]
        COL_onl = ["t","Cf_N","G_Cf"]
        SCL_off = DF.SCL_standard(jnp.zeros((1,4)),COL_off)
        SCL_onl = DF.SCL_standard(jnp.zeros((1,3)),COL_onl)
        SCL_off = eqx.tree_deserialise_leaves("slim_process_data/exp.off_scl.standard.eqx", SCL_off)
        SCL_onl = eqx.tree_deserialise_leaves("slim_process_data/exp.onl_scl.standard.eqx", SCL_onl)
        return SCL_off, SCL_onl
    elif method == "mean":
        COL_off = ["G","P","X","V_N"]
        COL_onl = ["t","Cf_N","G_Cf"]
        SCL_off = DF.SCL_mean(jnp.zeros((1,4)),COL_off)
        SCL_onl = DF.SCL_mean(jnp.zeros((1,3)),COL_onl)
        SCL_off = eqx.tree_deserialise_leaves("slim_process_data/exp.off_scl.mean.eqx", SCL_off)
        SCL_onl = eqx.tree_deserialise_leaves("slim_process_data/exp.onl_scl.mean.eqx", SCL_onl)

        # The concentration of G is always 0. therefore I have to replace the 0 in the avg with some reasonable value.
        # For now I think it's ok to hardcode it, but:
        # TODO: think of a proper way to do this.
        SCL_off = eqx.tree_at(
            where=lambda x: x.avg,
            pytree=SCL_off,
            replace=jnp.array([100., *SCL_off.avg[1:]], dtype=SCL_off.avg.dtype)
        )

        return SCL_off, SCL_onl
    else:
        raise NotImplementedError(f"Unknown method: {method}")

def get_fba_scalers(v_inp_flux_names,v_out_flux_names,method):

    SCL_fba_inp = DF.SCL_init_from_fba(v_inp_flux_names,method=method)
    SCL_fba_out = DF.SCL_init_from_fba(v_out_flux_names,method=method)
    return SCL_fba_inp, SCL_fba_out

def get_valid_sets():
    return [[name] for name in ['DoE1_R1', 'DoE1_R2', 'DoE1_R3', 'DoE1_R4', 'DoE2_R1', 'DoE2_R2','DoE2_R3', 'DoE2_R4', 'DoE3_R1', 'DoE3_R2', 'DoE3_R3', 'DoE3_R4']]

def SLIM_get_valid_sets():
    return [[f"exp{i}"] for i in range(1,10)]

def get_valid_sets_exclude_replicates():
    set_list = [
        ["DoE1_R2","DoE1_R4","DoE2_R2","DoE2_R4"],
        ["DoE1_R1"],
        ["DoE1_R3"],
        ["DoE2_R1"],
        ["DoE2_R3"],
        ["DoE3_R1"],
        ["DoE3_R2"],
        ["DoE3_R3"],
        ["DoE3_R4"]
    ]
    return set_list

def get_nnfba_model(model_path,verbose=True):
    # import model module
    MM = UF.import_module_from_path(model_path)
    # init model
    COLs = MM.get_columns_inp_out()
    # get scalers
    SCLs = MM.get_scalers(*COLs)
    # init model
    model = MM.model_construction(SCLs,verbose=verbose)
    model = eqx.tree_deserialise_leaves(model_path.strip(".py")+"/"+model_path.strip(".py").split("/")[-1]+".eqx", model)
    return model, COLs, SCLs

def get_idx_from_col(reaction_ids,reactions):
    reactions = np.array(reactions)
    idx = [jnp.where(i==reactions)[0][0] for i in reaction_ids]
    return jnp.array(idx).astype(int)

def print_and_write_CV_stats(y_true,best_preds,last_preds,tested_names,save_path="",id="",verbose=True):
    # [print(i.shape) for i in y_true]
    y_true = jnp.vstack(y_true)
    best_preds = jnp.vstack(best_preds)
    last_preds = jnp.vstack(last_preds)
    y_true = y_true.reshape(-1,y_true.shape[-2],y_true.shape[-1])
    best_preds = best_preds.reshape(-1,y_true.shape[-2],y_true.shape[-1])
    last_preds = last_preds.reshape(-1,y_true.shape[-2],y_true.shape[-1])
    # print(y_true.shape,best_preds.shape,last_preds.shape)

    output = {
        "y_true": y_true,
        "best_preds": best_preds,
        "last_preds": last_preds,
        "tested_names": tested_names
        }
    
    if len(save_path)>0:
        with open(save_path+f"analysis{id}.pkl","wb") as file:
            pickle.dump(output,file)

    try:
        name = save_path.split("/")[-2]
    except:
        name = ""
    if verbose:
        print("-------------------------------------------------")
        print("-------------------------------------------------")
        print("BEST")
        AF.NEW_print_stats((y_true,best_preds),name,header=True,verbose=True,metric="R2",info=f"{y_true.shape[0]} processes")

        print("-------------------------------------------------")
        print("LAST")
        AF.NEW_print_stats((y_true,last_preds),name,header=True,verbose=True,metric="R2",info=f"{y_true.shape[0]} processes")
        print("-------------------------------------------------")
        print("-------------------------------------------------")

def preprocess_data(exp_dfs,aug_dfs,valid_set,scalers,augmented=True,verbose=True):
    if augmented:
        if verbose:
            print("TRAINING WITH AUGUMENTED DATA")
        train_ = DF.PRE_process_data(aug_dfs,scalers,
                                    valid_set,
                                    inverse=False,
                                    yi_method="strip",
                                    Ycols=["G","P","X","V_N"],
                                    Ucols=["t","T","Cf_N","G_Cf"],
                                    verbose=False
                                    )
    else:
        if verbose:
            print("TRAINING WITH EXPERIMENTAL DATA")
        train_ = DF.PRE_process_data(exp_dfs,scalers,
                                    valid_set,
                                    inverse=False,
                                    yi_method="copy",
                                    Ycols=["G","P","X","V_N"],
                                    Ucols=["t","T","Cf_N","G_Cf"],
                                    verbose=False)
    
    valid_ = DF.PRE_process_data(exp_dfs,scalers,
                                 valid_set,
                                 inverse=True,
                                 yi_method="copy",
                                 Ycols=["G","P","X","V_N"],
                                 Ucols=["t","T","Cf_N","G_Cf"],
                                 verbose=False)
    return train_, valid_

def SLIM_preprocess_data(exp_dfs,aug_dfs,valid_set,scalers,augmented=True,verbose=True):
    raise NotImplementedError("The data preprocessing function is contained in the model script.")
    if augmented:
        if verbose:
            print("TRAINING WITH AUGUMENTED DATA")
        train_ = DF.PRE_process_data_SLIM_ALL(aug_dfs,scalers,
                                    valid_set,
                                    inverse=False,
                                    # here I always need "copy" -- so that t0=0 is included, because we only know SO42- at t0
                                    yi_method="copy",
                                    Ycols=["G","P","X","V_N"],
                                    Ucols=["t","Cf_N","G_Cf"],
                                    verbose=False
                                    )
    else:
        if verbose:
            print("TRAINING WITH EXPERIMENTAL DATA")
        train_ = DF.PRE_process_data_SLIM_ALL(exp_dfs,scalers,
                                    valid_set,
                                    inverse=False,
                                    yi_method="copy",
                                    Ycols=["G","P","X","V_N"],
                                    Ucols=["t","Cf_N","G_Cf"],
                                    verbose=False)
    
    valid_ = DF.PRE_process_data_SLIM_ALL(exp_dfs,scalers,
                                 valid_set,
                                 inverse=True,
                                 yi_method="copy",
                                 Ycols=["G","P","X","V_N"],
                                 Ucols=["t","Cf_N","G_Cf"],
                                 verbose=False)
    return train_, valid_

def PYSR_model_loader(mpm_version,
                      pysr_version,
                      dataset_name,
                      base_model,
                      set_name
                      ):
    import pysr
    
    if dataset_name == "optfed":
        data_base_path = "/mnt/y/code/250513_metabolic_process_models/prol_model_data/"
        script_base_path = "/mnt/y/code/250513_metabolic_process_models/prol_model_scripts/"
    elif dataset_name == "slim":
        data_base_path = "/mnt/y/code/250513_metabolic_process_models/slim_model_data/"
        script_base_path = "/mnt/y/code/250513_metabolic_process_models/slim_model_scripts/"
    else:
        raise NotImplementedError(f"Unknown dataset: {dataset_name}")
    
    pysr_path = f"{data_base_path}/{mpm_version}/{pysr_version}"
    model_path = f"{script_base_path}/{mpm_version}.py"
    
    # load metabolic bioprocess model
    base_model = eqx.nn.inference_mode(base_model)
    weights_path = data_base_path + f"/{mpm_version}/{set_name}_last.eqx"
    model = eqx.tree_deserialise_leaves(weights_path, base_model)
    model = eqx.nn.inference_mode(model)

    # load pysr model
    pysr_model = pysr.PySRRegressor.from_file(run_directory=f"{pysr_path}/{set_name}/")
    pysr_export = pysr_model.jax()

    # Load Modification Function
    pysr_script_path = f"{pysr_path}/script_{set_name}.py"
    with open(pysr_script_path,"r") as f:
            model_code = f.read()
    if "__name__ ==" in model_code:
        SR_module = UF.import_module_from_path(pysr_script_path)
        modification_function = SR_module.modify_inputs
    else:
        modification_function = UF.return_same_value

    # load scaling of pysr data (if any)
    if dataset_name == "optfed" and\
            int(mpm_version.split("_")[1][:3]) > 130 or\
                dataset_name == "slim":
        pysr_scl_inp = [DF.SCL_none()]*3
    else:
        raise NotImplementedError("PYSR scaling not implemented for this configuration.")
    

    # add everything into the wrapper
    PYSR_fba_obj = MF.PYSR_wrapper(pysr_export,pysr_scl_inp,modification_function)
    SR_model = eqx.tree_at(
        lambda m: m.HYB.ANN_fba_obj,  # Path to the leaf to replace
        model,                 # The original model
        PYSR_fba_obj           # The replacement module
    )
    return eqx.nn.inference_mode(SR_model)
