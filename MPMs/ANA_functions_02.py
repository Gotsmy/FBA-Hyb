import numpy as np
import matplotlib.pyplot as plt
import pickle
import jax
import jax.numpy as jnp
import diffrax as dfx
import os

import MPMs.UTIL_functions_01 as UF
import MPMs.DATA_functions_01 as DF

import importlib
importlib.reload(UF)
importlib.reload(DF)

def load_history(his_path):
    restored_history = DF.HIS([])
    restored_history.load(his_path)
    return restored_history

def plot_history(history,equal_scale=True,log=True,limit_r2=True,suptitle=""):
    def moving_average(arr, window_size):
        if window_size <= 0:
            raise ValueError("Window size must be positive")
        if window_size > len(arr):
            return []

        return [
            sum(arr[i:i + window_size]) / window_size
            for i in range(len(arr) - window_size + 1)
        ]

    his_length = len(history.metric_names)
    n_cols = 3
    n_rows = (his_length)//n_cols+1
    plt.figure(dpi=200,figsize=(n_cols*3,n_rows*3))
    plt.suptitle(suptitle)
    axes = []
    non_r2_axes = []
    for i in range(history.n):
        ax = plt.subplot(history.n//3+1,3,i+1)
        axes.append(ax)
        tx = history.train_epoch
        ty = history.train[:,i]
        vx = history.valid_epoch
        vy = history.valid[:,i]


        ax.plot(tx,ty,label="train",color="k",alpha=.2)
        ax.plot(moving_average(tx,np.max([1,len(tx)//10])),
                moving_average(ty,np.max([1,len(tx)//10])),
                label="train\n(mov avg)",color="k",alpha=1)

        ax.plot(vx,vy,label="valid",color="r",alpha=.2)
        ax.plot(moving_average(vx,np.max([1,len(tx)//10])),
                moving_average(vy,np.max([1,len(tx)//10])),
                label="valid\n(mov avg)",color="r",alpha=1)
        ax.set_ylabel(history.metric_names[i])

        if "R2" in history.metric_names[i].upper():
            if limit_r2:
                ax.set_ylim(0,1.1)
            ax.axhline(1,linestyle="--")
        else:
            non_r2_axes.append(ax)
            if log:
                ax.set_yscale("log")

    axes[0].legend(fontsize=7)
    if equal_scale and his_length > 1:
        ub = np.max([ax.get_ylim() for ax in non_r2_axes[1:]])
        lb = np.min([ax.get_ylim() for ax in non_r2_axes[1:]])
        if log:
            lb = np.max([lb,1e-6])
        for ax in non_r2_axes[:]:
            ax.set_ylim(lb,ub)
    plt.tight_layout()
    plt.show()

def compare_versions(versions,only_last_time_point=False,last=True,equal_comparison=True):
    """
    If last == False, then best.
    If equal_comparison only processes are considered that have been tested in all versions.
    """

    all_data = {}
    all_r2 = {}
    all_mae = {}
    min_processes = np.inf
    for i, version in enumerate(versions):
        _ = load_metrics(version,id="_sc")
        all_data[version] = _
        if len(_[-1]) < min_processes:
            min_processes = len(_[-1])
    if not equal_comparison:
        min_processes = 12

    print("--- R2 ---")
    for i, version in enumerate(versions):
        TMP1, best_preds, last_preds, tested_name = [j[:min_processes] for j in all_data[version]]
        
        TMP2 = last_preds if last else best_preds
        if only_last_time_point:
            TMP1 = TMP1[:,[-1],:]
            TMP2 = TMP2[:,[-1],:]

        info = get_info_str(version, tested_name)    
        all_r2[version] = print_stats(TMP1,TMP2,version,header=i==0,verbose=1,mode="R2",info=info)

    print("\n--- MAE ---")
    for i, version in enumerate(versions):
        TMP1, best_preds, last_preds, tested_name = [j[:min_processes] for j in all_data[version]]

        TMP2 = last_preds if last else best_preds
        if only_last_time_point:
            TMP1 = TMP1[:,[-1],:]
            TMP2 = TMP2[:,[-1],:]


        info = get_info_str(version, tested_name)    
        all_mae[version] = print_stats(TMP1,TMP2,version,header=i==0,verbose=1,mode="MAE",info=info)
    return all_data, all_r2, all_mae

def load_metrics(version,id=""):
    # check if OptFed or SLIM path
    filepath = f"/mnt/y/code/250513_metabolic_process_models/prol_model_data/{version}/analysis{id}.pkl"
    if not os.path.isfile(filepath):
        filepath = f"/mnt/y/code/250513_metabolic_process_models/slim_model_data/{version}/analysis{id}.pkl"

    with open(filepath,"rb") as file:
        analysis = pickle.load(file)

    y_true       = analysis["y_true"]
    best_preds   = analysis["best_preds"]
    last_preds   = analysis["last_preds"]
    tested_names = analysis["tested_names"]
    return y_true, best_preds, last_preds, tested_names

def get_info_str(version, tested_names):
    fba_type = get_fba_type(version)
    if len(tested_names) < 12:
        return f"{fba_type}; incomplete ({len(tested_names)} processes)"
    else:
        return f"{fba_type}"
    
def _OLD_get_fba_type(lines):
    for line in lines:
        if '"FBA"' in line and "()" in line:
            if "AMNfba()" in line:
                if "inf" in line:
                    fba_type = "infNNfba"
                elif "obj" in line:
                    fba_type = "objNNfba"
                else:
                    fba_type = "NNfba"
            elif "SRfba()" in line:
                if "auto" in line:
                    fba_type = "infSRfba"
                else:
                    fba_type = "SRfba"
            elif "NOfba()" in line:
                fba_type = "NOfba"
            else:
                fba_type = "unknown"
            if 'ALL' in line:
                fba_type += "_ALL"
            return fba_type
    return "unknown"

def get_fba_type(version):
    if "MPM" in version:
        if version.split("/")[0][-1] == "A":
            return "NO"
        elif version.split("/")[0][-1] == "B":
            if "ANN2SR" in version:
                return "SR"
            else:
                return "NN"
        else:
            return "unknown"
    else:
        if version.split("/")[0][-1] == "A":
            return "NO"
        elif version.split("/")[0][-1] == "C":
            if "ANN2SR" in version:
                return "SR"
            else:
                return "NN"
        else:
            return "unknown"
                
def print_stats(A,B,name,header=False,verbose=True,mode="R2",info=""):
    if mode == "R2":
        stats_function = UF.R2
    elif mode == "MAE":
        stats_function = UF.MAE
    elif mode == "RMSE":
        stats_function = UF.RMSE
    else:
        raise NotImplementedError

    labels = ["T (no V)","G","P","X","V"]
    stats = [stats_function(A[:,:,:3].flatten(),B[:,:,:3].flatten())]

    for nr, val in enumerate(["G","P","X","V"]):
        a = A[:,:,nr].flatten()
        b = B[:,:,nr].flatten()
        stats.append(stats_function(a,b))
    
    out_str = f"{name:>10} | "
    out_lab = f"{'  ':>10} | "
    for stat, label in zip(stats,labels):
        if stat < -1e4:
            out_str += f"{stat:8.1e} | "
        else:
            out_str += f"{stat:8.4f} | "
        out_lab += f"{label:8} | "

    if header:
        print(out_lab)
        print("-"*len(out_lab))
    if verbose:
        print(out_str,info)
    return jnp.array(stats)

def parse_data_to_model(model,data_,key,timepoints=None):
    """
    model : eqx.Module
        process model
    data : tuple
        of (yy, YT, uc, UT, DS, yi)
    key : jax.random.PRNGKey
    timepoints : Int, optional
        optional, Integer specifying the number of timepoints to use for prediction.
    """

    yy, YT, uc, UT, DS, yi = data_
    # only use y0 for prediction
    y0 = yy[:,0,:] # stat vars at YT[0]
    yi = yi[:,0,:] # initial state variables at t=0
    # times have to be 1-dimensional only
    if timepoints is not None:
        YT = YT[:,:,0]
        YT = jnp.array([jnp.linspace(_.min(),_.max(),timepoints) for _ in YT])
    else:
        YT = YT[:,:,0]
    UT = UT[:,:,0]
    # discrete steps
    DS = DS[:,:,0]

    dat2mod = YT, y0, yi, UT, uc, DS
    in_axes = (0, 0, 0, 0, 0, 0, None)
    if hasattr(model,"HYB"):
        yy_pred = jax.vmap(model, in_axes=in_axes)(*dat2mod, key)
    else:
        yy_pred = jax.vmap(model.AVG_call, in_axes=in_axes)(*dat2mod, key)
    return YT, yy_pred

def _get_fba_obj_per_timepoint(model, t, y, args):
    y, Y, u, U, yi, YI, key = model.HYB.unpack_inputs(t, y, args)
    fba_obj = model.HYB.ANN_fba_obj(jnp.concat([y,yi,u]),key=key)
    return jnp.concat([Y,YI,U]), fba_obj

def _get_flux_per_timepoint(model, t, y, args):
    y, Y, u, U, yi, YI, key = model.HYB.unpack_inputs(t, y, args)
    V_out = model.HYB.get_metabolic_fluxes(y, Y, u, U, yi, YI, key)
    return V_out

def _get_fba_obj_per_process(model, data_, key):
    yy, YT, uc, UT, DS, yi = data_
    
    if not hasattr(model.HYB,"nr_extra_metabolites"):
        tmp_yi = yi[0,:]
    else:
        tmp_yi = yi[0,:4]
    tmp_yy = yy[:,:]
    tmp_YT = YT[:,0]
    tmp_UT = UT[:,0]
    tmp_uc = uc[:,:,:]

    tmp_control = dfx.CubicInterpolation(tmp_UT, tmp_uc)
    tmp_args = (tmp_control,key,tmp_yi)
    fba_obj = jax.vmap(_get_fba_obj_per_timepoint, in_axes=(None, 0, 0, None))(model, tmp_YT, tmp_yy, tmp_args)
    return fba_obj

def _get_flux_per_process(model, data_, key):
    yy, YT, uc, UT, DS, yi = data_
    
    if not hasattr(model.HYB,"nr_extra_metabolites"):
        tmp_yi = yi[0,:]
    else:
        tmp_yi = yi[0,:4]
    tmp_yy = yy[:,:]
    tmp_YT = YT[:,0]
    tmp_UT = UT[:,0]
    tmp_uc = uc[:,:,:]

    tmp_control = dfx.CubicInterpolation(tmp_UT, tmp_uc)
    tmp_args = (tmp_control,key,tmp_yi)
    V_out = jax.vmap(_get_flux_per_timepoint, in_axes=(None, 0, 0, None))(model, tmp_YT, tmp_yy, tmp_args)
    return V_out

def get_metabolic_fluxes(model, data_, key):
    """
    model : eqx.Module process model
    data : tuple of (yy, YT, uc, UT, DS, yi)
    key : jax.random.PRNGKey
    """

    ALL_V_out = jax.vmap(_get_flux_per_process, in_axes=(None, 0, None))(model, data_, key)
    ALL_fba_obj = jax.vmap(_get_fba_obj_per_process, in_axes=(None, 0, None))(model, data_, key)
    return ALL_V_out, ALL_fba_obj

def inject_predictions_into_data(data_,yy,YT):
    # print(data_[1].shape, YT.shape)
    _, _, uc, UT, DS, yi = data_
    # expand YT to original shape
    if YT.ndim == 2:
        YT = YT[:,:,jnp.newaxis]
    data_ = yy, YT, uc, UT, DS, yi
    return data_    

# def NEW_get_values_per_process(model, data_, key):
#     yy, YT, uc, UT, DS, yi = data_
    
#     if not hasattr(model.HYB,"nr_extra_metabolites"):
#         tmp_yi = yi[0,:]
#     else:
#         tmp_yi = yi[0,:4]
#     tmp_yy = yy[:,:]
#     tmp_YT = YT[:,0]
#     tmp_UT = UT[:,0]
#     tmp_uc = uc[:,:,:]

#     tmp_control = dfx.CubicInterpolation(tmp_UT, tmp_uc)
#     tmp_args = (tmp_control,key,tmp_yi)
#     fba_obj = jax.vmap(_get_fba_obj_per_timepoint, in_axes=(None, 0, 0, None))(model, tmp_YT, tmp_yy, tmp_args)
#     return fba_obj

def NEW_get_rates_per_process(model, data_, key, debug=False):
    yy, YT, uc, UT, DS, yi = data_
    
    tmp_yi = yi[0,:4]
    tmp_yy = yy[:,:]
    tmp_YT = YT[:,0]
    tmp_UT = UT[:,0]
    tmp_uc = uc[:,:,:]

    HYB = get_HYB(model,debug=debug)
    
    # Extend tmp_yy if necessary
    if tmp_yy.shape[1] < HYB.NR_state_variables + HYB.NR_extra_variables:
        new_shape = (tmp_yy.shape[0], tmp_yy.shape[1] + HYB.NR_extra_variables)
        extended = jnp.full(new_shape, jnp.nan)

        # Copy the original array into the left-most columns
        tmp_yy = extended.at[:, :tmp_yy.shape[1]].set(tmp_yy)

    # print(tmp_YT.shape, tmp_yy.shape, tmp_UT.shape, tmp_uc.shape, tmp_yi.shape)

    tmp_control = dfx.CubicInterpolation(tmp_UT, tmp_uc)
    tmp_args = (tmp_control,key,tmp_yi)
    if hasattr(model,"HYB"):
        V_out = jax.vmap(model.HYB.dNdt_and_intermediates, in_axes=(0, 0, None))(tmp_YT, tmp_yy, tmp_args)[1]
    else:
        print(tmp_YT.shape, tmp_yy.shape)
        V_out = jax.vmap(model.AVG_get_intermediates, in_axes=(0, 0, None))(tmp_YT, tmp_yy, tmp_args)
    return V_out

def get_HYB(model,debug=False):
    if hasattr(model, 'HYB'):
        if debug:
            print("HYB module detected.")
        HYB = model.HYB
    else:
        if debug:
            print("HYB module NOT detected. Assuming Ensemble model.")
        HYB = model.models[0].HYB
    return HYB

def NEW_get_predictions(model,data_,key,mode,timepoints=1000,debug=False):
    """
    model : eqx.Module as process model
    data : tuple of (yy, YT, uc, UT, DS, yi)
    key : jax.random.PRNGKey
    mode : str, specifying the mode of prediction:
        - "true": returns true values/ rates predicted from true values
        - "pred": returns predicted values from y0/ yI with same timepoints as data
        - "smth": returns predicted values with timepoints specified
    timepoints : Int, optional, specifying the number of timepoints to use for prediction.
    debug : bool, optional
        If True, then the assert statements are ignored.

    Returns
    -------
    dict : intermediates
        Dictionary containing state values, rates, and intermediates from the model prediction.
    """

    assert mode in ["true", "pred", "smth"], "Mode must be 'true', 'pred', or 'smth'."

    HYB = get_HYB(model,debug=debug)

    if mode == "true":
        yy = data_[0]
        YY = HYB.SCL_yy.unscale(yy)
        YT = data_[1][:,:,0]
    else:        
        if mode == "pred":
            timepoints = None
        YT, yy = parse_data_to_model(model, data_, key, timepoints)
        yy_for_scaling = yy[:,:,:HYB.NR_state_variables]
        YY = HYB.SCL_yy.unscale(yy_for_scaling)
        data_ = inject_predictions_into_data(data_, yy, YT)


    intermediates = jax.vmap(NEW_get_rates_per_process, in_axes=(None, 0, None))(model, data_, key)
        
    # yy, YY, and YT are also tracked in intermediates,
    # therefore we don't need to return them separately.
    if not debug:
        assert jnp.allclose(intermediates["y"][:,:,:HYB.NR_state_variables], yy[:,:,:HYB.NR_state_variables])
        assert jnp.allclose(intermediates["Y"][:,:,:HYB.NR_state_variables], YY)
        assert jnp.allclose(intermediates["t"], YT)
    return intermediates

def vectorized_load_metrics(versions,equal_comparison, id):
    data = {}
    min_processes = np.inf
    
    for i, version in enumerate(versions):
        # load data in dictionary
        data[version] = load_metrics(version,id=id)
        # get minimum number of processes
        version_length = len(data[version][0])
        if version_length < min_processes:
            min_processes = version_length
    if not equal_comparison:
        min_processes = 12
    # trim all versions to the minimum number of processes
    for i, version in enumerate(versions):
        data[version] = [j[:min_processes] for j in data[version]]

    return data

def get_predictions(*args, **kwargs):
    raise NotImplementedError("This function has been replaced by NEW_get_predictions.")

# def get_predictions(model,data_,key,mode,timepoints=1000):
#     """
#     model : eqx.Module as process model
#     data : tuple of (yy, YT, uc, UT, DS, yi)
#     key : jax.random.PRNGKey
#     mode : str, specifying the mode of prediction:
#         - "true": returns true values
#         - "pred": returns predicted values with same timepoints as data
#         - "smth": returns predicted values with timepoints specified
#     timepoints : Int, optional, specifying the number of timepoints to use for prediction.

#     Returns
#     -------
#     tuple : (YY, YT)
#     """

#     assert mode in ["true", "pred", "smth"], "Mode must be 'true', 'pred', or 'smth'."

#     if mode == "true":
#         yy = data_[0]
#         YY = model.HYB.SCL_yy.unscale(yy)
#         YT = data_[1][:,:,0]
#         if hasattr(model.HYB,"nr_extra_metabolites"):
#             V_out = jnp.zeros_like(YY)* jnp.nan
#             outshape = (YY.shape[0],YY.shape[1],model.HYB.ANN_fba_obj.layers[-1].weight.shape[0])
#             fba_obj = jnp.zeros(outshape)* jnp.nan
#         else:
#             V_out, fba_obj = get_metabolic_fluxes(model, data_, key)
#         return YY, YT, V_out, yy, fba_obj

#     elif mode == "pred":
#         timepoints = None
    
#     YT, yy = parse_data_to_model(model, data_, key, timepoints)
#     if hasattr(model.HYB,"nr_extra_metabolites"):
#         nr_extra_metabolites = model.HYB.nr_extra_metabolites
#         yy_for_scaling = yy[:,:,:-nr_extra_metabolites]
#     else:
#         yy_for_scaling = yy
    
#     YY = model.HYB.SCL_yy.unscale(yy_for_scaling)
#     data_ = inject_predictions_into_data(data_, yy, YT)
#     V_out, fba_obj = get_metabolic_fluxes(model, data_, key)
#     return YY, YT, V_out, yy, fba_obj

def vectorized_load_metrics(versions,equal_comparison, id):
    data = {}
    min_processes = np.inf
    
    for i, version in enumerate(versions):
        # load data in dictionary
        data[version] = load_metrics(version,id=id)
        # get minimum number of processes
        version_length = len(data[version][0])
        if version_length < min_processes:
            min_processes = version_length
    if not equal_comparison:
        min_processes = 12
    # trim all versions to the minimum number of processes
    for i, version in enumerate(versions):
        data[version] = [j[:min_processes] for j in data[version]]

    return data

def get_metric_function(metric):
    if metric == "R2":
        metric_function = UF.R2
    elif metric == "MAE":
        metric_function = UF.MAE
    elif metric == "RMSE":
        metric_function = UF.RMSE
    elif metric == "NMAE":
        metric_function = UF.NMAE
    else:
        raise NotImplementedError
    return metric_function

def NEW_print_stats(UNSCALED, name, header=False, verbose=True, metric="R2", info=""):
    metric_function = get_metric_function(metric)

    # CALCULATING METRICS
    US_true, US_pred = UNSCALED
    labels = ["avg(GPX)","avg(PX)","G","P","X","V","P/X"]
    STATS = []

    # calculate G, P, X, V
    for j in range(US_true.shape[-1]):
        true = US_true[:,:,j].flatten()
        pred = US_pred[:,:,j].flatten()
        stat = metric_function(true, pred)
        STATS.append(stat)
    
    # calculate GPX, PX
    STATS = [np.mean(STATS[0:3]), np.mean(STATS[1:3])]+STATS
    
    # calculate P/X
    true = (US_true[:,:,1]/US_true[:,:,2]).flatten()
    pred = (US_pred[:,:,1]/US_pred[:,:,2]).flatten()
    stat = metric_function(true, pred)
    STATS.append(stat)
    STATS = jnp.array(STATS)

    # PRINTING
    out_str = f"{name:>25} | "
    out_lab = f"{'  ':>25} | "
    for stat, label in zip(STATS,labels):
        if stat < -1e4:
            out_str += f"{stat:8.1e} | "
        else:
            out_str += f"{stat:8.4f} | "
        out_lab += f"{label:8} | "

    if header:
        print(out_lab)
        print("-"*len(out_lab))
    if verbose:
        print(out_str,info)
    
    return STATS, labels

def NEW_compare_versions(versions,last_model=True,equal_comparison=True,metrics=[],verbose=True):
    assert len(versions) > 0, "At least one version must be provided."

    SC_data = vectorized_load_metrics(versions, equal_comparison, id="_sc")
    US_data = vectorized_load_metrics(versions, equal_comparison, id="_us")

    if verbose:
        print("Last Model       :", last_model)
        print("Equal Comparison :", equal_comparison)

    ALL_metrics = []
    for metric in metrics:
        TMP_metrics = {}
        print(f"--- {metric:4} ---")
        for i, version in enumerate(versions):
            # metric over all scaled values
            SC_true, best_preds, last_preds, tested_name = SC_data[version]
            SC_pred = last_preds if last_model else best_preds
            SCALED = (SC_true, SC_pred)

            US_true, best_preds, last_preds, tested_name = US_data[version]
            US_pred = last_preds if last_model else best_preds
            UNSCALED = (US_true, US_pred)

            info = get_info_str(version, tested_name)    
            _ = NEW_print_stats(UNSCALED,version,header=i==0,verbose=verbose,metric=metric,info=info)
            TMP_metrics[version] = _[0]  # only store the stats, not the labels
        ALL_metrics.append(TMP_metrics)
    return ALL_metrics, _[1] # labels once

def get_color_from_fba_type(fba_type):
    if "SR" in fba_type:
        return "#97755c"
    elif "NN" in fba_type:
        return "#008e8f"
    elif "NO" in fba_type:
        return "#646678"
    else:
        # unknown
        return "#000000"
    
def plot_compare_versions(versions, metrics, METRIC_values, METRIC_index=0, suptitle=""):
    figsize = (np.max([2,len(versions)/2]),4*len(metrics))
    figsize = (1+len(versions)/2,4*len(metrics))
    plt.figure(dpi=200,figsize=figsize)
    axes = []
    for i in range(len(metrics)):
        axes.append(plt.subplot(len(metrics),1,i+1))
        axes[i].set_title(metrics[i]+" - "+suptitle)

    offset = 0
    old_version = ""
    locs = []
    for i, version in enumerate(versions):
        if old_version != version.split("/")[0][-3:-1]:
            offset += .2
            old_version = version.split("/")[0][-3:-1]

        fba = get_fba_type(version)
        c = get_color_from_fba_type(fba)
        
        loc = i*.55+offset

        for i, metric in enumerate(metrics):
            value = METRIC_values[i][version][METRIC_index]
            text_value = np.max([value*1.05,.05,value+.05])
            axes[i].bar( loc,value,color=c,edgecolor="k",width=.5)
            axes[i].text(loc,text_value,f"{value:.2f}",va="center",ha="center")
        locs.append(loc)

    for ax in axes:
        ax.set_xticks(locs,[v.strip("MPM_")+f"\n{get_fba_type(v)}" for v in versions],fontsize=8,rotation=90)
        # increase y_max so the text fits inside the panel
        ylim = ax.get_ylim()
        ax.set_ylim(ylim[0],np.max([ylim[1]+.1,ylim[1]*1.1])) 
        ax.axhline(0,color="grey")
    plt.tight_layout()
    plt.show()

def NEW_print_summary(version,last_model=True,metric="R2",verbose=True):
    SC_data = load_metrics(version,id="_sc")
    US_data = load_metrics(version,id="_us")

    if last_model:
        info = f"LAST {version}"
        Y_PRED = US_data[2]
        Y_TRUE = US_data[0]
    else:
        info = f"BEST {version}"
        Y_PRED = US_data[1]
        Y_TRUE = US_data[0]
    
    if verbose:
        print(info)
        print("Last Model :", last_model)
        print("Metric     :", metric)

    stats = []
    for i, name in enumerate(SC_data[3]):
        UNSCALED = (Y_TRUE[[i],:,:], Y_PRED[[i],:,:])
        _ = NEW_print_stats(UNSCALED,name,verbose=verbose,metric=metric,header=i==0)
        stats.append(_[0])  # only store the stats, not the labels
    return jnp.array(stats), _[1]  # labels once









