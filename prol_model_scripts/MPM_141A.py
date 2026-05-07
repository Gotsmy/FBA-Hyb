
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax as dfx
import optax
import pickle
import importlib

import MPMs.MPM_functions_07 as MF
import MPMs.DATA_functions_01 as DF
import MPMs.UTIL_functions_01 as UF
import MPMs.HELPER_functions_02 as HF
import MPMs.ANA_functions_02 as AF
importlib.reload(UF)
importlib.reload(DF)
importlib.reload(MF)
importlib.reload(HF)
importlib.reload(AF)

#-------------------------------
#--- START MODEL FUNCTIONS -----
#-------------------------------

class HYB(eqx.Module):
    # NEURAL NET to predict fba objective
    ANN_fba_obj : eqx.Module = MF.trainable_field()
    # NEURAL NET to predict qG
    ANN_qG_flux : eqx.Module = MF.trainable_field()
    # NEURAL NET to predict base feed
    ANN_base_fd : eqx.Module = MF.trainable_field()

    # FBA surrogate to predict fba_out
    FBA : eqx.Module = MF.frozen_field()
    # Molar masses of G, P, X in g/mmol
    MOL : jnp.ndarray = MF.frozen_field()

    # Scaler for state variables
    SCL_yy : eqx.Module = MF.frozen_field()
    # Scaler for control variables
    SCL_uu : eqx.Module = MF.frozen_field()
    # Scaler for mod_ann_inp
    SCL_mod_ann_inp : eqx.Module = MF.frozen_field()
    
    # Index of relevant fluxes in v_out
    IDX_q_rel : jnp.ndarray = MF.frozen_field()
    # Index of extra fluxes in v_out
    IDX_v_ext : jnp.ndarray = MF.frozen_field()
    # Number of extra metabolites that are not in the state variables
    NR_extra_variables : int = MF.frozen_field()
    # Number of state variables
    NR_state_variables : int = MF.frozen_field()


    # Extra parameters for the switch to G-limited behaviour
    LIM_params : jnp.ndarray = MF.trainable_field()
    # Function on how to handle G-limitation
    LIM_handler : callable = MF.frozen_field()
    # Function on how to modify the ANN input
    MOD_ann_inp : callable = MF.frozen_field()
    

    def __init__(self, 
                 SCL_yy, SCL_uu, 
                 IDX_q_rel, IDX_v_ext,
                 ANN_base_fd,ANN_fba_obj, ANN_qG_flux, 
                 FBA_model,
                 MOL_masses,
                 LIM_handler,
                 MOD_ann_inp,
                 verbose=True):
        
        """
        This is the hybrid ODE part of my ODE, where I calculate
        state variable derivatives from metabolic rates. 
        """

        self.ANN_fba_obj = ANN_fba_obj # Neural Net to predict FBA objective
        self.ANN_qG_flux = ANN_qG_flux # Neural Net to predict FBA qG flux
        self.ANN_base_fd = ANN_base_fd # Neural Net to predict base feed rate

        self.FBA = FBA_model # Surrogate FBA model
        self.MOL = MOL_masses
        
        self.SCL_yy = SCL_yy
        self.SCL_uu = SCL_uu
        self.SCL_mod_ann_inp = DF.SCL_none(dim=ANN_fba_obj.in_size)
        
        self.IDX_q_rel = IDX_q_rel
        self.IDX_v_ext = IDX_v_ext
        self.NR_extra_variables = len(self.IDX_v_ext)
        self.NR_state_variables = SCL_yy.avg.shape[0]

        self.LIM_handler = LIM_handler
        self.LIM_params = jnp.array([5., 1.]) 
        self.MOD_ann_inp = MOD_ann_inp

        if verbose:
            MF.print_model_structure(self)
    
    def get_base_feed_rate(self, mod_ann_inp, q_rel, key):
        """
        Predict the base feed rate from state and control variables 
        & relevant fluxes.
        """
        Bf = self.ANN_base_fd(jnp.concat([q_rel,mod_ann_inp]),key=key)/10
        
        return Bf

    def get_relevant_fluxes(self,V_out):
        """
        Get relevant metabolic fluxes as indicated by the self.FBA.idx_out attribute.
        Relevant fluxes here are [qG, qP, qX]
        """

        Q_rel = jnp.take(V_out,self.IDX_q_rel)
        Q_rel = Q_rel*self.MOL/1000.
        return Q_rel
    
    
    def get_metabolic_fluxes(self, y, Y, u, U, yi, YI, key):
        """
        Calculates metabolic fluxes from state and control variables
        with NN & surrogate FBA. Does not function properly with NOfba

        Input
        -----
        y, Y ... SCALED/ UNSCALED state variabes
        u, U ... SCALED/ UNSCALED control variabes
        yi,YI... SCALED/ UNSCALED induction variables
        key  ... jax key

        Returns
        -------
        v_out ... vector of UNSCALED metabolic fluxes [qG, qP, qX, (qM, ...)] in g/(g h)
        """

        # --- STEP 1 --- #
        # get scaled FBA input values from 
        # 1. scaled state variables (y)
        # 2. initial state variables (yi)
        # 3. control variables (u)

        mod_ann_inp_raw = self.MOD_ann_inp(self,y, Y, u, U, yi, YI, key)
        mod_ann_inp = self.SCL_mod_ann_inp.scale(mod_ann_inp_raw)

        fba_obj = self.ANN_fba_obj(mod_ann_inp,key=key)
        qG_ann  = self.ANN_qG_flux(mod_ann_inp,key=key)

        # unscale qG_ann
        zeros = jnp.zeros(self.FBA.SCL_out.avg.shape[0]-1)
        QG_ann = self.FBA.SCL_out.unscale(jnp.concat([qG_ann,zeros]))[0]

        # --- STEP 2 --- #
        # handle G-limitation & combine to FBA input vector

        QG_lim = MF.get_qG_LIM(Y, U)
        QG_cmb = self.LIM_handler(QG_ann, QG_lim, self.LIM_params, Y[0])

        # scale again
        qG_cmb = self.FBA.SCL_out.scale(jnp.concat([QG_cmb,zeros]))[0:1]
        

        # --- STEP 3 --- #
        # calculate UNSCALED metabolic fluxes
        fba_inp = jnp.concat([qG_cmb, fba_obj])
        FBA_out = self.FBA.get_unscaled(fba_inp)

        return FBA_out, fba_inp, QG_ann, QG_lim, QG_cmb, mod_ann_inp, mod_ann_inp_raw
    
    def unpack_inputs(self,t, y, args):
        """
        Input
        -----
        t    ... un-scaled time
        y    ... scaled state variables at time t
        args ... spline, key
            control ... scaled spline object
            key    ... jax.random.key

        Returns
        -------
        y, Y ... SCALED/ UNSCALED state variabes
        u, U ... SCALED/ UNSCALED control variabes
        yi,YI... SCALED/ UNSCALED induction variables
        key  ... jax key
        """

        control, key, yi = args

        Y  = self.SCL_yy.unscale(y[:self.NR_state_variables])
        YI = self.SCL_yy.unscale(yi)
        u = control.evaluate(t)
        U = self.SCL_uu.unscale(u)

        return y, Y, u, U, yi, YI, key
    
    def __call__(self, t, y, args):
        """
        Wrapper for self.dNdt_and_intermediates() where only dNdt is returned.
        """
        return self.dNdt_and_intermediates(t, y, args)[0]
    
    def dNdt_and_intermediates(self, t, y, args):
        """
        Calculates derivatives from state and control variables
        with NN & surrogate FBA.

        Input
        -----
        t    ... un-scaled time
        y    ... scaled state variables at time t
        args ... spline, key
            control ... scaled spline objects
            key    ... jax.random.key

        Returns
        -------
        dy_dt ... vector of scaled state variable derivatives
        """

        y, Y, u, U, yi, YI, key = self.unpack_inputs(t, y, args)
        
        # metabolic rates are predicte from scaled variables
        _ = self.get_metabolic_fluxes(y, Y, u, U, yi, YI, key) # mmol/(gX h)
        FBA_out, fba_inp, QG_ann, QG_lim, QG_cmb, mod_ann_inp, mod_ann_inp_raw = _
        Q_rel = self.get_relevant_fluxes(FBA_out) # g/(g h)
        v_ext = jnp.take(self.FBA.SCL_out.scale(FBA_out),self.IDX_v_ext)
        Bf    = self.get_base_feed_rate(mod_ann_inp, Q_rel, key)
        Tf    = U[2] + Bf[0] # total feed rate [kg/h]
        Xr    = Y[2] - Y[1]  # active biomass [g/kg]

        # calculation of derivatives in unscaled space
        #        metabolism    dilution       feed
        dG_dt = -Q_rel[0]*Xr - Tf/Y[3]*Y[0] + U[2]/Y[3]*U[3]
        dP_dt =  Q_rel[1]*Xr - Tf/Y[3]*Y[1]
        dXr_dt=  Q_rel[2]*Xr - Tf/Y[3]*Xr
        dX_dt = dP_dt + dXr_dt
        dV_dt =                Tf
        dY_dt = jnp.array([dG_dt,dP_dt,dX_dt,dV_dt])
        dy_dt = self.SCL_yy.grad_scale(dY_dt)

        dE_dt = v_ext*Xr/100 - Tf/Y[3]*y[self.NR_state_variables:] # scaled and divided by 100
        dN_dt = jnp.concat([dy_dt,dE_dt])

        # track all intermediate values for analysis
        inter = {
            "t" : t,
            "y" : y,
            "Y" : Y,
            "u" : u,
            "U" : U,
            "yi": yi,
            "YI": YI,
            "mod_ann_inp_raw": mod_ann_inp_raw,
            "mod_ann_inp": mod_ann_inp,
            "FBA_inp": fba_inp,
            "QG_ann" : QG_ann,
            "QG_lim" : QG_lim,
            "QG_cmb" : QG_cmb,
            "V_out" : FBA_out,
            "Q_rel" : Q_rel,
            "v_ext" : v_ext,
            "Bf"    : Bf,
            "Tf"    : Tf,   
            "Xr"    : Xr,
            "dY_dt" : dY_dt,
            "dE_dt" : dE_dt,
            "dN_dt" : dN_dt,
        }

        return dN_dt, inter
    
class ODE(eqx.Module):
    HYB: eqx.Module

    def __init__(self,HYB_kwargs,verbose=True):
        """
        My  process ODE model.
        """

        self.HYB = HYB(**HYB_kwargs)
        if verbose:
            MF.print_model_structure(self)

    @eqx.filter_jit
    def get_solution(self, YT, y0, yi, UT, uc, DS, key):
        """
        Notes
        -----
        In this function, uppercase parameter letters indicate unnormalized
        values, lower case parameter letters indicated normalized values.

        n_yy ... number of state variables
        n_yt ... number of state variable time points
        n_uu ... number of control variables
        n_ut ... number of control variable time points
        
        Parameters
        ----------

        YT : jax.Array of shape (n_yt,)
            time steps
        y0 : jax.Array of shape (n_yy,)
            normalized state variable values at YT[0]
        yi : jax.Array of shape (n_yy,)
            normalized state variable values at t=0.
            these values are parsed as arguments to the NN.
            (these values may be the same as y0)
        UT : jax.Array of shape (n_ut,)
            time steps
        uc : jax.Array of shape (n_uu, n_ut-1, 4)
            control spline parameters
        
        """

        control = dfx.CubicInterpolation(UT, uc)
        y0 = jnp.concatenate([y0, jnp.zeros(self.HYB.NR_extra_variables)])
        stepsize_controller = dfx.ClipStepSizeController(
                              dfx.PIDController(rtol=1e-3, atol=1e-3),
                              jump_ts=DS)
        solution = dfx.diffeqsolve(
            dfx.ODETerm(self.HYB),
            dfx.Tsit5(),
            max_steps = 2**12,
            t0 = YT[0],
            t1 = YT[-1],
            y0 = y0,
            dt0 =  YT[1]-YT[0],
            stepsize_controller = stepsize_controller,
            saveat = dfx.SaveAt(ts=YT),
            args=(control,key,yi)
        )
        return solution
    
    def __call__(self, YT, y0, yi, UT, uc, DS, key):
        """
        This is just a wrapper for get_solution to
        return only the state variable values.
        """
        solution = self.get_solution(YT, y0, yi, UT, uc, DS, key)
        return solution.ys

# --- MODEL TRAINING FUNCTIONS --- #    

@eqx.filter_jit
@eqx.filter_value_and_grad
def grad_loss(model, data_, key):
    # yy, YT, uc, UT, DS, yi = data_
    losses = separate_losses(model, data_, key)
    return jnp.mean(jnp.array(losses))

@eqx.filter_jit
def parse_data_to_model(model,data_,key):
    yy, YT, uc, UT, DS, yi = data_
    # only use y0 for prediction
    y0 = yy[:,0,:] # stat vars at YT[0]
    yi = yi[:,0,:] # initial state variables at t=0
    # times have to be 1-dimensional only
    YT = YT[:,:,0]
    UT = UT[:,:,0]
    # discrete steps
    DS = DS[:,:,0]

    dat2mod = YT, y0, yi, UT, uc, DS
    in_axes = (0, 0, 0, 0, 0, 0, None)
    y_pred = jax.vmap(model, in_axes=in_axes)(*dat2mod, key)
    y_true = data_[0]  # yy, the true values
    return y_true, y_pred

@eqx.filter_jit
def separate_losses(model, data_, key):
    y_true, y_pred = parse_data_to_model(model, data_, key)
    
    # LOSS of G, P, X, V
    l_G = UF.MAE(y_true[:,:,0],y_pred[:,:,0])
    l_P = UF.MAE(y_true[:,:,1],y_pred[:,:,1])
    l_X = UF.MAE(y_true[:,:,2],y_pred[:,:,2])
    l_V = UF.MAE(y_true[:,:,3],y_pred[:,:,3])
    
    # NEGATIVITY LOSS
    # negativity violations of scaled state variables
    scaled_zeros = model.HYB.SCL_yy.scale(0)
    neg_vio_1 = jax.nn.relu(scaled_zeros - y_pred[:,:,:model.HYB.NR_state_variables])
    l_N1 = UF.MAE(jnp.zeros_like(neg_vio_1), neg_vio_1)
    # negativity violations of scaled extra metabolites
    neg_vio_2 = jax.nn.relu(0 - y_pred[:,:,model.HYB.NR_state_variables:])
    l_N2 = UF.MAE(jnp.zeros_like(neg_vio_2), neg_vio_2)
    l_N = l_N1 + l_N2
    return l_G, l_P, l_X, l_V, l_N

@eqx.filter_jit
def get_metrics(model, data_, key):
    """
    Returns metrics that can be parsed to HIS.
    """

    y_true, y_pred = parse_data_to_model(model, data_, key)

    y_pred = y_pred.reshape(-1,y_pred.shape[-1])
    y_true = y_true.reshape(-1,y_true.shape[-1])
    # R2   = UF.R2(y_true.flatten(),y_pred[:,:model.HYB.NR_state_variables].flatten())
    R2_G = UF.R2(y_true[:,0],y_pred[:,0])
    R2_P = UF.R2(y_true[:,1],y_pred[:,1])
    R2_X = UF.R2(y_true[:,2],y_pred[:,2])
    R2_V = UF.R2(y_true[:,3],y_pred[:,3])
    R2_PX = jnp.mean(jnp.array([R2_P,R2_X]))
    return R2_PX, R2_G, R2_P, R2_X, R2_V

@eqx.filter_jit
def evaluate(model, data_, key):
    """
    Calculates and returns loss and metrics.
    """
    
    l_G, l_P, l_X, l_V, l_neg = separate_losses(model, data_, key)
    loss = jnp.mean(jnp.array([l_G, l_P, l_X, l_neg]))
    R2_PX, R2_G, R2_P, R2_X, R2_V = get_metrics(model, data_, key)
    return loss, l_G, l_P, l_X, l_V, l_neg, R2_PX, R2_G, R2_P, R2_X, R2_V

def evaluate_write_history(model,train_,valid_,key,i,history,header=False):
    """
    Calculates loss and metrics, updates history.
    """
    # as the model is in inference mode the key will not be use,
    # so we can use the same key.
    history.update_valid(i,evaluate(model, valid_, key))
    history.update_train(i,evaluate(model, train_, key))
    history.writeout(header=header)
    return history

@eqx.filter_jit # eqx.nn.Dropout requires filter_jit
def make_step(data_, model, opt_state, key, optimizer):
    loss, grads = grad_loss(model, data_, key)
    parameters = eqx.filter(model,eqx.is_inexact_array)
    updates, opt_state = optimizer.update(grads, opt_state, parameters)
    # jax.debug.print("GRADS {x} UPDATES {y} 🤯", x=optax.global_norm(grads), y=optax.global_norm(updates))
    model = eqx.apply_updates(model, updates)

    # UPDATE SCALERS
    # 1. get intermediate values
    YT, yy = AF.parse_data_to_model(model, data_, key, 100)
    data_ = AF.inject_predictions_into_data(data_, yy, YT)
    intermediates = jax.vmap(AF.NEW_get_rates_per_process, in_axes=(None, 0, None))(model, data_, key)
    # 2. update scaler
    mod_ann_inp_raw = intermediates["mod_ann_inp_raw"].reshape(-1, intermediates["mod_ann_inp_raw"].shape[-1])
    new_SCL_ann = model.HYB.SCL_mod_ann_inp.update(mod_ann_inp_raw)
    # 3. replace scaler in model
    model = eqx.tree_at(lambda m: m.HYB.SCL_mod_ann_inp, model, new_SCL_ann)

    # jax.debug.print("SCALED {x} RAW {y} 🤯", 
    #                 x=optax.global_norm(intermediates["mod_ann_inp"]), 
    #                 y=optax.global_norm(intermediates["mod_ann_inp_raw"]))
    return loss, model, opt_state

@eqx.filter_jit # eqx.nn.Dropout requires filter_jit
def batching(k1,j,n):
    """
    Shuffels and returns fed-batches batched together.

    k1 ... key
    j  ... nr of fed-batches per training batch
    n  ... nr of fed-batch processes
    """
    r = jax.random.permutation(k1,jnp.arange(n))
    r = r[:r.shape[0]//j*j]
    r = r.reshape(-1,j)
    return r

def epoch_step(data_, model, opt_state, key, optimizer, processes_per_batch, total_processes):
    """
    Batches training data and does make_step per batch.
    """
    yy, YT, uc, UT, DS, yi = data_
    key1, key2 = jax.random.split(key,2)
    r = batching(key1,processes_per_batch,total_processes)
    for i in r:
        # _ = yy[i,:,:], YT[i,:,:], uc[i,:,:,:], UT[i,:,:], DS[i,:,:], yi[i,:,:]
        _ = [__[i] for __ in data_]
        loss, model, opt_state = make_step(_, model, opt_state, key2, optimizer)
    # loss, model, opt_state = make_step(data_, model, opt_state, key, optimizer)

    return loss, model, opt_state

def train(model, train_, valid_, train_kwargs):

    # generate training key
    key1 = jax.random.PRNGKey(train_kwargs["key_value"])

    # init optimizer
    parameters = eqx.filter(model,eqx.is_inexact_array)
    trainable_mask = MF.build_trainable_mask(parameters)
    processes_per_batch = train_kwargs["processes_per_batch"]
    # Gradient clipping is necessary otherwise they can explode,
    # which will make the integrator fail.
    optimizer = optax.chain(
        optax.adamw(learning_rate=train_kwargs["learning_rate"],
                    weight_decay=1e-3),
        train_kwargs["clip_by"],
        optax.transforms.freeze(trainable_mask),
    )
    opt_state = optimizer.init(parameters)

    # init total_processes variable used in batching(*)
    total_processes = train_[3].shape[0]

    # init evaluation objects
    n_head = round(np.max([train_kwargs["n_writeouts"]//5,1]))
    n_prnt = round(np.max([train_kwargs["n_epochs"]//train_kwargs["n_writeouts"],1]))
    metric_names = DF.HIS_get_return_variable_names(evaluate)
    history = DF.HIS(metric_names,precision=4)

    # init early stopping objects
    best_valid_loss = jnp.inf
    best_valid_epoch = 0
    best_valid_model = model
    cur_patience = 0

    # make sure the model is in trainig mode
    model = eqx.nn.inference_mode(model, value=False)

    for i in range(train_kwargs["n_epochs"]):
        key1, key2 = jax.random.split(key1,2)
        _, model, opt_state = epoch_step(train_, model, opt_state, key2, optimizer, processes_per_batch, total_processes)

        if i == 0 or i % n_prnt == 0:
                # wether to print a header
                print_header = (i==0) or i%(train_kwargs["n_epochs"]//(n_head)) == 0

                # inference
                inference_model = eqx.nn.inference_mode(model, value=True)
                key1, key2 = jax.random.split(key1,2)
                history = evaluate_write_history(inference_model,train_,valid_,key2, # params regarding model
                                                 i,history,print_header) # params regarding history

        # early stopping logic
        if train_kwargs["early_stopping"] and i > train_kwargs["disregard_first"]:
            inference_model = eqx.nn.inference_mode(model, value=True)
            _ = evaluate(inference_model,valid_,key2)
            # _ = loss, l_G, l_P, l_X, l_V, l_neg, R2_PX, R2_G, R2_P, R2_X, R2_V
            tmp_valid_loss = jnp.mean(jnp.array([_[2], _[3]])) # l_P, l_X
            if tmp_valid_loss > best_valid_loss*train_kwargs["grace_region"]:
                cur_patience += 1
            elif tmp_valid_loss < best_valid_loss:
                best_valid_loss = tmp_valid_loss
                best_valid_epoch = i
                best_valid_model = model
                cur_patience = 0
            if cur_patience == train_kwargs["max_patience"]:
                print(history.hline)
                print(f"EARLY STOPPING. No validation loss improvment for {train_kwargs['max_patience']} epochs. (best validation loss = {best_valid_loss:.4f} at epoch {best_valid_epoch})")
                print(history.hline)
                break
    inference_model = eqx.nn.inference_mode(model, value=True)
    history = evaluate_write_history(inference_model,train_,valid_,key2,i,history,False)
    print(history.hline)
    if train_kwargs["early_stopping"]:
        print("BEST VALID MODEL")
        inference_model = eqx.nn.inference_mode(best_valid_model, value=True)
        _ = evaluate_write_history(inference_model,train_,valid_,key2,best_valid_epoch,history,False)
        print(history.hline)
    return model, best_valid_model, history

def get_predictions(model,data_,key):
    y_true, y_pred = parse_data_to_model(model, data_, key)
    y_pred = y_pred[:,:,:model.HYB.NR_state_variables]

    Y_TRUE = model.HYB.SCL_yy.unscale(y_true)
    Y_PRED = model.HYB.SCL_yy.unscale(y_pred)
    return y_true, y_pred, Y_TRUE, Y_PRED

def run_model(ALL_data, ALL_kwargs):
    ODE_kwargs, TRAIN_kwargs = ALL_kwargs
    train_, valid_ = ALL_data
    # load data and init model
    model = ODE(**ODE_kwargs)

    # train model
    model, best_valid_model, history = train(model, train_, valid_, TRAIN_kwargs)

    key = jax.random.PRNGKey(13)
    best_pred = get_predictions(eqx.nn.inference_mode(best_valid_model), valid_, key)
    last_pred = get_predictions(eqx.nn.inference_mode(model           ), valid_, key)
    
    return model, best_valid_model, history, best_pred, last_pred

# --- CONFIGURATION OF HYBRID MODEL --- #

def NO_modify_ann_inp(HYB, y, Y, u, U, yi, YI, key=None):
    """
    returns [y,yi,u]
    """

    mod_ann_inp = jnp.concatenate([y,yi,u])
    return mod_ann_inp

def get_model_config(verbose=True):
    """
    Here all the important stuff of the model is defined.
    """
    # "NNfba", "SRfba", "NOfba"
    fba_type = 'NOfba'
    # "linear" or "ann"
    base_flux_type = "ann"
    # "none", "sigmoid", "min"
    LIM_handler_type = "none"
    # "none", "scaled", "unscaled"
    modify_ann_inp = "none"
    # True or false
    extra_metabolites = False

    if verbose:
        print("#-------------------- CONFIG --------------------#")
        print(f"fba_type          : {fba_type}")
        print(f"base_flux_type    : {base_flux_type}")
        print(f"LIM_handler_type  : {LIM_handler_type}")
        print(f"modify_ann_inp    : {modify_ann_inp}")
        print(f"extra_metabolites : {extra_metabolites}")
        print("#------------------------------------------------#")
    

    key = jax.random.PRNGKey(13)
    key_fba_obj, key_qG_flux, key_base_fd = jax.random.split(key,3)

    if modify_ann_inp == "none":
        MOD_ann_inp = NO_modify_ann_inp
        ANN_in_size = 12
    else:
        raise NotImplementedError(f"Unknown modify_ann_inp: {modify_ann_inp}")

    if fba_type == "SRfba":
        # initialize SR FBA model
        FBA_model = MF.SRfba()

        # define relevant fluxes (i.e., measured state variable fluxes)
        COL_q_rel = ["EX_glyc_e_i","protein_synthesis","BIOMASS_Ec_iJO1366_core_53p95M"]
        IDX_q_rel = HF.get_idx_from_col(COL_q_rel,FBA_model.COL_out)
        if extra_metabolites:
            COL_v_ext = ["EX_co2_e_o","EX_nh4_e_i","EX_o2_e_i","EX_so4_e_i"]
        else:
            COL_v_ext = []
        IDX_v_ext = HF.get_idx_from_col(COL_v_ext,FBA_model.COL_out)
        NR_extra_variables = len(IDX_v_ext)

        # output dimension of the ANN_fba_obj
        D_out = 3
    elif fba_type == "NOfba":
        # initialize empty FBA model
        COL_q_rel = ["EX_glyc_e_i","protein_synthesis","BIOMASS_Ec_iJO1366_core_53p95M"]
        IDX_q_rel = jnp.array([0,1,2]).astype(int) # just everything
        SCL_fba_out = DF.SCL_init_from_fba(COL_q_rel,method="mean")
        FBA_model = MF.NOfba(SCL_fba_out)
        COL_v_ext = [] #["EX_co2_e_o","EX_nh4_e_i","EX_o2_e_i","EX_so4_e_i"]
        IDX_v_ext = HF.get_idx_from_col(COL_v_ext,FBA_model.COL_out)
        NR_extra_variables = len(IDX_v_ext)

        # output dimension of the ANN_fba_obj
        D_out = 2
    else:
        raise NotImplementedError(f"Unknown fba_type: {fba_type}")
        
    if   base_flux_type == "linear":
        ANN_base_fd = MF.Bf_from_qX(jnp.array([.001]))
    elif base_flux_type == "ann":
        ANN_base_fd = eqx.nn.MLP(
            in_size    = ANN_in_size + 3 + NR_extra_variables,
            out_size   = 1,
            width_size = 8,
            depth      = 2,
            key        = key_base_fd,
            activation       = jax.nn.softplus,
            final_activation = UF.bounded_softplus,
        )
        ANN_base_fd = MF.INIT_xavier(ANN_base_fd, key_base_fd)
    else:
        raise NotImplementedError(f"Unknown base_flux_type: {base_flux_type}")

    if   LIM_handler_type == "none":
        LIM_handler = MF.G_LIM_none
    elif LIM_handler_type == "sigmoid":
        LIM_handler = MF.G_LIM_sigmoid
    elif LIM_handler_type == "min":
        LIM_handler = MF.G_LIM_min
    else:
        raise NotImplementedError(f"Unknown LIM_handler_type: {LIM_handler_type}")

    ANN_fba_obj = eqx.nn.MLP(
        in_size    = ANN_in_size + NR_extra_variables,
        out_size   = D_out,
        width_size = 16,
        depth      = 2,
        key        = key_fba_obj,
        activation       = jax.nn.softplus,
        final_activation = UF.bounded_softplus,
    )
    ANN_fba_obj = MF.INIT_xavier(ANN_fba_obj, key_fba_obj)

    ANN_qG_flux = eqx.nn.MLP(
        in_size    = ANN_in_size + NR_extra_variables,
        out_size   = 1,
        width_size = 8,
        depth      = 2,
        key        = key_qG_flux,
        activation       = jax.nn.softplus,
        final_activation = UF.bounded_softplus,
    )
    ANN_qG_flux = MF.INIT_xavier(ANN_qG_flux, key_qG_flux)

    scalers = HF.get_process_scalers(method="mean")
    SCL_off, SCL_onl = scalers
    HYB_kwargs = {
        "SCL_yy"    : SCL_off,   # Scaler of offline data.
        "SCL_uu"    : SCL_onl,   # Scaler of online data.
        "IDX_q_rel" : IDX_q_rel, # Index of relevant fluxes
        "IDX_v_ext" : IDX_v_ext, # Index of extra fluxes
        "FBA_model" : FBA_model,
        # molar masses G, P, X in g/mol
        # X is normalized to 1000 g/mol = 1 g/mmol 
        "MOL_masses": jnp.array([92.09382, 41949.43808, 1000.]), # g/mol 
        "ANN_fba_obj" : ANN_fba_obj,
        "ANN_qG_flux" : ANN_qG_flux,
        "ANN_base_fd" : ANN_base_fd, # ANN to predict base feed rate
        "LIM_handler" : LIM_handler, # how to handle G-limitation
        "MOD_ann_inp" : MOD_ann_inp, # functino that handles the ANN input modification
        "verbose"       : False,
                 }
    
    ODE_kwargs = {
        "verbose":  True,
        "HYB_kwargs": HYB_kwargs
                 }
    
    TRAIN_kwargs = {
            "n_epochs"            : 5000,
            "n_writeouts"         : 500,
            "key_value"           : 13,
            "learning_rate"       : 1e-3,
            "processes_per_batch" : 10,
            # EARLY STOPPING HYPERPARAMS
            "early_stopping"  : True,
            "disregard_first" : 100, # disregard first k epochs
            "max_patience"    : 50,  # stop, if there is no valid loss improvement for max_patience epochs
            "grace_region"    : 1.1, # don't count epochs where valid loss is within grace_region of best valid loss
            # GRADIENT CLIPPING HYPERPARAMS
            # "clip_by" : optax.clip(.1),
            "clip_by" : optax.clip_by_global_norm(10.),
        }
    
    ALL_kwargs = ODE_kwargs, TRAIN_kwargs
    return scalers, ALL_kwargs

def get_exp_dfs():
    exp_onl = pd.read_csv(str(REPO_ROOT / "prol_process_data" / "250804_exp_onl.csv"),index_col=0)
    exp_off = pd.read_csv(str(REPO_ROOT / "prol_process_data" / "250804_exp_off.csv"),index_col=0)
    exp_dsp = pd.read_csv(str(REPO_ROOT / "prol_process_data" / "250804_exp_dsp.csv"),index_col=0)

    return exp_off,exp_onl,exp_dsp

def get_aug_dfs():
    aug_onl = pd.read_csv(str(REPO_ROOT / "prol_process_data" / "250813_aug_onl.csv"),index_col=0)
    aug_off = pd.read_csv(str(REPO_ROOT / "prol_process_data" / "250813_aug_off.csv"),index_col=0)
    aug_dsp = pd.read_csv(str(REPO_ROOT / "prol_process_data" / "250813_aug_dsp.csv"),index_col=0)

    return aug_off,aug_onl,aug_dsp

def preprocess_data(exp_dfs,aug_dfs,valid_set,scalers,augmented=True,verbose=True):
    if augmented:
        if verbose:
            print("TRAINING WITH AUGUMENTED DATA")
        train_ = DF.PRE_process_data(aug_dfs,scalers,
                                    valid_set,
                                    inverse=False,
                                    yi_method='strip',
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

# --- MAIN CROSS VALIDATION LOOP --- #
if __name__ == "__main__":
    # cross validation sets
    cross_valid_sets = HF.get_valid_sets()

    # generate path and dir
    save_path = str(REPO_ROOT / "prol_model_data" / Path(__file__).stem) + "/"
    os.makedirs(save_path,exist_ok=True)

    # load data
    exp_dataframes = get_exp_dfs()
    aug_dataframes = get_aug_dfs()

    # get model config
    scalers, ALL_kwargs = get_model_config()

    SC_y_true = [] # scaled
    US_y_true = [] # unscaled
    SC_best_preds = [] # scaled
    US_best_preds = [] # unscaled
    SC_last_preds = [] # scaled
    US_last_preds = [] # unscaled
    tested_names = []

    for nr, valid_set in enumerate(cross_valid_sets):
        set_name = f"set_{nr+1:00.0f}"
        print("VALID SET:",valid_set,"SET NAME:", set_name)

        ALL_data = preprocess_data(exp_dataframes,aug_dataframes,valid_set,scalers,augmented=True)

        # init and train model
        _ = run_model(ALL_data, ALL_kwargs)
        model, best_valid_model, history, best_pred, last_pred = _
        
        # save model & history
        eqx.tree_serialise_leaves(save_path+set_name+"_last.eqx",model)
        eqx.tree_serialise_leaves(save_path+set_name+"_best.eqx",best_valid_model)
        history.sort()
        history.save(save_path+set_name+".history")

        # keep track of predictions & metrics over the LOO CV
        SC_y_true.append(best_pred[0])
        US_y_true.append(best_pred[2])
        SC_best_preds.append(best_pred[1])
        US_best_preds.append(best_pred[3])
        SC_last_preds.append(last_pred[1])
        US_last_preds.append(last_pred[3])
        tested_names.append(set_name)

        HF.print_and_write_CV_stats(SC_y_true,SC_best_preds,SC_last_preds,tested_names,save_path,id="_sc",verbose=False)
        HF.print_and_write_CV_stats(US_y_true,US_best_preds,US_last_preds,tested_names,save_path,id="_us",verbose=True)

        del model, best_valid_model, history, best_pred, last_pred
        # break

    print("DONE WITH CV")