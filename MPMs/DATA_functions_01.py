from pathlib import Path
import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax as dfx
import pandas as pd
import pickle

import MPMs.MPM_functions_07 as MF
import MPMs.UTIL_functions_01 as UF

REPO_ROOT = Path(__file__).resolve().parents[1]


# ----------------- SCALERS ----------------- #

class SCL_ADAPTIVE_mean(eqx.Module):
    avg: jax.Array
    count: jax.Array
    eps: float = 1e-8
    id_: str = "SCL.adaptive.mean"
    col: list

    def __init__(self, dim, col=[]):
        self.avg = jnp.ones(dim)
        self.count = jnp.array(0)
        self.col = [str(_) for _ in col]

    def scale(self, data):
        """
        Do mean scaling:
        x' = x / (mean + eps)
        """
        return data / (self.avg + self.eps)
    
    def scale_first(self, data):
        return data / (self.avg[:data.shape[-1]] + self.eps)

    def unscale(self, data):
        """
        Undo mean scaling:
        x = x' * (mean + eps)
        """
        return data * (self.avg + self.eps)
    
    def grad_scale(self, derivative):
        """
        Do mean scaling of derivative of init data:
        (dx/dt)' = dx/dt / (mean + eps)
        """
        return derivative / (self.avg + self.eps)
    
    def grad_unscale(self,derivative):
        """
        Undo mean scaling of derivative of init data:
        dx/dt = (dx/dt)' * (mean + eps)
        """
        return derivative * (self.avg + self.eps)
    
    def update(self, data, alpha=0.01):
        """
        Update scaler using new data.
        Uses JAX-compatible control flow (lax.cond) to avoid Python branching.
        """

        def no_update_fn(_):
            # jax.debug.print("Max count reached, no more updates.")
            return self

        def do_update_fn(_):
            data_arr = jnp.asarray(data)
            count_new = self.count + 1
            mean_batch = jnp.mean(data_arr, axis=0)
            avg_new = (1 - alpha) * self.avg + alpha * mean_batch

            return eqx.tree_at(
                lambda s: (s.avg, s.count),
                self,
                (avg_new, count_new),
            )

        return jax.lax.cond(self.count > 500, no_update_fn, do_update_fn, operand=None)
    
    # def update(self, data, alpha=0.01):
    #     """
    #     Update scaler using new data.
    #     """
    #     if self.count > 50_000:
    #         jax.debug.print("Max count reached, no more updates.")
    #         return self

    #     data = jnp.asarray(data)
    #     count_new = self.count + data.shape[0]
    #     mean_batch = jnp.mean(data, axis=0)
    #     avg_new = (1 - alpha) * self.avg + alpha * mean_batch
        
    #     # deviation = jnp.abs(self.avg - avg_batch) / (jnp.abs(self.avg) + self.eps)
    #     # alpha = jnp.ones(self.avg.shape) * alpha
    #     # jax.debug.print("DEV  {x}\nALPH {y}\nAVG  {z}", 
    #     #             x=jnp.array(deviation), 
    #     #             y=jnp.array(alpha),
    #     #             z=jnp.array(avg_new)
    #     #             )

    #     return eqx.tree_at(lambda s: (s.avg, s.count), 
    #                        self, 
    #                        (avg_new, count_new))

class SCL_ADAPTIVE_standard(eqx.Module):
    avg: jax.Array
    std: jax.Array
    count: jax.Array
    eps: float = 1e-8
    id_: str = "SCL.adaptive.standard"
    col: list

    def __init__(self, dim, col=[]):
        self.avg = jnp.zeros(dim)
        self.std = jnp.ones(dim)
        self.count = jnp.array(0)
        self.col = [str(_) for _ in col]

    def scale(self, data):
        return (data - self.avg) / (self.std + self.eps)

    def unscale(self, data):
        return data * (self.std + self.eps) + self.avg
    
    def grad_scale(self, derivative):
        return derivative / (self.std + self.eps)
    
    def grad_unscale(self, derivative):
        return derivative * (self.std + self.eps)
    
    def update(self, data, alpha=0.01):
        """
        Update scaler using new data.
        Uses JAX-compatible control flow (lax.cond) to avoid Python branching.
        """

        def no_update_fn(_):
            # jax.debug.print("Max count reached, no more updates.")
            return self

        def do_update_fn(_):
            data_arr = jnp.asarray(data)

            # Increment count (could also be + data_arr.shape[0] if you want batch-based)
            count_new = self.count + 1

            # Batch statistics
            avg_batch = jnp.mean(data_arr, axis=0)
            std_batch = jnp.std(data_arr, axis=0)

            # Exponential moving average of mean and variance
            avg_new = (1 - alpha) * self.avg + alpha * avg_batch
            var_new = (1 - alpha) * (self.std**2 + (self.avg - avg_new)**2) + alpha * std_batch**2
            std_new = jnp.sqrt(var_new + 1e-8)  # small epsilon for numerical safety

            return eqx.tree_at(
                lambda s: (s.avg, s.std, s.count),
                self,
                (avg_new, std_new, count_new),
            )

        # Only update the first 500 calls. After that, keep the scaler fixed.
        return jax.lax.cond(self.count > 500, no_update_fn, do_update_fn, operand=None)

    # def update(self, data, alpha=0.01):
    #     """
    #     Update scaler using new data.
    #     """

    #     data = jnp.asarray(data)
    #     count_new = self.count + data.shape[0]
    #     avg_batch = jnp.mean(data, axis=0)
    #     avg_new = (1 - alpha) * self.avg + alpha * avg_batch

    #     std_batch = jnp.std(data, axis=0)
    #     var_new = (1 - alpha) * (self.std**2 + (self.avg - avg_new)**2) + alpha * std_batch**2
    #     std_new = jnp.sqrt(var_new)

    #     # deviation = jnp.abs(self.avg - avg_batch) / (jnp.abs(self.avg) + self.eps)
    #     # alpha = jnp.ones(self.avg.shape) * alpha
    #     # jax.debug.print("DEV  {x}\nALPH {y}\nAVG  {z}", 
    #     #             x=jnp.array(deviation), 
    #     #             y=jnp.array(alpha),
    #     #             z=jnp.array(avg_new)
    #     #             )

    #     return eqx.tree_at(
    #         lambda s: (s.avg, s.std, s.count),
    #         self,
    #         (avg_new, std_new, count_new),
    #     )

class SCL_standard(eqx.Module):
    avg: jax.Array = MF.frozen_field()
    std: jax.Array = MF.frozen_field()
    eps: float = 1e-8
    id_: str = "SCL.standard"
    col: list

    def __init__(self, data, col=[]):
        """
        My  standard scaler.
        """

        data = jnp.asarray(data)
        self.avg = jnp.mean(data, axis=0)
        self.std = jnp.std(data, axis=0)
        # ensure that col is a list of strings
        self.col = [str(_) for _ in col]

    def scale(self, data):
        """
        Do standard scaling:
        x' = (x - mean) / (std + eps)
        """
        return (data - self.avg) / (self.std + self.eps)

    def unscale(self, data):
        """
        Undo standard scaling:
        x = x' * (std + eps) + mean
        """
        return data * (self.std + self.eps) + self.avg
    
    def grad_scale(self, derivative):
        """
        Do standard scaling of derivative of init data:
        (dx/dt)' = dx/dt / (std + eps)
        """
        return derivative / (self.std + self.eps)
    
    def grad_unscale(self,derivative):
        """
        Undo standard scaling of derivative of init data:
        dx/dt = (dx/dt)' * (std + eps)
        """
        return derivative * (self.std + self.eps)

class SCL_mean(eqx.Module):
    avg: jax.Array = MF.frozen_field()
    eps: float = 1e-8
    id_: str = "SCL.mean"
    col: list

    def __init__(self, data, col=[]):
        """
        My mean scaler.
        """

        data = jnp.asarray(data)
        self.avg = jnp.mean(data, axis=0)
        # ensure that col is a list of strings
        self.col = [str(_) for _ in col]

    def scale(self, data):
        """
        Do mean scaling:
        x' = x / (mean + eps)
        """
        return data / (self.avg + self.eps)

    def unscale(self, data):
        """
        Undo mean scaling:
        x = x' * (mean + eps)
        """
        return data * (self.avg + self.eps)
    
    def grad_scale(self, derivative):
        """
        Do mean scaling of derivative of init data:
        (dx/dt)' = dx/dt / (mean + eps)
        """
        return derivative / (self.avg + self.eps)
    
    def grad_unscale(self,derivative):
        """
        Undo mean scaling of derivative of init data:
        dx/dt = (dx/dt)' * (mean + eps)
        """
        return derivative * (self.avg + self.eps)

class SCL_none(eqx.Module):
    eps: float = 1e-8
    id_: str = "SCL.none"
    col: list
    dim: int

    def __init__(self,dim=None,col=[]):
        """
        A scaler object that does not scale.
        """
        # ensure that col is a list of strings
        self.col = [str(_) for _ in col]
        self.dim = dim

    def scale(self, data):
        """
        """
        return data 

    def unscale(self, data):
        """
        """
        return data
    
    def grad_scale(self, derivative):
        """
        """
        return derivative
    
    def grad_unscale(self,derivative):
        """
        """
        return derivative
    
    def update(self, data, alpha=0.01):
        """
        No scaling, no update.
        """
        return self

class SCL_logp1(eqx.Module):
    dim: int
    eps: float = 0.
    id_: str = "SCL.logp1"
    col: list

    def __init__(self, dim=None, col=[]):
        """
        My log + eps  scaler.
        eps = 1.
        """

        self.dim = dim
        # ensure that col is a list of strings
        self.col = [str(_) for _ in col]

    def scale(self, data):
        """
        Do log + 1 scaling:
        x' = log(x + 1)
        """
        return jnp.log(data + self.eps)

    def unscale(self, data):
        """
        Undo logp1 scaling:
        x = e ** x' - 1
        """
        return jnp.exp(data) - self.eps
    
    def grad_scale(self, derivative):
        raise NotImplementedError("Requires derivative and primitive. This is currently not supported.")
        # derivative_scaled = derivative_unscaled/(primitive_unscaled+1)
    
    def grad_unscale(self,derivative):
        raise NotImplementedError("Requires derivative and primitive. This is currently not supported.")
        # derivative_unscaled = derivative_scaled*(jnp.exp(primitive_scaled))

    def update(self, data, alpha=0.01):
        """
        No update.
        """
        return self

def SCL_init_from_fba(reactions,method):
    scaling_data = pd.read_csv(str(REPO_ROOT / "prol_process_data" / "fba_avg_std.csv"),index_col=0)
    v = scaling_data.loc[:,reactions].values
    v = jnp.asarray(v)
    reactions = [str(_) for _ in reactions]

    if method == "standard":
        return SCL_standard(v,reactions)
    elif method == "mean":
        return SCL_mean(v,reactions)
    elif method == "logp1":
        return SCL_logp1(v,reactions)
    else:
        raise NotImplementedError
    
def SCL_init_from_slim_fba(reactions,method):
    scaling_data = pd.read_csv(str(REPO_ROOT / "slim_fba_data" / "fba_avg_std_01_reduced.csv"),index_col=0)
    v = scaling_data.loc[:,reactions].values
    v = jnp.asarray(v)
    reactions = [str(_) for _ in reactions]

    if method == "standard":
        return SCL_standard(v,reactions)
    elif method == "mean":
        return SCL_mean(v,reactions)
    elif method == "logp1":
        return SCL_logp1(v,reactions)
    else:
        raise NotImplementedError

# ----------------- PREPROCESSING ----------------- #

def PRE_get_process(data,pname):
        loc = data["process"] == pname
        return data.loc[loc,:]

def PRE_find_filter_or_val(name,filter,inverse):
        """
        If inverse = True only the processes in filter are returned.
        VAL processes are never returned.
        """
        if inverse:
            return True if name.split("_aug_")[0] in filter else False
        else:
            return False if name.split("_aug_")[0] in filter or "VAL" in name else True
        
def PRE_filter_processes(dataframes,filter_processes,inverse,verbose):
    # --- FILTER PROCESSES --- #
    # Here i filter out all processes (and augmented processes derived) that are listed in filter_processes.
    # If inverse is True, only the processes in filter_processes are kept.
    import numpy as np

    off, onl, dsp = dataframes
    filter_onl = np.array([PRE_find_filter_or_val(name,filter_processes,inverse) for name in onl.process.values])
    filter_off = np.array([PRE_find_filter_or_val(name,filter_processes,inverse) for name in off.process.values])
    filter_dsp = np.array([PRE_find_filter_or_val(name,filter_processes,inverse) for name in dsp.process.values])
    tmp_off = off.loc[filter_off,:]
    tmp_onl = onl.loc[filter_onl,:]
    tmp_dsp = dsp.loc[filter_dsp,:]

    if verbose:
        print(f"off: {tmp_off.shape}, onl: {tmp_onl.shape}, dsp: {tmp_dsp.shape}")
    return tmp_off, tmp_onl, tmp_dsp

def PRE_select_columns(dataframes,verbose,
                       Ycols=["G","P","X","V_N"],
                       Ucols=["t","T","f_N","G_f"]
                       ):
    # --- SELECT COLUMNS --- #
    # here I use upper case to indicate UNSCALED values 
    # and lower case to indicate scaled values

    # Currently the the columns which are taken into account are hardcoded.
    # This may be smart to change in the future.
    # E.g., the scalers could have a list of columns for which they scale & 
    # they could be used for selecting the columns in the dataframes.

    tmp_off, tmp_onl, tmp_dsp = dataframes

    YY = jnp.array([PRE_get_process(tmp_off,name).loc[:,Ycols].values for name in tmp_off.process.unique()])
    YT = jnp.array([PRE_get_process(tmp_off,name).loc[:,["t"]              ].values for name in tmp_off.process.unique()])
        
    UU = jnp.array([PRE_get_process(tmp_onl,name).loc[:,Ucols].values for name in tmp_onl.process.unique()])
    UT = jnp.array([PRE_get_process(tmp_onl,name).loc[:,["t"]                ].values for name in tmp_onl.process.unique()])

    DS = jnp.array([PRE_get_process(tmp_dsp,name).loc[:,["discrete_steps"]].values for name in tmp_dsp.process.unique()])

    if verbose:
        print("YY:",YY.shape,"YT:",YT.shape,"UU:",UU.shape,"UT:",UT.shape,"DS:",DS.shape)

    return YY, YT, UU, UT, DS

def PRE_scale_data(YY,UU,scalers,verbose):
    SCL_off, SCL_onl = scalers

    # --- SCALE THE DATA --- #
    yy = SCL_off.scale(YY)
    uu = SCL_onl.scale(UU)
    if verbose:
        print("yy:",yy.shape,"uu:",uu.shape)
    return yy, uu

def PRE_calculate_control_spline(uu, UT, verbose):
    # --- CALCULATE DFX SPLINE --- #
    uc = jax.vmap(dfx.backward_hermite_coefficients)(UT[:,:,0], uu)
    uc = jnp.array(uc).swapaxes(0,1)
    if verbose:
        print("uc:",uc.shape)
    return uc

def PRE_calculate_initial_values(yy, YT, yi_method):
    if yi_method == "copy":
        YT = YT
        yy = yy
        yi = yy[:,[0],:]
        IT = YT[:,[0],:]
    elif yi_method == "strip":
        yi = yy[:,[0],:]
        IT = YT[:,[0],:]
        YT = YT[:,1:,:]
        yy = yy[:,1:,:]
    else:
        raise NotImplementedError
    assert jnp.all(jnp.isclose(IT,0)), "Initial time values should be zero, but found non-zero values."
    return yi, YT, yy

def PRE_process_data(dataframes,
                     scalers,
                     filter_processes,
                     inverse,
                     yi_method,
                     Ycols=["G","P","X","V_N"],
                     Ucols=["t","T","f_N","G_f"],
                     verbose=True):
    """

    Args:
    -----
    dataframes : Tuple of pd.DataFrames
        (onl, off, dsp) 
    scalers : Tuple of scalers
        (SCL_off, SCL_onl)
    """
    
    if verbose:
        off, onl, dsp = dataframes
        print(f"off: {off.shape}, onl: {onl.shape}, dsp: {dsp.shape}")

    # remove processes (and derived agumented ones) that are in filter_processes
    filtered_dataframes = PRE_filter_processes(dataframes,filter_processes,inverse,verbose)
    # get jax.Arrays from pd.DataFrames
    YY, YT, UU, UT, DS = PRE_select_columns(filtered_dataframes,verbose,Ycols,Ucols)
    # scale input data
    yy, uu = PRE_scale_data(YY,UU,scalers,verbose)
    # calculate control spline
    uc = PRE_calculate_control_spline(uu, UT, verbose)
    # separate intitial values from y-data
    yi, YT, yy = PRE_calculate_initial_values(yy, YT, yi_method)
    
    return yy, YT, uc, UT, DS, yi

def PRE_process_data_SLIM_ALL(dataframes,
                     scalers,
                     filter_processes,
                     inverse,
                     yi_method,
                     Ycols=["G","P","X","V_N"],
                     Ucols=["t","T","f_N","G_f"],
                     verbose=True):
    """

    Args:
    -----
    dataframes : Tuple of pd.DataFrames
        (onl, off, dsp) 
    scalers : Tuple of scalers
        (SCL_off, SCL_onl)
    """
    
    if verbose:
        off, onl, dsp = dataframes
        print(f"off: {off.shape}, onl: {onl.shape}, dsp: {dsp.shape}")

    # remove processes (and derived agumented ones) that are in filter_processes
    filtered_dataframes = PRE_filter_processes(dataframes,filter_processes,inverse,verbose)
    # get jax.Arrays from pd.DataFrames
    YY, YT, UU, UT, DS = PRE_select_columns(filtered_dataframes,verbose,Ycols,Ucols)
    # scale input data
    yy, uu = PRE_scale_data(YY,UU,scalers,verbose)
    # calculate control spline
    uc = PRE_calculate_control_spline(uu, UT, verbose)
    # separate intitial values from y-data
    yi, YT, yy = PRE_calculate_initial_values(yy, YT, yi_method)
    
    # add S0 values to yi

    lookup = get_SLIM_lookup()

    S0s = []
    # for process_A in aug_dataframes[0].process.unique():
    for process_A in filtered_dataframes[0].process.unique():
        for process_B in lookup.keys():
            if process_B in process_A:
                # print(process_A, process_B)
                S0s.append(lookup[process_B])
    S0s = jnp.array(S0s)
    # fill the other 3 rates with 0 initial vals
    fill_with_0 = jnp.zeros((S0s.shape[0], 1, 3))
    S0s_reshape = S0s.reshape(S0s.shape[0], 1, 1)
    yi = jnp.concatenate([yi, fill_with_0, S0s_reshape], axis=2)

    if verbose:
        print("yi:",yi.shape)

    return yy, YT, uc, UT, DS, yi

# ----------------- TRAINING HISTORY ----------------- #

class HIS():
    writeout_len = None
    precision = int

    def __init__(self,metric_names,precision=4):
        """
        My  history class.

        metrics ... ordered list of metric names
        """

        self.metric_names = metric_names
        self.n = len(self.metric_names)
        
        self.train = jnp.empty((0,self.n))
        self.valid = jnp.empty((0,self.n))

        self.train_epoch = jnp.empty((0))
        self.valid_epoch = jnp.empty((0))
        
        self.precision = precision

        pass

    def update_train(self,epoch,metric_values):
        """
        Update train metric array. 
        metric_values must have same length as metric_names.
        """
        self.train = jnp.vstack([self.train,jnp.array(metric_values,dtype=jnp.float32)])
        self.train_epoch = jnp.append(self.train_epoch,epoch)

    def update_valid(self,epoch,metric_values):
        """
        Update validation metric array. 
        metric_values must have same length as metric_names.
        """
        self.valid = jnp.vstack([self.valid,jnp.array(metric_values,dtype=jnp.float32)])
        self.valid_epoch = jnp.append(self.valid_epoch,epoch)

    def writeout(self,verbose=True,header=False):
        """
        Write out tracked metrics. If header=True, a table header will be printed as well.
        """
        epoch = self.train_epoch[-1]
        train_metrics = self.train[-1,:]
        valid_metrics = self.valid[-1,:]

        out_string = f"{epoch:6.0f} |"
        train_string = ""
        for i in range(self.n):
            train_string += f"| {train_metrics[i]:9.{self.precision}f}"[:11]+" "
        out_string += train_string+"|"
        valid_string = ""
        for i in range(self.n):
            valid_string += f"| {valid_metrics[i]:9.{self.precision}f}"[:11]+" "
        out_string += valid_string
        self.writeout_len = len(out_string)

        if header:
            hline = "-"*self.writeout_len
            self.hline = hline
            i = len(train_string[2:])
            i2 = i//2-2
            train_string = " "*i2+"train"+" "*i2
            valid_string = " "*i2+"valid"+" "*i2
            head1 = f"{'epoch':6} || {train_string[:i]}|| {valid_string[:i]}"

            head2_string = f"{'     ':6} |"
            for i in self.metric_names:
                head2_string += f"| {i:9} "
            head2_string += "|"
            for i in self.metric_names:
                head2_string += f"| {i:9} "
            print(hline)
            print(head1)
            print(head2_string)
            print(hline)
        if verbose:
            print(out_string)

    def save(self,filename):
        with open(filename,"wb") as file:
            pickle.dump(self.__dict__,file)

    def load(self,filename):
        with open(filename, "rb") as file:
            parameter_dict = pickle.load(file)
            for key, value in parameter_dict.items():
                setattr(self, key, value)

    def sort(self):
        """
        You can call HIS.update_*() out of epoch order. 
        Calling HIS.sort() ensures that the epoch order is restored, e.g., for plotting.
        """
        sorted_idx = jnp.argsort(self.train_epoch)
        self.train_epoch = self.train_epoch[sorted_idx]
        self.train = self.train[sorted_idx,:]

        sorted_idx = jnp.argsort(self.valid_epoch)
        self.valid_epoch = self.valid_epoch[sorted_idx]
        self.valid = self.valid[sorted_idx,:]

def HIS_get_return_variable_names(func):
    """
    Extract the names of variables in the return statement of a function.
    This is very helpful for auto-initializing HIS with the metrics function.
    """
    
    import ast
    import inspect

    source = inspect.getsource(func)  # Get the source code of the function
    tree = ast.parse(source)  # Parse the source code into an Abstract Syntax Tree (AST)
    for node in ast.walk(tree):
        if isinstance(node, ast.Return):
            if isinstance(node.value, ast.Tuple):  # If the return value is a tuple
                return [elt.id for elt in node.value.elts if isinstance(elt, ast.Name)]
            elif isinstance(node.value, ast.Name):  # Single return value
                return [node.value.id]
    return []

def get_SLIM_lookup():
    return {  "exp1" : 14.6060/.5,
                "exp2" : 7.7899/.5,
                "exp3" : 7.7899/.5,
                "exp4" : 7.7899/.5,
                "exp5" : 14.6060/.5,
                "exp6" : 14.6060/.5,
                "exp7" : 7.7899/.5,
                "exp8" : 7.7899/.5,
                "exp9" : 7.7899/.5,
                } # mmol/L





