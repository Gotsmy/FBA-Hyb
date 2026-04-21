import jax
import jax.numpy as jnp
import equinox as eqx
import MPMs.UTIL_functions_01 as UF
import importlib

importlib.reload(UF)


def frozen_field(**kwargs):
    """Convenience function to create a frozen field."""
    metadata = kwargs.get('metadata', {})
    metadata['frozen'] = True
    kwargs['metadata'] = metadata
    return eqx.field(**kwargs)

def trainable_field(**kwargs):
    """Convenience function to create a trainable field (default behavior)."""
    metadata = kwargs.get('metadata', {})
    metadata['frozen'] = False
    kwargs['metadata'] = metadata
    return eqx.field(**kwargs)

def build_trainable_mask(module, parent_frozen=False):
    """
    This function returns an eqx.Module that has the same structure as the input module,
    but with leaf values replaced by boolean values indicating if they are frozen.
    If a field is marked as frozen, it will be replaced with `True`, otherwise `False`.
    This mask can be parsed into optax.transforms.freeze() for optimization.
    """
    
    if not isinstance(module, eqx.Module):
        return parent_frozen
    
    def create_mask_value(value, effective_frozen):
        if isinstance(value, eqx.Module):
            return build_trainable_mask(value, parent_frozen=effective_frozen)
        elif isinstance(value, (list, tuple)):
            # raise NotImplementedError
            mask_list = [
                build_trainable_mask(v, parent_frozen=effective_frozen)
                if isinstance(v, eqx.Module) else effective_frozen
                for v in value
            ]
            return type(value)(mask_list)
        else:
            return bool(effective_frozen)
    
    updates = {}
    for name, field in module.__dataclass_fields__.items():
        value = getattr(module, name)
        # for debugging:
        # print(name,type(value),eqx.is_inexact_array(value))
        # I only want to apply the mask trainable parameters (i.e., jax.Arrays) or submodules (i.e., eqx.Modules).
        # However trainable parameters or submodules might be hidden in lists and tuples.
        if eqx.is_inexact_array(value) or isinstance(value,(eqx.Module, list, tuple)):
            field_frozen = field.metadata.get("frozen", False)
            effective_frozen = parent_frozen or field_frozen
            updates[name] = create_mask_value(value, effective_frozen)
    
    return eqx.tree_at(
        lambda m: tuple(getattr(m, k) for k in updates.keys()),
        module,
        tuple(updates[k] for k in updates.keys())
    )

def print_model_structure(module, prefix="", parent_frozen=False,
                          name_width=35, shape_width=14, status_width=20):
    """
    Print the complete structure of an eqx.Module with freeze status information.
    
    Args:
        module: The eqx.Module to analyze
        prefix: Prefix for field names (used internally for recursion)
        parent_frozen: Whether parent fields are frozen (used internally)
        name_width: Width of the name column (default: 26)
        shape_width: Width of the shape column (default: 14)
        status_width: Width of the status column (default: 20)
    """
    
    def _get_trainable_type(parent_frozen, field_frozen):
        """
        Determine the trainable type string based on freeze status.
        """
        if parent_frozen and field_frozen:
            return 'frozen'
        elif parent_frozen and not field_frozen:
            return 'frozen (inherited)'
        elif not parent_frozen and field_frozen:
            return 'frozen'
        else:
            return 'trainable'

    def _print_divider():
        """Print a divider line."""
        total_width = name_width + shape_width + status_width + 6  # 6 for borders and spaces
        print("|" + "-" * (total_width - 2) + "|")

    def _print_header():
        """Print the header with upper bound."""
        total_width = name_width + shape_width + status_width + 6  # 6 for borders and spaces
        print("=" * total_width)
        
        # Center the title
        title = "model structure"
        title_padding = (total_width - len(title) - 4) // 2  # 4 for "| " and " |"
        header_line = "|" + "-" * title_padding + f" {title} " + "-" * (total_width - title_padding - len(title) - 4) + "|"
        print(header_line)
        _print_divider()

    def _print_footer():
        """Print the footer with lower bound."""
        _print_divider()
        total_width = name_width + shape_width + status_width + 6  # 6 for borders and spaces
        print("=" * total_width)
    
    def _print_model_structure_recursive(module, prefix="", parent_frozen=False):
        """
        Recursively print the structure of an eqx.Module with freeze status.
        """
        
        if not isinstance(module, eqx.Module):
            return False
        
        has_printed = False
        
        for field_name, field_info in module.__dataclass_fields__.items():
            field_value = getattr(module, field_name)
            
            # Determine freeze status
            field_frozen = field_info.metadata.get("frozen", False)
            effective_frozen = parent_frozen or field_frozen
            
            # Build display name with proper indentation
            full_field_name = f"{prefix}.{field_name}" if prefix else field_name
            
            # Handle different field types
            if isinstance(field_value, eqx.Module):
                # Display the module field itself
                trainable_type = _get_trainable_type(parent_frozen, field_frozen)
                print(f"| {full_field_name:<{name_width}} {'(Module)':>{shape_width}} {trainable_type:>{status_width}} |")
                has_printed = True
                
                # Recurse into submodule
                sub_printed = _print_model_structure_recursive(field_value, full_field_name, effective_frozen)
                if sub_printed:
                    _print_divider()
                    
            elif isinstance(field_value, (list, tuple)):
                # Handle collections
                for i, item in enumerate(field_value):
                    item_name = f"{full_field_name}[{i}]"
                    
                    if isinstance(item, eqx.Module):
                        trainable_type = _get_trainable_type(parent_frozen, field_frozen)
                        print(f"| {item_name:<{name_width}} {'(Module)':>{shape_width}} {trainable_type:>{status_width}} |")
                        has_printed = True
                        
                        # Recurse into submodule
                        sub_printed = _print_model_structure_recursive(item, item_name, effective_frozen)
                        if sub_printed:
                            _print_divider()
                            
                    elif eqx.is_array(item):
                        trainable_type = _get_trainable_type(parent_frozen, field_frozen)
                        field_shape = str(item.shape)
                        print(f"| {item_name:<{name_width}} {field_shape:>{shape_width}} {trainable_type:>{status_width}} |")
                        has_printed = True
                        
            elif eqx.is_array(field_value):
                # Handle array fields
                trainable_type = _get_trainable_type(parent_frozen, field_frozen)
                field_shape = str(field_value.shape)
                print(f"| {full_field_name:<{name_width}} {field_shape:>{shape_width}} {trainable_type:>{status_width}} |")
                has_printed = True
        
        return has_printed
    
    # Main execution
    _print_header()
    has_content = _print_model_structure_recursive(module, prefix, parent_frozen)
    if not has_content:
        total_width = name_width + shape_width + status_width + 6
        no_content_msg = "No trainable parameters found"
        padding = total_width - len(no_content_msg) - 4  # 4 for "| " and " |"
        print(f"| {no_content_msg}{' ' * padding} |")
    _print_footer()

def debug_module_structure(module, prefix=""):
    """
    Recursively inspect and print the structure of an `eqx.Module` and its fields.

    This function is useful for debugging the internal structure of a module built using
    `equinox` (eqx), especially to identify and summarize non-array or unexpected values.
    Arrays and array-like structures are summarized by their shape and data type, and long
    lists/tuples are truncated for readability.

    Parameters
    ----------
    module : eqx.Module
        The module or value to inspect. If this is not an instance of `eqx.Module`,
        it is treated as a leaf node and printed directly.
    prefix : str, optional
        String prefix used to indicate the nested structure (used internally for recursion).
    """

    def format_value(value):
        """Format values for display, truncating arrays"""
        if hasattr(value, 'shape') and hasattr(value, 'dtype'):
            # It's an array-like object
            return f"Array(shape={value.shape}, dtype={value.dtype})"
        elif isinstance(value, (list, tuple)) and len(value) > 3:
            # Truncate long sequences
            return f"{type(value).__name__}[{len(value)} items]: {value[:3]}..."
        else:
            # For other values, show them directly
            return str(value)
    
    if not isinstance(module, eqx.Module):
        print(f"{prefix}Non-module leaf: {type(module)} = {format_value(module)}")
        return
    
    for name, field in module.__dataclass_fields__.items():
        value = getattr(module, name)
        new_prefix = f"{prefix}{name}."
        
        if isinstance(value, eqx.Module):
            debug_module_structure(value, new_prefix)
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                if isinstance(item, eqx.Module):
                    debug_module_structure(item, f"{new_prefix}[{i}].")
                else:
                    print(f"{new_prefix}[{i}]: {type(item)} = {format_value(item)}")
        else:
            print(f"{new_prefix[:-1]}: {type(value)} = {format_value(value)}")

def check_model_array_equality(A_model,B_model):
        """
        Takes two eqx.Module models with the same structure and prints out 
        "path jnp.all(jnp.isclose(A_leaf,B_leaf))" for all leafs that are jax.Arrays.

        Parameters
        ----------
        A_model : eqx.Module
        B_model : eqx.Module
        """

        A_paths = []
        A_leafs = []
        def A_collect_paths(path, leaf):
                A_paths.append(path)
                A_leafs.append(leaf)
        jax.tree_util.tree_map_with_path(A_collect_paths, A_model);

        B_paths = []
        B_leafs = []
        def B_collect_paths(path, leaf):
                B_paths.append(path)
                B_leafs.append(leaf)
        jax.tree_util.tree_map_with_path(B_collect_paths, B_model);

        for A_arr, B_arr, A_path, B_path in zip(A_leafs,B_leafs,A_paths,B_paths):
                A_key = "".join(str(k) for k in A_path).strip(".")
                B_key = "".join(str(k) for k in B_path).strip(".")
                assert A_key==B_key
                # print(A_key,"is array like",eqx.is_array_like(A_arr))
                if eqx.is_array_like(A_arr) and eqx.is_array_like(B_arr):
                    print(f"{A_key:40} is equal: {str(jnp.all(jnp.isclose(A_arr,B_arr))):10}")

# --- MODEL CLASSES --- #

class NOfba(eqx.Module):
    SCL_out : eqx.Module = frozen_field()
    COL_out : list = frozen_field()

    def __init__(self,SCL_out):
        """
        This just does re-scaling
        """
        self.SCL_out = SCL_out
        self.COL_out = self.SCL_out.col

    def __call__(self, fba_inp):
        """
        Inputs
        fba_inp ... SCALED values

        Returns
        fba_out ... SCALED values
        """

        fba_out = fba_inp
        return fba_out
    
    def get_unscaled(self, fba_inp):
        """
        Calls self() and then unscales the output.

        Inputs
        fba_inp ... SCALED values

        Returns
        FBA_out ... UNSCALED values
        """

        fba_out = self(fba_inp)
        # Scaling is not powerful enough to the the rates down to 
        # realistic values, so I divide by 10.
        # If this is not done, diffrax will hit max_steps.
        FBA_out = self.SCL_out.unscale(fba_out)/jnp.array([1.,1.,10.])
        return FBA_out
    
class SRfba(eqx.Module):
    SCL_inp : eqx.Module = frozen_field()
    SCL_out : eqx.Module = frozen_field()
    COL_inp : list = frozen_field()
    COL_out : list = frozen_field()

    def __init__(self): 
        """
        Symbolic Regression based surrogate FBA model.
        This model was derived from "run_SR_on_FBA_03.py".
        """
        import MPMs.DATA_functions_01 as DF

        # input values
        self.COL_inp = ['qG', 'nr_X', 'nr_P', 'nr_M',]
        # output values
        self.COL_out = ["EX_glyc_e_i",
                        "protein_synthesis",
                        'BIOMASS_Ec_iJO1366_core_53p95M',
                        "ATPM",
                        'EX_co2_e_o',
                        'EX_nh4_e_i',
                        'EX_o2_e_i',
                        'EX_so4_e_i']
        

        self.SCL_inp = DF.SCL_init_from_fba(self.COL_inp,method="mean") 
        self.SCL_out = DF.SCL_init_from_fba(self.COL_out,method="mean")

    def __call__(self, fba_inp):
        """
        Inputs
        fba_inp ... SCALED values of self.COL_inp

        Returns
        fba_out ... SCALED values of self.COL_out
        """
        
        n = self.normizer(fba_inp[1:4])
        n_X = n[0]
        n_P = n[1]
        n_M = n[2]

        qG = fba_inp[0]
        qP = qG * 1.2533545707500648*n_P /(0.7392875709919655*n_X + 0.2980381900321115*n_P + 0.31613452043825985*n_M) 
        qX = qG * 1.4466152118353457*n_X /(0.7392875709919655*n_X + 0.2980381900321115*n_P + 0.31613452043825985*n_M) 
        qM = qG * 1.2589659007754128*n_M /(0.7392875709919655*n_X + 0.2980381900321115*n_P + 0.31613452043825985*n_M) 
        qCO2 = qG * (0.7103896793658289*n_X + 0.14703757632448644*n_P + 1.4456742282955575     *n_M) / (1.298872281676593 *n_X + 0.5236437595294949*n_P + 0.5554295204483248*n_M) 
        qNH4 = qG * (3.262027935153786 *n_X + 1.4345848108836146 *n_P + -1.6645366599114816e-07*n_M) / (2.5132291325136933*n_X + 1.01319170879044  *n_P + 1.0747098312334038*n_M) 
        qO2  = qG * (1.1648615214015603*n_X + 0.34997064629049257*n_P + 1.7309298084472662     *n_M) / (1.8183790356051481*n_X + 0.7330725210391213*n_P + 0.7775782907021771*n_M) 
        qSO4 = qG * (3.547098615145487 *n_X + 0.1403078990559169 *n_P + 9.506688895607793e-05  *n_M) / (1.8962156727604698*n_X + 0.7634894500553386*n_P + 0.8107386190016588*n_M)
        fba_out =  jnp.array([qG,qP,qX,qM,qCO2,qNH4,qO2,qSO4])
        return fba_out
    
    def normizer(self,a):
        a = jnp.abs(a)
        s = jnp.sum(a)+1e-8
        return a/s
    
    def get_unscaled(self, fba_inp):
        """
        Calls self() and then unscales the output.

        Inputs
        fba_inp ... SCALED values of self.COL_inp

        Returns
        FBA_out ... UNSCALED values of self.COL_out
        """

        fba_out = self(fba_inp)
        FBA_out = self.SCL_out.unscale(fba_out)
        return FBA_out

def INIT_xavier(model, key):
    """
    Initializes the weights of an eqx.nn.Linear sub-module using Xavier initialization.
    """
    # return model
    def init_layer(layer):
        if isinstance(layer, eqx.nn.Linear):
            weight_key, bias_key = jax.random.split(key)
            new_weight = jax.nn.initializers.glorot_normal()(weight_key, layer.weight.shape)
            new_bias = jax.numpy.zeros_like(layer.bias) if layer.bias is not None else None
            return eqx.tree_at(lambda l: l.weight, layer, new_weight)
        return layer
    
    return jax.tree_util.tree_map(init_layer, model, is_leaf=lambda x: isinstance(x, eqx.nn.Linear))

class Bf_from_qX(eqx.Module):
    value : jax.Array = trainable_field()

    def __init__(self,value):
        """
        """
        self.value = jnp.array(value).astype(float)

    def __call__(self, x, key=None):
        """
        Inputs
        x = [v_rel, y, yi, u]
        qX = x[2]

        Returns
        Bf ... base feed
        """

        Bf = self.value*x[2]
        return Bf

class GPJX_wrapper(eqx.Module):
    GPJX_export : list = frozen_field()
    GPJX_train : list = frozen_field()
    SCL_inp : list = frozen_field()
    MODIFY_inputs : callable

    def __init__(self,GPJX_export,GPJX_train,SCL_inp,MODIFY_inputs=UF.return_same_value):
        """
        This wraps the GP jax expression so it can be used as an eqx.Module.
        """
        assert len(GPJX_export)==3
        assert len(SCL_inp)==3

        self.GPJX_export = GPJX_export
        self.GPJX_train = GPJX_train
        self.SCL_inp = SCL_inp
        self.MODIFY_inputs = MODIFY_inputs



    def __call__(self, inp, key=None):
        # because I did the SR on unscaled values, I have to unscale.

        yy = inp[0:4]
        yi = inp[4:8]
        uu = inp[8:12]
        # scale inputs
        YY = self.SCL_inp[0].unscale(yy)
        YI = self.SCL_inp[1].unscale(yi)
        UU = self.SCL_inp[2].unscale(uu)
        # scaled inputs    
        INP = jnp.concatenate([YY,YI,UU]).reshape(1,-1)
        # modify inputs if needed
        INP = self.MODIFY_inputs(INP)
        # compute outputs
        nX = self.GPJX_export[0].predict(INP,train_data=self.GPJX_train[0]).mean
        nP = self.GPJX_export[1].predict(INP,train_data=self.GPJX_train[1]).mean
        nM = self.GPJX_export[2].predict(INP,train_data=self.GPJX_train[2]).mean
        print(nX,nP,nM)
        fba_obj = jnp.concatenate([nX,nP,nM])
        fba_obj = jnp.nan_to_num(fba_obj, nan=0.0, posinf=0.0, neginf=0.0)

        return fba_obj

class PYSR_wrapper(eqx.Module):
    PYSR_export : list = frozen_field()
    SCL_inp : list = frozen_field()
    MODIFY_inputs : callable

    def __init__(self,PYSR_export,SCL_inp,MODIFY_inputs=UF.return_same_value):
        """
        This wraps the PYSR expression so it can be used as an eqx.Module.
        """
        assert len(PYSR_export)==3
        assert len(SCL_inp)==3

        self.PYSR_export = PYSR_export
        self.SCL_inp = SCL_inp
        self.MODIFY_inputs = MODIFY_inputs



    def __call__(self, inp, key=None):
        # because I did the SR on unscaled values, I have to unscale.

        yy = inp[0:4]
        yi = inp[4:8]
        uu = inp[8:12]
        # scale inputs
        YY = self.SCL_inp[0].unscale(yy)
        YI = self.SCL_inp[1].unscale(yi)
        UU = self.SCL_inp[2].unscale(uu)
        # scaled inputs    
        INP = jnp.concatenate([YY,YI,UU]).reshape(1,-1)
        # modify inputs if needed
        INP = self.MODIFY_inputs(INP)
        # compute outputs
        nX = self.PYSR_export[0]["callable"](INP,self.PYSR_export[0]["parameters"])
        nP = self.PYSR_export[1]["callable"](INP,self.PYSR_export[1]["parameters"])
        nM = self.PYSR_export[2]["callable"](INP,self.PYSR_export[2]["parameters"])
        fba_obj = jnp.concatenate([nX,nP,nM])
        fba_obj = jnp.nan_to_num(fba_obj, nan=0.0, posinf=0.0, neginf=0.0)

        return fba_obj

class MARCO_wrapper(eqx.Module):
    SP_export : list = frozen_field()
    SCL_inp : list = frozen_field()
    MODIFY_inputs : callable
    variables : list = frozen_field()
    nX : callable = frozen_field()
    nP : callable = frozen_field()
    nM : callable = frozen_field()

    def __init__(self,SP_export,SCL_inp,MODIFY_inputs=UF.return_same_value,NR_vars=9):
        """
        This wraps the Sympy expressions I got from Marco so it can be used as an eqx.Module.
        """
        import sympy as sp

        assert len(SP_export)==3
        assert len(SCL_inp)==3

        self.SP_export = SP_export
        self.SCL_inp = SCL_inp
        self.MODIFY_inputs = MODIFY_inputs
        STR_var = ""
        for i in range(NR_vars):
            STR_var += f"x_{i} "
        self.variables = sp.symbols(STR_var)

        self.nX = sp.lambdify(self.variables, self.SP_export[0], "jax")
        self.nP = sp.lambdify(self.variables, self.SP_export[1], "jax")
        self.nM = sp.lambdify(self.variables, self.SP_export[2], "jax")


    def __call__(self, inp, key=None):
        # because I did the SR on unscaled values, I have to unscale.

        yy = inp[0:4]
        yi = inp[4:8]
        uu = inp[8:12]
        # scale inputs
        YY = self.SCL_inp[0].unscale(yy)
        YI = self.SCL_inp[1].unscale(yi)
        UU = self.SCL_inp[2].unscale(uu)
        # scaled inputs    
        INP = jnp.concatenate([YY,YI,UU]).reshape(1,-1)
        # modify inputs if needed
        INP = self.MODIFY_inputs(INP)
        # compute outputs
        nX = self.nX(*INP)
        nP = self.nP(*INP)
        nM = self.nM(*INP)
        fba_obj = jnp.array([nX,nP,nM])
        fba_obj = jnp.nan_to_num(fba_obj, nan=0.0, posinf=0.0, neginf=0.0)

        return fba_obj

class KANSR_wrapper(eqx.Module):
    SP_export : list = frozen_field()
    MODIFY_inputs : callable
    variables : list = frozen_field()
    nX : callable = frozen_field()
    nP : callable = frozen_field()
    nM : callable = frozen_field()

    def __init__(self,SP_export,MODIFY_inputs=UF.return_same_value,NR_vars=12):
        """
        This wraps the KANSR Sympy expression so it can be used as an eqx.Module.
        """
        import sympy as sp

        assert len(SP_export)==3

        self.SP_export = SP_export
        self.MODIFY_inputs = MODIFY_inputs
        STR_var = ""
        for i in range(NR_vars):
            STR_var += f"x_{i} "
        self.variables = sp.symbols(STR_var)

        self.nX = sp.lambdify(self.variables, self.SP_export[0], "jax")
        self.nP = sp.lambdify(self.variables, self.SP_export[1], "jax")
        self.nM = sp.lambdify(self.variables, self.SP_export[2], "jax")


    def __call__(self, inp, key=None):
        # because I did the SR on unscaled values, I have to unscale.

        # modify inputs if needed
        INP = self.MODIFY_inputs(inp)
        # compute outputs
        nX = self.nX(*INP)
        nP = self.nP(*INP)
        nM = self.nM(*INP)
        fba_obj = jnp.array([nX,nP,nM])
        fba_obj = jnp.nan_to_num(fba_obj, nan=0.0, posinf=0.0, neginf=0.0)

        return fba_obj

class MLP_wrapper(eqx.Module):    
    ANN : eqx.Module = trainable_field()
    MODIFY_inputs : callable

    def __init__(self,ANN,MODIFY_inputs):
        """
        This wraps and equinox MLP where inputs can be modified 
        by a custom MODIFY_inputs function before passing to the ANN.
        """

        self.ANN = ANN
        self.MODIFY_inputs = MODIFY_inputs

    def __call__(self, inp, key=None):
        INP = self.MODIFY_inputs(inp)
        out = self.ANN(INP)
        return out

class Ensemble(eqx.Module):
    models: list[eqx.Module] = frozen_field()
    
    def __init__(self, models):
        """
        Initialize the ensemble with a list of models.
        
        Args:
            models (list[eqx.Module]): List of models to be included in the ensemble.
        """
        self.models = models

    def __call__(self, YT, y0, yi, UT, uc, DS, key):

        y = jnp.stack([model(YT, y0, yi, UT, uc, DS, key) for model in self.models], axis=0)
        return y
    
    def get_intermediates(self, t, y, args):
        intermediates_list  = [model.HYB.dNdt_and_intermediates(t, y, args)[1] for model in self.models]
        # Collect all keys (assuming all dicts have the same structure)
        keys = intermediates_list[0].keys()
        
        # Stack arrays along a new axis (ensemble axis)
        stacked = {
            k: jnp.stack([d[k] for d in intermediates_list], axis=0)
            for k in keys
        }
        return stacked

    def AVG_get_intermediates(self, t, y, args):
        stacked = self.get_intermediates(t, y, args)
        return {k: jnp.mean(v, axis=0) for k, v in stacked.items()}
    
    def AVG_call(self, YT, y0, yi, UT, uc, DS, key):
        y_ens = self(YT, y0, yi, UT, uc, DS, key)
        y_avg = jnp.mean(y_ens, axis=0)
        return y_avg

# --- G LIMITING FUNCTIONS --- #

def get_qG_LIM(Y, U):
    # glycerol feed rate, Gf = Cf_N * G_Cf in [gG/h]
    Gf = U[3]*U[2]        # [gG/h]
    Gf = Gf/92.09382*1000 # [mmolG/h]
    # total active biomass, XrV = (X-P)*V
    XrV = (Y[2]-Y[1])*Y[3] # [gXr]
    # max qG (when G is limiting)
    qG_lim = Gf/XrV # [mmolG/(gXr h)]
    return qG_lim

def SLIM_get_qG_LIM(Y, U):
    # glucose feed rate, Gf = Cf_N * G_Cf in [gG/h]
    Gf = U[2]*U[1]          # [gG/h]
    Gf = Gf/180.15588*1000  # [mmolG/h]
    # total active biomass, XrV = (X-P)*V
    XrV = (Y[2]-Y[1])*Y[3] # [gXr]
    # max qG (when G is limiting)
    qG_lim = Gf/XrV # [mmolG/(gXr h)]
    return qG_lim

def G_LIM_none(qG_ann, qG_lim, LIM_params, G):
        return jnp.array([qG_ann])

def G_LIM_sigmoid(qG_ann, qG_lim, LIM_params, G):
    # scale qG to process max if G is low (smooth with sigmoid)

    # combine qG_lim (process) and qG_ann (NN prediction) with sigmoid
    alpha = UF.custom_sigmoid(G, switch=LIM_params[0], stretch=LIM_params[1])
    qG_cmb = (1-alpha)*qG_lim + alpha*qG_ann
    # jax.debug.print("alpha: {alpha}",alpha=alpha,)
    return jnp.array([qG_cmb])

def G_LIM_min(qG_ann, qG_lim, LIM_params, G):
    # take the minimum of qG_max and qG_ann
    # this behaviour does not allow for consumption of G that once accumulated,
    # but in practice this is not observed for this data set
    
    # combine qG_lim (process) and qG_ann (NN prediction)
    qG_cmb = jnp.min(jnp.array([qG_lim, qG_ann]))# *.99 + qG_ann * 0.01
    return jnp.array([qG_cmb])

def G_LIM_fixed(qG_ann, qG_lim, LIM_params, G):
    return jnp.array([qG_lim])

# --- SLIM MODEL CLASSES --- #

class SLIM_SRfba(eqx.Module):
    SCL_inp : eqx.Module = frozen_field()
    SCL_out : eqx.Module = frozen_field()
    COL_inp : list = frozen_field()
    COL_out : list = frozen_field()

    def __init__(self):
        """
        Symbolic Regression based surrogate FBA model.
        This model was derived from "slim_fba_data/run_SR_on_FBA_03.py".
        """
        import MPMs.DATA_functions_01 as DF

        # input values
        self.COL_inp = ['qG', 'nr_X', 'nr_P', 'nr_M',]
        # output values
        self.COL_out = ["EX_glc__D_e",
                        "pDNA_synthesis",
                        'BIOMASS_Ec_iML1515_core_75p37M',
                        "ATPM",
                        'EX_co2_e',
                        'EX_nh4_e',
                        'EX_o2_e',
                        'EX_so4_e',]
        
        self.SCL_inp = DF.SCL_init_from_slim_fba(self.COL_inp,method="mean") 
        self.SCL_out = DF.SCL_init_from_slim_fba(self.COL_out,method="mean")

    def __call__(self, fba_inp):
        """
        Inputs
        fba_inp ... SCALED values

        Returns
        fba_out ... SCALED values
        """
        
        n = self.normizer(fba_inp[1:4])
        n_X = n[0]
        n_P = n[1]
        n_M = n[2]

        qG = fba_inp[0]
        qX = qG * 2.095925756666534*n_X /(1.10909581775274*n_X + 0.06282378505932679*n_P + 0.856382771607087*n_M) 
        qP = qG * 1.5774883284049748*n_P /(1.10909581775274*n_X + 0.06282378505932679*n_P + 0.856382771607087*n_M) 
        qM = qG * 1.9885886903097858*n_M /(1.10909581775274*n_X + 0.06282378505932679*n_P + 0.856382771607087*n_M) 
        qCO2 = qG * (1.2082933347251283*n_X + 0.031406825524812980*n_P +  2.4389345018027253*n_M) /(2.0273005373419446*n_X + 0.11774348504170798*n_P + 1.5547353239688702*n_M) 
        qNH4 = qG * (1.9515541853597302*n_X + 0.214745234308724600*n_P + -0.0007234803861126038*n_M) /(1.1874825647695029*n_X + 0.06552739969547203*n_P + 0.911539718451724*n_M) 
        qO2  = qG * (0.9545539690035202*n_X + 0.047285966317114195*n_P +  2.0881011826979528*n_M) /(1.7127767420491968*n_X + 0.09842279611123259*n_P + 1.3054046578313026*n_M) 
        qSO4 = qG * (5.8760980162677820*n_X + 0.000509871455802084*n_P + -0.0015773984660723997*n_M) /(3.111874219100181*n_X + 0.1769442128559814*n_P + 2.3968877820508285*n_M)
        fba_out =  jnp.array([qG,qP,qX,qM,qCO2,qNH4,qO2,qSO4])
        return fba_out
    
    def normizer(self,a):
        a = jnp.abs(a)
        s = jnp.sum(a)+1e-8
        return a/s
    
    def get_unscaled(self, fba_inp):
        """
        Calls NOfba() and then unscales the output.

        Inputs
        fba_inp ... SCALED values

        Returns
        FBA_out ... UNSCALED values
        """

        fba_out = self(fba_inp)
        # qSO4 is a uptake, so I need to invert the sign
        FBA_out = self.SCL_out.unscale(fba_out)*jnp.array([1,1,1,1,1,1,1,-1])
        return FBA_out

class RETURN_value(eqx.Module):
    value : jax.Array = frozen_field()

    def __init__(self,value):
        """
        Stores singular float value and returns it as jax.Array when called.
        """
        self.value = jnp.array([float(value)])

    def __call__(self, x, key=None):
        """
        Stores singular float value and returns it as jax.Array when called.
        """
        
        return self.value




