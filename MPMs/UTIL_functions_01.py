import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
import time
import importlib.util

# ----------------- ACTIVATION FUNCTIONS ----------------- #

def bounded_softplus(x, alpha=10):
    x = jax.nn.tanh(jax.nn.softplus(x)/alpha)*alpha
    return x

def tanh_t3(x):
    x = jax.nn.tanh(x/3)*3
    return x

def tanh_t2(x):
    x = jax.nn.tanh(x/2)*2
    return x

def smooth_abs(x):
    """
    Smooth absolute value function.
    """
    eps = 1e-8
    return jnp.sqrt(jnp.square(x) + eps)

def custom_sigmoid(x, switch=0, stretch=1):
    return jax.nn.sigmoid((x-switch)/stretch)

def return_same_value(x):
    return x

def sigmoid_t3(x):
    return jax.nn.sigmoid(x/3)*3

# ----------------- EVALUATION FUNCTIONS ----------------- #

def RMSE(y_true, y_pred):
    """
    Root mean squared error.
    """
    return jax.lax.cond(
        y_true.size == 0,  # Empty array
        lambda: 0.0,
        lambda: jnp.sqrt(jnp.mean(jnp.square(y_true - y_pred)))
    )

def rRMSE(y_true, y_pred):
    """
    Relative Root mean squared error in %
    """
    return jax.lax.cond(
        y_true.size == 0,  # Empty array
        lambda: 0.0,
        lambda: jnp.mean(jnp.sqrt(jnp.square(y_true - y_pred))/y_true)*100
    )

def MSE(y_true, y_pred):
    """
    Mean of squared errors.
    """
    
    return jax.lax.cond(
        y_true.size == 0,  # Empty array
        lambda: 0.0,
        lambda: jnp.mean(jnp.square(y_true - y_pred))
    )

def SSE(y_true, y_pred):
    """
    Sum of squared errors.
    """
    return jnp.sum(jnp.square(y_true - y_pred))

def MAE(y_true, y_pred):
    """
    Mean absolute error.
    """

    return jax.lax.cond(
        y_true.size == 0,  # Empty array
        lambda: 0.0,
        lambda: jnp.mean(jnp.abs(y_true - y_pred))
    )

def NMAE(y_true, y_pred):
    """
    Normalized mean absolute error.
    """
    return jax.lax.cond(
        y_true.size == 0,  # Empty array
        lambda: 0.0,
        lambda: jnp.mean(jnp.abs(y_true - y_pred))/jnp.mean(jnp.abs(y_true))
    )

def R2(y_true, y_pred):
    RSS = jnp.sum(jnp.square(y_true - y_pred))
    TSS = jnp.sum(jnp.square(y_true - jnp.mean(y_true)))
    is_zero_TSS = jnp.isclose(TSS, 0.0)

    def return_inf(_):
        return jnp.inf

    def compute_r2(_):
        return 1 - RSS / TSS

    return jax.lax.cond(is_zero_TSS, return_inf, compute_r2, operand=None)

# ----------------- TIMING ----------------- #

class timing:
    def __init__(self, comment="Start"):
        self.last_time = time.perf_counter()
        elapsed = 0.
        i, c = self.get_indent(comment)
        self.i = i
        print(f"{' '*i}{c:20}: {elapsed:10.2f} s")

    def __call__(self, comment=""):
        current_time = time.perf_counter()
        i, c = self.get_indent(comment)
        elapsed = (current_time - self.last_time)
        print(f"{' '*self.i}{c:20}: {elapsed:10.2f} s")
        self.last_time = current_time

    def get_indent(self,comment):
        c = comment.strip(" ")
        i = len(comment) - len(c)
        return i, c

# ----------------- IMPORT MODEL FILE ----------------- #

def import_module_from_path(filepath):
    """
    takes filepath as string and returns python module
    """
    module_name = os.path.splitext(os.path.basename(filepath))[0]
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# ----------------- OTHER ----------------- #

def squeeze_3D_array(a):
    assert len(a.shape) == 3
    return a.reshape(-1,a.shape[-1])

def squeeze(a):
    """
    Squeezes array to last 2 dimensions.
    """
    return a.reshape(-1,a.shape[-1])

def unsqueeze(a):
    """
    Adds a dimension at axis 0.
    """
    return a[None,:,:]

def diag(ax,ls="--",c="k",zorder=-1,**kwargs):
    """
    draw diagonal line in plt axis
    """
    values = ax.get_xlim(), ax.get_ylim()
    min_ = np.min(values)
    max_ = np.max(values)
    ax.plot([min_,max_],[min_,max_],ls=ls,c=c,zorder=zorder,**kwargs)
    ax.set_xlim(min_,max_)
    ax.set_ylim(min_,max_)

# --- basic differentiation --- #

def diff(x):
    return x[1:] - x[:-1]

def mean(x):
    return (x[1:] + x[:-1]) / 2






