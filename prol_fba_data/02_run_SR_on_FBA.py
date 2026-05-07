import os
os.environ['JAX_PLATFORMS'] = 'cpu'
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import pysr
import jax
import jax.numpy as jnp

import MPMs.DATA_functions_01 as DF

def get_scalers(col_inp, col_out):
    SCL_inp = DF.SCL_init_from_fba(col_inp,method="mean") 
    SCL_out = DF.SCL_init_from_fba(col_out,method="mean")
    SCLs = (SCL_inp, SCL_out)
    return SCLs

def get_columns_inp_out():
    # input columns
    COL_inp = np.array(['qG', 'nr_X', 'nr_P', 'nr_M',])
    # output columns
    COL_out = np.array(["EX_glyc_e_i","protein_synthesis",'BIOMASS_Ec_iJO1366_core_53p95M',"ATPM",
                        'EX_co2_e_o','EX_nh4_e_i','EX_o2_e_i','EX_so4_e_i',])
    # COL_out = np.array(['BIOMASS_Ec_iJO1366_core_53p95M',"protein_synthesis"])
    return COL_inp, COL_out

def get_FBA_data(col_inp, col_out):
    FBAdata = pd.read_csv(str(REPO_ROOT / "prol_fba_data" / "01_obj_fba_data.csv"),index_col=0)
    FBAdata = FBAdata.iloc[:,:]
    
    # remove combined objective fluxes
    real_fluxes = []
    for f in FBAdata.columns[4:]:
        if not "combined_obj" in f:
            real_fluxes.append(f)
    INPdata = FBAdata.iloc[:,:4]
    FBAdata = FBAdata.loc[:,real_fluxes]

    y_data = FBAdata.loc[:,col_out]
    x_data = INPdata.loc[:,col_inp]

    return x_data.values, y_data.values

def split_scale(X,Y,SCLs,validation_split=.1):

    # select random values for train and valid data set
    key = jax.random.PRNGKey(13)
    ID = jax.random.choice(key, X.shape[0], shape=(int(Y.shape[0]*validation_split),), replace=False)
    X_valid, Y_valid  = X[ID,:], Y[ID,:]
    X_train = jnp.delete(X, ID, axis=0)
    Y_train = jnp.delete(Y, ID, axis=0)

    # scaling data set
    SCL_inp, SCL_out = SCLs
    
    x_train = SCL_inp.scale(X_train)
    y_train = SCL_out.scale(Y_train)
    x_valid = SCL_inp.scale(X_valid)
    y_valid = SCL_out.scale(Y_valid)

    # ensure that everything is a jax array
    x_train = jnp.array(x_train)
    y_train = jnp.array(y_train)
    x_valid = jnp.array(x_valid)
    y_valid = jnp.array(y_valid)

    return (x_train, y_train), (x_valid, y_valid)

if __name__ == "__main__":
    # --- DEFINE INPUT AND OUTPUT COLUMSN --- #
    COL_inp, COL_out = get_columns_inp_out()

    # --- LOAD DATA --- #
    X, Y = get_FBA_data(COL_inp, COL_out)
    print("X.shape",X.shape,"Y.shape",Y.shape)

    # --- GET SCALERS --- #
    SCLs = get_scalers(COL_inp, COL_out)

    # --- PREPROCESS DATA --- #
    train_, valid_ = split_scale(X,Y,SCLs)

    # --- DEFINE MODEL --- #
    model = pysr.PySRRegressor(
                binary_operators=["*","+","-","/"],
                # unary_operators=["exp","log"], # was commented out
                niterations=1000,
                populations=200,
                population_size=200,
                output_directory=str(REPO_ROOT / "prol_fba_data" / "sr_models"),
                run_id=__file__.split("/")[-1].strip(".py"),
                # constraints={"*":(-1,1)}, # was commented in
                # complexity_of_variables=2, # was commented in
                # nested_constraints={"*":{"*":0}}, # was commented in
                turbo=True,
                bumper=True
            )

    # --- TRAIN MODEL --- #
    print("START FIT")
    model.fit(*train_,variable_names=COL_inp.tolist())

    print("FINISHED")
    print(model)





