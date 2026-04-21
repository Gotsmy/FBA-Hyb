#!/home/mgotsmy/anaconda3/envs/pysr/bin/python

import jax.numpy as jnp
import pandas as pd
import pysr

def modify_inputs(ann_inp):
    # ann_inp = jnp.array(ann_inp)
    # IDX = jnp.array([ 0,  1,  2,  3,  9])
    # mod_ann_inp = jnp.take(ann_inp, IDX, axis=1)

    # # GENERATIONS
    # XV_I = ann_inp[:,6]*ann_inp[:,7] # XV_I =  X_I * V_N_I [gX]
    # XV   = ann_inp[:,2]*ann_inp[:,3] # XV   =  X   * V_N   [gX]
    # N = jnp.log(XV/XV_I)/jnp.log(2)

    # # G-FEED/Volume
    # Gf_N = ann_inp[:,10]*ann_inp[:,11] # Gf_N = G_Cf * Cf_N   [gG/h]
    # Gf_V = Gf_N/ann_inp[:,3]           # Gf_V = Gf_N / V_N    [gG/(h kgV)]
    # mod_ann_inp = jnp.concatenate([mod_ann_inp,N[:,None],Gf_V[:,None]],axis=1)
    mod_ann_inp = jnp.array(ann_inp)
    return mod_ann_inp

if __name__ == "__main__":
    #--- SCRIPT INPUT ---#
    version = "MPM_146B"


    #--- END INPUT ---#

    if "SLIM" in version:
        base_path = "/mnt/y/code/250513_metabolic_process_models/slim_model_data/"
    else:
        base_path = "/mnt/y/code/250513_metabolic_process_models/model_data/"

    data_path = f"{base_path}/{version}/CV_ann_fba_obj.csv"
    pysr_path = f"{base_path}/{version}/ANN2SR_03//"


    df = pd.read_csv(data_path)
    set_names = df["set"].unique()

    for set_name in [set_names[1]]:
        print("set_name",set_name)
        tmp = df[df["set"]==set_name]
        mod_ann_inp = modify_inputs(tmp.iloc[:,:tmp.shape[1]-4].values)
        # mod_ann_inp = pd.DataFrame(mod_ann_inp,columns=['G', 'P', 'X', 'V_N', 'T',"n","Gf_V"])

        
        model = pysr.PySRRegressor(
                        binary_operators=["*","+","/"],
                        # unary_operators=["exp","log"],
                        niterations=100,
                        populations=48,
                        population_size=100,
                        output_directory=pysr_path,
                        run_id=set_name,
                        constraints={
                            "*": (-1, 5),
                            "/": (-1, 5),
                        },
                        complexity_of_variables=2,  # This is fine - controls variable combinations
                        nested_constraints={
                            "*": {
                                "*": 2,  # INCREASED: Allow x*x, x*x*x terms (polynomial)
                                "/": 0,  # Keep: Don't allow division inside multiplication
                                "+": 1,  # Keep: Allow addition inside multiplication
                            },
                            "/": {
                                "*": 2,  # CHANGED: Allow x*x in numerators (for x*x/(p+x) terms)
                                "/": 0,  # Keep: Don't allow division inside division
                                "+": 2,  # Keep: Allow addition in denominators
                            },
                            "+": {
                                "*": 2,  # INCREASED: Allow polynomial terms in sums
                                "/": 1,  # Keep: Allow division in addition terms
                                "+": 1,  # Keep: Allow nested addition
                            },
                        },
                        turbo=True,
                        bumper=True,
                        progress=True
                    )
        model.fit(mod_ann_inp,tmp.iloc[:,tmp.shape[1]-4:-1])

    print("DONE")
