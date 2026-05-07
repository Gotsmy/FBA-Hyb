from pathlib import Path
import cobra
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import qmc
import snek

REPO_ROOT = Path(__file__).resolve().parents[1]

def build_model(qG, nr_X, nr_P, nr_M, tmp_model):
    nr_X, nr_P, nr_M = normizer(nr_X, nr_P, nr_M)
    # add new objective reaction
    reaction = cobra.Reaction('combined_obj')
    reaction.name = 'combined_obj_function'
    reaction.lower_bound = 0.
    reaction.upper_bound = 1000.
    reaction.add_metabolites({
        xxx_equivalent: -nr_X,
        atp_equivalent: -nr_M,
        pro_equivalent: -nr_P})
    tmp_model.add_reactions([reaction])
    snek.set_objective(tmp_model,"combined_obj","max")
    snek.set_bounds(tmp_model,"EX_glc__D_e",-qG,-qG)
    return tmp_model

def load_model():
    # load model
    model = cobra.io.read_sbml_model(str(REPO_ROOT / "slim_fba_data" / "iML1515_pDNA.xml"))
    model.solver = "cplex"


    # glucose as main C source
    snek.set_bounds(model,"EX_glc__D_e",0,10)
    # remove the ATPM constraint
    snek.set_bounds(model,"ATPM",0,1000)
    # add pseudo-metabolites to model
    pro_equivalent = cobra.Metabolite('pro_equ',name='product equivalent',compartment='c')
    xxx_equivalent = cobra.Metabolite('xxx_equ',name='biomass equivalent',compartment='c')
    atp_equivalent = cobra.Metabolite('atp_equ',name='maintenance equivalent',compartment='c')
    
    model.reactions.BIOMASS_Ec_iML1515_core_75p37M.add_metabolites({xxx_equivalent:1})
    model.reactions.ATPM.add_metabolites({atp_equivalent:1})
    model.reactions.pDNA_synthesis.add_metabolites({pro_equivalent:1})

    return model, xxx_equivalent, atp_equivalent, pro_equivalent

def normizer(a,b,c):
    m = a+b+c
    return a/m, b/m, c/m


if __name__ == "__main__":
    # sample solution space (i.e., lowest glucose uptake and )
    # dimensions = qG, qX, qP, qM
    sampler  = qmc.LatinHypercube(d=4)
    n_samples = 10_000
    # n_samples = 100
    norm_sample = sampler.random(n_samples)
    # these are approx 2x the max value observed in prot L DoE
    scaled_sample = qmc.scale(norm_sample,[0,0,0,0],[20,.5,.01,100])

    # load model
    model, xxx_equivalent, atp_equivalent, pro_equivalent = load_model()

    fba_list = []
    for qG, nr_X, nr_P, nr_M in tqdm(scaled_sample):
        with model as tmp:
            tmp = build_model(qG, nr_X, nr_P, nr_M, tmp)
            try:
                sol = cobra.flux_analysis.pfba(tmp)
            except:
                print(qG, nr_X, nr_P, nr_M)
            fba_list.append(np.concatenate([
                np.array([qG,nr_X,nr_P,nr_M]),
                sol.fluxes.values.flatten()]))

    with model as tmp:
        tmp = build_model(1,1,1,1,tmp)
        df = pd.DataFrame(fba_list,columns=["qG","nr_X","nr_P","nr_M"]+[r.id for r in tmp.reactions])

    print("saving dataframes")
    df.to_csv(str(REPO_ROOT / "slim_fba_data" / "250813_obj_fba_01.csv"))
    print("done")
