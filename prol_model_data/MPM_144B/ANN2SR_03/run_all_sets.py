
import os
import time
import random
from multiprocessing import Pool

def run_optimization(script):
    time.sleep(random.uniform(0,200))
    os.system(f"/home/mgotsmy/anaconda3/envs/jax/bin/python {script}")

scripts = ['/mnt/y/code/250513_metabolic_process_models/model_data//MPM_144B/ANN2SR_03///script_set_1.py', '/mnt/y/code/250513_metabolic_process_models/model_data//MPM_144B/ANN2SR_03///script_set_2.py', '/mnt/y/code/250513_metabolic_process_models/model_data//MPM_144B/ANN2SR_03///script_set_3.py', '/mnt/y/code/250513_metabolic_process_models/model_data//MPM_144B/ANN2SR_03///script_set_4.py', '/mnt/y/code/250513_metabolic_process_models/model_data//MPM_144B/ANN2SR_03///script_set_5.py', '/mnt/y/code/250513_metabolic_process_models/model_data//MPM_144B/ANN2SR_03///script_set_6.py', '/mnt/y/code/250513_metabolic_process_models/model_data//MPM_144B/ANN2SR_03///script_set_7.py', '/mnt/y/code/250513_metabolic_process_models/model_data//MPM_144B/ANN2SR_03///script_set_8.py', '/mnt/y/code/250513_metabolic_process_models/model_data//MPM_144B/ANN2SR_03///script_set_9.py', '/mnt/y/code/250513_metabolic_process_models/model_data//MPM_144B/ANN2SR_03///script_set_10.py', '/mnt/y/code/250513_metabolic_process_models/model_data//MPM_144B/ANN2SR_03///script_set_11.py', '/mnt/y/code/250513_metabolic_process_models/model_data//MPM_144B/ANN2SR_03///script_set_12.py']

with Pool(processes=2) as p:
    results = list(p.imap_unordered(run_optimization, scripts))
print("DONE")