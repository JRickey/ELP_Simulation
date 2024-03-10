import pandas as pd
import numpy as np
import math
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from modules.data_reading import read_csv, merge_dataframes
import modules.simulator_components as sim

base_path = Path.cwd()

def main_simulation():
    # Step 1: Read and prepare data
    dt_coef_path = base_path / 'cloglog' / 'default_coef_final.csv'
    pp_coef_path = base_path / 'cloglog' / 'prepaid_coef_final.csv'
    test_data_path = base_path / 'loans' / 'test_data_final.csv'
    df_msa_path = base_path / 'msa_data' / 'HPI_PO_metro_name.csv'
    rate15y_path = base_path / 'rate_HPI_process' / '15_rate.csv'
    rate30y_path = base_path / 'rate_HPI_process' / '30_rate.csv'
    hpi_path = base_path / 'rate_HPI_process' / 'HPI.csv'
    msa_state_adj_path = base_path / 'rate_HPI_process' / 'dispersion_matrix.csv'

    dt_coef = read_csv(dt_coef_path)
    pp_coef = read_csv(pp_coef_path)
    data = read_csv(test_data_path)
    df_msa = read_csv(r"path\to\HPI_PO_metro_name.csv")
    
    data = merge_dataframes(data, df_msa[['MSA', 'metro_name']], on='MSA',
                            fill_na={'metro_name': data['prop_state']},
                            rename_columns={'metro_name': 'msa_state_name'})
    
    # Additional steps to process data and prepare for simulation...
    
    # Step 2: Perform simulation
    dt_hpa_param, pp_hpa_param, dt_time_param, pp_time_param = sim.extract_parameters(dt_coef, pp_coef)
    h0_dt = sim.compute_h0(dt_time_param)
    h0_pp = sim.compute_h0(pp_time_param)
    
    # Simulation loop...
    
    # Step 3: Analyze and visualize results
    # Visualization and analysis logic here...
