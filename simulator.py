from pathlib import Path
import pandas as pd
import os
import concurrent.futures
import math
import numpy as np
import time
import logging

def configure_logger():
    # Create or get the logger
    logger = logging.getLogger(__name__)  # __name__ gives each file its logger, or use a custom name
    logger.setLevel(logging.DEBUG)  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    # Create handlers (console and file)
    c_handler = logging.StreamHandler()  # Console handler
    f_handler = logging.FileHandler('simulation.log')  # File handler
    c_handler.setLevel(logging.INFO)  # Console handler level
    f_handler.setLevel(logging.DEBUG)  # File handler level
    
    # Create formatters and add them to handlers
    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)
    
    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger

logger = configure_logger()

def read_csv(filepath):
    return pd.read_csv(filepath, index_col=None)

def compute_h0(time_param):
    return [math.exp(i) for i in time_param.iloc[:,0]]

def compute_cheng(x, y):
    return (x * y).sum()



def simulate_loan_defaults(n, data, rate15y_path, rate30y_path, dt_params, pp_params, h0_dt, h0_pp, hpa_path, msa_state, msa_state_adj, original_rate):
    """
    Simulate loan defaults and prepayments for n loans using given parameters.

    Parameters:
    - n: Number of loans to simulate.
    - data: DataFrame containing loan data.
    - dt_params: Parameters for default timing including 'unchanged' part.
    - pp_params: Parameters for prepayment timing including 'unchanged' part.
    - h0_dt: Baseline hazard rate for default timing.
    - h0_pp: Baseline hazard rate for prepayment timing.
    - hpa_path: Home Price Appreciation path data.
    - msa_state: Original MSA values.
    - msa_state_adj: MSA state adjustment factors.
    - original_rate: Original interest rates of the loans.
    
    Returns:
    - Tuple of DataFrames for default rates and prepayment rates.
    """
    
    # Initialize the arrays for default and prepayment rates
    default_rate = [0] * n
    pp_rate = [0] * n
    start = 0
    #logging.info(f"Data size: {data.shape}, Rate15y size: {rate15y_path.shape}, Rate30y size: {rate30y_path.shape}")
    pp_coef_param = pp_params.iloc[42:]['Estimate'] # series
    dt_coef_param = dt_params.iloc[41:]['Estimate']
    dt_hpa_param = dt_params.iloc[40,0] # value
    pp_hpa_param = pp_params.iloc[40,0] # value
    pp_refi_param = pp_params.iloc[41,0] # value
    
    # Iterate through each loan
    for j in range(n):
        #pid = os.getpid()
        #logger.info(f'[{pid}] Processing loan {j} of {range(n)}')
        # Calculate unchanged part of the model for each loan
        Dflt_Xb_unchanged = compute_cheng(dt_coef_param,data.iloc[j + start])
        PP_Xb_unchanged = compute_cheng(pp_coef_param,data.iloc[j + start])
        #logger.info('unchanged part calculated)')
        # Extract necessary data for the current loan
        msa_state_adj_temp = msa_state_adj[msa_state[j]].tolist()
        loan_term_years = data.loc[j, 'loan_term_years']
        original_interest_rate = original_rate[j]

        # Determine if loan is closer to 15 or 30 years and select the rate path accordingly
        diff_15 = abs(loan_term_years - 15)
        diff_30 = abs(loan_term_years - 30)
        condition = diff_15 <= diff_30
        #logging.info(f"Loan {j+1} term condition: {'15 years' if condition else '30 years'}")
        
        # Initialize containers for simulation results
        default_rate[j] = [0] * len(hpa_path.columns)
        pp_rate[j] = [0] * len(hpa_path.columns)
        
        # Simulate across all paths
        for k in range(len(hpa_path.columns)):  # Assuming hpa_path columns correspond to different paths
            rate_temp = rate15y_path.iloc[:, k].tolist() if condition else rate30y_path.iloc[:, k].tolist()
            hpa_path_temp = hpa_path.iloc[:, k].tolist()
            
            # Containers for hazard rates and probabilities
            Dflt_Hazard = [0] * 41
            PP_Hazard = [0] * 41
            Surv_Prob = [1] * 41
            Est_Period_Dflts = [0] * 41
            Est_Period_PPs = [0] * 41

            # Simulate each quarter
            for i in range(1, 41):
                # Calculation logic for default and prepayment probabilities
                #logging.info(f"Processing quarter {i} for loan {j+1}, path {k+1}")
                hpa_temp = hpa_path_temp[i-1]
                msa = msa_state_adj_temp[i-1]

                Dflt_Xb = Dflt_Xb_unchanged + hpa_temp * dt_hpa_param + msa * dt_hpa_param
                PP_Xb = PP_Xb_unchanged + hpa_temp * pp_hpa_param + (original_interest_rate - rate_temp[i-1]) * pp_refi_param
                
                Dflt_Hazard[i] = 1 - math.exp(-h0_dt[i-1] * math.exp(Dflt_Xb))
                PP_Hazard[i] = 1 - math.exp(-h0_pp[i-1] * math.exp(PP_Xb))
                Surv_Prob[i] = Surv_Prob[i-1] * (1 - Dflt_Hazard[i] - PP_Hazard[i])
                
                Est_Period_Dflts[i] = Dflt_Hazard[i] * Surv_Prob[i-1]
                Est_Period_PPs[i] = PP_Hazard[i] * Surv_Prob[i-1]

                # Update cumulative probabilities
                default_rate[j][k] += Est_Period_Dflts[i]
                pp_rate[j][k] += Est_Period_PPs[i]
    dt_prob = pd.DataFrame(default_rate)
    pp_prob = pd.DataFrame(pp_rate)
    return dt_prob, pp_prob

def parallel_simulate_loan_defaults(path_index, data, rate15y_path, rate30y_path, dt_params, pp_params, h0_dt, h0_pp, hpa_path, msa_state, msa_state_adj, original_rate):
    """
    Adjusted wrapper function for simulating loan defaults and prepayments in parallel, focusing on path subsets.
    """
    # Adjust to process a single path based on path_index
    subset_rate15y_path = rate15y_path.iloc[:, path_index:path_index+1]
    subset_rate30y_path = rate30y_path.iloc[:, path_index:path_index+1]
    subset_hpa_path = hpa_path.iloc[:, path_index:path_index+1]
    
    logger.info(f'Processing path index {path_index}')
    
    # Note: 'data', 'msa_state', and 'original_rate' are used without subsetting since all loans are processed
    return simulate_loan_defaults(len(data), data, subset_rate15y_path, subset_rate30y_path, dt_params, pp_params, h0_dt, h0_pp, subset_hpa_path, msa_state, msa_state_adj, original_rate)

def main_simulation():
    logger.info("Starting the simulation.")
    base_path = Path.cwd()

    logger.info("Loading and preparing data.")
    try:
        # Reading and preparing data
        dt_params = pd.read_csv(base_path / 'cloglog' / 'default_coef_final.csv', index_col=0)
        dt_params.columns=['Estimate']
        logger.info('loaded default')
        pp_params = pd.read_csv(base_path / 'cloglog' / 'prepaid_coef_final.csv', index_col=0)
        pp_params.columns=['Estimate']
        logger.info('loaded prepay')
        loans = pd.read_csv(base_path / 'loans' / 'test_data_final.csv', index_col=0)
        logger.info('loaded loans')
        df_msa = pd.read_csv(base_path / 'msa_data' / 'HPI_PO_metro_name.csv', index_col=None)
        logger.info('loaded MSA')
        rate15y_path = pd.read_csv(base_path / 'rate_HPI_process' / '15_rate.csv', index_col=None)
        logger.info('loaded 15 year rate')
        rate30y_path = pd.read_csv(base_path / 'rate_HPI_process' / '30_rate.csv', index_col=None)
        logger.info('loaded 30 year rate')
        hpa_path = pd.read_csv(base_path / 'rate_HPI_process' / 'HPI.csv', index_col=None)
        logger.info('loaded home path')
        msa_state_adj = pd.read_csv(base_path / 'rate_HPI_process' / 'dispersion_matrix.csv', index_col=None)
        logger.info('loaded dispersion matrix')

        # Limit to the first 80 paths and 100 loans
        rate15y_path = rate15y_path.iloc[:, :80]
        rate30y_path = rate30y_path.iloc[:, :80]
        hpa_path = hpa_path.iloc[:, :80]
        #loans = loans.iloc[:100,:]
        original_rate = loans['orig_interest_rate'].tolist()

        loans = pd.merge(loans, df_msa[['MSA', 'metro_name']], on='MSA', how='left')
        loans = loans.rename(columns={'metro_name': 'msa_state_name'})
        loans['msa_state_name'] = loans['msa_state_name'].fillna(loans['prop_state'])
        msa_state = loans['msa_state_name'].tolist()
        h0_dt = []
        h0_pp = []
        pp_time_param = pp_params.iloc[:40] # dataframe
        dt_time_param = dt_params.iloc[:40] # dataframe
        for i in dt_time_param.iloc[:,0]:
            h0_dt.append(math.exp(float(i)))
        for i in pp_time_param.iloc[:,0]:
            h0_pp.append(math.exp(float(i)))
    except Exception as e:
        logger.error(f"Error during data loading: {e}")
        raise

    n = len(loans)  # Total number of loans
    n= int(n)
    num_workers = os.cpu_count()  # Using all available CPU cores
    logger.info(f"Executing simulation in parallel across {num_workers} workers.")
    total_paths = rate15y_path.shape[1]  # Number of columns

    total_paths = rate15y_path.shape[1]  # Number of columns
    path_indices = range(total_paths)  # Generate a range for the total number of paths


    results = []
    start_times = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
    # Process paths in batches of the number of workers
        for i in range(0, total_paths, num_workers):
            batch_paths = path_indices[i:i+num_workers]
            futures = {}
            start_times = {}

            # Submitting tasks for each path in the current batch
            for path_index in batch_paths:
                # Adjust `parallel_simulate_loan_defaults` to accept a single path index (or a small range if needed)
                future = executor.submit(parallel_simulate_loan_defaults, path_index, loans, rate15y_path, rate30y_path, dt_params, pp_params, h0_dt, h0_pp, hpa_path, msa_state, msa_state_adj, original_rate)
                futures[future] = path_index
                start_times[future] = time.time()
        
             # Wait for the current batch of futures to complete before moving on
            for future in concurrent.futures.as_completed(futures):
                duration = time.time() - start_times[future]
                path_index = futures[future]
                logger.info(f"Path {path_index} completed in {duration:.2f} seconds.")
                results.append(future.result())

    logger.info("Simulation completed successfully.")
    
    final_default_rate, final_pp_rate = pd.concat([r[0] for r in results], axis=1), pd.concat([r[1] for r in results],axis=1)
    logger.info("Results aggregation completed successfully.")
    # Use the first row as headers
    final_default_rate.columns = final_default_rate.iloc[0]
    final_pp_rate.columns = final_pp_rate.iloc[0]

    # Drop the first row now that it's been set as headers
    final_default_rate = final_default_rate.iloc[1:].reset_index(drop=True)
    final_pp_rate = final_pp_rate.iloc[1:].reset_index(drop=True)

    # Set new integer-based column labels
    final_default_rate.columns = range(final_default_rate.shape[1])
    final_pp_rate.columns = range(final_pp_rate.shape[1])

    # Define the base path for the results folder
    results_path = Path('results')
    # Create the results folder if it doesn't exist
    results_path.mkdir(parents=True, exist_ok=True)

    # Define file paths for the CSV files
    default_rate_csv = results_path / 'final_default_rate.csv'
    pp_rate_csv = results_path / 'final_pp_rate.csv'

    # Save the DataFrames to CSV files
    final_default_rate.to_csv(default_rate_csv, index=False)
    final_pp_rate.to_csv(pp_rate_csv, index=False)

    logger.info(f"Saved final default rate to {default_rate_csv}")
    logger.info(f"Saved final prepay rate to {pp_rate_csv}")


if __name__ == "__main__":
        main_simulation()
        logger.info("Simulation pipeline completed successfully.")
