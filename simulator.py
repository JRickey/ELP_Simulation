from pathlib import Path
import pandas as pd
import concurrent.futures
import math
import time
import numpy as np
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from tqdm import tqdm
import os
import logging
import sys

# Determine if the app is "frozen" (packaged) or running in a development environment
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    # If the app is running in a bundled environment, use the _MEIPASS attribute
    # provided by PyInstaller to find resources within the app bundle.
    # For py2app, the resource path is slightly different; adjust as necessary.
    base_path = Path(sys._MEIPASS)
else:
    # If running in a development environment, use the current working directory
    base_path = Path(__file__).parent

def configure_logger():
    # Create or get the logger
    logger = logging.getLogger(__name__)  # __name__ gives each file its logger, or use a custom name
    logger.setLevel(logging.DEBUG)  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    # Create handlers (console and file)
    c_handler = logging.StreamHandler()  # Console handler
    log_file_path = os.path.expanduser('~/Desktop/simulation.log')  # Saves log to the user's Desktop
    f_handler = logging.FileHandler(log_file_path)
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

def cheng_vectorized(df, coef):
    # Align DataFrame columns with coefficients
    # Ensure that we only select columns that exist in both the DataFrame and the coefficients
    aligned_df = df.loc[:, df.columns.intersection(coef.index)]
    # Reorder df columns to match coef order
    aligned_df = aligned_df[coef.index]
    # Perform the dot product
    result = aligned_df.dot(coef)
    return result

def simulate_loan_defaults(n, data, rate15y_path, rate30y_path, dt_params, pp_params, h0_dt, h0_pp, hpa_path, msa_state, msa_state_adj, original_rate, pathnum, Dflt_Xb_unchanged, PP_Xb_unchanged):
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
    dt_hpa_param = dt_params.iloc[40,0] # value
    pp_hpa_param = pp_params.iloc[40,0] # value
    pp_refi_param = pp_params.iloc[41,0] # value

    # Iterate through each loan
    for j in tqdm(range(n), desc=f'Path number #{pathnum}'):
        # Extract necessary data for the current loan
        msa_state_adj_temp = msa_state_adj[msa_state[j]].tolist()
        loan_term_years = data.loc[j, 'loan_term_years']
        spread_orig = data['spread_orig'].tolist()

        # Determine if loan is closer to 15 or 30 years and select the rate path accordingly
        diff_15 = abs(loan_term_years - 15)
        diff_30 = abs(loan_term_years - 30)
        condition = diff_15 <= diff_30
        
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
            sum_rate = 0

            current_dflt = 0
            current_PP = 0
            # Simulate each quarter
            for i in range(1, 41):
                # Calculation logic for default and prepayment probabilities
                #logging.info(f"Processing quarter {i} for loan {j+1}, path {k+1}")
                hpa_temp = hpa_path_temp[i-1]
                msa = msa_state_adj_temp[i-1]
                sum_rate = sum_rate + rate_temp[i-1]
                Dflt_Xb = Dflt_Xb_unchanged[j] + hpa_temp * dt_hpa_param + msa * dt_hpa_param
                PP_Xb = PP_Xb_unchanged[j] + (hpa_temp + msa) * pp_hpa_param + (spread_orig[j+ start] - sum_rate)* pp_refi_param
                
                Dflt_Hazard[i] = 1 - math.exp(-h0_dt[i-1] * math.exp(Dflt_Xb))
                PP_Hazard[i] = 1 - math.exp(-h0_pp[i-1] * math.exp(PP_Xb))
                Surv_Prob[i] = Surv_Prob[i-1] * (1 - Dflt_Hazard[i] - PP_Hazard[i])
                
                Est_Period_Dflts[i] = Dflt_Hazard[i] * Surv_Prob[i-1]
                Est_Period_PPs[i] = PP_Hazard[i] * Surv_Prob[i-1]

                # Update cumulative probabilities
                default_rate[j][k] = Est_Period_Dflts[i]
                pp_rate[j][k] = Est_Period_PPs[i]

    dt_prob = pd.DataFrame(default_rate)
    pp_prob = pd.DataFrame(pp_rate)
    return dt_prob, pp_prob

def parallel_simulate_loan_defaults(path_index, data, rate15y_path, rate30y_path, dt_params, pp_params, h0_dt, h0_pp, hpa_path, msa_state, msa_state_adj, original_rate, Dflt_Xb_unchanged, PP_Xb_unchanged):
    """
    Adjusted wrapper function for simulating loan defaults and prepayments in parallel, focusing on path subsets.
    """
    # Adjust to process a single path based on path_index
    subset_rate15y_path = rate15y_path.iloc[:, path_index:path_index+1]
    subset_rate30y_path = rate30y_path.iloc[:, path_index:path_index+1]
    subset_hpa_path = hpa_path.iloc[:, path_index:path_index+1]
    
    # Note: 'data', 'msa_state', and 'original_rate' are used without subsetting since all loans are processed
    return simulate_loan_defaults(len(data), data, subset_rate15y_path, subset_rate30y_path, dt_params, pp_params, h0_dt, h0_pp, subset_hpa_path, msa_state, msa_state_adj, original_rate, path_index, Dflt_Xb_unchanged, PP_Xb_unchanged)

 # Function to generate chunks of loans
def chunked_loans(loans, chunk_size):
    for i in range(0, len(loans), chunk_size):
        yield loans.iloc[i:i + chunk_size].reset_index(drop=True)
    
def main_simulation():
    logger.info("Starting the simulation.")
    main_start_time = time.time()

    logger.info("Loading and preparing data.")
    try:
        # Reading and preparing data
        import os

        dt_params = pd.read_csv(os.path.join(base_path, 'cloglog', 'default_coef_final.csv'), index_col=0)
        dt_params.columns = ['Estimate']
        logger.info('loaded default')

        pp_params = pd.read_csv(os.path.join(base_path, 'cloglog', 'prepaid_coef_final.csv'), index_col=0)
        pp_params.columns = ['Estimate']
        # Assuming pp_params is already loaded as per your code
        # Check the current index name at row 43 (keeping in mind Python uses 0-based indexing)
        current_index_name_at_row_41 = pp_params.index[41]

        # If the current index name is 'refi_incentive', change it to 'rate_spread'
        if current_index_name_at_row_41 == 'refi_incentive':
        # Create a new index object with the changed name
            new_index = pp_params.index.tolist()
            new_index[41] = 'rate_spread'  # Changing the name at row 43
            pp_params.index = new_index
        else:
            logger.info(f"Index name at row 42 is not 'refi_incentive'. It is '{current_index_name_at_row_41}'")

        # Verify the change
        logger.info(pp_params.index[41])  # This should print 'rate_spread'

        logger.info('loaded prepay')

        loans = pd.read_csv(os.path.join(base_path, 'loans', 'loans.csv'), index_col=0)
        logger.info('loaded loans')

        df_msa = pd.read_csv(os.path.join(base_path, 'msa_data', 'HPI_PO_metro_name.csv'), index_col=None)
        logger.info('loaded MSA')

        rate15y_path = pd.read_csv(os.path.join(base_path, 'rate_HPI_process', '15_rate.csv'), index_col=None)
        logger.info('loaded 15 year rate')

        rate30y_path = pd.read_csv(os.path.join(base_path, 'rate_HPI_process', '30_rate.csv'), index_col=None)
        logger.info('loaded 30 year rate')

        hpa_path = pd.read_csv(os.path.join(base_path, 'rate_HPI_process', 'HPI.csv'), index_col=None)
        logger.info('loaded home path')

        msa_state_adj = pd.read_csv(os.path.join(base_path, 'rate_HPI_process', 'dispersion_matrix.csv'), index_col=None)
        logger.info('loaded dispersion matrix')


        # Limit to the first 80 paths and 100 loans
        rate15y_path = rate15y_path.iloc[:, :8]
        rate30y_path = rate30y_path.iloc[:, :8]
        hpa_path = hpa_path.iloc[:, :8]
        loans = loans.iloc[:400,:]
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

    logger.info('Precomputing unchanged part for all loans')
    pp_coef_param = pp_params.iloc[42:]['Estimate']
    dt_coef_param = dt_params.iloc[41:]['Estimate']
    # Given indices of the DataFrame
    loan_columns = ['FICO', 'first_pmt', 'fthb', 'maturity_dte', 'MSA', 'num_units',
                'oCLTV', 'oDTI', 'oUPB', 'oLTV', 'orig_interest_rate', 'prop_state',
                'orig_loan_term', 'num_borrowers', 'report_month', 'delinquency_status',
                'zero_bal', 'zero_bal_date', 'Default', 'Prepaid', 'quarter',
                'Current_UPB', 'Prev_Quarter_UPB', 'RUPB', 'year_quarter', 'orig_year',
                'pmms30', 'pmms15', 'loan_term_years', 'pmms', 'refi_incentive',
                'index_sa', 'median_UPB_MSA', 'median_UPB_State', 'spread_orig',
                'loan_term_loan_term_1015', 'loan_term_loan_term_15',
                'loan_term_loan_term_20', 'loan_term_loan_term_25',
                'loan_term_loan_term_2530', 'loan_term_loan_term_30', 'multi_borrowers',
                'multi_units', 'channel_B', 'channel_C', 'channel_R', 'channel_T',
                'loan_purpose_C', 'loan_purpose_N', 'loan_purpose_P', 'occ_status_I',
                'occ_status_P', 'occ_status_S', 'prop_type_CO', 'prop_type_CP',
                'prop_type_MH', 'prop_type_PU', 'prop_type_SF', 'HPA', 'HPI_LTV',
                'year', 'time_seq', 'FICO_Below_680', 'FICO_Above_680', 'DTI_Below_30',
                'DTI_Above_30', 'CLTV_below_80', 'CLTV_Between_80_and_95',
                'CLTV_above_95', 'DTI_Below_15', 'DTI_Between_15_and_30',
                'DTI_Between_30_and_43', 'DTI_above_43', 'FICO_Between_680_and_800',
                'FICO_Above_800', 'CLTV_above_80', 'msa_state_name']

    # For dt_coef_param
    dt_columns_needed = [col for col in dt_coef_param.index if col in loan_columns]
    # For pp_coef_param
    pp_columns_needed = [col for col in pp_coef_param.index if col in loan_columns]

    # Convert the pandas dataframe to a dask dataframe

    with ProgressBar():
        st_time = time.time()
        meta=pd.Series(dtype='float')
        dt_dask_loans = dd.from_pandas(loans[dt_columns_needed], npartitions=os.cpu_count())
        # Adjusting the computation for the default model coefficients
        Dflt_Xb_unchanged = dt_dask_loans.map_partitions(lambda df: cheng_vectorized(df, dt_coef_param), meta=meta).compute(scheduler='processes')
        logger.info(f'dflt_xb_unchanged calculated in {time.time() - st_time} seconds')
        
        st_time = time.time()
        # Adjusting the computation for the prepayment model coefficients
        pp_dask_loans = dd.from_pandas(loans[pp_columns_needed], npartitions=os.cpu_count())
        PP_Xb_unchanged = pp_dask_loans.map_partitions(lambda df: cheng_vectorized(df, pp_coef_param), meta=meta).compute(scheduler='processes')
        logger.info(f'pp_xb_unchanged calculated in {time.time() - st_time} seconds')
                
    results = []
    start_times = {}
    loans_chunk_size = max(len(loans) // (10), 1)

    final_default_rate = []
    final_pp_rate = []
    for chunk in chunked_loans(loans, loans_chunk_size):
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Process paths in batches of the number of workers
            for i in tqdm(range(0, total_paths, num_workers), desc='Main Progress'):
                batch_paths = path_indices[i:i+num_workers]
                futures = {}
                start_times = {}

                # Submitting tasks for each path in the current batch
                for path_index in batch_paths:
                    # Adjust `parallel_simulate_loan_defaults` to accept a single path index (or a small range if needed)
                    future = executor.submit(parallel_simulate_loan_defaults, path_index, chunk, rate15y_path, rate30y_path, dt_params, pp_params, h0_dt, h0_pp, hpa_path, msa_state, msa_state_adj, original_rate, Dflt_Xb_unchanged, PP_Xb_unchanged)
                    futures[future] = path_index
                    start_times[future] = time.time()
            
                # Wait for the current batch of futures to complete before moving on
                for future in concurrent.futures.as_completed(futures):
                    duration = time.time() - start_times[future]
                    path_index = futures[future]
                    logger.info(f"Path {path_index} completed in {duration:.2f} seconds.")
                    results.append(future.result())
        
        sub_default_rate, sub_pp_rate = pd.concat([r[0] for r in results], axis=1), pd.concat([r[1] for r in results],axis=1)

    final_default_rate.append(sub_default_rate)
    final_pp_rate.append(sub_pp_rate)
    final_default_rate = pd.concat(final_default_rate, axis=1)
    final_pp_rate = pd.concat(final_pp_rate, axis=1)
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
    desktop_path = Path.home() / 'Desktop'
    results_folder_name = 'SimulationResults'  # Name of the results folder to create on the Desktop
    results_path = desktop_path / results_folder_name

    # Create the results folder on the Desktop if it doesn't exist
    results_path.mkdir(parents=True, exist_ok=True)

    # Define file paths for the CSV files within the results folder on the Desktop
    default_rate_csv = results_path / 'final_default_rate.csv'
    pp_rate_csv = results_path / 'final_pp_rate.csv'

    # Save the DataFrames to CSV files
    final_default_rate.to_csv(default_rate_csv, index=False)
    final_pp_rate.to_csv(pp_rate_csv, index=False)

    logger.info(f"Saved final default rate to {default_rate_csv}")
    logger.info(f"Saved final prepay rate to {pp_rate_csv}")
    logger.info(f'Simulation completed in {time.time() - main_start_time}')

if __name__ == "__main__":
        main_simulation()

