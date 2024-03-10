import math
import tqdm
import pandas as pd
import numpy as np

def compute_h0(time_param):
    return [math.exp(i) for i in time_param.iloc[:,0]]

def compute_cheng(x, y):
    return sum(x[i] * y[x.index[i]] for i in range(len(x)))

def calculate_loan_payments(data, n_periods=123, frequency=3):
    monthly_balances = []
    for index, row in data.iterrows():
        loan_amount = row['oUPB']
        interest_rate = row['orig_interest_rate'] / (12 * 100)
        loan_term = 180 if abs(row['loan_term_years'] - 15) <= abs(row['loan_term_years'] - 30) else 360
        monthly_payment = loan_amount * interest_rate / (1 - (1 + interest_rate) ** -loan_term)
        
        balance = loan_amount
        loan_balances = []
        for i in range(1, n_periods + 1):
            interest = balance * interest_rate
            balance -= (monthly_payment - interest)
            if i % frequency == 0:
                loan_balances.append(balance)
        
        monthly_balances.append(loan_balances)
    return pd.DataFrame(monthly_balances)

def simulate_loan_defaults(n, data, rate15y_path, rate30y_path, dt_params, pp_params, h0_dt, h0_pp, hpa_path, msa_state, msa_state_adj, original_rate):
    """
    Simulate loan defaults and prepayments for n loans using given parameters.

    Parameters:
    - n: Number of loans to simulate.
    - data: DataFrame containing loan data.
    - dt_params: Parameters for default timing.
    - pp_params: Parameters for prepayment timing.
    - h0_dt: Baseline hazard rate for default timing.
    - h0_pp: Baseline hazard rate for prepayment timing.
    - hpa_path: Home Price Appreciation path data.
    - msa_state: original msa values
    - msa_state_adj: MSA state adjustment factors.
    - original_rate: Original interest rates of the loans.
    
    Returns:
    - Tuple of DataFrames for default rates and prepayment rates.
    """
    
    # Initialize the arrays for default and prepayment rates
    default_rate = [0] * n
    pp_rate = [0] * n
    
    # Iterate through each loan
    for j in tqdm(range(n)):
        # Extract necessary data for the current loan
        msa_state_adj_temp = msa_state_adj[msa_state[j]].tolist()
        loan_term_years = data.loc[j, 'loan_term_years']
        original_interest_rate = original_rate[j]

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

            # Simulate each quarter
            for i in range(1, 41):
                # Calculation logic for default and prepayment probabilities
                hpa_temp = hpa_path_temp[i-1]
                msa = msa_state_adj_temp[i-1]

                Dflt_Xb = dt_params['unchanged'] + hpa_temp * dt_params['hpa_param'] + msa * dt_params['msa_param']
                PP_Xb = pp_params['unchanged'] + hpa_temp * pp_params['hpa_param'] + (original_interest_rate - rate_temp[i-1]) * pp_params['refi_param']
                
                Dflt_Hazard[i] = 1 - math.exp(-h0_dt[i-1] * math.exp(Dflt_Xb))
                PP_Hazard[i] = 1 - math.exp(-h0_pp[i-1] * math.exp(PP_Xb))
                Surv_Prob[i] = Surv_Prob[i-1] * (1 - Dflt_Hazard[i] - PP_Hazard[i])
                
                Est_Period_Dflts[i] = Dflt_Hazard[i] * Surv_Prob[i-1]
                Est_Period_PPs[i] = PP_Hazard[i] * Surv_Prob[i-1]

                # Update cumulative probabilities
                default_rate[j][k] += Est_Period_Dflts[i]
                pp_rate[j][k] += Est_Period_PPs[i]

    # Convert lists to DataFrames for easier manipulation and analysis
    dt_prob = pd.DataFrame(default_rate)
    pp_prob = pd.DataFrame(pp_rate)
    
    return dt_prob, pp_prob


def calculate_discounted_losses(monthly_balance, dt_prob, annual_discount_rate=0.05, severity=0.25):
    """
    Calculate the discounted losses for each loan across different paths.
    
    Parameters:
    - monthly_balance: DataFrame containing the monthly balance for each loan.
    - dt_prob: DataFrame containing the default probabilities for each loan across different paths.
    - annual_discount_rate: The annual discount rate to be used in discounting the losses.
    - severity: The loss severity rate to be applied to the unpaid balance (UPB).
    
    Returns:
    - A DataFrame containing the total discounted loss for each loan-path combination,
      transposed so rows are loans and columns are paths.
    """
    
    # Calculate the quarterly discount rate based on the annual discount rate
    quarterly_discount_rate = (1 + annual_discount_rate) ** (1/4) - 1
    
    # Initialize a dictionary to store the discounted results for each loan
    results_discounted = {}
    
    # Iterate over each loan
    for i in tqdm(range(len(monthly_balance))):
        # Calculate the Unpaid Principal Balance (UPB) by applying the loss severity
        UPB = pd.Series(monthly_balance.iloc[i, :]) * severity
        
        # Initialize a list to store discounted total losses for this loan across all paths
        loan_results_discounted = []
        
        # Iterate over each path
        for j in range(len(dt_prob.columns)):
            # Extract the default probability for the current path
            prob = pd.Series(dt_prob.iloc[i, j])
            
            # Calculate the loss by multiplying the UPB by the default probability
            loss = UPB * prob
            
            # Generate an array of quarters for discounting purposes
            quarters = np.arange(1, len(loss) + 1)
            
            # Calculate the discounted loss for each quarter
            discounted_loss = loss / ((1 + quarterly_discount_rate) ** quarters)
            
            # Sum up the discounted losses to get the total discounted loss for the current path
            total_discounted_loss = discounted_loss.sum()
            
            # Store the total discounted loss for this path
            loan_results_discounted.append(total_discounted_loss)
        
        # Store the discounted results for this loan
        results_discounted[i] = loan_results_discounted
    
    # Convert the results dictionary into a DataFrame
    dt_total_discounted_loss = pd.DataFrame(results_discounted)
    
    # Transpose the DataFrame so that rows represent loans and columns represent paths
    dt_total_discounted_loss_transposed = dt_total_discounted_loss.T
    
    return dt_total_discounted_loss_transposed

