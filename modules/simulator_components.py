import math
import tqdm
import pandas as pd

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

def simulate_loan_defaults(n, data, dt_params, pp_params, h0_dt, h0_pp, paths, rate_paths, hpa_path, msa_state_adj, original_rate):
    # Initialize the arrays for default and prepayment rates
    default_rate, pp_rate = [0] * n, [0] * n
    for j in tqdm(range(n)):  # Simulate for each loan
        # Calculation logic here...
        pass  # Replace with detailed simulation steps
    
    return pd.DataFrame(default_rate), pd.DataFrame(pp_rate)

def calculate_discounted_losses(n, monthly_balance, dt_prob, annual_discount_rate=0.05, severity=0.25):
    quarterly_discount_rate = (1 + annual_discount_rate) ** (1/4) - 1
    results_discounted = {}
    for i in tqdm(range(n)):
        # Discounted loss calculation logic here...
        pass
    return pd.DataFrame(results_discounted)
