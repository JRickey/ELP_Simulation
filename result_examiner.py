import pandas as pd
import numpy as np
import seaborn as sns
import os
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

def visualize_results():
    base_path = Path.cwd()
    dt_prob = pd.read_csv(os.path.join(base_path, 'SimulationResults', 'final_default_rate.csv'))
    pp_prob = pd.read_csv(os.path.join(base_path, 'SimulationResults', 'final_pp_rate.csv'))
    dt_params = pd.read_csv(os.path.join(base_path, 'cloglog', 'default_coef_final.csv'), index_col=0)
    dt_params.columns = ['Estimate']
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
    loans = pd.read_csv(os.path.join(base_path, 'loans', 'loans.csv'), index_col=0)
    df_msa = pd.read_csv(os.path.join(base_path, 'msa_data', 'HPI_PO_metro_name.csv'), index_col=None)
    rate15y_path = pd.read_csv(os.path.join(base_path, 'rate_HPI_process', '15_rate.csv'), index_col=None)
    rate30y_path = pd.read_csv(os.path.join(base_path, 'rate_HPI_process', '30_rate.csv'), index_col=None)
    hpa_path = pd.read_csv(os.path.join(base_path, 'rate_HPI_process', 'HPI.csv'), index_col=None)
    msa_state_adj = pd.read_csv(os.path.join(base_path, 'rate_HPI_process', 'dispersion_matrix.csv'), index_col=None)

    data = pd.read_csv(Path.cwd() / 'loans' / 'loans.csv')
    data = pd.merge(data, df_msa[['MSA', 'metro_name']], on='MSA', how='left')
    data = data.rename(columns={'metro_name': 'msa_state_name'})
    data['msa_state_name'] = data['msa_state_name'].fillna(data['prop_state'])
    msa_state = data['msa_state_name'].tolist()
    data.head()

    Monthly_Balance_Cas = []
    sumUPB = 0

    for index, row in data.iterrows():
        sumUPB += row['oUPB']  # Sum up the original UPB for all loans, if needed.
        Loan_Amount = row['oUPB']
        Interest_rate = row['orig_interest_rate'] 
        loan_term_years = row['loan_term_years']  
        if abs(loan_term_years - 15) <= abs(loan_term_years - 30):
            Loan_Term = 180  # 15 years * 12 months
        else:
            Loan_Term = 360  # 30 years * 12 months

        R = 1 + (Interest_rate) / (12 * 100)  # Monthly interest rate
        X = Loan_Amount * (R**Loan_Term) * (1 - R) / (1 - R**Loan_Term)  # Monthly payment
        Monthly_Balance = []

        for i in range(1, 123):  # Adjusted loop to run through the quarters
            Interest = Loan_Amount * (R - 1)
            Loan_Amount -= (X - Interest)
            if i % 3 == 0:  # Calculate every quarter, starting from the 3rd month
                Monthly_Balance.append(Loan_Amount)

        Monthly_Balance_Cas.append(Monthly_Balance)
        
    Monthly_Balance_Cas=pd.DataFrame(Monthly_Balance_Cas)

    annual_discount_rate = 0.05
    quarterly_discount_rate = (1 + annual_discount_rate)**(1/4) - 1

    results_discounted = {}

    for i in tqdm(range(len(data))):  # Iterate over loans
        UPB = pd.Series(Monthly_Balance_Cas.iloc[i, :]) * 0.25 # loss severity  # Get UPB for the loan
        loan_results_discounted = []  # Initialize a list to store discounted total losses for this loan
        
        for j in range(500):  # Iterate over paths
            prob = pd.Series(dt_prob.iloc[i, j]) # multiple marginal loss rate for the path
            loss = UPB * prob  # Calculate the loss
            quarters = np.arange(1, len(loss) + 1)
            discounted_loss = loss / ((1 + quarterly_discount_rate) ** quarters)
            total_discounted_loss = discounted_loss.sum()
            loan_results_discounted.append(total_discounted_loss)  # Store the total discounted loss for this path
        
        results_discounted[i] = loan_results_discounted  # Store the discounted results for this loan

    # Each cell now contains the total discounted loss for that loan-path combination
    dt_total_discounted_loss = pd.DataFrame(results_discounted)

    # Transpose the DataFrame so rows are loans and columns are paths
    dt_total_discounted_loss_transposed = dt_total_discounted_loss.T

    # Display the first few rows to verify the structure
    dt_total_discounted_loss_transposed.head()

    # Calculate the sum of discounted losses for each path across all loans
    total_loss_per_path = dt_total_discounted_loss_transposed.sum()

    # Calculate the loss percentage of each path as a fraction of sumUPB
    loss_percentage = (total_loss_per_path / sumUPB) * 100

    # Convert the series to a numpy array for plotting
    loss_percentage_array = loss_percentage.values

    # Plotting
    plt.figure(figsize=(8, 8))
    sns.distplot(loss_percentage_array, hist=True, kde=True, bins=40, color='blue', 
                hist_kws={'edgecolor':'black'}, kde_kws={'linewidth': 2.5})

    # It's better to dynamically set the max_y_value based on the plot data
    max_y_value = plt.gca().get_ylim()[1]  # Dynamically get the current max y value

    # Adding vertical lines for percentiles
    percentiles = [0.5, 0.9, 0.95, 0.99]
    colors = ['red', 'blue', 'green', 'orange']
    labels = ['50th percentile', '90th percentile', '95th percentile', '99th percentile']

    for percentile, color, label in zip(percentiles, colors, labels):
        plt.vlines(np.percentile(loss_percentage_array, percentile*100), 0, max_y_value, color=color, linestyles='dashed', label=label)

    plt.ylabel('Frequency')
    plt.xlabel("Loss of portfolio as % of oUPB")
    plt.title("Loss Distribution")
    plt.legend()

    plt.show()

    # After calculating loss_percentage
    result_dt = pd.DataFrame(loss_percentage)

    # Knowing the 99th percentile losses
    print("50th percentile:", result_dt.quantile(0.50).values)
    print("90th percentile:", result_dt.quantile(0.90).values)
    print("95th percentile:", result_dt.quantile(0.95).values)
    print("99th percentile:", result_dt.quantile(0.99).values)

    result_dt = result_dt.sort_values(by=0)
    result_dt.head()

    annual_discount_rate = 0.05
    quarterly_discount_rate = (1 + annual_discount_rate)**(1/4) - 1

    results_discounted = {}

    for i in tqdm(range(len(data))):  # Iterate over loans
        UPB = pd.Series(Monthly_Balance_Cas.iloc[i, :])  # Get UPB for the loan
        loan_results_discounted = []  # Initialize a list to store discounted total losses for this loan
        
        for j in range(500):  # Iterate over paths
            prob = pd.Series(pp_prob.iloc[i, j]) * 0.25  # multiple loss severity for the path
            loss = UPB * prob  # Calculate the loss
            quarters = np.arange(1, len(loss) + 1)
            discounted_loss = loss / ((1 + quarterly_discount_rate) ** quarters)
            total_discounted_loss = discounted_loss.sum()
            loan_results_discounted.append(total_discounted_loss)  # Store the total discounted loss for this path
        
        results_discounted[i] = loan_results_discounted  # Store the discounted results for this loan

    # Each cell now contains the total discounted loss for that loan-path combination
    pp_total_discounted_loss = pd.DataFrame(results_discounted)

    # Transpose the DataFrame so rows are loans and columns are paths
    pp_total_discounted_loss_transposed = pp_total_discounted_loss.T

    # Calculate the sum of discounted losses for each path across all loans
    total_loss_per_path = pp_total_discounted_loss_transposed.sum()

    # Calculate the loss percentage of each path as a fraction of sumUPB
    loss_percentage = (total_loss_per_path / sumUPB) * 100

    # Convert the series to a numpy array for plotting
    loss_percentage_array = loss_percentage.values

    # Plotting
    plt.figure(figsize=(8, 8))
    sns.distplot(loss_percentage_array, hist=True, kde=True, bins=40, color='blue', 
                hist_kws={'edgecolor':'black'}, kde_kws={'linewidth': 2.5})

    # It's better to dynamically set the max_y_value based on the plot data
    max_y_value = plt.gca().get_ylim()[1]  # Dynamically get the current max y value

    # Adding vertical lines for percentiles
    percentiles = [0.5, 0.9, 0.95, 0.99]
    colors = ['red', 'blue', 'green', 'orange']
    labels = ['50th percentile', '90th percentile', '95th percentile', '99th percentile']

    for percentile, color, label in zip(percentiles, colors, labels):
        plt.vlines(np.percentile(loss_percentage_array, percentile*100), 0, max_y_value, color=color, linestyles='dashed', label=label)

    plt.ylabel('Frequency')
    plt.xlabel("Loss of portfolio as % of oUPB")
    plt.title("Loss Distribution")
    plt.legend()

    plt.show()
    # Calculate the sum of discounted losses for each path across all loans for both default and prepaid
    combined_total_loss_per_path = dt_total_discounted_loss_transposed.sum() + pp_total_discounted_loss_transposed.sum()

    # Calculate the loss percentage of the combined path as a fraction of sumUPB
    combined_loss_percentage = (combined_total_loss_per_path / sumUPB) * 100

    # Convert the series to a numpy array for plotting
    combined_loss_percentage_array = combined_loss_percentage.values

    # Plotting the combined loss distribution
    plt.figure(figsize=(8, 8))
    sns.histplot(combined_loss_percentage_array, kde=True, bins=40, color='blue', edgecolor='black', linewidth=2.5)

    # It's better to dynamically set the max_y_value based on the plot data
    combined_max_y_value = plt.gca().get_ylim()[1]  # Dynamically get the current max y value

    # Adding vertical lines for percentiles of the combined loss distribution
    percentiles = [0.5, 0.9, 0.95, 0.99]
    colors = ['red', 'blue', 'green', 'orange']
    labels = ['50th percentile', '90th percentile', '95th percentile', '99th percentile']

    for percentile, color, label in zip(percentiles, colors, labels):
        plt.axvline(x=np.percentile(combined_loss_percentage_array, percentile*100), ymin=0, ymax=combined_max_y_value, color=color, linestyle='dashed', label=label)

    plt.ylabel('Frequency')
    plt.xlabel("Combined Loss of portfolio as % of OUPB")
    plt.title("Combined Loss Distribution")
    plt.legend()

    plt.show()

if __name__ == "__main__":
        visualize_results()