import pandas as pd
import math
import numpy as np

# --- Step 1: Inputs & Constants ---

# Constant epsilon to prevent ln(0)
EPSILON = 1e-12
def load_config_and_data(data_csv_path, vars_csv_path):
    """
    Loads the main data from environmental.csv and the variables config.
    Filters for variables that are marked "Include" == 1.
    """
    try:
        # Load the data from environmental.csv
        df_data = pd.read_csv(data_csv_path)
        df_vars = pd.read_csv(vars_csv_path)
    except FileNotFoundError as e:
        print(f"Error: Could not find file {e.filename}")
        return None, None, None

    df_vars_included = df_vars[df_vars["Include (1/0)"] == 1].copy()
    
    if df_vars_included.empty:
        print("Error: No variables marked for inclusion in variables.csv")
        return None, None, None

    # Use Display_Name
    feature_columns = df_vars_included["Display_Name"].tolist()
    
    missing_cols = [col for col in feature_columns if col not in df_data.columns]
    if missing_cols:
        print(f"Error: Data file '{data_csv_path}' is missing required columns: {missing_cols}")
        print("Please check your 'variables.csv' and 'environmental.csv'.")
        return None, None, None

    df_features = df_data[feature_columns]
    
    return df_features, df_vars_included, len(df_features)

def normalize_data(df_features, df_vars_config):
    """
    Step 2: Normalization (Z_ij)
    """
    df_normalized = pd.DataFrame()
    
    for _, var_row in df_vars_config.iterrows():
        col_name = var_row["Display_Name"] 
        min_val = var_row["Min"]
        max_val = var_row["Max"]
        norm_type = var_row["Norm_Type (inc/dec)"]
        
        x = df_features[col_name]
        
        if max_val - min_val == 0:
            z = pd.Series([0] * len(x), index=x.index)
        elif norm_type == 'inc':
            z = (x - min_val) / (max_val - min_val)
        elif norm_type == 'dec':
            z = (max_val - x) / (max_val - min_val)
        else:
            z = pd.Series([0] * len(x), index=x.index)
        
        df_normalized[col_name] = z.clip(0, 1)
            
    return df_normalized

def calculate_propositions(df_normalized, epsilon):
    """
    Step 3: Proposition (P_ij)
    """
    df_proposition = pd.DataFrame()
    
    for col_name in df_normalized.columns:
        z = df_normalized[col_name]
        sum_z = z.sum()
        
        if sum_z == 0:
            p = z + epsilon
        else:
            p = (z / sum_z) + epsilon
        
        df_proposition[col_name] = p
    return df_proposition

def calculate_entropy(df_proposition, k):
    """
    Step 4: Entropy (E_j)
    """
    entropy_values = {}
    
    if k == 0:
        return {col_name: 0.0 for col_name in df_proposition.columns}

    for col_name in df_proposition.columns:
        p = df_proposition[col_name]
        
        # --- *** MODIFIED LINE *** ---
        # We check if x > 0 before applying math.log(x) to avoid the domain error
        p_log_p = p.apply(lambda x: x * math.log(x) if x > 0 else 0)
        
        e_j = -k * p_log_p.sum()
        entropy_values[col_name] = e_j
        
    return entropy_values

def calculate_diversification(entropy_values):
    """
    Step 5: Diversification (d_j)
    """
    diversification_values = {}
    for col_name, e_j in entropy_values.items():
        diversification_values[col_name] = 1 - e_j
    return diversification_values

def calculate_entropy_weights(diversification_values):
    """
    Step 6: Entropy Weight (W_j)
    """
    weights = {}
    sum_d = sum(diversification_values.values())
    
    if sum_d == 0:
        num_features = len(diversification_values)
        if num_features > 0:
            return {col: 1.0 / num_features for col in diversification_values}
        else:
            return {}

    for col_name, d_j in diversification_values.items():
        weights[col_name] = d_j / sum_d
    return weights

def process_entropy_computation(data_csv_path, vars_csv_path, epsilon_val):
    """
    Main orchestration function to run all steps and save output files.
    """
    df_features, df_vars_config, n_rows = load_config_and_data(data_csv_path, vars_csv_path)
    
    if df_features is None:
        return

    if n_rows == 0:
        print("Error: No data rows found.")
        return
    
    # --- Run Computations & Save Step 2 ---
    df_normalized = normalize_data(df_features, df_vars_config)
    
    # --- MODIFIED: Use .map instead of .applymap ---
    df_normalized = df_normalized.map(
        lambda x: round(x, 3) if isinstance(x, float) and math.isfinite(x) else x
    )
    df_normalized.to_csv("Output_Data/normalized.csv", index=False)

    # --- Run Computations & Save Step 3 ---
    df_proposition = calculate_propositions(df_normalized, epsilon_val)

    # --- MODIFIED: Use .map instead of .applymap ---
    df_proposition = df_proposition.map(
        lambda x: round(x, 3) if isinstance(x, float) and math.isfinite(x) else x
    )
    df_proposition.to_csv("Output_Data/proposition.csv", index=False)

    # --- k CALCULATION ---
    k = 0
    n = 0
    if 'Idle_Ratio' in df_proposition.columns:
        n = np.sum(np.array(df_proposition['Idle_Ratio']) > EPSILON) 
        
        if n > 1:
            k = round(1 / math.log(n), 3) 
            print(f"Calculated 'k'={k} using n={n} (from Idle_Ratio > EPSILON)")
        else:
            k = 0 
            print(f"Warning: 'n' for k-calculation is {n}. Setting k=0.")
    else:
        print("Warning: 'Idle_Ratio' not in proposition matrix. Using n_rows for k.")
        n = n_rows
        if n > 1:
            k = round(1 / math.log(n), 3)
        else:
            k = 0
    # --- END OF k CALCULATION ---

    # --- Run Steps 4, 5, 6 ---
    # calculate_entropy will now work correctly
    entropy_values = calculate_entropy(df_proposition, k)
    diversification_values = calculate_diversification(entropy_values)
    entropy_weights = calculate_entropy_weights(diversification_values)
    
    # --- Create and Save Final Entropy.csv ---
    df_entropy_results = pd.DataFrame({
        'E_j (Entropy)': entropy_values,
        'd_j (Diversification)': diversification_values,
        'W_j (Entropy Weight)': entropy_weights
    })
    
    # --- MODIFIED: Use .map instead of .applymap ---
    df_entropy_results = df_entropy_results.map(
        lambda x: round(x, 3) if isinstance(x, float) and math.isfinite(x) else x
    )
    df_entropy_results.to_csv("Output_Data/entropy.csv", index=True, index_label="Feature")

    print("\nEntropy calculation complete.")

# --- Execute the Process ---
if __name__ == "__main__":
    DATA_FILE = "Output_Data/environment.csv"
    VARS_FILE = "Input_Data/variables.csv"
    
    process_entropy_computation(DATA_FILE, VARS_FILE, EPSILON)