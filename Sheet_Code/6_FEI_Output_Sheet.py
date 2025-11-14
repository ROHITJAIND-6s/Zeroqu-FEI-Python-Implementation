import pandas as pd
import numpy as np
import math

# --- Configuration ---
# Corrected typo: environment.csv -> environmental.csv
ENV_FILE = "Output_Data/environment.csv" 
VARS_FILE = "Input_Data/variables.csv"
WEIGHTS_FILE = "Output_Data/weights.csv"
OUTPUT_FILE = "Output_Data/FEI.csv"

# --- Loading Helper Functions ---

def load_environmental_data(filepath):
    """
    Loads the main data from environmental.csv.
    """
    try:
        df = pd.read_csv(filepath)
        df['Date'] = df['Date'].fillna("Unknown")
        df['Leg_ID'] = df['Leg_ID'].fillna(0)
        return df
    except FileNotFoundError:
        print(f"Error: File not found '{filepath}'")
        return None

def load_variables_config(filepath):
    """
    Loads the variables config (Min, Max, Norm_Type) using Display_Name as the key.
    """
    try:
        df = pd.read_csv(filepath)
        df = df.set_index('Display_Name')
        return df
    except FileNotFoundError:
        print(f"Error: File not found '{filepath}'")
        return None
    except KeyError:
        print(f"Error: 'Display_Name' column not found in '{filepath}'")
        return None

def load_final_weights(filepath):
    """
    Loads the final normalized weights from weights.csv.
    """
    try:
        df = pd.read_csv(filepath)
        df = df.set_index('Variable') # 'Variable' column holds the Display_Name
        return df[df['Include'] == 1]['Final_Weight_Normalized']
    except FileNotFoundError:
        print(f"Error: File not found '{filepath}'")
        return None
    except KeyError:
        print(f"Error: 'Variable' or 'Include' column not found in '{filepath}'")
        return None

# --- Calculation Helper Functions ---

def normalize_features(df_env, df_vars, s_weights):
    """
    Step 2: Normalization (Z_ij)
    Normalizes all features based on the variables config.
    """
    print("Normalizing features...")
    df_normalized = pd.DataFrame()
    for display_name, weight in s_weights.items():
        if display_name not in df_vars.index:
            print(f"Warning: '{display_name}' in weights.csv but not in variables.csv. Skipping.")
            continue
        
        if display_name not in df_env.columns:
            print(f"Warning: Column '{display_name}' not found in {ENV_FILE}. Skipping.")
            continue
            
        var_config = df_vars.loc[display_name]
        min_val = var_config['Min']
        max_val = var_config['Max']
        norm_type = var_config['Norm_Type (inc/dec)']
        
        x = df_env[display_name].fillna(0).astype(float)
        
        z = pd.Series(0.0, index=x.index) 
        if max_val - min_val != 0:
            if norm_type == 'inc':
                z = (x - min_val) / (max_val - min_val)
            elif norm_type == 'dec':
                z = (max_val - x) / (max_val - min_val)
        
        df_normalized[display_name] = z.clip(0, 1)
    return df_normalized

def calculate_weighted_contributions(df_normalized, s_weights):
    """
    Step 3: Weighted Feature Contribution
    Multiplies each normalized feature by its final weight.
    """
    print("Calculating weighted contributions...")
    df_weighted_contributions = pd.DataFrame()
    for display_name in df_normalized.columns:
        weight = s_weights.loc[display_name]
        df_weighted_contributions[display_name + "Wt"] = df_normalized[display_name] * weight
    return df_weighted_contributions

def calculate_risk_band(score):
    """
    Step 7: Risk Band Classification
    Categorizes the cumulative FEI score.
    """
    try:
        score = float(score)
    except (ValueError, TypeError):
        return "Unknown" # Handle non-numeric
        
    if score < 2:
        return "Low"
    elif score < 4.5:
        return "Moderate"
    elif score < 7:
        return "High"
    else:
        return "Very High"

def calculate_all_fei_scores(df_base, df_weighted_contributions, s_weights):
    """
    Steps 4, 5, 6, and 7:
    Calculates Daily_FEI, FEI_Cum, FEI_Cum_Selected, and Risk_Band.
    """
    print("Calculating FEI scores...")
    df_scores = df_base.copy()
    
    # Step 4: Daily FEI Calculation
    sum_final_weights = s_weights.sum()
    sum_weighted_contributions = df_weighted_contributions.sum(axis=1)
    
    if sum_final_weights == 0:
        potential_fei = 0.0
    else:
        potential_fei = sum_weighted_contributions / sum_final_weights
    
    idle_wt_column_name = 'Idle_Ratio' + 'Wt'
    
    if idle_wt_column_name in df_weighted_contributions:
        # Corrected typo in variable name here
        df_scores['Daily_FEI'] = np.where(
            df_weighted_contributions[idle_wt_column_name] == 0, 
            0,
            potential_fei
        )
    else:
        print(f"Warning: '{idle_wt_column_name}' not found. Calculating Daily_FEI without IF(K=0) check.")
        df_scores['Daily_FEI'] = potential_fei

    # Step 5: Cumulative FEI
    df_scores['FEI_Cum'] = df_scores.groupby('Leg_ID')['Daily_FEI'].cumsum()

    # Step 6: FEI_Cum_Selected
    df_scores['FEI_Cum_Selected'] = df_scores['FEI_Cum']

    # Step 7: Risk Band Classification
    df_scores['Risk_Band'] = df_scores['FEI_Cum_Selected'].apply(calculate_risk_band)
    
    return df_scores

def assemble_final_dataframe(df_fei_scores, df_normalized, df_weighted_contributions):
    """
    Step 9: Assembles the final DataFrame in the correct column order.
    """
    return pd.concat([
        df_fei_scores[['Date', 'Leg_ID']],
        df_normalized,
        df_weighted_contributions,
        df_fei_scores.drop(columns=['Date', 'Leg_ID'])
    ], axis=1)

def round_and_save(df_final_output, output_file):
    """
    Step 10: Rounds all numeric values to 3 decimals and saves to CSV.
    """
    df_rounded = df_final_output.map(
        lambda x: round(x, 3) if isinstance(x, float) and math.isfinite(x) else x
    )
    df_rounded.to_csv(output_file, index=False)

# --- Main Orchestration ---

def main():
    """
    Main function to load all data, calculate FEI, and save the final report.
    """
    
    # 1. Load all input files
    df_env = load_environmental_data(ENV_FILE)
    df_vars = load_variables_config(VARS_FILE)
    s_weights = load_final_weights(WEIGHTS_FILE)

    if df_env is None or df_vars is None or s_weights is None:
        print("Aborting due to file loading errors.")
        return

    # 2. Create the base DataFrame for scores
    df_fei_base = df_env[['Date', 'Leg_ID']].copy()

    # 3. Perform Normalization (Z_ij)
    df_normalized = normalize_features(df_env, df_vars, s_weights)

    # 4. Calculate Weighted Feature Contribution
    df_weighted_contributions = calculate_weighted_contributions(df_normalized, s_weights)

    # 5. Calculate all FEI scores
    df_fei_scores = calculate_all_fei_scores(df_fei_base, df_weighted_contributions, s_weights)

    # 6. Assemble the final DataFrame
    df_final_output = assemble_final_dataframe(df_fei_scores, df_normalized, df_weighted_contributions)

    # 7. Round and Save
    round_and_save(df_final_output, OUTPUT_FILE)

    print(f"\nSuccessfully generated final report: '{OUTPUT_FILE}'")
    print("All values rounded to 3 decimal places.")

if __name__ == "__main__":
    main()