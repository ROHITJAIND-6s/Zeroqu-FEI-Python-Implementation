import pandas as pd
import numpy as np
import math

# --- Configuration ---
ENV_FILE = "Output_Data/environment.csv"
VARS_FILE = "Input_Data/variables.csv"
WEIGHTS_FILE = "Output_Data/weights.csv"
OUTPUT_FILE = "Output_Data/FEI.csv"

# --- Helper Functions ---
# (These are unchanged)

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
    Loads the variables config. We need Min, Max, and Norm_Type.
    We will use Display_Name as the key.
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

def calculate_risk_band(score):
    """
    Step 7: Risk Band Classification
    Categorizes the cumulative FEI score.
    """
    if score < 2:
        return "Low"
    elif score < 4.5:
        return "Moderate"
    elif score < 7:
        return "High"
    else:
        return "Very High"

# --- Main Script Execution ---

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

    df_fei_output = pd.DataFrame()
    # --- MODIFICATION ---
    # df_raw_features is no longer needed
    # df_raw_features = pd.DataFrame() 
    df_normalized = pd.DataFrame()
    df_weighted_contributions = pd.DataFrame()

    # 2. Link Date/Leg_ID
    df_fei_output['Date'] = df_env['Date']
    df_fei_output['Leg_ID'] = df_env['Leg_ID']

    # 3. Perform Normalization (Z_ij)
    print("Normalizing features...")
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
        
        # Get raw data
        x = df_env[display_name].fillna(0).astype(float)
        
        # --- MODIFICATION ---
        # No longer storing raw features
        # df_raw_features[display_name] = x
        
        # Step 2: Normalization
        z = pd.Series(0.0, index=x.index) 
        if max_val - min_val != 0:
            if norm_type == 'inc':
                z = (x - min_val) / (max_val - min_val)
            elif norm_type == 'dec':
                z = (max_val - x) / (max_val - min_val)
        
        # Store the normalized value (Z_ij)
        df_normalized[display_name] = z.clip(0, 1)

    # 4. Step 3: Weighted Feature Contribution
    print("Calculating weighted contributions...")
    for display_name in df_normalized.columns:
        weight = s_weights.loc[display_name]
        df_weighted_contributions[display_name + "Wt"] = df_normalized[display_name] * weight

    # 5. Step 4: Daily FEI Calculation
    print("Calculating Daily FEI...")
    sum_final_weights = s_weights.sum()
    sum_weighted_contributions = df_weighted_contributions.sum(axis=1)
    
    if sum_final_weights == 0:
        potential_fei = 0.0
    else:
        potential_fei = sum_weighted_contributions / sum_final_weights
    
    idle_wt_column_name = 'Idle_Ratio' + 'Wt'
    
    if idle_wt_column_name in df_weighted_contributions:
        df_fei_output['Daily_FEI'] = np.where(
            df_weighted_contributions[idle_wt_column_name] == 0, 
            0,
            potential_fei
        )
    else:
        print(f"Warning: '{idle_wt_column_name}' not found. Calculating Daily_FEI without IF(K=0) check.")
        df_fei_output['Daily_FEI'] = potential_fei

    # 6. Step 5: Cumulative FEI
    print("Calculating Cumulative FEI...")
    df_fei_output['FEI_Cum'] = df_fei_output.groupby('Leg_ID')['Daily_FEI'].cumsum()

    # 7. Step 6: FEI_Cum_Selected
    df_fei_output['FEI_Cum_Selected'] = df_fei_output['FEI_Cum']

    # 8. Step 7: Risk Band Classification
    print("Assigning Risk Bands...")
    df_fei_output['Risk_Band'] = df_fei_output['FEI_Cum_Selected'].apply(calculate_risk_band)

    # 9. Assemble the final DataFrame
    # --- MODIFICATION ---
    # Replaced df_raw_features with df_normalized
    df_final_output = pd.concat([
        df_fei_output[['Date', 'Leg_ID']],
        df_normalized, # <--- CORRECTED
        df_weighted_contributions,
        df_fei_output.drop(columns=['Date', 'Leg_ID'])
    ], axis=1)


    df_final_output = df_final_output.map(
        lambda x: round(x, 3) if isinstance(x, float) and math.isfinite(x) else x
    )
    # 10. Save the final file
    df_final_output.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSuccessfully generated final report: '{OUTPUT_FILE}'")

if __name__ == "__main__":
    main()