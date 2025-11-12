import pandas as pd
import math  

# --- Configuration ---
ALPHA = 0.5  # Constant for Alpha (AHP weight share)

# File paths
AHP_FILE = "Output_Data/ahp.csv"
ENTROPY_FILE = "Output_Data/entropy.csv"
VARS_FILE = "Input_Data/variables.csv"
OUTPUT_FILE = "Output_Data/weights.csv"

# --- Helper Functions ---

def load_ahp_weights(filepath):
    """
    Loads AHP weights and resets the index to create
    a 'Variable' column.
    """
    try:
        df_ahp = pd.read_csv(filepath, index_col="Criteria")
        # Convert to float, as AHP.csv might have strings
        ahp_weights_series = pd.to_numeric(df_ahp.loc['AHP Weight (raw)'], errors='coerce').dropna()
        
        df = ahp_weights_series.to_frame(name="AHP_Weight")
        df = df.reset_index()
        df = df.rename(columns={"index": "Variable"})
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found '{filepath}'")
        return None
    except KeyError:
        print(f"Error: Could not find row 'AHP Weight (raw)' in '{filepath}'")
        return None

def load_entropy_weights(filepath):
    """
    Loads Entropy weights and resets the index to create
    a 'Variable' column.
    """
    try:
        df_entropy = pd.read_csv(filepath, index_col="Feature")
        df = df_entropy[['W_j (Entropy Weight)']].rename(
            columns={"W_j (Entropy Weight)": "Entropy_Weight"}
        )
        
        df = df.reset_index()
        df = df.rename(columns={"Feature": "Variable"})
        return df

    except FileNotFoundError:
        print(f"Error: File not found '{filepath}'")
        return None
    except KeyError:
        print(f"Error: Could not find column 'W_j (Entropy Weight)' in '{filepath}'")
        return None

def load_variables_config(filepath):
    """
    MODIFIED: Loads 'Include' flag using 'Display_Name' as the key
    and renames it to 'Variable' for merging.
    """
    try:
        df_vars = pd.read_csv(filepath)
        
        df = df_vars[['Display_Name', 'Include (1/0)']].copy()
        
        df = df.rename(columns={
            "Display_Name": "Variable", 
            "Include (1/0)": "Include"
        })
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found '{filepath}'")
        return None
    except KeyError:
        print(f"Error: Could not find 'Display_Name' or 'Include (1/0)' in '{filepath}'")
        return None

# --- Main Script Execution ---

def main():
    """
    Main function to load all data, merge, calculate weights, and save.
    """
    
    # 1. Load all three input files
    df_ahp = load_ahp_weights(AHP_FILE)
    df_entropy = load_entropy_weights(ENTROPY_FILE)
    df_vars = load_variables_config(VARS_FILE) 

    if df_ahp is None or df_entropy is None or df_vars is None:
        print("Aborting due to file loading errors.")
        return

    # 2. Merge the DataFrames
    df_final = pd.merge(df_vars, df_ahp, on="Variable", how="left")
    df_final = pd.merge(df_final, df_entropy, on="Variable", how="left")
    
    # Handle NaNs
    df_final['Include'] = df_final['Include'].fillna(0)
    df_final['AHP_Weight'] = df_final['AHP_Weight'].fillna(0)
    df_final['Entropy_Weight'] = df_final['Entropy_Weight'].fillna(0)

    # 3. Perform Calculations
    df_final['Final_Weight'] = (ALPHA * df_final['AHP_Weight']) + \
                             ((1 - ALPHA) * df_final['Entropy_Weight'])
    
    df_final['Active_Unnorm'] = df_final['Final_Weight'] * df_final['Include']
    
    sum_active_unnorm = df_final['Active_Unnorm'].sum()
    
    if sum_active_unnorm == 0:
        df_final['Final_Weight_Normalized'] = 0.0
    else:
        df_final['Final_Weight_Normalized'] = df_final['Active_Unnorm'] / sum_active_unnorm
        
    # --- NEW STEP: Round all float values ---
    # We apply rounding to all numeric columns before saving
    df_final = df_final.map(
        lambda x: round(x, 3) if isinstance(x, (float, int)) and math.isfinite(x) else x
    )

    # 4. Save the final file
    output_columns = [
        'Variable',
        'AHP_Weight', 
        'Entropy_Weight', 
        'Final_Weight', 
        'Include', 
        'Active_Unnorm', 
        'Final_Weight_Normalized'
    ]
    
    df_output = df_final[output_columns]
    
    df_output.to_csv(OUTPUT_FILE, index=False)
    
    print(f"Successfully calculated and saved final weights to '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    main()