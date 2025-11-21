import pandas as pd
import numpy as np
import math

# --- Configuration ---
ENV_FILE = "Output_Data/environment.csv" 
VARS_FILE = "Input_Data/variables.csv"
WEIGHTS_FILE = "Output_Data/weights.csv"
OUTPUT_FILE = "Output_Data/FEI.csv"

# --- Loading Helper Functions ---

def load_environmental_data(filepath):
    """Loads environment data and fills missing values for Date and Leg_ID."""
    try:
        df = pd.read_csv(filepath)
        df['Date'] = df['Date'].fillna("Unknown")
        df['Leg_ID'] = df['Leg_ID'].fillna(0)
        return df
    except FileNotFoundError:
        print(f"Error: File not found '{filepath}'")
        return None

def load_variables_config(filepath):
    """Loads variable configuration (Min, Max, etc.) indexed by Display_Name."""
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
    """Loads final normalized weights indexed by Variable name."""
    try:
        df = pd.read_csv(filepath)
        df = df.set_index('Variable')
        return df[df['Include'] == 1]['Final_Weight_Normalized']
    except FileNotFoundError:
        print(f"Error: File not found '{filepath}'")
        return None
    except KeyError:
        print(f"Error: 'Variable' or 'Include' column not found in '{filepath}'")
        return None

# --- Calculation Helper Functions ---

def normalize_features(df_env, df_vars, s_weights):
    """Normalizes features (Z_ij) based on Min/Max and inc/dec logic."""
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
    """Calculates weighted contribution for each feature."""
    print("Calculating weighted contributions...")
    df_weighted_contributions = pd.DataFrame()
    for display_name in df_normalized.columns:
        weight = s_weights.loc[display_name]
        df_weighted_contributions[display_name + "Wt"] = df_normalized[display_name] * weight
    return df_weighted_contributions

def calculate_risk_band_standard(score):
    """Determines standard risk band based on FEI score thresholds."""
    try:
        score = float(score)
    except (ValueError, TypeError):
        return "Unknown"
    if score < 2: return "Low"
    elif score < 4.5: return "Moderate"
    elif score < 7: return "High"
    else: return "Very High"

def calculate_risk_band_final(score):
    """Determines final risk band based on new thresholds (20, 40, 60)."""
    try:
        score = float(score)
    except (ValueError, TypeError):
        return "Unknown"
    
    if score < 20: return "Low"
    elif score < 40: return "Moderate"
    elif score < 60: return "High"
    else: return "Very High"

def calculate_conditional_cumulative_fei(df_calc, idle_wt_column_name, df_weighted_contributions):
    """Calculates cumulative FEI with reset logic: =IF(IdleWt=0, 0, Prev_Cum + Daily_FEI)."""
    cumulative_values = []
    prev_cum = 0.0
    prev_leg = None
    
    if idle_wt_column_name in df_weighted_contributions:
        idle_wts = df_weighted_contributions[idle_wt_column_name].fillna(0).values
    else:
        idle_wts = np.ones(len(df_calc)) 

    daily_feis = df_calc['Daily_FEI'].fillna(0).values
    leg_ids = df_calc['Leg_ID'].values

    for i in range(len(df_calc)):
        current_leg = leg_ids[i]
        idle_wt = idle_wts[i]
        daily_fei = daily_feis[i]

        if current_leg != prev_leg:
            prev_cum = 0.0

        if idle_wt == 0:
            current_cum = 0.0
        else:
            current_cum = prev_cum + daily_fei
        
        cumulative_values.append(current_cum)
        prev_cum = current_cum
        prev_leg = current_leg
        
    return cumulative_values

def calculate_final_cumulative_fei(df_calc):
    """Calculates continuous cumulative FEI without reset: Row N = Prev_Daily + Prev_Cum."""
    cumulative_values = []
    
    # Trackers for previous values
    prev_cum_final = 0.0
    prev_daily_final = 0.0
    
    daily_finals = df_calc['Daily_FEI_Final'].fillna(0).values
    
    for i in range(len(df_calc)):
        current_daily = daily_finals[i]
        
        if i == 0:
            # First row of the entire file: Cum = Daily
            current_cum = current_daily
        else:
            # Subsequent rows: Prev_Cum + Prev_Daily
            current_cum = prev_cum_final + prev_daily_final
            
        cumulative_values.append(current_cum)
        
        # Update trackers
        prev_cum_final = current_cum
        prev_daily_final = current_daily
        
    return cumulative_values

def calculate_extended_fei_metrics(df_scores, df_env):
    """Computes Coating Risk, Hull Risk, Management Risk, and Final FEI metrics."""
    print("Calculating Final Risk Metrics...")
    df_calc = df_scores.copy()
    
    coating_eff = df_env.get('Coating_Effectiveness')
    hull_maint = df_env.get('Hull_Maintenance_Score')
    
    if coating_eff is None or hull_maint is None:
        print("Warning: Missing Effectiveness/Maintenance scores. Using NaNs.")
        df_calc['Coating_Effectiveness'] = np.nan
        df_calc['Hull_Maintenance_Score'] = np.nan
    else:
        df_calc['Coating_Effectiveness'] = coating_eff
        df_calc['Hull_Maintenance_Score'] = hull_maint

    # T: Coating_Risk_Factor
    df_calc['Coating_Risk_Factor'] = df_calc['Coating_Effectiveness'].apply(
        lambda x: np.nan if pd.isna(x) or x == "" else (1 + 0.5 * (1 - float(x)))
    )

    # U: Hull_Risk_Factor
    df_calc['Hull_Risk_Factor'] = df_calc['Hull_Maintenance_Score'].apply(
        lambda x: 1.0 if pd.isna(x) or x == "" else (1 + 0.5 * (1 - float(x)))
    )

    # V: Management_Risk_Factor
    df_calc['Management_Risk_Factor'] = df_calc['Coating_Risk_Factor'] * df_calc['Hull_Risk_Factor']

    # W: Daily_FEI_Final
    df_calc['Daily_FEI_Final'] = df_calc['Daily_FEI'] * df_calc['Management_Risk_Factor']

    # X: FEI_Cum_Final (Continuous accumulation)
    df_calc['FEI_Cum_Final'] = calculate_final_cumulative_fei(df_calc)

    # Y: Risk Band Final
    df_calc['VesselFoulingExposure_Risk_Band_Final'] = df_calc['FEI_Cum_Final'].apply(calculate_risk_band_final)

    return df_calc

def calculate_all_fei_scores(df_base, df_weighted_contributions, s_weights, df_env):
    """Orchestrates calculation of all FEI scores (Standard and Extended)."""
    print("Calculating Standard FEI scores...")
    df_scores = df_base.copy()
    
    # --- Step 4: Daily FEI Calculation ---
    sum_final_weights = s_weights.sum()
    sum_weighted_contributions = df_weighted_contributions.sum(axis=1)
    
    if sum_final_weights == 0:
        potential_fei = 0.0
    else:
        potential_fei = sum_weighted_contributions / sum_final_weights
    
    idle_wt_column_name = 'Idle_Ratio' + 'Wt'
    
    if idle_wt_column_name in df_weighted_contributions:
        # Only result in NaN if input is explicitly NaN. 0 is valid.
        df_scores['Daily_FEI'] = np.where(
            df_weighted_contributions[idle_wt_column_name].isna(),
            np.nan, 
            potential_fei
        )
    else:
        df_scores['Daily_FEI'] = potential_fei

    # --- Step 5: Cumulative FEI (Conditional) ---
    print("Calculating Conditional Cumulative FEI...")
    df_scores['FEI_Cum'] = calculate_conditional_cumulative_fei(
        df_scores, idle_wt_column_name, df_weighted_contributions
    )
    
    # Step 6 & 7
    df_scores['FEI_Cum_Selected'] = df_scores['FEI_Cum']
    df_scores['Risk_Band'] = df_scores['FEI_Cum_Selected'].apply(calculate_risk_band_standard)
    
    # --- New Final FEI Metrics ---
    df_final_scores = calculate_extended_fei_metrics(df_scores, df_env)
    
    return df_final_scores

def assemble_final_dataframe(df_fei_scores, df_normalized, df_weighted_contributions):
    """Combines normalized features, weighted contributions, and calculated scores into one DataFrame."""
    score_cols = [
        'Daily_FEI', 'FEI_Cum', 'FEI_Cum_Selected', 'Risk_Band',
        'Coating_Effectiveness', 'Hull_Maintenance_Score',
        'Coating_Risk_Factor', 'Hull_Risk_Factor', 
        'Management_Risk_Factor', 'Daily_FEI_Final', 
        'FEI_Cum_Final', 'VesselFoulingExposure_Risk_Band_Final'
    ]
    
    existing_score_cols = [c for c in score_cols if c in df_fei_scores.columns]
    
    return pd.concat([
        df_fei_scores[['Date', 'Leg_ID']],
        df_normalized,
        df_weighted_contributions,
        df_fei_scores[existing_score_cols]
    ], axis=1)

def round_and_save(df_final_output, output_file):
    """Rounds numeric values to 4 decimal places and saves to CSV."""
    df_rounded = df_final_output.map(
        lambda x: round(x, 4) if isinstance(x, (float, np.float64)) and math.isfinite(x) else x
    )
    df_rounded.to_csv(output_file, index=False)

# --- Main Orchestration ---

def main():
    """Main execution flow to load data, calculate FEI, and save report."""
    
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

    # 5. Calculate all FEI scores (Standard + New Final Metrics)
    df_fei_scores = calculate_all_fei_scores(df_fei_base, df_weighted_contributions, s_weights, df_env)

    # 6. Assemble the final DataFrame
    df_final_output = assemble_final_dataframe(df_fei_scores, df_normalized, df_weighted_contributions)

    # 7. Round and Save
    round_and_save(df_final_output, OUTPUT_FILE)

    print(f"\nSuccessfully generated final report: '{OUTPUT_FILE}'")
    print("All values rounded to 3 decimal places.")

if __name__ == "__main__":
    main()