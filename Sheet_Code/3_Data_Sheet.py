import pandas as pd
from datetime import datetime
import math
import sys

# --- Configuration ---
INPUT_DATA_FILE = "Input_Data/data.csv"
SEASON_LOOKUP_FILE = "Input_Data/lkp_season.csv"
NICHE_SETTINGS_FILE = "Output_Data/niche_settings.csv"
OUTPUT_FILE = "Output_Data/environment.csv"
COATING_CONFIG_FILE = "Input_Data/CoatingConfig.csv"
HULL_HUSBANDRY_FILE = "Input_Data/HullHusbandry.csv"

# --- Helper Functions ---

def load_niche_settings(filepath):
    try:
        df_niche = pd.read_csv(filepath)
        if "Niche_Score" in df_niche.columns:
            niche_value = df_niche.loc[0, "Niche_Score"]
            if pd.notna(niche_value):
                return float(niche_value)
    except FileNotFoundError:
        print(f"Warning: '{filepath}' not found. Using default 0.725")
        return 0.725
    except Exception as e:
        print(f"Warning: Error reading '{filepath}'. Using default 0.725. Error: {e}")
        return 0.725
    return 0.725

def calculate_idle_days_leg(df_combined):
    idle_days_leg = []
    prev_idle_leg = 0
    prev_idle_days = 0

    for idx, row in df_combined.iterrows():
        current_idle_leg = row['Idle_Leg']
        daily_idle = (row['Idle Hours'] + row['Slow hours (<4 kn)']) / 24

        if current_idle_leg == 0:
            current_val = 0.0
            idle_days_leg.append(current_val)
            prev_idle_leg = 0
            prev_idle_days = 0
        else:
            if prev_idle_leg == 0:
                current_val = daily_idle
            else:
                current_val = prev_idle_days + daily_idle

            idle_days_leg.append(current_val)
            prev_idle_leg = current_idle_leg
            prev_idle_days = current_val

    return idle_days_leg

def idle_ratio(idle_hours, slow_hours):
    return (idle_hours + slow_hours) / 24

def idle_leg(idle_ratio):
    return 1 if idle_ratio > 0 else 0

def region(swt):
    if pd.isna(swt):
        return ""
    if swt > 25:
        return "Tropical"
    elif swt > 20:
        return "Subtropical"
    elif swt > 10:
        return "Temperate"
    elif swt > 5:
        return "Cold"
    else:
        return "Polar"

def region_risk(region):
    risk_map = {
        "Tropical": 1.0, "Subtropical": 0.8, "Temperate": 0.6,
        "Cold": 0.4, "Polar": 0.2, "": ""
    }
    return risk_map.get(region, "")

def season(date, latitude, df_lookup):
    if pd.isna(date) or date == 0:
        return "", ""
    if pd.isna(latitude):
        return "", ""

    try:
        date_str = str(date).split(" ")[0]
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        MM = date_obj.strftime("%m")
    except (ValueError, TypeError):
        return "", ""

    if abs(latitude) <= 15:
        zone = "NTrop" if latitude >= 0 else "STrop"
    else:
        zone = "N" if latitude >= 0 else "S"

    key = zone + ":" + str(MM)

    if "Key" in df_lookup.columns:
        row = df_lookup.loc[df_lookup["Key"].astype(str).str.strip() == key]
        if not row.empty:
            return row.iloc[0]["Season"], row.iloc[0]["SeasonFactor"]

    return "", ""

def process_data_row(date, leg_id, latitude, longitude, sea_water_temp,
                     idle_hours, slow_hours, salinity, df_lookup,
                     niche_score_val, niche_area_val):

    idle_ratio_value = idle_ratio(idle_hours, slow_hours)
    idle_leg_value = idle_leg(idle_ratio_value)
    region_value = region(sea_water_temp)
    region_risk_value = region_risk(region_value)
    season_name, season_factor = season(date, latitude, df_lookup)

    return pd.Series({
        "Idle_Ratio": idle_ratio_value,
        "Idle_Leg": idle_leg_value,
        "Region": region_value,
        "Region_Risk": region_risk_value,
        "Season": season_name,
        "Season_Risk": season_factor,
        "Niche_Score": niche_score_val,
        "Niche_Area": niche_area_val
    })

def compute_coating_effectiveness(df_env, coating_config_row):
    family = str(coating_config_row.get('Coating_Family', '')).strip()
    design_life = float(coating_config_row.get('Design_Life_Months', 0) or 0)
    idle_threshold = float(coating_config_row.get('Idle_Threshold_Days', 0) or 0)
    idle_sensitivity = float(coating_config_row.get('Idle_Sensitivity', 0) or 0)
    app_quality = float(coating_config_row.get('Application_Quality_Score', 1) or 1)

    def age_component(age):
        if design_life == 0: return 1.0
        ratio = age / design_life
        if family in {"SPC_STD", "SPC_EXT", "HYBRID_CDP_SPC"}:
            if design_life == 0: return 1.0
            if ratio <= 0.8: return 1 - 0.25 * ratio
            return max(0.0, 0.8 - 1.5 * (ratio - 0.8))
        if family in {"CDP", "SOLUBLE_CONV"}:
            return max(0.3, 1 - 0.8 * ratio)
        if family == "FRC":
            return max(0.5, 1 - 0.7 * ratio)
        return max(0.4, 1 - 0.6 * ratio)

    def speed_component(stw):
        if family in {"SPC_STD", "SPC_EXT", "HYBRID_CDP_SPC"}:
            if pd.isna(stw): return 1.0
            if stw < 3: return 0.7
            if stw < 6: return 0.7 + (stw - 3) * (0.15 / 3)
            if stw < 10: return 0.85 + (stw - 6) * (0.15 / 4)
            return 1.0
        if family == "FRC":
            if pd.isna(stw): return 1.0
            if stw < 8: return 0.6
            if stw < 14: return 0.6 + (stw - 8) * (0.4 / 6)
            return 1.0
        return 1.0

    def idle_component(idle_days):
        if idle_threshold == 0: return 1.0
        if idle_days <= idle_threshold: return 1.0
        decay = 1 - idle_sensitivity * (idle_days - idle_threshold)
        return max(0.0, decay)

    results = []
    for _, row in df_env.iterrows():
        if pd.isna(row.get('Date')) or str(row.get('Date')).strip() == "":
            results.append("")
            continue
        age = float(row.get('Coating_Age', 0) or 0)
        # Use 'SST_degC' because the dataframe has been renamed by the time this runs
        stw = row.get('STW') 
        idle_days = float(row.get('IdleDays_Leg', 0) or 0)
        
        val = age_component(age) * speed_component(stw) * idle_component(idle_days) * app_quality
        val = max(0.0, min(1.0, val))
        results.append(round(val, 4))
    return results

def compute_hull_maintenance_score(df_env, hull_decay_row, last_drydock_date, last_clean_date):
    """
    Compute Hull_Maintenance_Score per row.
    Matches Excel Formula:
    = 0.6 * EXP(-(Date - LastDrydock)/DD_Decay) + 0.4 * EXP(-(Date - LastClean)/Clean_Decay)
    """
    # 1. Get Decay Constants (Matches the XLOOKUP parts)
    if hull_decay_row.empty:
        dd_decay = 1825.0  # Default from formula
        clean_decay = 365.0 # Default from formula
    else:
        # Fetch values, default to formula fallback if missing/0
        dd_decay = float(hull_decay_row.get('DD_Decay_days', 0) or 1825.0)
        clean_decay = float(hull_decay_row.get('Clean_Decay_days', 0) or 365.0)

    def parse_date(val):
        try:
            return datetime.strptime(str(val).split(" ")[0], "%Y-%m-%d").date()
        except Exception:
            return None

    results = []
    for _, r in df_env.iterrows():
        # Current Date ($A2)
        data_date = parse_date(r.get('Date'))
        if data_date is None:
            results.append("")
            continue
        
        # --- Part 1: Drydock Calculation ---
        # Excel: IFERROR($A2 - MAXIFS("Drydock"), 999)
        if last_drydock_date is not None and data_date >= last_drydock_date:
            diff_drydock = (data_date - last_drydock_date).days
        else:
            diff_drydock = 999  # Fallback if date missing or invalid relative to current
            
        # --- Part 2: Hull Clean Calculation ---
        # Excel: IFERROR($A2 - MAXIFS("HullClean"), 999)
        if last_clean_date is not None and data_date >= last_clean_date:
            diff_clean = (data_date - last_clean_date).days
        else:
            diff_clean = 999  # Fallback if date missing or invalid relative to current
        
        # --- Exponential Decay Formula ---
        part_dd = math.exp(- diff_drydock / dd_decay)
        part_clean = math.exp(- diff_clean / clean_decay)
        
        # Weighted Sum
        score = 0.6 * part_dd + 0.4 * part_clean
        
        # Clamp result between 0 and 1 (MAX(0, MIN(1, ...)))
        score = max(0.0, min(1.0, score))
        results.append(round(score, 4))
        
    return results

def get_user_choice_and_dates(coating_df):
    unique_families = coating_df['Coating_Family'].unique().tolist()
    
    print("\n--- Coating Family Selection ---")
    print("Select the Coating Family applied to the vessel:")
    for i, family in enumerate(unique_families):
        print(f"{i + 1}. {family}")
    
    while True:
        try:
            choice = int(input("Enter the number corresponding to your choice: "))
            if 1 <= choice <= len(unique_families):
                selected_family = unique_families[choice - 1]
                break
            else:
                print(f"Please enter a number between 1 and {len(unique_families)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    print(f"\nSelected Coating Family: {selected_family}")
    
    print("\n--- Maintenance Dates ---")
    print("Please enter dates in YYYY-MM-DD format.")
    
    def get_date_input(prompt_text):
        while True:
            val = input(prompt_text)
            try:
                return datetime.strptime(val.strip(), "%Y-%m-%d").date()
            except ValueError:
                print("Invalid format. Please use YYYY-MM-DD.")

    last_dd = get_date_input("Enter Last Drydock Date: ")
    last_clean = get_date_input("Enter Last Hull Cleaning Date: ")
    
    return selected_family, last_dd, last_clean

# --- Main Script Execution ---

def main():
    niche_value = load_niche_settings(NICHE_SETTINGS_FILE)
    Niche_Score = niche_value
    Niche_area = niche_value
    print(f"Loaded Niche_Score / Niche_Area: {niche_value}")

    try:
        df_main_original = pd.read_csv(INPUT_DATA_FILE)
    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_DATA_FILE}' not found.")
        return

    df_processing = df_main_original.fillna(0)

    try:
        df_lookup = pd.read_csv(SEASON_LOOKUP_FILE)
    except FileNotFoundError:
        print(f"Error: Lookup file '{SEASON_LOOKUP_FILE}' not found.")
        return

    print("Processing basic metrics...")

    # NOTE: We pass STW here, which will be renamed to SST_degC later
    metrics_df = df_processing.apply(
        lambda row: process_data_row(
            row['Date'],
            row['Leg_ID'],
            row['Lat'],
            row['Lon'],
            row.get('STW', 0),
            row['Idle Hours'],
            row['Slow hours (<4 kn)'],
            row['Salinity'],
            df_lookup,
            Niche_Score,
            Niche_area
        ),
        axis=1
    )

    print("Calculating cumulative IdleDays_Leg...")
    df_for_calc = pd.DataFrame()
    df_for_calc['Idle Hours'] = df_processing['Idle Hours']
    df_for_calc['Slow hours (<4 kn)'] = df_processing['Slow hours (<4 kn)']
    df_for_calc['Idle_Leg'] = metrics_df['Idle_Leg']

    metrics_df['IdleDays_Leg'] = calculate_idle_days_leg(df_for_calc)

    num_rows = len(df_main_original)
    df_blankspace = pd.DataFrame({
        ' ': [''] * num_rows,
        '  ': [''] * num_rows
    })

    df_main_original = df_main_original.rename(columns={
        'SST': 'SST_degC',
        'Salinity': 'Salinity_psu'
    })

    df_final = pd.concat([df_main_original, df_blankspace, metrics_df], axis=1)

    try:
        coating_config_df = pd.read_csv(COATING_CONFIG_FILE)
        hull_husbandry_df = pd.read_csv(HULL_HUSBANDRY_FILE)
    except FileNotFoundError as e:
        print(f"Error loading config files: {e}")
        return

    selected_family, last_dd_date, last_clean_date = get_user_choice_and_dates(coating_config_df)

    coating_row = coating_config_df[coating_config_df['Coating_Family'] == selected_family].iloc[0]
    
    hull_row = hull_husbandry_df[hull_husbandry_df['Coating_Type'] == selected_family]
    if hull_row.empty:
        print(f"Warning: No hull husbandry data found for {selected_family}. Using defaults.")
        hull_row = pd.Series({}) 
    else:
        hull_row = hull_row.iloc[0]

    print(f"Computing scores for family: {selected_family}...")
    
    if 'Coating_Age' in df_final.columns:
        df_final['Coating_Effectiveness'] = compute_coating_effectiveness(df_final, coating_row)
    else:
        print("Error: 'Coating_Age' column missing. Cannot calculate Effectiveness.")

    df_final['Hull_Maintenance_Score'] = compute_hull_maintenance_score(
        df_final, hull_row, last_dd_date, last_clean_date
    )

    cols_to_remove = ['Speed_Ratio', 'Coat_Age', 'Coating_Age_Months'] 
    df_final = df_final.drop(
        columns=[c for c in cols_to_remove if c in df_final.columns],
        errors='ignore'
    )

    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"Successfully processed and saved results to '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    main()