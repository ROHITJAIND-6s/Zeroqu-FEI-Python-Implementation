import pandas as pd
from datetime import datetime

# --- Helper Functions ---
# (These are unchanged)

def load_niche_settings(filepath):
    """
    Loads the Niche_Score from the 'niche_settings.csv' file
    by reading the 'Niche_Score' column.
    """
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
    
    print(f"Warning: Could not find 'Niche_Score' in '{filepath}'. Using default 0.725")
    return 0.725

def idle_ratio(idle_hours, slow_hours):
    """Calculates Idle_Ratio from 'Idle Hours' and 'Slow Hours'."""
    return (idle_hours + slow_hours) / 24

def idle_leg(idle_ratio):
    """Determines if the leg had any idle time."""
    return 1 if idle_ratio > 0 else 0

def region(swt):
    """Determines the Region based on Sea Water Temperature (swt)."""
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
    """Gets the risk factor for a given Region."""
    risk_map = {
        "Tropical": 1.0,
        "Subtropical": 0.8,
        "Temperate": 0.6,
        "Cold": 0.4,
        "Polar": 0.2,
        "": ""
    }
    return risk_map.get(region, "")

def season(date, latitude, df_lookup):
    """
    Looks up the Season and SeasonFactor from the lkp_season.csv.
    """
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
            season_name = row.iloc[0]["Season"]
            season_factor = row.iloc[0]["SeasonFactor"]
            return season_name, season_factor
            
    return "", ""

# --- Main Processing Function ---
# (This is unchanged)

def process_data_row(date, leg_id, speed_ratio, coat_age, latitude, 
                     longitude, sea_water_temp, idle_hours, slow_hours, salinity, df_lookup,
                     niche_score_val, niche_area_val): 
    """
    Processes a single row of data and returns the computed metrics.
    """
    
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

# --- Main Script Execution ---

def main():
    """
    Main function to load data, process it, and save the combined result.
    """
    
    # 1. Load Niche_Score
    niche_settings_file = "Output_Data/niche_settings.csv"
    niche_value = load_niche_settings(niche_settings_file)
    Niche_Score = niche_value
    Niche_area = niche_value
    print(f"Loaded Niche_Score / Niche_Area: {niche_value} from '{niche_settings_file}'")

    # 2. Load input data
    try:
        df_main_original = pd.read_csv("Input_Data/data.csv")
    except FileNotFoundError:
        print("Error: Input file 'data.csv' not found.")
        return

    # 3. Create a processing-safe version (fill empty with 0)
    df_processing = df_main_original.fillna(0) 

    # 4. Load lookup table
    try:
        df_lookup = pd.read_csv("Input_Data/lkp_season.csv")
    except FileNotFoundError:
        print("Error: Lookup file 'lkp_season.csv' not found.")
        return

    

    # 5. Calculate metrics (using the processing-safe dataframe)
    #    The lambda function still uses the *original* column names
    metrics_df = df_processing.apply(
        lambda row: process_data_row(
            row['Date'],
            row['Leg_ID'],
            row['Speed_Ratio'],
            row['Coat_Age'],
            row['Lat'],
            row['Lon'],
            row['SST'],
            row['Idle Hours'],
            row['Slow hours (<4 kn)'],
            row['Salinity'],
            df_lookup,
            Niche_Score,
            Niche_area
        ),
        axis=1
    )

    # 6. Create the two empty columns
    num_rows = len(df_main_original)
    df_blankspace = pd.DataFrame({
        ' ': [''] * num_rows,
        '  ': [''] * num_rows
    })

    # 7. **NEW STEP**: Rename columns in the original dataframe
    #    This ensures the concatenated file has the standardized names
    df_main_original = df_main_original.rename(columns={
        'Coat_Age': 'Coating_Age_months',
        'SST': 'SST_degC',
        'Salinity': 'Salinity_psu'
    })
    print("Standardized column names (e.g., 'Coat_Age' -> 'Coating_Age_Months').")


    # 8. Combine all DataFrames: [Renamed Original Data] + [Blanks] + [Metrics]
    df_final = pd.concat([df_main_original, df_blankspace, metrics_df], axis=1)

    # 9. Save the final combined file
    output_filename = "Output_Data/environment.csv"
    df_final.to_csv(output_filename, index=False)
    
    print(f"Successfully processed and combined data into '{output_filename}'.")

if __name__ == "__main__":
    main()