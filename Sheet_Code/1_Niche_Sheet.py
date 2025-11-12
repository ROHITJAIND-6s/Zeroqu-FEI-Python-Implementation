import pandas as pd
import math # Import math for isnan check

# 1. Hardcode the data from the NicheSettings sheet
data = [
    {'Niche Area': 'Sea Chest', 'Weight': 0.25, 'Fouling Factor': 0.9},
    {'Niche Area': 'Bow Thruster', 'Weight': 0.15, 'Fouling Factor': 0.8},
    {'Niche Area': 'Stern Thruster', 'Weight': 0.1, 'Fouling Factor': 0.7},
    {'Niche Area': 'Rudder', 'Weight': 0.1, 'Fouling Factor': 0.7},
    {'Niche Area': 'Others', 'Weight': 0.4, 'Fouling Factor': 0.6},
]

# 2. Create the main DataFrame for niche areas
df_niche_areas = pd.DataFrame(data)

# 3. Calculate 'Contribution' (Weight * Fouling_Factor) and round it
df_niche_areas['Contribution'] = round(df_niche_areas['Weight'] * df_niche_areas['Fouling Factor'], 3)

# 4. Calculate 'Niche_Score' (Sum of Contribution) and round it
niche_score_value = round(df_niche_areas['Contribution'].sum(), 3)

# 5. Add the 'Niche_Score' as a new column
df_niche_areas['Niche_Score'] = pd.NA

# 6. Set the Niche_Score value in the first row
df_niche_areas.loc[0, 'Niche_Score'] = niche_score_value

# 7. Save the single DataFrame to the CSV
output_filename = "Output_Data/niche_settings.csv"

# --- MODIFIED: Apply rounding to all floats before saving ---
# This ensures all floats (Weight, Fouling Factor, Contribution) are rounded
df_niche_areas = df_niche_areas.map(
    lambda x: round(x, 3) if isinstance(x, float) and math.isfinite(x) else x
)

df_niche_areas.to_csv(output_filename, index=False)

print(f"Successfully calculated Niche_Score ({niche_score_value}) and saved to '{output_filename}'.")