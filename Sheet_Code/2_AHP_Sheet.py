import math
import pandas as pd

# Step 1 — Manual Inputs / Constants
R_FACTOR = 1.3
N_CRITERIA = 8
RI = 1.41  # Random Index for n=8

# Ranking (Importance Order)
RANKINGS = {
    "Idle_Ratio": 8,
    "SST_degC": 6,
    "Salinity_psu": 7,
    "Speed_Ratio": 3,
    "Coating_Age_months": 5,
    "Region_Risk": 4,
    "Season_Risk": 1,
    "Niche_Score": 2
}

# Ensure consistent ordering for matrix operations
CRITERIA_LIST = list(RANKINGS.keys())

# --- ORIGINAL FUNCTIONS (UNCHANGED) ---

def build_pairwise_matrix(r_factor, rankings, criteria_list):
    # Step 2 — Pairwise Comparison Matrix Construction
    matrix = []
    for row_crit in criteria_list:
        row = []
        for col_crit in criteria_list:
            val = pow(r_factor, rankings[row_crit] - rankings[col_crit])
            row.append(val)
        matrix.append(row)
    return matrix

def calculate_geometric_means(matrix, n):
    # Step 3 — Geometric Mean Calculation
    geo_means = []
    for row in matrix:
        product = 1.0
        for val in row:
            product *= val
        geo_means.append(pow(product, 1/n))
    return geo_means

def calculate_ahp_weights(geo_means):
    # Step 4 — AHP Weight (Raw)
    total_geo_mean = sum(geo_means)
    return [gm / total_geo_mean for gm in geo_means]

def calculate_consistency_vectors(matrix, ahp_weights, n):
    """
    MODIFIED: This function now returns the A.w and lambda_i vectors
    (like columns K & L in your image) and the final lambda_max.
    """
    # Step 5 (Aw vector) & Step 6 (Lambda Calculation)
    aw_values = []
    lambda_i_values = []
    
    for i in range(n):
        # A.w[i] = SUMPRODUCT(Matrix[i, :], AHP_Weights)
        aw_val = sum(matrix[i][j] * ahp_weights[j] for j in range(n))
        aw_values.append(aw_val)
        
        # λ_i = (A·w[i]) / AHP_Weight[i]
        lambda_i = aw_val / ahp_weights[i]
        lambda_i_values.append(lambda_i)
    
    # λ_max = AVERAGE(λ_i[all])
    lambda_max = sum(lambda_i_values) / n
    
    return aw_values, lambda_i_values, lambda_max

def calculate_consistency(lambda_max, n, ri):
    # Step 7 — Consistency Index (CI)
    ci = (lambda_max - n) / (n - 1)
    # Step 8 — Consistency Ratio (CR)
    cr = ci / ri
    return ci, cr

def process_ahp_computation_and_save(r_factor, n, rankings, criteria_list, ri):
    """
    MODIFIED: This function now builds a DataFrame, rounds all values,
    and saves it to 'ahp.csv'.
    """
    # Orchestration of all steps
    matrix = build_pairwise_matrix(r_factor, rankings, criteria_list)
    geo_means = calculate_geometric_means(matrix, n)
    ahp_weights = calculate_ahp_weights(geo_means)
    aw_values, lambda_i_values, lambda_max = calculate_consistency_vectors(matrix, ahp_weights, n)
    ci, cr = calculate_consistency(lambda_max, n, ri)

    # --- Build the DataFrame ---

    # 1. Create the main 8x8 matrix
    df = pd.DataFrame(matrix, columns=criteria_list, index=criteria_list)

    # 2. Add the 'A.w' and 'lambda_i' columns
    df['A.w'] = aw_values
    df['lambda_i'] = lambda_i_values

    # 3. Add the 'Geometric Mean' and 'AHP Weight' rows
    df.loc['Geometric Mean', criteria_list] = geo_means
    df.loc['AHP Weight (raw)', criteria_list] = ahp_weights
    
    # 4. Create the 'Ranking' row
    rank_series = pd.Series([rankings[crit] for crit in criteria_list], index=criteria_list, name="Ranking")
    df_ranking = pd.DataFrame(rank_series).T

    # 5. Create the summary (CI, CR) rows
    summary_index = ['lambda_max', 'n (number of criteria)', 'CI (Consistency Index)', 'RI (Random Index)', 'CR (Consistency Ratio)']
    summary_data = [lambda_max, n, ci, ri, cr]
    df_summary = pd.DataFrame(summary_data, index=summary_index, columns=[criteria_list[0]])

    # 6. Combine all parts in the correct order
    df_final = pd.concat([df_ranking, df, df_summary])

    # --- NEW STEP: Apply round(x, 3) ---
    # This lambda function rounds the number to 3 decimal places.
    df_final_rounded = df_final.map(
        lambda x: round(x, 3) if isinstance(x, float) and math.isfinite(x) else x
    )

    # 7. Save to CSV
    output_filename = "Output_Data/ahp.csv"
    # Save the new rounded DataFrame
    df_final_rounded.to_csv(output_filename, index_label="Criteria") 
    

# Execute the process
process_ahp_computation_and_save(R_FACTOR, N_CRITERIA, RANKINGS, CRITERIA_LIST, RI)