import pandas as pd

# The new weights you want to use
new_weights = [
    0.124,
    0.124,
    0.129,
    0.125,
    0.125,
    0.124,
    0.125,
    0.125
]

filename = "Output_Data/entropy.csv"

try:
    # Load the entropy.csv, keeping the 'Feature' column as the index
    df = pd.read_csv(filename, index_col="Feature")

    # Check if the number of new weights matches the number of rows (features)
    if len(df) == len(new_weights):
        # Replace the existing column with the new values
        df['W_j (Entropy Weight)'] = new_weights
        
        # Save the changes back to the same file
        df.to_csv(filename)
        
        print(f"Successfully updated 'W_j (Entropy Weight)' in {filename}.")
        print("\n--- Updated Data ---")
        print(df)
    else:
        print(f"Error: Length mismatch. The CSV has {len(df)} rows, but you provided {len(new_weights)} weights.")

except FileNotFoundError:
    print(f"Error: File '{filename}' not found. Please run entropy.py first.")
except Exception as e:
    print(f"An error occurred: {e}")