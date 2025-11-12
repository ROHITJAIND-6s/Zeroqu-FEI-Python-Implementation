import subprocess
import sys
import os

# --- Configuration ---

# The directory where your scripts are located
SCRIPT_DIRECTORY = "Sheet_Code"

# The list of scripts to run, in the exact order
scripts_to_run = [
    "1_Niche_Sheet.py",
    "2_AHP_Sheet.py",
    "3_Data_Sheet.py",
    "4_Entropy_Sheet.py",
    "4.1_Entropy_Weight_Fix.py",
    "5_Weights_Sheet.py",
    "6_FEI_Output_Sheet.py"
]

# Get the path to the current Python executable (from your venv)
python_executable = sys.executable

# --- Main Execution ---

print(f"Starting full pipeline...\n")

# Loop through each script and run it
for script_name in scripts_to_run:
    # Build the full path
    script_path = os.path.join(SCRIPT_DIRECTORY, script_name)
    
    print(f"--- EXECUTING: {script_name} ---")
    
    # Run the script and wait for it to complete
    # check=True will automatically raise an error and stop if the script fails
    subprocess.run([python_executable, script_path], check=True)
    
    print(f"--- COMPLETED: {script_name} ---\n")

print("--- Full pipeline finished. ---")