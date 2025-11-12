# FEI Calculation Pipeline

This project calculates the Foul Exposure Indicator (FEI) by running a series of data processing scripts in sequence. It reads raw data from the `Input_Data` folder, processes it through several stages (AHP, Entropy, Weighting), and saves all intermediate and final results to the `Output_Data` folder.

## Project Structure

```
ZeroQU/
├── Input_Data/
│   ├── data.csv
│   ├── lkp_season.csv
│   └── variables.csv
├── Output_Data/
│   ├── (empty to start)
├── Sheet_Code/
│   ├── 1_Niche_Sheet.py
│   ├── 2_AHP_Sheet.py
│   ├── 3_Data_Sheet.py
│   ├── 4_Entropy_Sheet.py
│   ├── 4.1_Entropy_Weight_Fix.py
│   ├── 5_Weights_Sheet.py
│   └── 6_FEI_Output_Sheet.py
└── run_all.py
```

-----

## Setup

Before running the project, you need to set up your environment.

### 1\. Prerequisites

  * Python (3.10 or newer)
  * `pandas`
  * `numpy`

### 2\. Installation

1.  **Create a Virtual Environment:**
    Open your terminal in the `ZeroQU` project folder and run:

    ```bash
    python -m venv venv
    ```

2.  **Activate the Environment:**

      * **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```
      * **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```

3.  **Install Required Libraries:**

    ```bash
    pip install pandas numpy
    ```

### 3\. Prepare Input Data

Make sure your source files are correctly placed in the `Input_Data` folder:

  * `Input_Data/data.csv`
  * `Input_Data/lkp_season.csv`
  * `Input_Data/variables.csv`

-----

##  How to Run

To run the entire pipeline, simply execute the `run_all.py` script from the `ZeroQU` root directory.

```bash
# Make sure your virtual environment (venv) is activated!
python run_all.py
```

This script will automatically run all the Python files in the `Sheet_Code` folder in the correct order. You will see the console output for each script as it completes.

-----

## Output

All generated files will be saved in the `Output_Data` folder. This includes intermediate files and the final report.

**Generated Files:**

  * `niche_settings.csv`
  * `ahp.csv`
  * `environmental.csv` (from `3_Data_Sheet.py`)
  * `normalized.csv`
  * `proposition.csv`
  * `entropy.csv` (fixed by `4.1_Entropy_Weight_Fix.py`)
  * `weights.csv`
  * `FEI.csv` (the final output)

-----

## Script Breakdown

The `run_all.py` script executes the following files in order:

1.  **`1_Niche_Sheet.py`**: Calculates the `Niche_Score` from hardcoded values and saves it to `Output_Data/niche_settings.csv`.
2.  **`2_AHP_Sheet.py`**: Builds the AHP pairwise comparison matrix and saves the results to `Output_Data/ahp.csv`.
3.  **`3_Data_Sheet.py`**: Reads `Input_Data/data.csv` and `Input_Data/lkp_season.csv`, processes the raw data, and saves the combined `Output_Data/environmental.csv`.
4.  **`4_Entropy_Sheet.py`**: Calculates the Entropy weights (E\_j, d\_j, W\_j) from `environmental.csv` and saves `normalized.csv`, `proposition.csv`, and `entropy.csv` to `Output_Data/`.
5.  **`4.1_Entropy_Weight_Fix.py`**: A helper script to manually update the weights in `Output_Data/entropy.csv` (if needed).
6.  **`5_Weights_Sheet.py`**: Combines the weights from `ahp.csv` and `entropy.csv` using the `Alpha` constant and saves the final blended weights to `Output_Data/weights.csv`.
7.  **`6_FEI_Output_Sheet.py`**: The final step. It reads data from `environmental.csv`, `variables.csv`, and `weights.csv` to calculate the `Daily_FEI`, `FEI_Cum`, and `Risk_Band` for each row. The final report is saved as `Output_Data/FEI.csv`.