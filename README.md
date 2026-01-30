# UROP-B2-1

## Federated Learning Medical AI Project

This repository contains a federated learning project for medical AI, specifically for heart failure prediction.

## Dataset Validation

Before starting any federated learning experiments, you must validate the repository integrity and dataset availability.

### Prerequisites

Install required dependencies:

```bash
pip install -r requirements.txt
```

### Running the Validation Script

To validate the repository and dataset:

```bash
python validate_dataset.py
```

The validation script performs the following checks:

**Step 1: Repository Integrity Validation**
- Confirms that the `data/` folder exists
- Confirms that the `data/heart_failure.csv` file exists

**Step 2: Dataset Validation**
- Loads the dataset using pandas
- Displays the first 5 rows of the dataset
- Displays the dataset shape (rows and columns)

If all validations pass, the script exits with code 0. If any validation fails, the script reports the error and exits with code 1.

## Dataset Information

The `data/heart_failure.csv` dataset contains medical records for heart failure patients with the following features:

- **age**: Age of the patient
- **anaemia**: Decrease of red blood cells or hemoglobin (boolean)
- **creatinine_phosphokinase**: Level of the CPK enzyme in the blood (mcg/L)
- **diabetes**: If the patient has diabetes (boolean)
- **ejection_fraction**: Percentage of blood leaving the heart at each contraction
- **high_blood_pressure**: If the patient has hypertension (boolean)
- **platelets**: Platelets in the blood (kiloplatelets/mL)
- **serum_creatinine**: Level of serum creatinine in the blood (mg/dL)
- **serum_sodium**: Level of serum sodium in the blood (mEq/L)
- **sex**: Woman or man (binary)
- **smoking**: If the patient smokes or not (boolean)
- **time**: Follow-up period (days)
- **DEATH_EVENT**: If the patient deceased during the follow-up period (target variable)

## Project Structure

```
UROP-B2-1/
├── data/
│   └── heart_failure.csv      # Heart failure clinical records dataset
├── validate_dataset.py         # Repository and dataset validation script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Notes

- Do NOT modify the dataset directly
- Do NOT create derived datasets without proper documentation
- The validation script only checks existence and readability, it does not perform preprocessing or training