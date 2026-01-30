"""
Federated Learning Medical AI Project - Dataset Validation Script

This script validates the repository integrity and dataset availability
for the heart failure prediction project.

Requirements:
- Verify data/ folder exists
- Verify data/heart_failure.csv exists
- Load and validate dataset using pandas
- Display basic dataset information
"""

import os
import sys
import pandas as pd


def validate_repository_integrity():
    """
    Step 1: Verify repository integrity
    - Confirm that a folder named `data/` exists.
    - Confirm that the file `data/heart_failure.csv` exists.
    """
    print("=" * 60)
    print("STEP 1: Repository Integrity Validation")
    print("=" * 60)
    
    # Check data/ folder
    data_folder = "data/"
    if os.path.exists(data_folder) and os.path.isdir(data_folder):
        print(f"✓ Folder '{data_folder}' exists")
    else:
        print(f"✗ ERROR: Folder '{data_folder}' does not exist")
        return False
    
    # Check data/heart_failure.csv file
    data_file = "data/heart_failure.csv"
    if os.path.exists(data_file) and os.path.isfile(data_file):
        print(f"✓ File '{data_file}' exists")
    else:
        print(f"✗ ERROR: File '{data_file}' does not exist")
        return False
    
    print("\nRepository integrity validation: PASSED\n")
    return True


def validate_dataset():
    """
    Step 2: Dataset validation
    - Load the dataset using pandas.
    - Print the first 5 rows.
    - Print dataset shape.
    - If the file does not exist or cannot be loaded, STOP execution and report the error.
    """
    print("=" * 60)
    print("STEP 2: Dataset Validation")
    print("=" * 60)
    
    data_file = "data/heart_failure.csv"
    
    try:
        # Load the dataset using pandas
        df = pd.read_csv(data_file)
        print(f"✓ Dataset loaded successfully\n")
        
        # Print dataset shape
        print(f"Dataset Shape: {df.shape}")
        print(f"  - Rows: {df.shape[0]}")
        print(f"  - Columns: {df.shape[1]}\n")
        
        # Print the first 5 rows
        print("First 5 rows of the dataset:")
        print("-" * 60)
        print(df.head())
        print("-" * 60)
        
        print("\nDataset validation: PASSED")
        return True
        
    except FileNotFoundError:
        print(f"✗ ERROR: File '{data_file}' not found")
        return False
    except pd.errors.EmptyDataError:
        print(f"✗ ERROR: File '{data_file}' is empty")
        return False
    except pd.errors.ParserError as e:
        print(f"✗ ERROR: Failed to parse '{data_file}': {e}")
        return False
    except Exception as e:
        print(f"✗ ERROR: Failed to load dataset: {e}")
        return False


def main():
    """Main function to run all validation steps"""
    print("\n" + "=" * 60)
    print("FEDERATED LEARNING MEDICAL AI PROJECT")
    print("Dataset Validation")
    print("=" * 60 + "\n")
    
    # Step 1: Verify repository integrity
    if not validate_repository_integrity():
        print("\n" + "=" * 60)
        print("VALIDATION FAILED: Repository integrity check failed")
        print("=" * 60)
        sys.exit(1)
    
    # Step 2: Dataset validation
    if not validate_dataset():
        print("\n" + "=" * 60)
        print("VALIDATION FAILED: Dataset validation failed")
        print("=" * 60)
        sys.exit(1)
    
    # All validation passed
    print("\n" + "=" * 60)
    print("ALL VALIDATIONS PASSED SUCCESSFULLY")
    print("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()
