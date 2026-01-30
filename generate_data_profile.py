"""
Dataset Inspection Script

This script performs a complete inspection of the heart failure dataset
and generates a structured markdown report.

Constraints:
- Does NOT modify the dataset
- Does NOT remove or impute values
- Does NOT train any model
"""

import pandas as pd
import os


def inspect_dataset(data_path):
    """
    Perform complete dataset inspection.
    
    Args:
        data_path: Path to the CSV dataset
        
    Returns:
        dict: Dictionary containing all inspection results with keys:
            - column_names: List of column names
            - data_types: Dict mapping column names to pandas dtypes
            - missing_values: Dict mapping column names to missing value counts
            - total_records: Integer count of total rows
            - target_variable: String name of the target variable
            - class_distribution: Dict mapping class labels to counts
            - classification_type: String ('binary' or 'multi-class')
    
    Raises:
        FileNotFoundError: If the dataset file doesn't exist
        pd.errors.EmptyDataError: If the dataset is empty
        pd.errors.ParserError: If the CSV file is malformed
        KeyError: If the target variable column is not found
    """
    try:
        # Load the dataset
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError(f"Dataset file is empty: {data_path}")
    except pd.errors.ParserError as e:
        raise pd.errors.ParserError(f"Failed to parse dataset: {e}")
    
    # Define target variable (as documented in README.md)
    target_variable = 'DEATH_EVENT'
    
    # Validate target variable exists
    if target_variable not in df.columns:
        raise KeyError(f"Target variable '{target_variable}' not found in dataset. Available columns: {df.columns.tolist()}")
    
    # Extract inspection information
    inspection_results = {
        'column_names': df.columns.tolist(),
        'data_types': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'total_records': len(df),
        'target_variable': target_variable,
        'class_distribution': df[target_variable].value_counts().to_dict()
    }
    
    # Determine classification type
    num_classes = df[target_variable].nunique()
    inspection_results['classification_type'] = 'binary' if num_classes == 2 else 'multi-class'
    
    return inspection_results


def generate_markdown_report(results, output_path):
    """
    Generate a structured markdown report from inspection results.
    
    Args:
        results: Dictionary containing inspection results (from inspect_dataset)
        output_path: Path where the report should be saved (directories will be created if needed)
        
    Raises:
        OSError: If there are permission issues or disk space problems
    """
    report_lines = []
    
    # Title
    report_lines.append("# Dataset Profile Report")
    report_lines.append("")
    report_lines.append("**Dataset**: Heart Failure Clinical Records")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # Total number of patient records
    report_lines.append("## Summary Statistics")
    report_lines.append("")
    report_lines.append(f"- **Total Patient Records**: {results['total_records']}")
    report_lines.append(f"- **Number of Features**: {len(results['column_names'])}")
    report_lines.append("")
    
    # Target variable identification
    report_lines.append("## Target Variable")
    report_lines.append("")
    report_lines.append(f"- **Target Variable**: `{results['target_variable']}`")
    report_lines.append(f"- **Classification Type**: {results['classification_type'].capitalize()} Classification")
    report_lines.append("")
    
    # Class distribution
    report_lines.append("## Class Distribution")
    report_lines.append("")
    report_lines.append(f"Distribution of `{results['target_variable']}`:")
    report_lines.append("")
    for class_label, count in sorted(results['class_distribution'].items()):
        percentage = (count / results['total_records']) * 100
        report_lines.append(f"- **Class {class_label}**: {count} records ({percentage:.2f}%)")
    report_lines.append("")
    
    # Column names
    report_lines.append("## Column Names")
    report_lines.append("")
    report_lines.append("The dataset contains the following columns:")
    report_lines.append("")
    for i, col in enumerate(results['column_names'], 1):
        report_lines.append(f"{i}. `{col}`")
    report_lines.append("")
    
    # Data types
    report_lines.append("## Data Types")
    report_lines.append("")
    report_lines.append("| Column | Data Type |")
    report_lines.append("|--------|-----------|")
    for col in results['column_names']:
        dtype = str(results['data_types'][col])
        report_lines.append(f"| `{col}` | {dtype} |")
    report_lines.append("")
    
    # Missing values
    report_lines.append("## Missing Values")
    report_lines.append("")
    total_missing = sum(results['missing_values'].values())
    if total_missing == 0:
        report_lines.append("**No missing values detected in the dataset.**")
        report_lines.append("")
    else:
        report_lines.append("| Column | Missing Count |")
        report_lines.append("|--------|---------------|")
        for col in results['column_names']:
            missing_count = results['missing_values'][col]
            report_lines.append(f"| `{col}` | {missing_count} |")
        report_lines.append("")
    
    # Conclusion
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## Conclusion")
    report_lines.append("")
    report_lines.append(f"This dataset contains {results['total_records']} patient records with {len(results['column_names'])} features. ")
    report_lines.append(f"The target variable for prediction is `{results['target_variable']}`, making this a **{results['classification_type']} classification** task. ")
    
    if total_missing == 0:
        report_lines.append("The dataset is complete with no missing values, making it ready for analysis and modeling.")
    else:
        report_lines.append(f"The dataset contains {total_missing} missing values across various columns that may need to be addressed during preprocessing.")
    
    report_lines.append("")
    
    # Write to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✓ Report generated successfully: {output_path}")


def main():
    """Main function to execute dataset inspection and report generation"""
    data_path = "data/heart_failure.csv"
    output_path = "reports/data_profile.md"
    
    print("=" * 60)
    print("DATASET INSPECTION")
    print("=" * 60)
    print()
    
    # Check if dataset exists
    if not os.path.exists(data_path):
        print(f"✗ ERROR: Dataset not found at {data_path}")
        return
    
    print(f"Inspecting dataset: {data_path}")
    print()
    
    try:
        # Perform inspection
        results = inspect_dataset(data_path)
        
        # Display summary
        print(f"Total Records: {results['total_records']}")
        print(f"Total Features: {len(results['column_names'])}")
        print(f"Target Variable: {results['target_variable']}")
        print(f"Classification Type: {results['classification_type'].capitalize()}")
        print()
        
        # Generate report
        generate_markdown_report(results, output_path)
        
        print()
        print("=" * 60)
        print("INSPECTION COMPLETE")
        print("=" * 60)
        
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError, KeyError) as e:
        print(f"✗ ERROR: {e}")
        print()
        print("=" * 60)
        print("INSPECTION FAILED")
        print("=" * 60)
    except Exception as e:
        print(f"✗ UNEXPECTED ERROR: {e}")
        print()
        print("=" * 60)
        print("INSPECTION FAILED")
        print("=" * 60)


if __name__ == "__main__":
    main()
