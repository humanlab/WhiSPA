import os, argparse
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel


TABLES_DIR = '/cronus_data/rrao/WhiSPA/tables/'


def paired_ttest(gt_df, baseline_df, pred_df):
    """
    Perform paired t-tests for each column (excluding 'user_id') between 
    baseline predictions and model predictions relative to ground truth.
    Includes only rows where 'user_id' is present in all three DataFrames.
    """
    # Find common user_ids across all three DataFrames
    common_user_ids = set(gt_df['user_id']).intersection(baseline_df['user_id'], pred_df['user_id'])

    # Filter the DataFrames to include only common user_ids
    gt_df = gt_df[gt_df['user_id'].isin(common_user_ids)].reset_index(drop=True)
    baseline_df = baseline_df[baseline_df['user_id'].isin(common_user_ids)].reset_index(drop=True)
    pred_df = pred_df[pred_df['user_id'].isin(common_user_ids)].reset_index(drop=True)

    # Find shared columns across all DataFrames
    shared_columns = [col for col in pred_df.columns if col != 'user_id']
    results = []

    for col in shared_columns:
        # Compute residuals (absolute errors)
        baseline_res = np.abs(gt_df[col] - baseline_df[col])
        pred_res = np.abs(gt_df[col] - pred_df[col])

        # Paired t-test
        t_stat, p_val = ttest_rel(baseline_res, pred_res)

        # Append results for this column
        results.append({"Column": col, "t-statistic": t_stat, "p-value": p_val})
    
    # Convert to DataFrame for better presentation
    result_df = pd.DataFrame(results).sort_values(by="p-value").reset_index(drop=True)
    return result_df

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Perform paired t-tests for residual errors across datasets.")
    parser.add_argument("--gt_csv", required=True, type=str, help="Path to the ground truth CSV file")
    parser.add_argument("--baseline_csv", required=True, type=str, help="Path to the baseline predictions CSV file")
    parser.add_argument("--pred_csv", required=True, type=str, help="Path to the predictions CSV file")
    args = parser.parse_args()

    # Prepare filepaths
    gt_path = os.path.join(TABLES_DIR, args.gt_csv)
    baseline_path = os.path.join(TABLES_DIR, args.baseline_csv)
    pred_path = os.path.join(TABLES_DIR, args.pred_csv)

    # Load the CSV files
    gt_df = pd.read_csv(gt_path).sort_values(by='user_id').reset_index(drop=True)
    baseline_df = pd.read_csv(baseline_path).drop(columns=['id']).rename(columns={'group_id': 'user_id'}).pivot(index="user_id", columns="feat", values="value").reset_index().sort_values(by='user_id')
    baseline_df.columns.name = None
    pred_df = pd.read_csv(pred_path).drop(columns=['id']).rename(columns={'group_id': 'user_id'}).pivot(index="user_id", columns="feat", values="value").reset_index().sort_values(by='user_id')
    pred_df.columns.name = None

    # Perform paired t-tests
    print("Paired T-Test Results:")
    print(paired_ttest(gt_df, baseline_df, pred_df))


if __name__ == "__main__":
    main()
