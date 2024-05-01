import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import mlflow


def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_input", type=str, help="path to input data")
    parser.add_argument("--data_output", type=str, help="path to output data")
    args = parser.parse_args()

    # Start Logging
    mlflow.start_run()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    print("input data:", args.data_input)

    crime_df = pd.read_csv(args.data_input, header=1, index_col=0)

    mlflow.log_metric("num_samples_original_data:", credit_df.shape[0])
    mlflow.log_metric("num_features_original_data", credit_df.shape[1] - 1)
    
    crime_df.drop(
        [
            'Case Number',
            'Ward',
            'Community Area',
            'District',
            'Location Description',
            'Description',
            'Location',
            'Block',
            'IUCR'
        ],
        axis=1,
        inplace=True
    )
    
    mlflow.log_metric("num_samples_preprocessed_data:", credit_df.shape[0])
    mlflow.log_metric("num_features_preprocessed_data", credit_df.shape[1] - 1)

    # output paths are mounted as folder, therefore, we are adding a filename to the path
    credit_df.to_csv(os.path.join(args.data_output, "crime_data_preprocessed.csv"), index=False)

    # Stop Logging
    mlflow.end_run()


if __name__ == "__main__":
    main()
