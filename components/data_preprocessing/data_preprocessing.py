import os
import argparse
import pandas as pd
import logging
import mlflow


def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data", type=str, help="path to input data")
    parser.add_argument("--preprocessed_data", type=str, help="path to output data")
    args = parser.parse_args()

    # start Logging
    mlflow.start_run()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    print("input data:", args.raw_data)

    crime_df = pd.read_csv(args.raw_data)
    
    mlflow.log_metric("num_samples_original_data:", crime_df.shape[0])
    mlflow.log_metric("num_features_original_data", crime_df.shape[1])
    
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
            'IUCR',
            'Beat',
            'FBI Code',
            'ID'
        ],
        axis=1,
        inplace=True
    )
    crime_df = crime_df[~crime_df['X Coordinate'].isna()]
    
    mlflow.log_metric("num_samples_preprocessed_data:", crime_df.shape[0])
    mlflow.log_metric("num_features_preprocessed_data", crime_df.shape[1])

    # output path mounted as folder -> add filename to the path
    crime_df.to_csv(os.path.join(args.preprocessed_data, "crime_data_preprocessed.csv"), index=False)

    # stop Logging
    mlflow.end_run()


if __name__ == "__main__":
    main()
