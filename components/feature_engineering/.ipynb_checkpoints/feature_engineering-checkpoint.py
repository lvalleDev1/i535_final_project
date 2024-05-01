import os
import argparse
import pandas as pd
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

    mlflow.log_metric("num_samples_original_data:", crime_df.shape[0])
    mlflow.log_metric("num_features_original_data", crime_df.shape[1] - 1)
    
    crime_df['Date'] = crime_df['Date'].str[0:10]
    crime_df['Datetime'] = pd.to_datetime(crime_df['Date'], format='%m/%d/%Y')
    crime_df['Day Name'] = crime_df['Datetime'].dt.day_name()
    
    crime_df['Day'] = crime_df['Date'].str[3:5]
    crime_df['Month'] = crime_df['Date'].str[0:2]
    
    crime_df['Updated On Day'] = crime_df['Updated On'].str[3:5]
    crime_df['Updated On Month'] = crime_df['Updated On'].str[0:2]
    crime_df['Updated On Year'] = crime_df['Updated On'].str[6:10]
    
    crime_df.drop(
        [
            'Date',
            'Updated On',
            'Datetime',
        ],
        axis=1,
        inplace=True
    )
    
    mlflow.log_metric("num_samples_preprocessed_data:", crime_df.shape[0])
    mlflow.log_metric("num_features_preprocessed_data", crime_df.shape[1] - 1)

    # output paths are mounted as folder, therefore, we are adding a filename to the path
    crime_df.to_csv(os.path.join(args.data_output, "crime_data_feature_engineered.csv"), index=False)

    # Stop Logging
    mlflow.end_run()


if __name__ == "__main__":
    main()
