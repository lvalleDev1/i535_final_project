import os
import argparse
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import logging
import mlflow

def select_first_file(path):
    """Selects first file in folder, use under assumption there is only one file in folder
    Args:
        path (str): path to directory or file to choose
    Returns:
        str: full path of selected file
    """
    files = os.listdir(path)
    return os.path.join(path, files[0])


# start Logging
mlflow.start_run()


def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed_data", type=str, help="path to input data")
    parser.add_argument("--train_data", type=str, help="path to train data")
    parser.add_argument("--test_data", type=str, help="path to test data")
    args = parser.parse_args()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    print("input data:", args.preprocessed_data)
    
    crime_df = pd.read_csv(select_first_file(args.preprocessed_data))
    
    print(f"log 1 - crime_df.shape: {crime_df.shape}")
    print(f"log 2 - crime_df.columns: {crime_df.columns}")

    mlflow.log_metric("num_samples_preprocessed_data:", crime_df.shape[0])
    mlflow.log_metric("num_features_preprocessed_data", crime_df.shape[1])
    
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
    
    mlflow.log_metric("num_samples_feature_engineered_data:", crime_df.shape[0])
    mlflow.log_metric("num_features_feature_engineered_data", crime_df.shape[1])
    
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(crime_df['Primary Type'])
    crime_df['Primary Type Cat'] = label_encoder.transform(crime_df['Primary Type'])
    label_encoder.fit(crime_df['Arrest'])
    crime_df['Arrest Cat'] = label_encoder.transform(crime_df['Arrest'])
    label_encoder.fit(crime_df['Domestic'])
    crime_df['Domestic Cat'] = label_encoder.transform(crime_df['Domestic'])
    label_encoder.fit(crime_df['Day Name'])
    crime_df['Day Name Cat'] = label_encoder.transform(crime_df['Day Name'])
    
    df_hotencoded = crime_df[
        [
            'Primary Type Cat',
            'Domestic Cat',
            'Day Name Cat',
            'Day',
            'Month',
            'Year',
            'Updated On Day',
            'Updated On Month',
            'Updated On Year',
            'X Coordinate',
            'Y Coordinate',
            'Latitude',
            'Longitude',
            'Arrest Cat'
        ]
    ]
    
    mlflow.log_metric("num_samples_hotencoded_data:", crime_df.shape[0])
    mlflow.log_metric("num_features_hotencoded_data", crime_df.shape[1])
    
    train_df, test_df = train_test_split(
        df_hotencoded,
        test_size=0.2,
    )

    # output paths mounted as folder -> add filename to the path
    train_df.to_csv(os.path.join(args.train_data, "training_data.csv"), index=False)
    test_df.to_csv(os.path.join(args.test_data, "testing_data.csv"), index=False)

    # Stop Logging
    mlflow.end_run()


if __name__ == "__main__":
    main()
