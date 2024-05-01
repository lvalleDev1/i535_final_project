import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import os
import pandas as pd
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

# enable autologging
mlflow.sklearn.autolog()

os.makedirs("./outputs", exist_ok=True)


def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="path to train data")
    parser.add_argument("--test_data", type=str, help="path to test data")
    parser.add_argument("--registered_model_name", type=str, help="model name")
    parser.add_argument("--model", type=str, help="path to model file")
    args = parser.parse_args()

    # paths mounted as folder -> select file from folder
    train_df = pd.read_csv(select_first_file(args.train_data))
    test_df = pd.read_csv(select_first_file(args.test_data))

    X_train = train_df[
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
        ]
    ].to_numpy()
    
    y_train = train_df[
        [
           'Arrest Cat' 
        ]
    ].to_numpy()
    
    X_test = test_df[
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
        ]
    ].to_numpy()

    y_test = test_df[
        [
           'Arrest Cat' 
        ]
    ].to_numpy()
    
    y_train = y_train.ravel()
    y_test = y_test.ravel()
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    
    svm = LinearSVC(C=0.0001)
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)

    print(classification_report(y_test, y_pred))

    # registering the model to the workspace
    print("Registering the model via MLFlow")
    mlflow.sklearn.log_model(
        sk_model=svm,
        registered_model_name=args.registered_model_name,
        artifact_path=args.registered_model_name,
    )

    # saving the model to a file
    mlflow.sklearn.save_model(
        sk_model=svm,
        path=os.path.join(args.model, "trained_model"),
    )

    # stop Logging
    mlflow.end_run()


if __name__ == "__main__":
    main()
