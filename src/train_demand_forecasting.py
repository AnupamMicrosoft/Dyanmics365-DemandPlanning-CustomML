# Demand Forecasting Model Training Script

import argparse
import os
import pandas as pd
import sys
sys.path.append('/anaconda/envs/azureml_py38/lib/python3.8/site-packages')
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import joblib
import mlflow
import mlflow.xgboost

def main():
    parser = argparse.ArgumentParser()

    # Define arguments for the script
    parser.add_argument('--data-path', type=str, help='Path to the input CSV file')
    parser.add_argument('--model-output', type=str, default='demand_forecast_model',
                        help='Path to output the trained model')

    # Enable auto-logging to MLflow
    mlflow.start_run()
    mlflow.xgboost.autolog()

    args = parser.parse_args()

    
    # Read the first two rows for parameters
    with open(args.data_path, 'r') as file:
        first_line = file.readline().strip().split(',')
        second_line = file.readline().strip().split(',')

    # Create dictionaries for standard and custom parameters
    standard_params = ['DateGranularity', 'PeriodStartDate', 'PeriodEndDate', 'PredictionTimeWindow', 'MeasureColumn', 'TimeColumn']
    params = dict(zip(first_line, second_line))

    # Validate and convert standard parameter values
    params['PeriodStartDate'] = pd.to_datetime(params['PeriodStartDate'])
    params['PeriodEndDate'] = pd.to_datetime(params['PeriodEndDate'])
    params['PredictionTimeWindow'] = int(params['PredictionTimeWindow'])

    # Define the TimeColumn and MeasureColumn based on standard parameters
    TimeColumn = params['TimeColumn']
    MeasureColumn = params['MeasureColumn']

    # Read and process the data
    data = pd.read_csv(args.data_path, skiprows=lambda x: x in [0, 1, 2] or pd.isna(x), header=0)
    print("Raw Dimensions Data (first 5 rows):")
    print(data.head())
    # Date range check
    print(f"Date range in the file: {data[TimeColumn].min()} to {data[TimeColumn].max()}")

    # Convert TimeColumn to datetime
    data[TimeColumn] = pd.to_datetime(data[TimeColumn], errors='coerce')
    # Filter data based on the date range
    if not data[(data[TimeColumn] >= params['PeriodStartDate']) & (data[TimeColumn] <= params['PeriodEndDate'])].empty:
        data = data[(data[TimeColumn] >= params['PeriodStartDate']) & (data[TimeColumn] <= params['PeriodEndDate'])]
    else:
        print("No data within the specified date range. Please check the PeriodStartDate and PeriodEndDate.")
    # Aggregate data based on DateGranularity


    if params['DateGranularity'] == 'M':
        data[TimeColumn] = data[TimeColumn].dt.to_period('M').dt.to_timestamp()
    elif params['DateGranularity'] == 'W':
        data[TimeColumn] = data[TimeColumn].dt.to_period('W').dt.to_timestamp()
    # Add conditions for other granularities if required

    # Feature engineering: Extract year, month, day as separate columns
    # Assuming 'data' is your DataFrame and 'TimeColumn' holds the correct column name
    # Confirm the correct column name for the time column
    print(f"Time column as per parameters: '{TimeColumn}'")

    # Print the actual column names from the DataFrame for verification
    print("Actual column names in DataFrame:")
    print(data.columns.tolist())

    # Check if TimeColumn exists in the DataFrame and convert it to datetime
    if TimeColumn in data.columns:
        data[TimeColumn] = pd.to_datetime(data[TimeColumn], errors='coerce')

        # Feature engineering: Extract year, month, day as separate columns
        data['Year'] = data[TimeColumn].dt.year
        data['Month'] = data[TimeColumn].dt.month
        data['Day'] = data[TimeColumn].dt.day

        # Now you can drop the original time column as it's been replaced by more specific features
        data.drop(TimeColumn, axis=1, inplace=True)
    else:
        print(f"Column '{TimeColumn}' not found in the data. Please check the DataFrame columns.")
    # List of columns to drop
    columns_to_drop = ['_Value', 'ProductVariantName', 'WarehouseLocationName', 'P1', 'P2']

    # Drop only if the column exists in the DataFrame
    columns_to_drop = [col for col in columns_to_drop if col in data.columns]


    # Now drop the columns
    X = data.drop(columns_to_drop, axis=1)
    y = data['_Value']

    # Ensure that the target variable '_Value' is converted to numeric
    y = pd.to_numeric(y, errors='coerce')

    # Drop any rows with NaN in the target variable
    data.dropna(subset=['_Value'], inplace=True)

     # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = XGBRegressor()
    model.fit(X_train, y_train)
        # Evaluate the model
    accuracy = model.score(X_test, y_test)
    print(f'Accuracy of XGBoost regressor on test set: {accuracy:.2f}')
    mlflow.log_metric('Accuracy', accuracy)

    # Register the model
    registered_model_name = "DemandForecastingModel"
    print("Registering the model via MLflow")
    mlflow.xgboost.log_model(
        xgb_model=model,
        registered_model_name=registered_model_name,
        artifact_path=registered_model_name
    )

    # Optionally, save the model to a file
    joblib.dump(model, os.path.join(args.model_output, "demand_forecast_model.joblib"))

    mlflow.end_run()

if __name__ == '__main__':
    main()
    
    
    
