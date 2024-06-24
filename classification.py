from pycaret.classification import *
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import argparse
import os

def preprocess_data(train_data_path, train_labels_path, test_data_path):

    # Load features from CSV file
    X_train = pd.read_csv(train_data_path)
    Y_train = pd.read_csv(train_labels_path, header=None)
    X_test = pd.read_csv(test_data_path)

    # Check if the length of features and labels match
    if len(X_train) != len(Y_train):
        raise ValueError(f"Training data and labels have different lengths: {len(X_train)} != {len(Y_train)}")
    
    train_len = len(X_train)
    test_len = len(X_test)

    # Combine the training and test data to preprocess them together
    data = pd.concat([X_train, X_test], axis=0)

    # Fill missing values with NaN
    data[data == '?'] = np.nan

    # Fill missing values with the mode of the column
    categorical = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
    for col in categorical:
        data[col].fillna(data[col].mode()[0], inplace=True)

    # Encode categorical features
    for feature in categorical:
            le = preprocessing.LabelEncoder()
            data[feature] = le.fit_transform(data[feature])

    # Standardize the features
    scaler = StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns = data.columns)

    # Split the data back into training and test data
    X_train = data.iloc[:train_len, :]
    X_test = data.iloc[train_len:, :]

    # Concatenate the features and labels
    dataset = pd.concat([X_train, Y_train], axis=1)
    dataset.columns = list(X_train.columns) + ['label']

    # Save the combined and preprocessed data to a new CSV file
    dataset.to_csv('processed_data/processed_train_data.csv', index=False)
    X_test.to_csv('processed_data/processed_test_data.csv', index=False)

    print("Train Data has been successfully processed and saved to 'processed_train_data.csv'.")
    print("Test data has been successfully processed and saved to 'processed_test_data.csv'.")

def train(data_path):

    # Load the data
    data = pd.read_csv(data_path)

    # Initialize the setup
    exp = setup(data, target='label', session_id=1, experiment_name='Adult Census Income Classification', pca=True, pca_components=0.95, system_log=False)

    best = compare_models()

    return best

def predict(data_path, model):
    # Load the data
    data = pd.read_csv(data_path)

    # Make predictions
    predictions = predict_model(model, data)

    return predictions


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train', type=str, help='Path to the train data file', required=False, default='traindata.csv')
    argparser.add_argument('--train_labels', type=str, help='Path to the train data labels file', required=False, default='trainlabel.txt')
    argparser.add_argument('--test', type=str, help='Path to the test data file', required=False, default='testdata.csv')
    args = argparser.parse_args()

    # Create a new directory to save the pre-processed data and predictions
    os.makedirs('processed_data', exist_ok=True)
    os.makedirs('predictions', exist_ok=True)
    
    # Preprocess the training data
    preprocess_data(args.train, args.train_labels, args.test)

    # Train the model
    model = train('processed_data/processed_train_data.csv')

    # Make predictions on the test data
    predictions = predict('processed_data/processed_test_data.csv', model)

    # Save the predictions to a CSV file
    predictions.to_csv('predictions/predictions.csv', index=False)
    predictions[['prediction_label']].to_csv('predictions/testlabel.txt', index=False, header=False)