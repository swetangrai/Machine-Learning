import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle

def main():
    # Step 1: Read the Data
    print("Step 1: Reading the data...")
    df = pd.read_csv('C:\\NLP_With_Krishh\\annclassification\\Churn_Modelling.csv')
    print(f"Data shape: {df.shape}")
    print(df.head())
    print("Step 1 completed.\n")

    # Step 2: Drop the First Three Columns
    print("Step 2: Dropping the first three columns...")
    df.drop(df.columns[[0, 1, 2]], axis=1, inplace=True)
    print(f"Data shape after dropping: {df.shape}")
    print("Step 2 completed.\n")

    # Step 3: Label Encoding for Geography Column
    print("Step 3: Label encoding for Geography column...")
    le = LabelEncoder()
    df['Geography'] = le.fit_transform(df['Geography'])
    print("Geography column encoded.")
    print("Step 3 completed.\n")

    # Step 4: Handle Dummy Variable Trap (integrated with Step 5)

    # Step 5: One-Hot Encoding for Gender Column (with drop='first' to handle dummy trap)
    print("Step 5: One-hot encoding for Gender column...")
    ohe = OneHotEncoder(drop='first', sparse_output=False)
    encoded = ohe.fit_transform(df[['Gender']])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(['Gender']))
    df = pd.concat([df, encoded_df], axis=1)
    print("Gender column one-hot encoded with dummy trap handled.")
    print("Step 5 completed.\n")

    # Step 6: Drop Non-Encoded Gender Column
    print("Step 6: Dropping non-encoded Gender column...")
    df.drop('Gender', axis=1, inplace=True)
    print("Original Gender column dropped.")
    print("Step 6 completed.\n")

    # Step 7: Test-Train Split for ANN Model Building
    print("Step 7: Performing test-train split...")
    X = df.drop('Exited', axis=1)
    y = df['Exited']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print("Step 7 completed.\n")

    # Step 8: Standardize Using Standard Scaler
    print("Step 8: Standardizing the data...")
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    print("Data standardized.")
    print("Step 8 completed.\n")

    # Step 9: Save Encoders and Standard Scalers in Pickle Files
    print("Step 9: Saving encoders and scalers...")
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    with open('onehot_encoder.pkl', 'wb') as f:
        pickle.dump(ohe, f)
    with open('standard_scaler.pkl', 'wb') as f:
        pickle.dump(sc, f)
    print("Encoders and scalers saved as pickle files.")
    print("Step 9 completed.\n")

    # Additional: Save processed data
    print("Saving processed data...")
    pd.DataFrame(X_train).to_csv('X_train.csv', index=False)
    pd.DataFrame(X_test).to_csv('X_test.csv', index=False)
    pd.DataFrame(y_train).to_csv('y_train.csv', index=False)
    pd.DataFrame(y_test).to_csv('y_test.csv', index=False)
    print("Processed data saved as CSV files.")

if __name__ == "__main__":
    main()
