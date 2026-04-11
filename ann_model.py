import tensorflow as tf
from tensorflow import keras
from keras_tuner import RandomSearch
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def build_model(hp):
    model = keras.Sequential()
    # Input layer
    model.add(keras.layers.Dense(
        hp.Int('units_0', min_value=32, max_value=512, step=32),
        activation=hp.Choice('activation', values=['relu', 'tanh']),
        input_shape=(10,)
    ))
    # Hidden layers
    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(keras.layers.Dense(
            hp.Int(f'units_{i+1}', min_value=32, max_value=512, step=32),
            activation=hp.Choice('activation', values=['relu', 'tanh'])
        ))
    # Output layer
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(hp.Float('lr', min_value=1e-4, max_value=1e-2, sampling='log')),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    print("Loading preprocessed data...")
    X_train = pd.read_csv('X_train.csv').values
    X_test = pd.read_csv('X_test.csv').values
    y_train = pd.read_csv('y_train.csv').values.ravel()
    y_test = pd.read_csv('y_test.csv').values.ravel()
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    print("Setting up Keras Tuner...")
    tuner = RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=15,
        executions_per_trial=1,
        directory='tuner_results',
        project_name='ann_churn'
    )

    print("Defining callbacks...")
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    print("Starting hyperparameter tuning...")
    tuner.search(
        X_train, y_train,
        epochs=50,
        validation_split=0.2,
        callbacks=[early_stopping]
    )

    print("Getting best hyperparameters...")
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Best hyperparameters:")
    print(best_hps.values)

    print("Building and training final model...")
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(
        X_train, y_train,
        epochs=100,
        validation_split=0.2,
        callbacks=[early_stopping]
    )

    print("Evaluating model...")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")

    # Predictions
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("Saving model...")
    model.save('ann_churn_model.h5')
    print("Model saved as 'ann_churn_model.h5'")

if __name__ == "__main__":
    main()
