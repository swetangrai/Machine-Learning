### Detailed Instructions for AI Agent: Create ANN Model Training Script

**Objective:** Create a Python script that builds, tunes, and trains an ANN classification model for the Churn_Modelling dataset. Use the preprocessed data (X_train.csv, X_test.csv, y_train.csv, y_test.csv) and saved scalers/encoders if needed. The script should use Keras Tuner for hyperparameter optimization, include early stopping, and train the final model with the best parameters found.

**Requirements:**
- Use Python 3.x with TensorFlow/Keras and Keras Tuner.
- Load preprocessed data from CSVs.
- Build a sequential ANN model with tunable hyperparameters (layers, units, activation, learning rate).
- Use Keras Tuner with `max_trials=15`.
- Include early stopping callback (monitor 'val_loss', patience=10).
- After tuning, train the model with best hyperparameters.
- Evaluate on test data and save the trained model (e.g., as HDF5 or SavedModel).
- Print progress, best hyperparameters, and evaluation metrics.
- Handle binary classification (Exited: 0/1).

**Step-by-Step Instructions for the AI Agent:**

1. **Import Libraries:**
   - Import necessary libraries: `tensorflow`, `keras`, `keras_tuner`, `pandas`, `numpy`, `sklearn.metrics` (for evaluation).
   - Code example: `import tensorflow as tf; from tensorflow import keras; from keras_tuner import RandomSearch; import pandas as pd; import numpy as np`

2. **Load Preprocessed Data:**
   - Load X_train, X_test, y_train, y_test from CSV files.
   - Convert to numpy arrays if needed.
   - Code example: `X_train = pd.read_csv('X_train.csv').values; y_train = pd.read_csv('y_train.csv').values.ravel()`

3. **Define Model Builder Function for Keras Tuner:**
   - Create a function that builds the ANN model with hyperparameters.
   - Use `hp` for tunable params: number of layers (1-3), units per layer (32-512), activation ('relu', 'tanh'), learning rate (0.001-0.1).
   - Model: Sequential with Dense layers, input shape from X_train, output layer with sigmoid for binary classification.
   - Compile with Adam optimizer and binary_crossentropy loss.
   - Code example: `def build_model(hp): model = keras.Sequential(); model.add(keras.layers.Dense(hp.Int('units', 32, 512), activation=hp.Choice('activation', ['relu', 'tanh']), input_shape=(X_train.shape[1],))); ... model.compile(optimizer=keras.optimizers.Adam(hp.Float('lr', 1e-4, 1e-2)), loss='binary_crossentropy', metrics=['accuracy']); return model`

4. **Set Up Keras Tuner:**
   - Use `RandomSearch` with the build_model function, max_trials=15, executions_per_trial=1, directory='tuner_results', project_name='ann_churn'.
   - Code example: `tuner = RandomSearch(build_model, objective='val_accuracy', max_trials=15, executions_per_trial=1, directory='tuner_results', project_name='ann_churn')`

5. **Define Callbacks:**
   - Early stopping: monitor='val_loss', patience=10, restore_best_weights=True.
   - Code example: `early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)`

6. **Run Hyperparameter Tuning:**
   - Search with tuner.search(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[early_stopping]).
   - Code example: `tuner.search(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[early_stopping])`

7. **Get Best Hyperparameters:**
   - Retrieve best hyperparameters.
   - Print them.
   - Code example: `best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]; print(best_hps.values)`

8. **Build and Train Final Model:**
   - Build model with best hyperparameters.
   - Train on full train data with early stopping.
   - Code example: `model = tuner.hypermodel.build(best_hps); history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, callbacks=[early_stopping])`

9. **Evaluate Model:**
   - Evaluate on X_test, y_test.
   - Print accuracy, loss, classification report.
   - Code example: `loss, accuracy = model.evaluate(X_test, y_test); print(f'Accuracy: {accuracy}')`

10. **Save the Model:**
    - Save as HDF5 or SavedModel.
    - Code example: `model.save('ann_churn_model.h5')`

**Additional Guidelines:**
- Structure the script with a main function or sequential execution.
- Print progress messages (e.g., "Tuning started").
- Assume data is preprocessed and scaled.
- If needed, load scalers from pickle/ for inverse transforms, but not required here.
- Test the script for errors (e.g., shapes, imports).
- Expected output: Tuned model, saved file, evaluation metrics.

**Expected Output:**
- A complete Python script file (e.g., `ann_model.py`).
- Tuner results directory.
- Saved model file.
- Console output with best params and metrics.