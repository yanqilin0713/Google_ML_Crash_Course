#@title Code - Load dependencies

# data
import numpy as np
import pandas as pd

# machine learning
import tensorflow as tf
from tensorflow import keras
import ml_edu.experiment
import ml_edu.results

# data visualization
import plotly.express as px
import matplotlib.pyplot as plt

# Load Dataset
chicago_taxi_dataset = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv")

# Updates dataframe to use specific columns.
training_df = chicago_taxi_dataset.loc[:, ('TRIP_MILES', 'TRIP_SECONDS', 'FARE', 'COMPANY', 'PAYMENT_TYPE', 'TIP_RATE')]

print('Read dataset completed successfully.')
print('Total number of rows: {0}\n\n'.format(len(training_df.index)))
training_df.head(200)

print('Total number of rows: {0}\n\n'.format(len(training_df.index)))
training_df.describe(include='all')

# What is the maximum fare?
max_fare = training_df['FARE'].max()
print("What is the maximum fare? 				Answer: ${fare:.2f}".format(fare = max_fare))

# What is the mean distance across all trips?
mean_distance = training_df['TRIP_MILES'].mean()
print("What is the mean distance across all trips? 		Answer: {mean:.4f} miles".format(mean = mean_distance))

# How many cab companies are in the dataset?
num_unique_companies =  training_df['COMPANY'].nunique()
print("How many cab companies are in the dataset? 		Answer: {number}".format(number = num_unique_companies))

# What is the most frequent payment type?
most_freq_payment_type = training_df['PAYMENT_TYPE'].value_counts().idxmax()
print("What is the most frequent payment type? 		Answer: {type}".format(type = most_freq_payment_type))

# Are any features missing data?
missing_values = training_df.isnull().sum().sum()
print("Are any features missing data? 				Answer:", "No" if missing_values == 0 else "Yes")

#@title Code - View correlation matrix
print(training_df.corr(numeric_only=True))

#@title Code - Define ML functions

def create_model(
    settings: ml_edu.experiment.ExperimentSettings,
    metrics: list[keras.metrics.Metric],
) -> keras.Model:
  """Create and compile a simple linear regression model."""
  # Describe the topography of the model.
  # The topography of a simple linear regression model
  # is a single node in a single layer.
  inputs = {name: keras.Input(shape=(1,), name=name) for name in settings.input_features}
  concatenated_inputs = keras.layers.Concatenate()(list(inputs.values()))
  outputs = keras.layers.Dense(units=1)(concatenated_inputs)
  model = keras.Model(inputs=inputs, outputs=outputs)

  # Compile the model topography into code that Keras can efficiently
  # execute. Configure training to minimize the model's mean squared error.
  model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=settings.learning_rate),
                loss="mean_squared_error",
                metrics=metrics)

  return model


def train_model(
    experiment_name: str,
    model: keras.Model,
    dataset: pd.DataFrame,
    label_name: str,
    settings: ml_edu.experiment.ExperimentSettings,
) -> ml_edu.experiment.Experiment:
  """Train the model by feeding it data."""

  # Feed the model the feature and the label.
  # The model will train for the specified number of epochs.
  features = {name: dataset[name].values for name in settings.input_features}
  label = dataset[label_name].values
  history = model.fit(x=features,
                      y=label,
                      batch_size=settings.batch_size,
                      epochs=settings.number_epochs)

  return ml_edu.experiment.Experiment(
      name=experiment_name,
      settings=settings,
      model=model,
      epochs=history.epoch,
      metrics_history=pd.DataFrame(history.history),
  )

print("SUCCESS: defining linear regression functions complete.")

# #@title Code - Experiment 1
#
# # The following variables are the hyperparameters.
# settings_1 = ml_edu.experiment.ExperimentSettings(
#     learning_rate = 0.01,
#     number_epochs = 20,
#     batch_size = 500,
#     input_features = ['TRIP_MILES']
# )
#
# metrics = [keras.metrics.RootMeanSquaredError(name='rmse')]
#
# model_1 = create_model(settings_1, metrics)
#
# experiment_1 = train_model('one_feature', model_1, training_df, 'FARE', settings_1)
#
# ml_edu.results.plot_experiment_metrics(experiment_1, ['rmse'])
# ml_edu.results.plot_model_predictions(experiment_1, training_df, 'FARE')
# dense_layer = next(
#     layer for layer in model_1.layers if layer.get_weights()
# )
#
# weights, bias = dense_layer.get_weights()
#
# for feature, weight in zip(settings_1.input_features, weights):
#     print(f"{feature}: {weight[0]}")
#
# print(f"Bias: {bias[0]}")


# @title Code - Experiment 3

# The following variables are the hyperparameters.
settings_3 = ml_edu.experiment.ExperimentSettings(
    learning_rate = 0.001,
    number_epochs = 20,
    batch_size = 50,
    input_features = ['TRIP_MILES', 'TRIP_MINUTES']
)

training_df['TRIP_MINUTES'] = training_df['TRIP_SECONDS']/60

metrics = [keras.metrics.RootMeanSquaredError(name='rmse')]

model_3 = create_model(settings_3, metrics)

experiment_3 = train_model('two_features', model_3, training_df, 'FARE', settings_3)

ml_edu.results.plot_experiment_metrics(experiment_3, ['rmse'])
ml_edu.results.plot_model_predictions(experiment_3, training_df, 'FARE')

for layer in model_3.layers:
    params = layer.get_weights()
    if params:   # only layers with weights
        weights, bias = params
        print(f"Layer: {layer.name}")
        print("Weights:")
        print(weights)
        print("Bias:")
        print(bias)

dense_layer = next(
    layer for layer in model_3.layers if layer.get_weights()
)

weights, bias = dense_layer.get_weights()

for feature, weight in zip(settings_3.input_features, weights):
    print(f"{feature}: {weight[0]}")

print(f"Bias: {bias[0]}")

# y_true = training_df['FARE'].values
# X1 = {"TRIP_MILES": training_df["TRIP_MILES"].to_numpy(dtype="float32")}
# y_pred_1 = experiment_1.model.predict(X1).squeeze()
# X3 = {
#     "TRIP_MILES": training_df["TRIP_MILES"].to_numpy(dtype="float32"),
#     "TRIP_MINUTES": training_df["TRIP_MINUTES"].to_numpy(dtype="float32"),
# }
# y_pred_3 = experiment_3.model.predict(X3).squeeze()
#
# rmse_1 = np.sqrt(np.mean((y_true - y_pred_1.squeeze())**2))
# rmse_3 = np.sqrt(np.mean((y_true - y_pred_3.squeeze())**2))
#
# plt.figure()
# plt.bar(['Experiment 1', 'Experiment 3'], [rmse_1, rmse_3])
# plt.ylabel('RMSE')
# plt.title('RMSE Comparison Across Experiments')
# plt.show()


# Validate Model
def format_currency(x):
  return "${:.2f}".format(x)

def build_batch(df, batch_size):
  batch = df.sample(n=batch_size).copy()
  batch.set_index(np.arange(batch_size), inplace=True)
  return batch

def predict_fare(model, df, features, label, batch_size=50):
  batch = build_batch(df, batch_size)
  predicted_values = model.predict_on_batch(x={name: batch[name].values for name in features})

  data = {"PREDICTED_FARE": [], "OBSERVED_FARE": [], "L1_LOSS": [],
          features[0]: [], features[1]: []}
  for i in range(batch_size):
    predicted = predicted_values[i][0]
    observed = batch.at[i, label]
    data["PREDICTED_FARE"].append(format_currency(predicted))
    data["OBSERVED_FARE"].append(format_currency(observed))
    data["L1_LOSS"].append(format_currency(abs(observed - predicted)))
    data[features[0]].append(batch.at[i, features[0]])
    data[features[1]].append("{:.2f}".format(batch.at[i, features[1]]))

  output_df = pd.DataFrame(data)
  return output_df

def show_predictions(output):
  header = "-" * 80
  banner = header + "\n" + "|" + "PREDICTIONS".center(78) + "|" + "\n" + header
  print(banner)
  print(output)
  return

output = predict_fare(experiment_3.model, training_df, experiment_3.settings.input_features, 'FARE')
show_predictions(output)
