# Methodology summary

## Structured literature review

The literature review is structured around four themes: environmental TinyML, multicriteria decision analysis for IoT, edge-cloud architectures, and public digital governance. It is not presented as a PRISMA systematic review.

## Contextual detection

For each station, the contextual score uses only the previous 24 observations:

`z(t) = abs(x(t) - mean_previous_24(t)) / std_previous_24(t)`

A record is flagged when `z(t) >= 2.5`. The current observation is excluded from the rolling reference.

## Prediction task

The target is positive when at least one of the next three hourly PM2.5 observations exceeds 25 µg/m³. The feature vector contains the current PM2.5 value, three lags, rolling statistics based only on previous observations, and calendar variables.

## Temporal evaluation

Each station is ordered chronologically and split into 70% training, 10% validation, and 20% testing. The scaler is fitted on training data only. The classification threshold is selected on validation data by maximizing F1; the test set is evaluated once.

## Network and quantization

The network uses 11 inputs, dense layers with 16 and 8 ReLU units, and one sigmoid output. It is trained for 50 epochs with Adam, binary cross-entropy, and training-set class weights. The model is exported to full-integer INT8 TensorFlow Lite using a representative sample drawn only from training data.

## Sensitivity analyses

FDTE weight sensitivity uses 100,000 weight vectors drawn from a uniform Dirichlet distribution. TCO sensitivity independently varies Edge and cloud totals by ±20% across 100,000 Monte Carlo simulations.
