Overview of the Analysis

The purpose of this analysis was to build a deep learning model using TensorFlow and Keras to predict whether an organization funded by Alphabet Soup will be successful. This binary classification problem required preprocessing the dataset, designing and training a neural network model, evaluating its performance, and optimizing it to achieve the highest possible accuracy.

Data Preprocessing

-Target Variable: The target variable for this model was the IS_SUCCESSFUL column, which indicates whether an organization was successful.
-Feature Variables: The features for the model included all other columns except for EIN and NAME, as they do not contribute to predicting the target.
-Removed Variables: The EIN and NAME columns were removed from the dataset as they are identifiers and not relevant for prediction.

Compiling, Training, and Evaluating the Model

  Model Architecture:
    -The neural network consisted of three layers:
    -Input Layer: Corresponding to the number of features in the dataset.
    -First Hidden Layer: 80 neurons with the ReLU activation function.
    -Second Hidden Layer: 30 neurons with the ReLU activation function.
    -Output Layer: 1 neuron with the sigmoid activation function for binary classification.
    -The original model achieved 72.84% accuracy with 56.2% loss.

  Optimization Attempts

  -Increased and decreased the number of neurons in hidden layers to provide the model with more capacity. 
  -Added an additional hidden layer to introduce more complexity to the model.
  -Tried different activation functions, including tanh and leaky ReLU, to see if they improved performance.
  -Adjusted the number of epochs, testing both increased and reduced training durations.
  -Tuned the batch size to check if different batch processing affected learning stability.
  -Tried one hot and embedded encoding of categorical variables.
  -Created ask_amt bins
  -Converted income_amt to numeric midpoints of the original ranges.

Summary

Overall, the deep learning model successfully classified organizations with a reasonable accuracy of 73.25%. While the target accuracy of 75% was not met, the model still demonstrated good predictive capabilities.

Alternative Model Recommendation

A different machine learning model, such as Random Forest or Gradient Boosting (XGBoost), could be a viable alternative for solving this classification problem. These models often handle categorical variables more effectively, require less feature scaling, and can sometimes outperform deep learning models on structured tabular data. Additionally, hyperparameter tuning and feature engineering could further enhance performance.
