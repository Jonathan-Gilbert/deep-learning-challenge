Alphabet Soup Funding Predictability: A Deep Learning Approach
Overview of the Analysis

The purpose of this analysis is to build a deep learning model that can predict whether applicants will be successful if funded by Alphabet Soup. This involves:

    Data Preprocessing: Identifying and preparing the relevant data for model training.
    Model Development: Constructing a neural network architecture using TensorFlow/Keras.
    Hyperparameter Tuning: Optimizing the model's performance by adjusting its settings.
    Model Evaluation: Assessing the model's predictive accuracy on a test dataset.

Results
Data Preprocessing

    Target Variable: IS_SUCCESSFUL (binary: 1 for successful, 0 for unsuccessful)
    Features:
        APPLICATION_TYPE
        AFFILIATION
        CLASSIFICATION
        USE_CASE
        ORGANIZATION
        STATUS
        INCOME_AMT
        SPECIAL_CONSIDERATIONS
        ASK_AMT
    Variables Removed: EIN and NAME were removed as they are neither targets nor informative features for prediction. These are unique identifiers and do not contribute to the model's learning.

Compiling, Training, and Evaluating the Model

    Neural Network Architecture:
        Layers: 4 Dense layers were used.
        Neurons: The number of neurons in each layer was determined through hyperparameter tuning using Keras Tuner.
            First layer: Varied between 32 and 512 neurons.
            Second layer: Varied between 32 and 512 neurons.
            Third layer: Varied between 32 and 512 neurons.
            Output layer: 1 neuron with a sigmoid activation function for binary classification.
        Activation Functions: ReLU activation was used for the hidden layers to introduce non-linearity and improve learning. A sigmoid activation was used for the output layer to produce a probability between 0 and 1, representing the likelihood of success.
    Target Model Performance: The target model performance was to achieve an accuracy above 75%.
    Steps to Increase Model Performance:
        Hyperparameter Tuning: Keras Tuner's Hyperband algorithm was employed to find the optimal number of neurons in each layer and the learning rate for the Adam optimizer.
        Early Stopping: Implemented early stopping to prevent overfitting and reduce training time. The model stopped training when the validation loss did not improve for a certain number of epochs (patience).
        Validation Split: A portion of the training data was reserved for validation, allowing us to monitor the model's performance on unseen data during training.
    Outcomes:
        The hyperparameter search yielded the best configuration for the model, leading to improved accuracy.
        Early stopping helped prevent overfitting and ensured the model generalized well to new data.

Conclusion

The deep learning model developed for Alphabet Soup funding prediction showed promising results. Through careful data preprocessing, model selection, and hyperparameter tuning, we were able to achieve 73.24% and were unable to achieve the target of 75%.

This model could be a valuable tool for Alphabet Soup in making more informed funding decisions. By predicting the likelihood of success, resources can be allocated more efficiently to support organizations with the highest potential for positive impact.

Report generated via Google Colab using Gemini.
