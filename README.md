%%markdown
# Heart Disease Prediction Project

## Project Overview

This project aims to build a machine learning model to predict the presence of heart disease based on various health indicators. The goal is to develop an accurate and reliable model that can assist in the early identification of individuals at risk of heart disease.

The dataset used for this project is named `heart.csv`. It contains a collection of health-related attributes for individuals, including age, sex, chest pain type, resting blood pressure, cholesterol levels, fasting blood sugar, resting electrocardiogram results, maximum heart rate achieved, exercise-induced angina, oldpeak (ST depression), and the slope of the peak exercise ST segment. The dataset also includes the target variable, 'HeartDisease', which indicates whether the individual has heart disease (1) or not (0).

%%markdown
## Exploratory Data Analysis (EDA)

Exploratory Data Analysis was conducted to understand the structure, content, and basic statistics of the dataset, and to identify potential issues like missing values, duplicates, and outliers.

1.  **Initial Data Inspection:** We started by checking the shape of the DataFrame to see the number of rows and columns (`df.shape`), examining the data types and non-null counts for each column using `df.info()`, and viewing the first few rows with `df.head()` to get a sense of the data format. Descriptive statistics were generated using `df.describe()` to understand the central tendency, dispersion, and shape of the numerical features.
2.  **Handling Missing Values and Duplicates:** We checked for missing values using `df.isnull().sum()` and confirmed there were none. Duplicate rows were identified using `df.duplicated().sum()`, and it was found that there were no duplicates in the dataset. For the `Cholesterol` and `RestingBP` columns, values of 0 were treated as missing or erroneous data and were imputed with the mean of the non-zero values in their respective columns.
3.  **Distribution of Numerical Features:** Histograms with KDE plots were generated for the numerical features 'Age', 'RestingBP', 'Cholesterol', and 'MaxHR' to visualize their distributions and identify any skewness or potential outliers.
4.  **Relationship between Categorical Features and Heart Disease:** Count plots were used to explore the relationship between categorical features ('Sex', 'ChestPainType', 'FastingBS') and the target variable 'HeartDisease'. These plots showed the distribution of heart disease cases within each category.
5.  **Relationship between Numerical Features and Heart Disease:** Box plots and violin plots were used to visualize the relationship between 'HeartDisease' and continuous variables like 'Cholesterol' and 'Age', helping to understand if there are significant differences in these attributes between individuals with and without heart disease.
6.  **Correlation Analysis:** A heatmap of the correlation matrix for numerical features was generated using `df.corr(numeric_only=True)` and `seaborn.heatmap` to visualize the linear relationships between these variables.

%%markdown
## Data Preprocessing and Cleaning

Before training machine learning models, the data underwent several preprocessing and cleaning steps to handle categorical variables and prepare the features for modeling.

1.  **Handling Categorical Variables:** Categorical features in the dataset were converted into a numerical format suitable for machine learning algorithms using one-hot encoding. The `pd.get_dummies()` function was used for this purpose. The `drop_first=True` parameter was set to avoid multicollinearity, which occurs when predictor variables are highly correlated with each other, potentially causing issues in some models.
2.  **Type Conversion:** The resulting columns from one-hot encoding were initially of boolean type (`True`/`False`). These were explicitly converted to integer type (`1`/`0`) to ensure compatibility with the machine learning libraries and models used.
3.  **Feature and Target Separation:** The dataset was split into features (independent variables, denoted as `X`) and the target variable (dependent variable, denoted as `y`), which is 'HeartDisease'.
4.  **Data Splitting:** The `X` and `y` data were split into training and testing sets using `train_test_split` from `sklearn.model_selection`. A `test_size` of 0.2 (20% of the data) was allocated for the test set, and `stratify=y` was used to ensure that the proportion of the target variable ('HeartDisease') is the same in both the training and testing sets as in the original dataset. This is particularly important for imbalanced datasets. A `random_state` was set for reproducibility.
5.  **Feature Scaling:** Numerical features (`X_train` and `X_test`) were scaled using `StandardScaler` from `sklearn.preprocessing`. This process standardizes the features by removing the mean and scaling to unit variance. The scaler was fitted *only* on the training data (`X_train_scaled = scaler.fit_transform(X_train)`) to learn the scaling parameters (mean and standard deviation) from the training distribution. This fitted scaler was then used to transform both the training and testing data (`X_test_scaled = scaler.transform(X_test)`). This ensures that the test data is scaled based on the training data's characteristics, preventing data leakage from the test set into the training process.

%%markdown
## Modeling and Evaluation

Several machine learning models were trained and evaluated to predict heart disease.

1.  **Models Used:** The following classification models were employed:
    *   Logistic Regression
    *   K-Nearest Neighbors (KNN)
    *   Gaussian Naive Bayes
    *   Decision Tree Classifier
    *   Support Vector Machine (SVM) with RBF Kernel

2.  **Training Process:** Each model was trained using the scaled training data (`X_train_scaled`) and the corresponding training labels (`y_train`). The `fit()` method was called on each model instance with the scaled training data to learn the underlying patterns and relationships.

3.  **Prediction:** After training, each model was used to make predictions on the scaled test data (`X_test_scaled`) using the `predict()` method.

4.  **Evaluation Metrics:** The performance of each model was evaluated using the following metrics:
    *   **Accuracy:** The proportion of correctly classified instances (both positive and negative) out of the total number of instances in the test set. It is calculated as (True Positives + True Negatives) / Total Instances.
    *   **F1 Score:** The harmonic mean of precision and recall. It is a good metric for evaluating models on datasets with imbalanced classes, as it considers both false positives and false negatives. The F1 Score is calculated as 2 \* (Precision \* Recall) / (Precision + Recall).

The accuracy and F1 Score for each model were calculated and stored in the `results` list.


%%markdown
## Results

The trained models were evaluated based on their Accuracy and F1 Score on the test dataset. The performance of each model is summarized below:


display(pd.DataFrame(results))
	Model	 Accuracy	 F1 Score
0	Logistic Regression	0.8750	0.8878
1	KNN	0.8859	0.8986
2	Naive Bayes	0.8696	0.8788
3	Decision Tree	0.7500	0.7604
4	SVM (RBF Kernel)	0.8641	0.8804



%%markdown

From the results, the K-Nearest Neighbors (KNN) model achieved the highest Accuracy (0.8859) and F1 Score (0.8986), indicating it is the best-performing model among those tested for this dataset and problem. Logistic Regression and Naive Bayes also showed strong performance, while the Decision Tree had the lowest performance metrics. The SVM with RBF kernel performed comparably to Logistic Regression and Naive Bayes.


%%markdown
## How to Use the Saved Model

To use the saved model for making predictions on new data, you will need the saved model file (`KNN_heart.pkl`), the fitted scaler object (`scaler.pkl`), and the list of original column names (`columns.pkl`).

1.  **Load the Saved Artifacts:** Use the `joblib.load()` function to load the saved model, scaler, and column names.

    ```python
    import joblib
    loaded_model = joblib.load('KNN_heart.pkl')
    loaded_scaler = joblib.load('scaler.pkl')
    loaded_columns = joblib.load('columns.pkl')
    ```

2.  **Prepare New Data for Prediction:** Your new data for prediction should be in a pandas DataFrame format, with the same column names and in the same order as the original training data (`X`). Ensure that the categorical columns are already one-hot encoded and numerical columns are in their raw form before scaling.

    Let's assume your new data is in a DataFrame called `new_data_df`.

3.  **Scale the New Data:** Apply the loaded scaler to your new data DataFrame. The `transform()` method should be used for new data.

    ```python
    new_data_scaled = loaded_scaler.transform(new_data_df[loaded_columns])
    ```
    Note: Make sure the columns in `new_data_df` are in the same order as `loaded_columns`.

4.  **Make Predictions:** Use the loaded model's `predict()` method on the scaled new data to get the predictions.

    ```python
    predictions = loaded_model.predict(new_data_scaled)
    ```

5.  **Interpret Predictions:** The `predictions` array will contain the predicted class for each instance in your new data. A value of `0` indicates the model predicts no heart disease, and `1` indicates the model predicts heart disease.

6.  
