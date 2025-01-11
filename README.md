Analysis and Classification of Student Data
This project involves analyzing and classifying a student dataset to predict their academic outcomes using various machine learning models. The workflow includes data preprocessing, splitting the dataset, model building, evaluation, and visualization.

1. Importing Libraries
The required libraries such as numpy, pandas, matplotlib, seaborn, and machine learning modules from scikit-learn are imported. These libraries facilitate data manipulation, visualization, and the implementation of machine learning models.

2. Reading the Dataset
The student dataset is loaded using Pandas, and basic information such as shape and column names is extracted. This helps understand the dataset's structure and features.
Dataset Shape: (4424, 37)
Features: 37, including student demographics, prior qualifications, and academic performance indicators.

3. Preprocessing
To enhance the model's performance and efficiency, preprocessing steps are applied:

Dropping Irrelevant Features: Removed features like Marital status, Nacionality, and Educational special needs to reduce noise.
Feature Scaling: Standardized numerical features to ensure they contribute equally to the model.
Label Encoding: Categorical variables like Daytime/evening attendance and Previous qualification were encoded into numerical values for compatibility with machine learning algorithms.

4. Splitting the Dataset
The dataset was divided into training (70%) and testing (30%) sets using train_test_split.
This ensures a proper evaluation of the model's performance on unseen data.

5. Building and Evaluating Models
Four machine learning algorithms were implemented, and their performance was evaluated:

A. Logistic Regression
Achieved an accuracy of 100%.
Precision, Recall, and F1-Score: 1.00 across all classes.
Confusion Matrix demonstrated perfect classification.

B. Support Vector Machine (SVM)
Kernel: linear
Accuracy: 100%.
Classification Report: Metrics were perfect across all classes.

C. K-Nearest Neighbors (KNN)
Number of Neighbors: 5.
Accuracy: 97%.
Slight misclassification observed in class 1, but overall performance was high.

D. Decision Tree
Criterion: Entropy, Max Depth: 2.
Achieved an accuracy of 100%.

The Decision Tree was visualized to understand the splitting criteria.


6. Visualizing Models
Decision Tree Visualization: The structure of the decision tree was visualized for training, testing, and overall data. The generated tree provided insights into how features like Curricular units and GDP influenced the outcomes.
Performance Metrics: Confusion Matrices and Classification Reports were generated for each model to highlight the strengths and areas of improvement.
7. Model Deployment
Models and preprocessing scalers were saved using joblib for future use.


Summary
All models performed exceptionally well, with Logistic Regression, SVM, and Decision Tree achieving 100% accuracy. KNN provided slightly lower accuracy but demonstrated robust results. This project showcases the power of machine learning in predicting student outcomes with the right preprocessing and feature selection.
