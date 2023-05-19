# Expert System Project
This project is the application of expert system course at college using ML algorithms using Spyder and Anaconda.

# Machine Learning Classification Project

This project aims to build and evaluate different machine learning classification models using the Titanic dataset. The dataset contains information about passengers aboard the Titanic, including features like age, gender, fare, and whether they survived or not.

## Dataset

The dataset used for this project is the Titanic dataset from Kaggle. You can download the dataset from the following link: [Titanic Dataset](https://www.kaggle.com/c/titanic/data?select=train.csv).

Please make sure to download the `train.csv` file from the dataset.

The dataset contains the following columns:

- `Survived`: Whether the passenger survived (0 = No, 1 = Yes)
- `Pclass`: Ticket class (1 = 1st class, 2 = 2nd class, 3 = 3rd class)
- `Sex`: Gender of the passenger
- `Age`: Age of the passenger
- `SibSp`: Number of siblings/spouses aboard
- `Parch`: Number of parents/children aboard
- `Fare`: Passenger fare
- `Embarked`: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## Data Preprocessing

Before building the models, the dataset underwent some preprocessing steps, including:

- Handling missing values: Missing values in the `Age` column were filled with the median age, while the `Cabin` column was dropped due to many missing values. The missing values in the `Embarked` column were filled with the most common value.
- Converting categorical variables: The categorical variables (`Sex` and `Embarked`) were converted into numerical values using one-hot encoding.

## Models and Evaluation

Several classification models were evaluated using cross-validation and the training data. The following models were used:

- Logistic Regression
- Linear Discriminant Analysis
- K-Nearest Neighbors
- Decision Tree
- Gaussian Naive Bayes
- Support Vector Machines

The models were evaluated using stratified k-fold cross-validation and the accuracy metric. Boxplots were used to compare the performance of each model.

## Best Model Performance

The best-performing model was selected based on cross-validation results, and its performance was evaluated on the testing data. The selected model was Support Vector Machines (SVM) with an automatic gamma value.

The performance metrics of the best model on the testing data are as follows:

- Accuracy Score: 0.8071748878923767
- Confusion Matrix: 

|   | Column 1 | Column 2 |
|---|----------|----------|
| Row 1 | 121      | 18       |
| Row 2 | 25       | 59       |

- Classification Report: 

|   | Precision | Recall | F1-Score | Support |
|---|-----------|--------|----------|---------|
| 0 |   0.83    |  0.87  |   0.85   |   139   |
| 1 |   0.77    |  0.70  |   0.73   |   84    |
|---|-----------|--------|----------|---------|
| Accuracy  |           |        |   0.81   |   223   |


---

Feel free to explore the code provided to understand the implementation details and reproduce the results.




