# Load libraries
from pandas import read_csv
import pandas as pd
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Load dataset
dataset = read_csv('~/Downloads/train.csv')


print(dataset.shape)
print(dataset.head())

print(dataset.describe())

#number of missing values in each column
print(dataset.isnull().sum())

#filling the missing values in Age column using the median value
median_age = dataset['Age'].median()
dataset['Age'].fillna(median_age, inplace=True)

#we can drop the cabin column, it has many missing values
dataset.drop('Cabin', axis=1, inplace=True)

#for the embarked column we can fill missing values with the most common value
most_common_embarked = dataset['Embarked'].mode()[0]
dataset['Embarked'].fillna(most_common_embarked, inplace=True)

#converting categorical variables ("Sex", "Embarked") to numerical 
dataset = pd.get_dummies(dataset, columns=['Sex', 'Embarked'])

X = dataset[['Pclass', 'Sex_female', 'Sex_male', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]
y = dataset['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Scale the training and testing data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define a list of models
models = []
models.append(('LR', LogisticRegression(random_state=0)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier(random_state=0)))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# Evaluate each model using cross-validation
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
    cv_results = cross_val_score(model, X_train_scaled, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Compare the performance of each model using boxplots
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

# Fit the best model on the training data and evaluate its performance on the testing data
best_model = SVC(gamma='auto')
best_model.fit(X_train_scaled, y_train)
y_pred = best_model.predict(X_test_scaled)

print("Accuracy Score:")
print(accuracy_score(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Classification Report:")
print(classification_report(y_test, y_pred))
