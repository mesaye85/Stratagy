import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Path: main.py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV



# create a function to load the data from the csv files "electoral-democracy-index.csv", "gdp per capita.csv", "gini-index.csv", "military_rankings.csv"
def load_data():
    data = pd.read_csv("electoral-democracy-index.csv")
    data1 = pd.read_csv("gdp per capita.csv")
    data2 = pd.read_csv("gini-index.csv")
    data3 = pd.read_csv("military_rankings.csv")
    return data, data1, data2, data3

# create a function to merge the data from the csv files "electoral-democracy-index.csv", "gdp per capita.csv", "gini-index.csv", "military_rankings.csv"
def merge_data(data, data1, data2, data3):
    data = pd.merge(data, data1, on='Country')
    data = pd.merge(data, data2, on='Country')
    data = pd.merge(data, data3, on='Country')
    return data

# create a function to clean the data
def clean_data(data):
    data = data.dropna()
    data = data.drop(['Country Code', 'Year'], axis=1)
    return data

# create a function to split the data into training and testing sets
def split_data(data):
    X = data.drop(['Status'], axis=1)
    y = data['Status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

# create a function to train the model
def train_model(X_train, y_train):
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42)
    classifier.fit(X_train, y_train)
    return classifier

# create a function to predict the model
def predict_model(classifier, X_test):
    y_pred = classifier.predict(X_test)
    return y_pred


# create a function to evaluate the model
def evaluate_model(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return cm

# create a function to perform k-fold cross validation
def k_fold_cross_validation(classifier, X_train, y_train):
    accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
    print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
    print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
    return accuracies

# create a function to perform grid search
def grid_search(classifier, X_train, y_train): 
    