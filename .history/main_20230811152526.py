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
    parameters = [{'n_estimators': [10, 100, 1000], 'criterion': ['entropy', 'gini']}]
    grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10, n_jobs=-1)
    grid_search = grid_search.fit(X_train, y_train)
    best_accuracy = grid_search.best_score_
    best_parameters = grid_search.best_params_
    print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
    print("Best Parameters:", best_parameters)
    return best_accuracy, best_parameters

# create a function to perform random search
def random_search(classifier, X_train, y_train):
    parameters = [{'n_estimators': [10, 100, 1000], 'criterion': ['entropy', 'gini']}]
    random_search = RandomizedSearchCV(estimator=classifier, param_distributions=parameters, scoring='accuracy', cv=10, n_jobs=-1)
    random_search = random_search.fit(X_train, y_train)
    best_accuracy = random_search.best_score_
    best_parameters = random_search.best_params_
    print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
    print("Best Parameters:", best_parameters)
    return best_accuracy, best_parameters

# create a function to plot the confusion matrix
def plot_confusion_matrix(cm):
    plt.figure(figsize=(24, 16))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.show()

# create a function to plot the accuracy
def plot_accuracy(accuracies):
    plt.figure(figsize=(24, 16))
    sns.distplot(accuracies)
    plt.title('Accuracy')
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    plt.show()

# create a function to plot the classification report
def plot_classification_report(classification_report):
    plt.figure(figsize=(24, 16))
    sns.heatmap(classification_report, annot=True, cmap='Blues', fmt='g')
    plt.title('Classification Report')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.show()

# create a function to plot the ROC curve
def plot_roc_curve(fpr, tpr, auc):
    plt.figure(figsize=(24, 16))
    plt.plot(fpr, tpr, color='red', label='ROC (AUC = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()

# create a function to plot the precision-recall curve
def plot_precision_recall_curve(precision, recall, average_precision):
    plt.figure(figsize=(24, 16))
    plt.plot(recall, precision, color='red', label='Precision-Recall (AP = %0.2f)' % average_precision)
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower right')
    plt.show()

# create a function to plot the learning curve
def plot_learning_curve(classifier, X_train, y_train):
    train_sizes, train_scores, test_scores = learning_curve(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 50))
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.figure(figsize=(24, 16))
    plt.plot(train_sizes, train_mean, color='red', marker='o', markersize=5, label='Training Accuracy')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='red')
    plt.plot(train_sizes, test_mean, color='blue', marker='o', markersize=5, label='Validation Accuracy')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='blue')
    plt.title('Learning Curve')
    plt.xlabel('Training Examples')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()

# create a function to plot the validation curve
def plot_validation_curve(classifier, X_train, y_train):
    param_range = np.arange(1, 100, 1)
    train_scores, test_scores = validation_curve(estimator=classifier, X=X_train, y=y_train, param_name='n_estimators', param_range=param_range, cv=10, n_jobs=-1)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.figure(figsize=(24, 16))
    plt.plot(param_range, train_mean, color='red', marker='o', markersize=5, label='Training Accuracy')
    plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='red')
    plt.plot(param_range, test_mean, color='blue', marker='o', markersize=5, label='Validation Accuracy')
    plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='blue')
    plt.title('Validation Curve')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()

# create a function to plot the decision boundary
def plot_decision_boundary(classifier, X_train, y_train):
    X_set, y_set = X_train, y_train
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01), 
                         np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
    plt.figure(figsize=(24, 16))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), 
                 alpha=0.75, cmap=ListedColormap(('red', 'blue')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set==j, 0], X_set[y_set==j, 1], c=ListedColormap(('red', 'blue'))(i), label=j)
    plt.title('Decision Boundary')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.show()


# print the accuracy
print('Accuracy: %.2f%%' % (accuracy * 100))

# print the confusion matrix
print('Confusion Matrix:')
print(confusion_matrix)

# print the classification report
print('Classification Report:')
print(classification_report)

# plot the confusion matrix
plot_confusion_matrix(confusion_matrix)

# plot the classification report
plot_classification_report(classification_report)

# plot the ROC curve
plot_roc_curve(fpr, tpr, auc)

# plot the precision-recall curve
plot_precision_recall_curve(precision, recall, average_precision)

# plot the learning curve
plot_learning_curve(classifier, X_train, y_train)

# plot the validation curve
plot_validation_curve(classifier, X_train, y_train)

# plot the decision boundary
plot_decision_boundary(classifier, X_train, y_train)

# print the best parameters
print('Best Parameters:')
print(grid_search.best_params_)

# print the best score
print('Best Score: %.2f%%' % (grid_search.best_score_ * 100))

# print the best estimator
print('Best Estimator:')
print(grid_search.best_estimator_)

# print the best index
print('Best Index:')
print(grid_search.best_index_)



#use the data to create a relationship matrix between the US and a nation
# plot the relationship matrix
plot_relationship_matrix(relationship_matrix)






    #create a matrix of the relationship between the US and a nation
    #use the data to adjust relationsips of the rest of the nations based on the relationship matrix

#use the data  to adjust relationsips of the rest of the nations based on the relationship matrix







