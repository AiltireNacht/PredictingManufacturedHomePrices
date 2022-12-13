# Predict Manufactured Home Prices
# Random Forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix  
import numpy as np
import matplotlib.pyplot as plt
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

def main():
    print("Predicting Housing Prices using Naïve Bayes Classifier")
    print("=======================")
    print("PRE-PROCESSING")
    print("=======================")

    # Load data from relevant files
    print("Loading training data...")
    training = np.loadtxt(os.path.join(ROOT, 'trainingdata.csv'), delimiter=',', dtype=int)
    training_data = training[:, :-1]
    training_labels = training[:, -1]
    print("Loading testing data...")
    testing = np.loadtxt(os.path.join(ROOT, 'testingdata.csv'), delimiter=',', dtype=int)
    testing_data = testing[:, :-1]
    testing_labels = testing[:, -1]

    # Extract useful parameters
    # Make a list of possible values for sqft
    sqft_categories = np.unique(training_data[:, 2])
    sqft_categories = np.unique(sqft_categories - (sqft_categories % 100))
    sqft_categories = np.delete(sqft_categories, 22)

    #convert values to indices
    training_data[:, 2] = (training_data[:, 2] / 100) - 4
    training_data[training_data[:, 2] > 21, 2] = 21
    training_data[training_data[:, 2] < 0, 2] = 0

    testing_data[:, 2] = (testing_data[:, 2] / 100) - 4
    testing_data[testing_data[:, 2] > 21, 2] = 21
    testing_data[testing_data[:, 2] < 0, 2] = 0

    #Make a list of possible values for price
    possible_prices = np.unique(training_labels)
    possible_prices = np.unique(possible_prices - (possible_prices % 25000))
    
    #convert values to indices
    training_labels = np.floor(training_labels / 25000).astype(int)
    testing_labels = np.floor(testing_labels / 25000).astype(int)

    # create classifier object  
    classifier= RandomForestClassifier(n_estimators= 15, criterion="entropy")  
    classifier.fit(training_data, training_labels)
    
    # predict the test result
    y_pred= classifier.predict(testing_data) 

    print("Calculating accuracy...")
    print("Accuracy: {:.2f}%".format(classifier.score(testing_data, testing_labels) * 100)) 
    
    # create confusion matrix to determine correct predictions
    cm = confusion_matrix(testing_labels, y_pred)
    print(cm)
    
    
    plt.figure(1)
    plt.scatter(testing_data[:, 0], y_pred, edgecolor='k', s=100)
    plt.title("Predict Manufactured Housing Prices")
    plt.xlabel('Region')
    plt.ylabel('Price')
    
    plt.figure(2)
    plt.scatter(testing_data[:, 0], testing_labels, edgecolor='k', s=100)
    plt.title("Predict Manufactured Housing Prices")
    plt.xlabel('Region')
    plt.ylabel('Price')

    print("Predictions by region: ")
    print("Northeast: ")
    for i in range(len(np.bincount(y_pred[np.where(testing_data[:, 0] == 1)]))):
        print(np.bincount(y_pred[np.where(testing_data[:, 0] == 1)])[i], end=" ")
        print(np.bincount(testing_labels[np.where(testing_data[:, 0] == 1)])[i])
    
    print()
    print("Midwest: ")
    for i in range(len(np.bincount(y_pred[np.where(testing_data[:, 0] == 2)]))):
        print(np.bincount(y_pred[np.where(testing_data[:, 0] == 2)])[i], end=" ")
        print(np.bincount(testing_labels[np.where(testing_data[:, 0] == 2)])[i])

    print()
    print("South: ")
    for i in range(len(np.bincount(y_pred[np.where(testing_data[:, 0] == 3)]))):
        print(np.bincount(y_pred[np.where(testing_data[:, 0] == 3)])[i], end=" ")
        print(np.bincount(testing_labels[np.where(testing_data[:, 0] == 3)])[i])

    print()
    print("West: ")
    for i in range(len(np.bincount(y_pred[np.where(testing_data[:, 0] == 4)]))):
        print(np.bincount(y_pred[np.where(testing_data[:, 0] == 4)])[i], end=" ")
        print(np.bincount(testing_labels[np.where(testing_data[:, 0] == 4)])[i])

    plt.show()


    plt.figure(1)
    plt.scatter(testing_data[:, 1], y_pred, edgecolor='k', s=100)
    plt.title("Predicted Manufactured Housing Prices")
    plt.xlabel('Year')
    plt.ylabel('Price')
    
    plt.figure(2)
    plt.scatter(testing_data[:, 1], testing_labels, edgecolor='k', s=100)
    plt.title("Predicted Manufactured Housing Prices")
    plt.xlabel('Year')
    plt.ylabel('Price')

    plt.show()
    
    plt.figure(1)
    plt.scatter(testing_data[:, 2], y_pred, edgecolor='k', s=100)
    plt.title("Predicted Manufactured Housing Prices")
    plt.xlabel('SQFT')
    plt.ylabel('Price')
    
    plt.figure(2)
    plt.scatter(testing_data[:, 2], testing_labels, edgecolor='k', s=100)
    plt.title("Predicted Manufactured Housing Prices")
    plt.xlabel('SQFT')
    plt.ylabel('Price')

    plt.show()
    
    plt.figure(1)
    plt.scatter(testing_data[:, 3], y_pred, edgecolor='k', s=100)
    plt.title("Predicted Manufactured Housing Prices")
    plt.xlabel('Bedrooms')
    plt.ylabel('Price')
    
    plt.figure(2)
    plt.scatter(testing_data[:, 3], testing_labels, edgecolor='k', s=100)
    plt.title("Predicted Manufactured Housing Prices")
    plt.xlabel('Bedrooms')
    plt.ylabel('Price')

    print("Predictions by bedroom count: ")
    print("Two or less: ")
    for i in range(len(np.bincount(y_pred[np.where(testing_data[:, 3] == 1)]))):
        print(np.bincount(y_pred[np.where(testing_data[:, 3] == 1)])[i], end=" ")
        print(np.bincount(testing_labels[np.where(testing_data[:, 3] == 1)])[i])
    
    print()
    print("Three or more: ")
    for i in range(len(np.bincount(y_pred[np.where(testing_data[:, 3] == 3)]))):
        print(np.bincount(y_pred[np.where(testing_data[:, 3] == 3)])[i], end=" ")
        print(np.bincount(testing_labels[np.where(testing_data[:, 3] == 3)])[i])

    plt.show()
    

if __name__ == "__main__":
    main()