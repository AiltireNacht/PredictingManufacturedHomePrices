#NaiveBayes.py
"""Predicting housing prices using Naive Bayes model."""

import os
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB

ROOT = os.path.dirname(os.path.abspath(__file__))

def main():
    print("Predicting Housing Prices using Naïve Bayes Classifier")
    print()
    print("=======================")
    print()
    print("PRE-PROCESSING...")

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
    # Convert values in data to indices for regions
    training_data[:, 0] = training_data[:, 0] - 1
    testing_data[:, 0] = testing_data[:, 0] - 1

    # Convert values in data to indices for shipdates
    training_data[:, 1] = ((training_data[:, 1] % 100) + 12 * np.floor(((training_data[:, 1] - 201401) / 100)) - 1).astype(int)
    testing_data[:, 1] = ((testing_data[:, 1] % 100) + 12 * np.floor(((testing_data[:, 1] - 201401) / 100)) - 1).astype(int)

    # Make a list of possible values for sqft
    sqft_categories = np.unique(training_data[:, 2])
    sqft_categories = np.unique(sqft_categories - (sqft_categories % 100))
    sqft_categories = np.delete(sqft_categories, 22)

    # Convert values in data to indices for sqft
    training_data[:, 2] = (training_data[:, 2] / 100) - 4
    training_data[training_data[:, 2] > 21, 2] = 21
    training_data[training_data[:, 2] < 0, 2] = 0

    testing_data[:, 2] = (testing_data[:, 2] / 100) - 4
    testing_data[testing_data[:, 2] > 21, 2] = 21
    testing_data[testing_data[:, 2] < 0, 2] = 0

    # Convert values in data to indices for bedrooms
    training_data[:, 3] = training_data[:, 3] - 1
    training_data[np.where(training_data[:, 3] == 2), 3] =  training_data[np.where(training_data[:, 3] == 2), 3] - 1

    testing_data[:, 3] = testing_data[:, 3] - 1
    testing_data[np.where(testing_data[:, 3] == 2), 3] =  testing_data[np.where(testing_data[:, 3] == 2), 3] - 1

    # Make a list of possible values for price
    possible_prices = np.unique(training_labels)
    possible_prices = np.unique(possible_prices - (possible_prices % 25000))
    
    # Convert values in labels to indices for price
    training_labels = np.floor(training_labels / 25000).astype(int)

    testing_labels = np.floor(testing_labels / 25000).astype(int)

    # A couple more variables
    num_test = len(testing_labels)


    print("TRAINING...")

    clf = GaussianNB()
    clf.fit(training_data, training_labels)

    print("TESTING...")

    pred = clf.predict(testing_data)

    print("PERFORMANCE: ")

    print("Accuracy: {:.2f}%".format((np.sum(pred == testing_labels) / num_test) * 100))
    print("Confusion Matrix: ")
    cm = confusion_matrix(testing_labels, pred)
    print('\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in cm]))



if __name__ == "__main__":
    main()