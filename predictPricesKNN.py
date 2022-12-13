# Predict Manufactured Home Prices
# K-Nearest Neighbor

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


ROOT = os.path.dirname(os.path.abspath(__file__))

def main():

    # import data
    print("Loading training data...")
    training = np.loadtxt(os.path.join(ROOT, 'trainingdata.csv'), delimiter=',', dtype=int)
    x_train = training[:, :-1]
    y_train = training[:, -1]
    print("Loading testing data...")
    testing = np.loadtxt(os.path.join(ROOT, 'testingdata.csv'), delimiter=',', dtype=int)
    x_test = testing[:, :-1]
    y_test = testing[:, -1]

    x_train[:, 2] = (x_train[:, 2] / 100) - 4
    x_train[x_train[:, 2] > 21, 2] = 21
    x_train[x_train[:, 2] < 0, 2] = 0

    x_test[:, 2] = (x_test[:, 2] / 100) - 4
    x_test[x_test[:, 2] > 21, 2] = 21
    x_test[x_test[:, 2] < 0, 2] = 0

    # make a list of all possible values for price
    possible_prices = np.unique(y_train)
    possible_prices = np.unique(possible_prices - (possible_prices % 25000))

    # convert values to indices
    y_train = np.floor(y_train / 25000).astype(int)
    y_test = np.floor(y_test / 25000).astype(int)

    # train k-NN
    knn = KNeighborsClassifier(n_neighbors=8)
    knn.fit(x_train, y_train)

    # Predict on the test data
    pred = knn.predict(x_test)
    pred = pred.astype(int)

    # Calculate the accuracy of the model using test data
    print("Calculating accuracy...")
    print("Accuracy: {:.2f}%".format(knn.score(x_test, y_test) * 100))

    # show results
    cm = confusion_matrix(y_test, pred)
    print(cm)
    plt.figure(1)
    plt.scatter(x_test[:, 0], pred,  edgecolor='k', s=100)
    plt.title("Predicted Manufactured Housing Prices")
    plt.xlabel('Region')
    plt.ylabel('Price')

    plt.figure(2)
    plt.scatter(x_test[:, 0], y_test, edgecolor='k', s=100)
    plt.title("Actual Manufactured Housing Prices")
    plt.xlabel('Region')
    plt.ylabel('Price')

    print("Predictions by region: ")
    print("Northeast: ")
    for i in range(len(np.bincount(pred[np.where(x_test[:, 0] == 1)]))):
        print(np.bincount(pred[np.where(x_test[:, 0] == 1)])[i], end=" ")
        print(np.bincount(y_test[np.where(x_test[:, 0] == 1)])[i])
    
    print()
    print("Midwest: ")
    for i in range(len(np.bincount(pred[np.where(x_test[:, 0] == 2)]))):
        print(np.bincount(pred[np.where(x_test[:, 0] == 2)])[i], end=" ")
        print(np.bincount(y_test[np.where(x_test[:, 0] == 2)])[i])

    print()
    print("South: ")
    for i in range(len(np.bincount(pred[np.where(x_test[:, 0] == 3)]))):
        print(np.bincount(pred[np.where(x_test[:, 0] == 3)])[i], end=" ")
        print(np.bincount(y_test[np.where(x_test[:, 0] == 3)])[i])

    print()
    print("West: ")
    for i in range(len(np.bincount(pred[np.where(x_test[:, 0] == 4)]))):
        print(np.bincount(pred[np.where(x_test[:, 0] == 4)])[i], end=" ")
        print(np.bincount(y_test[np.where(x_test[:, 0] == 4)])[i])

    plt.show()

    plt.figure(3)
    plt.scatter(x_test[:, 1], pred, edgecolor='k', s=100)
    plt.title("Predicted Manufactured Housing Prices")
    plt.xlabel('Year')
    plt.ylabel('Price')

    plt.figure(4)
    plt.scatter(x_test[:, 1], y_test, edgecolor='k', s=100)
    plt.title("Actual Manufactured Housing Prices")
    plt.xlabel('Year')
    plt.ylabel('Price')

    plt.show()

    plt.figure(5)
    plt.scatter(x_test[:, 2], pred, edgecolor='k', s=100)
    plt.title("Predicted Manufactured Housing Prices")
    plt.xlabel('SQFT')
    plt.ylabel('Price')

    plt.figure(6)
    plt.scatter(x_test[:, 2], y_test, edgecolor='k', s=100)
    plt.title("Actual Manufactured Housing Prices")
    plt.xlabel('SQFT')
    plt.ylabel('Price')

    plt.show()

    plt.figure(7)
    plt.scatter(x_test[:, 3], pred, edgecolor='k', s=100)
    plt.title("Predicted Manufactured Housing Prices")
    plt.xlabel('Bedrooms')
    plt.ylabel('Price')

    plt.figure(8)
    plt.scatter(x_test[:, 3], y_test, edgecolor='k', s=100)
    plt.title("Actual Manufactured Housing Prices")
    plt.xlabel('Bedrooms')
    plt.ylabel('Price')

    print("Predictions by bedroom count: ")
    print("Two or less: ")
    for i in range(len(np.bincount(pred[np.where(x_test[:, 3] == 1)]))):
        print(np.bincount(pred[np.where(x_test[:, 3] == 1)])[i], end=" ")
        print(np.bincount(y_test[np.where(x_test[:, 3] == 1)])[i])
    
    print()
    print("Three or more: ")
    for i in range(len(np.bincount(pred[np.where(x_test[:, 3] == 3)]))):
        print(np.bincount(pred[np.where(x_test[:, 3] == 3)])[i], end=" ")
        print(np.bincount(y_test[np.where(x_test[:, 3] == 3)])[i])


    plt.show()



if __name__ == "__main__":
    main()