# Predicting Manufactured Home Prices

## INTRODUCTION

### REQUIRED LIBRARIES
- numpy
- os
- sklearn
- matplotlib

The goal of our project is to predict manufactured housing prices using data from the US Census. 

  The data includes:  
1. Region

   This is separated into 4 sections: NorthEast, Midwest, South, and West each represented by a number 1 - 4
2. Shipmonth

   Written as YYYYMM
3. Square Footage

4. Number of Bedrooms

    This is separated into two categories: 1, if the number of bedrooms is 2 or less, and 2, if the number of bedrooms is 3 or more


The files have been modified to better suit our needs for this project. If you wish to read up on the original data and its documentation, you can check it out [here](https://www.census.gov/data/datasets/2021/econ/MHS/puf.html)

For all files, make sure the data is in the same folder (not a subfolder). 

If you want to change the data for this project, it can be updated in any of the files by changing the string in the np.loadtxt lines. So long as it is formatted the same way, no other changes need to be made.

## K NEAREST NEIGHBOR

##### How to run the code:
To the run the code, all that you need is the data and press run. 

##### What the code does:
First the K-NN code trains the algorithm using the training data and the eight nearest neighbors. 
Then we use the algorithm to predict the output for the test data. The accuracy is also calculated by comparing the initial test output to the predicted output. 

The results are then shown using a confusion matrix to compare the accuracy of the test output compared to the predicted output

##### Meaning of the results:
The results shown in the confusion matrix show that the predicted values are fairly accurate and most of the values were correctly predicted. 

## NAIVE BAYES
##### How to run the code:
Ensure the data exits in the folder, and run the code

##### What the code does:
The code separates the data so that square footage is labeled by categories of 100 ft (eg. 400 is used to repesent any data between 400 and 499) and prices are labeled by categories of 25,000 dollars (eg. 25,000 represents data from 25,000 to 49,999). Then further changes all labels to indicies for ease of parsing.

The code then implements the Gaussian Naive Bayes algorithm using sklearn's GaussianNB library.

##### Results
The code has a prediction rate of approximately 44% and corresponding decisions are shown in the confusion matrix.

## RANDOM FOREST
##### How to run the code:
The only thing you need to run the code is the data. 

##### What the code does:
First the Random Forest code trains the algorithm using the Random Forest Classifier which uses the training data and labels. 

Then we use the algorithm to predict the output for the test data.

The results are then shown using a confusion matrix to compare the accuracy of the test output compared to the predicted output

##### Meaning of the results:
The results are shown in a confusion matrix. Most of the data was correctly predicted as you can see in the confusion matrix. 
