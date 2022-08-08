# Credit_Risk_Analysis
## Overview of the Analysis
In this challenge, machine learning was used to determine credit card risk. Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Therefore, different techniques were used to train and evaluate models with unbalanced classes. These methods were selected form imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling.

## Data overview and preparation
Using the credit card credit dataset from Lending Club, a peer-to-peer lending services company, various algorithms were used to classify high-risk and low-risk loans.  The data was provided in csv format with about 100 columns and 116k rows and included loan information such as the loan amount, loan term, interest rate. 

A screenshot of modified data frame is shown below:
![image](https://user-images.githubusercontent.com/58461542/183506574-9ba4433b-e978-49cc-b63b-c2d4548c3dca.png)

 
The data cleaning included items such as dropping the null columns and rows, converting the interest rate to numerical values, and converting the target column values to low_risk and high_risk based on their values. Out of 68,817 applications, 347 of them were considered as high_risk (denoted as “0”) and the rest were low_risk loans (denoted as “1”). The data was split into training and testing subsets using the “train_test_split” function from the “sklearn library”.
The following algorithms used for ML classification:
* Over sampling- RandomOverSampler and SMOTE algorithms,
* Undersample - ClusterCentroids algorithm.
* Combinatorial - SMOTEENN algorithm
* Reduce bias algorithms- BalancedRandomForestClassifier and EasyEnsembleClassifier, 
For each algorithm, the following steps were followed:
* Use the resampled data to train a logistic regression model.
* Calculate the balanced accuracy score from “sklearn.metrics”.
* Print the confusion matrix from “sklearn.metrics”.
* Generate a classification report using the “imbalanced_classification_report” from “imblearn”.


## Results
As screen shot of the results for RandomOverSampler method is provided below. Note the accuracy score is relatively low (62%). The algorithm performs poorly in dealing with the minority class (high-risk loans) with low F1 scores of 0.02. The low F1 score was mainly due to the very low precision score of 0.01 indicating a large number of false-positive prediction for high-risk loans (mis-identifying many low-risk loans as high-risk)

![image](https://user-images.githubusercontent.com/58461542/183506746-69b28746-ba65-4713-8125-c50df1fb45da.png)

As screen shot of the results for SMOTE method is provided below. Note the accuracy score is relatively low (65%). The algorithm performs poorly in dealing with the minority class (high-risk loans) with low F1 scores of 0.02. The low F1 score was mainly due to the very low precision score of 0.01 indicating a large number of false-positive prediction for high-risk loans (mis-identifying many low-risk loans as high-risk)

 ![image](https://user-images.githubusercontent.com/58461542/183506783-aea1cbfa-cf26-4e32-91e6-16847ca12e71.png)

As screen shot of the results for ClusterCentroids method is provided below.   This method results in lowest accuracy score of all methods used. This is because with under sampling (form majority class), the prediction of majority class become more inaccurate. This results in recalls core of 0.47 for “low-risk” loans. This means only close to 50% of low risk loans are identified correctly which is not acceptable. 
 
 ![image](https://user-images.githubusercontent.com/58461542/183506817-9dcacced-909c-4713-80b6-8dc701f88bc2.png)

As screen shot of the results for SMOTEENN method is provided below. The SMOTEENN algorithm showed a relatively low accuracy score 64%. The model performed poorly in dealing with the minority class (high-risk loans) with a low F1 score of 0.02 which is mainly due to low precision score of 0.01. The recall score of 0.7 indicates that 30% of the high-risk loans were falsely predicted as low-risk loans. The performance of this model is better compared to the oversampling and undersampling techniques. 
 
![image](https://user-images.githubusercontent.com/58461542/183506846-3c5a6a51-291c-478b-af52-7ff206e8932d.png)

As screen shot of the results for BalancedRandomForestClassifier method is provided below. Both ensemble algorithms showed higher accuracy, precesion, recall and F-1 score compared to the ones discussed in previous sections. The recall score was very high for the model with the Easy Ensemble AdaBoost classifier indicating a very low number of false-negative outputs. The recall score of 0.91 means that only 9% of the high-risk loans was falsely predicted as low-risk loans. Both models performed very well predicting the low-risk loans with high F1, recall and precision scores.

![image](https://user-images.githubusercontent.com/58461542/183506884-73686a48-6b9e-46cc-8280-c1cd3fa240bf.png)
 
As screen shot of the results for EasyEnsembleClassifier method is provided below:
 
 ![image](https://user-images.githubusercontent.com/58461542/183506919-975f2861-f1ef-4f54-a426-8e41adab9727.png)

## Summary

A summary of various algorithms used to classify the high risk and low risk loans are presented in the following tables.

* Considering the imbalanced population, it is expected that over-sampling methods and under sampling methods would result better classification of majority class (Low-risk loans) than minority class (high-risk loans). The higher precision score for low-risk loans verifies this expectation.
* All six models do a reasonably decent job in classifying the low-risk loan applications (note the high precision score of 1.00). However, model 6 results in much better classification of low-risk loans shown by recall score of 0.97.  This means only 3% of “low-risk” loans were identified incorrectly as “high-risk”
* It is noted that Algorithms No. 5 and 6 (based on bias reduction) provide the highest F1 scores. The improvement is particularly significant for “high-risk loan”. 
* For identification of high-risk loans, it is noted that model 6 provides the highest recall score of 91%. That means 91% of “high-risk” loans are identified correctly and only 9% of “high-risk” loans were incorrectly identified as “low-risk”.  In this regard, Model 6 performance is significantly better than other models. Based on recall scores, all other classification techniques are probably not acceptable. For example, the next best model (models 4-5), still classify 30% of high-risk loans as low risk.


Summary for performance of various ML learning techniques for High-Risk Loans

| No  | Method |Algorithm |Accuracy |Precision |Recall |F1|
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 1  | Over-sampling  | RandomOverSampler  | 0.62  | 0.01  | 0.60  | 0.02  |
| 2  | Over-sampling  | SMOTE  | 0.65  | 0.01  | 0.64  |0.02  |
| 3  | Under-sampling  | ClusterCentroids  | 0.51  | 0.01  | 0.56  |0.01  |
| 4  | combinatorial |	SMOTEENN  | 0.64  | 0.01  | 0.70  | 0.02  |
| 5  | Bias reduction  | BalancedRandomForestClassifier  | 0.79  | 0.03  | 0.69  |0.06  |
| 6  | Bias reduction  | EasyEnsembleClassifier  | 0.92  | 0.07  | 0.91  |0.14  |


Summary for performance of various ML learning techniques for Low-Risk Loans

| No  | Method |Algorithm |Accuracy |Precision |Recall |F1|
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 1  | Over-sampling  | RandomOverSampler  | 0.62  | 1.00  | 0.65  | 0.79  |
| 2  | Over-sampling  | SMOTE  | 0.65  | 1.00  | 0.66  |0.79  |
| 3  | Under-sampling  | ClusterCentroids  | 0.51  | 1.00  | 0.47  |0.64  |
| 4  | combinatorial |	SMOTEENN  | 0.64  | 1.00  | 0.57  | 0.73  |
| 5  | Bias reduction  | BalancedRandomForestClassifier  | 0.79  | 1.00  | 0.89 |0.94  |
| 6  | Bias reduction  | EasyEnsembleClassifier  | 0.92  | 1.00  | 0.94  |0.97  |

