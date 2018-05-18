# Bankruptcy-Prediction
This project aims to apply machine learning algorithms to identify the companies that went bankrupt using certain measures. 

Data Description
Our dataset was downloaded from Kaggle.
Specifically, there are 83 financial and non-financial characteristic variables for 422 companies in 14 different time periods. The dimension of the dataset is 3673 by 86. 
Attached below is the link to the Kaggle dataset: 
https://www.kaggle.com/shebrahimi/financial-distress

Data Preprocessing
Step 1: Outlier Detection & Treatment
Step 2: Identify bankruptcy
Step 3: Centering& Scaling
Step 4: Data Splitting  

Model Implementation
Since the financial distress index is numeric and continuous, we can set threshold on the data and conduct classification. We may also try clustering if our computational power allows. For this part, we will split the training set and testing set by company. 
Model Explanation 1:
Bankrupted companies:  32.2%,
Non-bankrupted companies: 67.8%
Since it is imbalanced and skewed, we used SMOTE and ROSE to balance the data. We found that he results for using SMOTE were better.
Model Explanation 2:
Using Flexible discriminant analysis as the benchmark, we see that the validation sensitivity has increased from 33.78% to 56.25% which is an improvement. 

Key terms Explanation:
Extreme Gradient Boost is a machine learning technique for regression and classification problems, which produces a prediction model in the form of an ensembe of weak prediction models.

Sensitivity and specificity are two useful criteria to evaluate the predictive accuracy of a binary classifier.

Sensitivity = (Number of True Positives) ÷ (Number of True Positive + Number of False Negatives)

Conclusion & Takeaways
Extreme Gradient Boosting has the highest sensitivity of 56.25%.
Exploratory Data Analysis and Data Preprocessing are extremely important as they are the essential steps before fitting any model. Without those steps, precision would be significantly lowered and the difficulty of conducting the fitting would be really hard.
Overall, we find that Extreme Gradient Boost is the best fitted model.
