# Jesse Moore
## Phase 3 Project - Customer Churn
### Instructor: Mark Barbour
### Blog Post: [Clarifying Confusion Tables](https://medium.com/@jessecmoore222/clarifying-confusion-tables-0918c091f337)

# Predictive Modeling for Customer Churn
![Customer Churn](images/pk-1.png)

## Overview

### SyriaTel wishes to understand more about why customers cancel their service ("churn") to help them develop a more effective retention strategy. In this report, I analyze their data using predictive modeling techniques to develop insight into their customer churn.

---

## Business Understanding

### Problem Statement

Customer retention, or 'Churn', is a critical challenge for our client, as high churn rates lead to significant revenue loss. Attracting a new customer costs [5 to 7 times more than keeping an existing customer](https://www.forbes.com/councils/forbesbusinesscouncil/2022/12/12/customer-retention-versus-customer-acquisition/). Currently, we lack a reliable method to predict which customers are likely to leave and why.

### Stakeholder Questions

- What features play the largest role in determining whether a customer will churn?
- What is the percentage of customers who churn?
- What states or area codes have the highest churn?
- What role do features such as voicemail, international plans/calls, and customer service calls play in our churn?

### Business Objective

The goal of this analysis is to develop a machine learning model that can accurately identify at-risk customers based on historical data. This will allow the company to take proactive measures to improve retention. By optimizing for recall, we aim to minimize false negatives and ensure that as many churn-prone customers as possible are correctly identified.

A false negative (Type II error) occurs when the model incorrectly predicts that a customer will stay, but in reality, they churn. In other words, the model fails to identify an at-risk customer. The False Negative Rate (FNR) is calculated as:

$$
\text{False Negative Rate (FNR)} = \frac{\text{False Negatives (FN)}}{\text{False Negatives (FN)} + \text{True Positives (TP)}}
$$

Where:
- **False Negatives (FN)**: The number of churned customers that the model incorrectly predicted as staying.
- **True Positives (TP)**: The number of churned customers that the model correctly predicted as churning.

For our churn model, minimizing false negatives is crucial because failing to identify a churn-prone customer means losing an opportunity to intervene and retain them.

To visualize this, we will use a confusion matrix, which provides a breakdown of the model’s predictions:

- **Top-left (True Positives - TP)**: Correctly identified churned customers.
- **Top-right (False Positives - FP)**: Customers incorrectly predicted to churn when they actually stayed.
- **Bottom-left (False Negatives - FN)**: Customers incorrectly predicted to stay when they actually churned.
- **Bottom-right (True Negatives - TN)**: Correctly identified retained customers.

By examining the confusion matrix, we can assess the trade-offs between precision and recall, ensuring that our model prioritizes correctly identifying at-risk customers.

---

## Data Understanding

### Data Source

Our dataset, SyriaTel's own internally collected data, can be found here: [Churn in Telecom's Dataset](https://www.kaggle.com/datasets/becksddf/churn-in-telecoms-dataset). It has been saved locally as `cust_churn.csv`.

### Data Description

The dataset contains 3,333 records of customer information, including:

- **Customer's state**, account length, area code, and phone number.
- **Plans**: international and/or voicemail plans.
- **Usage data**: number of voicemail messages received, total minutes, total calls, and charges for daytime, evening, nighttime, and international calls.
- **Target variable**: whether the customer ultimately churned.

We can see that the dataset has 3333 entries of 21 columns with no null values. Most of the columns are numeric, while `state`, `phone number`, `international plan`, and `voice mail plan` are categorical.

### Data Cleaning

- **Scaling**: Numeric values need to be scaled because they vary widely.
- **Categorical Encoding**: One-hot encoding will be applied to the `state` column, while `international plan`, `voice mail plan`, and `churn` will be mapped to binary values (0 or 1).
- **Dropping Irrelevant Columns**: `phone number` will be dropped as it is unique for each customer and doesn’t provide valuable insights.

### Churn Percentage Analysis

![Churn Counts](images/pchurn_counts.png)

The dataset is highly imbalanced, with a significant difference between the number of churned and non-churned customers. As a remedy for imbalance, we will apply class weights where applicable, instead of using SMOTE, which could lead to overfitting.

---

## Data Preparation

We engineered three new features: **total calls**, **total minutes**, and **total charges** by summing related columns. We also mapped categorical variables to binary values and dropped irrelevant columns.

---

## Modeling

### Baseline Model

The first step is to build a **baseline model** using just one feature, **customer service calls**. We’ll evaluate the performance, and then scale up by adding more features and tuning hyperparameters using GridSearchCV.

#### Model Metrics (Baseline)

- **Validation Accuracy**: 0.852
- **Precision**: 0.480
- **Recall**: 0.247
- **F1-Score**: 0.327

![Baseline Model](images/baseline_model.png)

While this model’s recall is low, we can see that it avoids overfitting early, which is critical in ensuring that we don’t miss important insights about our data.

---

## Second Model: Hyperparameter Tuning

We adjusted the **max_depth** of the Decision Tree to generalize better.

#### Model Metrics (Second Tuning)

- **Best Parameters**: `max_depth = 5`
- **Validation Accuracy**: 0.852
- **Precision**: 0.480
- **Recall**: 0.247
- **F1-Score**: 0.327

![Second Model](images/second_model.png)

Unfortunately, this did not improve recall, so we proceeded with adding more features.

---

## Third Model: Adding `total calls`

We added the `total calls` feature to see if it improves the model.

#### Model Metrics (Third Model)

- **Best Parameters**: `max_depth = 10`
- **Validation Accuracy**: 0.715
- **Precision**: 0.204
- **Recall**: 0.330
- **F1-Score**: 0.252

![Third Model](images/third_model.png)

There was an improvement in recall, but not by much.

---

## Fourth Model: Adding More Features

We added several more features: **voice mail plan**, **total charges**, and **customer service calls**.

#### Model Metrics (Fourth Model)

- **Best Parameters**: `max_depth = None`
- **Validation Accuracy**: 0.913
- **Precision**: 0.724
- **Recall**: 0.649
- **F1-Score**: 0.685

![Fourth Model](images/fourth_model.png)

This resulted in a significant improvement in recall, reducing the number of false negatives.

---
# Fifth Model

## We will now add min_samples_split to our gridsearch, knowing that a higher number here can help decrease overfitting.


- **Best Parameters**: {'model__criterion': 'entropy', 'model__max_depth': None, 'model__min_samples_split': 2}
- **Validation Accuracy**:  0.889
- **Precision**: 0.612
- **Recall**: 0.649
- **F1-Score**: 0.630
## While this had no effect on our recall, it actually caused our Accuracy and Precision to go down. 
![fifth model](images/fifth_model.png)

# Sixth Model

## Now we will adjust our hyperparameters again, trying to see if the higher number of 'min_samples_split' will lead a higher recall score, and also adding in min_samples_leaf. 

- **Best Parameters**:  {'model__criterion': 'entropy', 'model__max_depth': 20, 'model__min_samples_leaf': 11, 'model__min_samples_split': 2}
- **Validation Accuracy**:  0.817
- **Precision**: 0.419 
- **Recall**: 0.670 
- **F1-Score**:  0.516
## Another slight improvement to our recall score but not enough.
![sixth model](images/sixth_model.png)


# Seventh Model
- **Best Parameters**:  {'model__criterion': 'entropy', 'model__max_depth': 18, 'model__min_samples_leaf': 11, 'model__min_samples_split': 2}
- **Validation Accuracy**: 0.817
- **Precision**:  0.419
- **Recall**: 0.670
- **F1-Score**:  0.516
  
## As nothing seems to be improving, let us run all of our features and see what our recall is.
![seventh model](images/seventh_model.png)


# Eighth Model
![eigth model](images/eight_model.png)


# Ninth Model

## Let's try reducing our 'min_samples_leaf' hyperparameter.

# defining our param grid for grid searching. 

- **Best Parameters**:  {'model__criterion': 'gini', 'model__max_depth': None, 'model__max_features': 'log2', 'model__min_samples_leaf': 4, 'model__min_samples_split': 2}
- **Validation Accuracy**: 0.748
- **Precision**:  0.349
- **Recall**: 0.845
- **F1-Score**:  0.494
  
## This is a fairly good model, let's test it on our holdout set!



## Final Model: Hyperparameter Tuning and Feature Set Optimization

After tuning several hyperparameters and using the full set of features, we arrived at the following results:

#### Model Metrics (Final Model)

Test Set Final Evaluation:
    Accuracy: 0.744 | Precision: 0.340 | Recall: 0.833 | F1-Score: 0.483

![Final Model](images/final_model.png)

---

## Final Conclusions

In this analysis, we developed a predictive model for SyriaTel’s customer churn, achieving an 83.3% recall rate. The top features influencing churn include **total minutes**, **international plan**, and **customer service calls**.

Looking forward, we could incorporate external economic data, such as trends in customer locations or broader economic conditions, to improve prediction accuracy further.

---

### **Next Steps:**

- **Model Deployment**: Deploy the model for real-time churn prediction and customer retention.
- **Further Tuning**: Test other algorithms like Random Forest and XGBoost to compare performance.
- **Monitoring**: Implement model performance monitoring and periodic retraining as new data comes in.

---

### **Additional Notes**:

- All code and analysis can be found in the Jupyter notebook.
- Visualizations were generated using `matplotlib` and `seaborn`.

---

### **Contact**:
Jesse Moore  
[GitHub: JesseMooreDS](https://github.com/JesseMooreDS)  
[Medium Blog](https://medium.com/@jessecmoore222/clarifying-confusion-tables-0918c091f337)

---

