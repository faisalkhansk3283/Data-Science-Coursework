#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


lendingClub_data = pd.read_csv("C:\\Users\\juver\\Downloads\\Lending.csv")


# In[3]:


lendingClub_data 


# In[4]:


lendingClub_data.columns


# In[5]:


lendingClub_data.describe()


# In[6]:


import matplotlib.pyplot as plt

# column names of dataset
columns = lendingClub_data.columns

# columns for subplots
num_rows = len(columns)
num_cols = 3

# subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 20))

# Flatten the axes array 
axes = axes.flatten()

# Plot histograms for each column
for i, column in enumerate(columns):
    if i >= len(columns):
        break
    ax = axes[i]
    ax.hist(lendingClub_data[column], bins=20, edgecolor='k')
    ax.set_title(f'Histogram of {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')

# Remove any empty
for i in range(len(columns), num_rows * num_cols):
    fig.delaxes(axes[i])

# Adjust spacing between subplots
fig.tight_layout()

# Display subplots
plt.show()


# In[7]:


#converting into numeric values
lendingClub_data = lendingClub_data.replace({'residence_property': {'Own':0,'Rent':1}})

import matplotlib.pyplot as plt

# column names
columns = lendingClub_data.columns

# Define the number of rows and columns for subplots
num_rows = len(columns) 
num_cols = 3

# subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 20))

# Flatten the array
axes = axes.flatten()

#  boxplots 
for i, column in enumerate(columns):
    if i >= len(columns):
        break
    ax = axes[i]
    ax.boxplot(lendingClub_data[column])
    ax.set_title(f'Boxplot of {column}')
    ax.set_xlabel(column)

# Remove any empty subplots
for i in range(len(columns), num_rows * num_cols):
    fig.delaxes(axes[i])

# Adjust spacing between subplots
fig.tight_layout()

# Display the plots
plt.show()
 


# In[8]:


lendingClub_data.dropna(inplace=True)


# In[9]:


lendingClub_data 


# In[10]:


correlation_matrix = lendingClub_data.corr()
correlation_with_target = correlation_matrix['loan_default'].sort_values(ascending=False)


# In[11]:


correlation_matrix


# In[12]:


correlation_with_target


# All the coulumn fields are not correlated with the dependent variable loan_default
# only loan_amt is correlated with pct_loan_income

# In[13]:



loan_default_counts = lendingClub_data['loan_default'].value_counts()


# In[14]:


loan_default_counts


# from the above counts its clearly visible that our dependent variable has high class imbalance

# In[15]:


#converting into numeric values
lendingClub_data = lendingClub_data.replace({'residence_property': {'Own':0,'Rent':1}})


# In[16]:


#based on the correlation and all the analysis we come to conclusion of what we willl be using as input and target
target_variable = lendingClub_data['loan_default']
input_variable = lendingClub_data.loc[:,lendingClub_data.columns!='loan_default']


# In[17]:


target_variable


# In[18]:


input_variable


# In[19]:


from sklearn.model_selection import train_test_split


# In[20]:


x_train,x_temp,y_train,y_temp=train_test_split(input_variable , target_variable , test_size=0.3 , random_state=10)
x_dev, x_test, y_dev, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=10)


# In[21]:


x_train


# In[22]:


y_train


# In[23]:


print(x_test)


# In[24]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report, precision_score, recall_score

# logistic regression model on the training data
logistic_model = LogisticRegression()
logistic_model.fit(x_train, y_train)

#predictions on the validation data
y_dev_predictions = logistic_model.predict(x_dev)

# Evaluate on the validation data
accuracy = accuracy_score(y_dev, y_dev_predictions)
f1 = f1_score(y_dev, y_dev_predictions)
precision = precision_score(y_dev, y_dev_predictions)
recall = recall_score(y_dev, y_dev_predictions)
conf_matrix = confusion_matrix(y_dev, y_dev_predictions)

# evaluation metrics
print("Dev Accuracy:", accuracy)
print("Dev F1 Score:", f1)
print("Dev Precision:", precision)
print("Dev Recall:", recall)
print("Dev Confusion Matrix:\n", conf_matrix)

# classification report 
classification_rep = classification_report(y_dev, y_dev_predictions)
print("Classification Report:\n", classification_rep)

# Make predictions on the test data and evaluate the model on the test set
y_test_predictions = logistic_model.predict(x_test)
test_accuracy = accuracy_score(y_test, y_test_predictions)
test_f1 = f1_score(y_test, y_test_predictions)
test_precision = precision_score(y_test, y_test_predictions)
test_recall = recall_score(y_test, y_test_predictions)
test_conf_matrix = confusion_matrix(y_test, y_test_predictions)

# Print the test set evaluation metrics
print("Test Set Accuracy:", test_accuracy)
print("Test Set F1 Score:", test_f1)
print("Test Set Precision:", test_precision)
print("Test Set Recall:", test_recall)
print("Test Set Confusion Matrix:\n", test_conf_matrix)


# In[25]:


#train ROC plot
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

#predictions on the train data
y_train_scores = logistic_model.predict_proba(x_train)[:, 1]

# ROC curve
fpr, tpr, thresholds = roc_curve(y_train, y_train_scores)

# AUC (Area Under the Curve) for ROC
roc_auc = roc_auc_score(y_train, y_train_scores)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic (No Sampling) Receiver Operating Characteristic (ROC) Curve for train data')
plt.legend(loc='lower right')
plt.show()


# In[26]:


#dev ROC plot
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# predictions on the validation data 
y_dev_scores = logistic_model.predict_proba(x_dev)[:, 1]

# ROC curve
fpr, tpr, thresholds = roc_curve(y_dev, y_dev_scores)

# AUC (Area Under the Curve) for ROC
roc_auc = roc_auc_score(y_dev, y_dev_scores)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic (No Sampling)Receiver Operating Characteristic (ROC) Curve for dev data')
plt.legend(loc='lower right')
plt.show()


# In[27]:


#test ROC plot
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# predictions on the test data 
y_test_scores = logistic_model.predict_proba(x_test)[:, 1]

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_test_scores)

# AUC (Area Under the Curve) for ROC
roc_auc = roc_auc_score(y_test, y_test_scores)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic (No Sampling)Receiver Operating Characteristic (ROC) Curve for test data')
plt.legend(loc='lower right')
plt.show()


# In[28]:


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report, precision_score, recall_score

# Split the data into training, validation (development), and test sets
x_train, x_temp, y_train, y_temp = train_test_split(input_variable, target_variable, test_size=0.3, random_state=10)
x_dev, x_test, y_dev, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=10)

# the Gaussian Naive Bayes model on the training data
naive_bayes_model = GaussianNB()
naive_bayes_model.fit(x_train, y_train)

# predictions on the validation data
y_dev_predictions = naive_bayes_model.predict(x_dev)

# Evaluate the model
accuracy = accuracy_score(y_dev, y_dev_predictions)
f1 = f1_score(y_dev, y_dev_predictions)
precision = precision_score(y_dev, y_dev_predictions)
recall = recall_score(y_dev, y_dev_predictions)
conf_matrix = confusion_matrix(y_dev, y_dev_predictions)

# evaluation metrics
print("Dev Accuracy:", accuracy)
print("Dev F1 Score:", f1)
print("Dev Precision:", precision)
print("Dev Recall:", recall)
print("Dev Confusion Matrix:\n", conf_matrix)

# classification report
classification_rep = classification_report(y_dev, y_dev_predictions)
print("Classification Report:\n", classification_rep)

# predictions and evaluate the model on the test data
y_test_predictions = naive_bayes_model.predict(x_test)
test_accuracy = accuracy_score(y_test, y_test_predictions)
test_f1 = f1_score(y_test, y_test_predictions)
test_precision = precision_score(y_test, y_test_predictions)
test_recall = recall_score(y_test, y_test_predictions)
test_conf_matrix = confusion_matrix(y_test, y_test_predictions)

#test set evaluation metrics
print("Test Set Accuracy:", test_accuracy)
print("Test Set F1 Score:", test_f1)
print("Test Set Precision:", test_precision)
print("Test Set Recall:", test_recall)
print("Test Set Confusion Matrix:\n", test_conf_matrix)


# In[29]:


#train ROC plot
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# predictions on the train data 
y_train_scores = naive_bayes_model.predict_proba(x_train)[:, 1]

# ROC curve
fpr, tpr, thresholds = roc_curve(y_train, y_train_scores)

# AUC (Area Under the Curve) for ROC
roc_auc = roc_auc_score(y_train, y_train_scores)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Naive (No Sampling) Receiver Operating Characteristic (ROC) Curve for train data')
plt.legend(loc='lower right')
plt.show()


# In[30]:


#dev ROC plot
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# predictions on the validation data 
y_dev_scores = naive_bayes_model.predict_proba(x_dev)[:, 1]

# ROC curve
fpr, tpr, thresholds = roc_curve(y_dev, y_dev_scores)

# AUC (Area Under the Curve) for ROC
roc_auc = roc_auc_score(y_dev, y_dev_scores)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Naive (No Sampling)Receiver Operating Characteristic (ROC) Curve for dev data')
plt.legend(loc='lower right')
plt.show()


# In[31]:


#test ROC plot
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# predictions on the test data
y_test_scores = naive_bayes_model.predict_proba(x_test)[:, 1]

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_test_scores)

# AUC (Area Under the Curve) for ROC
roc_auc = roc_auc_score(y_test, y_test_scores)

# ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Naive (No Sampling)Receiver Operating Characteristic (ROC) Curve for test data')
plt.legend(loc='lower right')
plt.show()


# In[32]:


pip install -U imbalanced-learn


# In[33]:


from imblearn.under_sampling import RandomUnderSampler


# In[34]:



# imblearn provides this technique RandomUnderSampler
rus = RandomUnderSampler(random_state=0)

# random undersampling is used to balance the dataset
X_resampled, y_resampled = rus.fit_resample(input_variable, target_variable)


# In[35]:


X_resampled


# In[36]:


y_resampled


# In[37]:


type(y_resampled)


# In[38]:


value_counts = y_resampled.value_counts()


# In[39]:


value_counts


# Now data is nicely balanced

# In[40]:


x_train,x_temp,y_train,y_temp=train_test_split(X_resampled , y_resampled , test_size=0.3 , random_state=10)
x_dev, x_test, y_dev, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=10)


# In[41]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report, precision_score, recall_score

# logistic regression model
logistic_model2 = LogisticRegression()
logistic_model2.fit(x_train, y_train)

# predictions on the validation data
y_dev_predictions = logistic_model2.predict(x_dev)

# Evaluate on the validation data
accuracy = accuracy_score(y_dev, y_dev_predictions)
f1 = f1_score(y_dev, y_dev_predictions)
precision = precision_score(y_dev, y_dev_predictions)
recall = recall_score(y_dev, y_dev_predictions)
conf_matrix = confusion_matrix(y_dev, y_dev_predictions)

# evaluation metrics
print("Dev Accuracy:", accuracy)
print("Dev F1 Score:", f1)
print("Dev Precision:", precision)
print("Dev Recall:", recall)
print("Dev Confusion Matrix:\n", conf_matrix)

# classification report 
classification_rep = classification_report(y_dev, y_dev_predictions)
print("Classification Report:\n", classification_rep)

# predictions and evaluate the model on the test data
y_test_predictions = logistic_model2.predict(x_test)
test_accuracy = accuracy_score(y_test, y_test_predictions)
test_f1 = f1_score(y_test, y_test_predictions)
test_precision = precision_score(y_test, y_test_predictions)
test_recall = recall_score(y_test, y_test_predictions)
test_conf_matrix = confusion_matrix(y_test, y_test_predictions)

# test set evaluation metrics
print("Test Set Accuracy:", test_accuracy)
print("Test Set F1 Score:", test_f1)
print("Test Set Precision:", test_precision)
print("Test Set Recall:", test_recall)
print("Test Set Confusion Matrix:\n", test_conf_matrix)


# In[42]:


#train ROC plot
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# predictions on the train data 
y_train_scores = logistic_model2.predict_proba(x_train)[:, 1]

# ROC curve
fpr, tpr, thresholds = roc_curve(y_train, y_train_scores)

# AUC (Area Under the Curve) for ROC
roc_auc = roc_auc_score(y_train, y_train_scores)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Sampling Receiver Operating Characteristic (ROC) Curve for train data')
plt.legend(loc='lower right')
plt.show()


# In[43]:


#dev ROC plot
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# predictions on the validation data 
y_dev_scores = logistic_model2.predict_proba(x_dev)[:, 1]

# ROC curve
fpr, tpr, thresholds = roc_curve(y_dev, y_dev_scores)

# AUC (Area Under the Curve) for ROC
roc_auc = roc_auc_score(y_dev, y_dev_scores)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Sampling Receiver Operating Characteristic (ROC) Curve for dev data')
plt.legend(loc='lower right')
plt.show()


# In[44]:


#test ROC plot
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# predictions on the test data
y_test_scores = logistic_model2.predict_proba(x_test)[:, 1]

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_test_scores)

# AUC (Area Under the Curve) for ROC
roc_auc = roc_auc_score(y_test, y_test_scores)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Sampling Receiver Operating Characteristic (ROC) Curve for test data')
plt.legend(loc='lower right')
plt.show()


# In[45]:


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report, precision_score, recall_score


# Gaussian Naive Bayes model 
naive_bayes_model2 = GaussianNB()
naive_bayes_model2.fit(x_train, y_train)

# predictions on the validation data
y_dev_predictions = naive_bayes_model2.predict(x_dev)

# Evaluate the model
accuracy = accuracy_score(y_dev, y_dev_predictions)
f1 = f1_score(y_dev, y_dev_predictions)
precision = precision_score(y_dev, y_dev_predictions)
recall = recall_score(y_dev, y_dev_predictions)
conf_matrix = confusion_matrix(y_dev, y_dev_predictions)

# Evaluation metrics
print("Dev Accuracy:", accuracy)
print("Dev F1 Score:", f1)
print("Dev Precision:", precision)
print("Dev Recall:", recall)
print("Dev Confusion Matrix:\n", conf_matrix)

# classification report 
classification_rep = classification_report(y_dev, y_dev_predictions)
print("Classification Report:\n", classification_rep)

# predictions and evaluate the model on the test data
y_test_predictions = naive_bayes_model2.predict(x_test)
test_accuracy = accuracy_score(y_test, y_test_predictions)
test_f1 = f1_score(y_test, y_test_predictions)
test_precision = precision_score(y_test, y_test_predictions)
test_recall = recall_score(y_test, y_test_predictions)
test_conf_matrix = confusion_matrix(y_test, y_test_predictions)

# test set evaluation metrics
print("Test Set Accuracy:", test_accuracy)
print("Test Set F1 Score:", test_f1)
print("Test Set Precision:", test_precision)
print("Test Set Recall:", test_recall)
print("Test Set Confusion Matrix:\n", test_conf_matrix)


# In[46]:


#train ROC plot
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# predictions on the train data 
y_train_scores = naive_bayes_model2.predict_proba(x_train)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_train, y_train_scores)

# Calculate the AUC (Area Under the Curve) for ROC
roc_auc = roc_auc_score(y_train, y_train_scores)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Naive Sampling Receiver Operating Characteristic (ROC) Curve for train data')
plt.legend(loc='lower right')
plt.show()


# In[47]:


#dev ROC plot
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# predictions on the validation data 
y_dev_scores = naive_bayes_model2.predict_proba(x_dev)[:, 1]

# ROC curve
fpr, tpr, thresholds = roc_curve(y_dev, y_dev_scores)

# AUC (Area Under the Curve) for ROC
roc_auc = roc_auc_score(y_dev, y_dev_scores)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Naive Sampling Receiver Operating Characteristic (ROC) Curve for dev data')
plt.legend(loc='lower right')
plt.show()


# In[48]:


#test ROC plot
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# predictions on the test data 
y_test_scores = naive_bayes_model2.predict_proba(x_test)[:, 1]

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_test_scores)

# AUC (Area Under the Curve) for ROC
roc_auc = roc_auc_score(y_test, y_test_scores)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Naive Sampling Receiver Operating Characteristic (ROC) Curve for test data')
plt.legend(loc='lower right')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




