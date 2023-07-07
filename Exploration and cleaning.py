#!/usr/bin/env python
# coding: utf-8

# # Project: Dataset Exploration and Cleaning
# 
# Dataset: Titanic: Machine Learning from Disaster - Kaggle (https://www.kaggle.com/c/titanic/data) "smartCard-inline")

# In[1]:


# importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# In[2]:


# READ DATASET
data=pd.read_csv(r"F:\train.csv")
data


# In[3]:


yes = data[data['Survived'] == 1].shape[0]
no = data[data['Survived'] == 0].shape[0]

# Print the counts
print("Number of Survivors:", yes)
print("Number of Deceased:", no)


# # Step 1: Remove duplicates from the dataset.

# In[4]:


# Check for duplicates
duplicates = data.duplicated()
# Remove duplicates
data = data.drop_duplicates()
# Print the results
num_duplicates = duplicates.sum()
print("Number of duplicates:", num_duplicates)


# # Step 2: Handle missing values by imputing or removing them.

# In[5]:


#checking null values
data.isnull().sum()


# In[6]:


#replacing with mean values
data["Age"].fillna(data["Age"].mean(), inplace = True)
data["Cabin"].fillna(data["Cabin"].mode().iloc[0], inplace = True)
data["Embarked"].fillna(data["Embarked"].mode().iloc[0], inplace=True)
#confirming there are no null values
data.isnull().sum()


# # Step 3: Check and handle outliers in the data.

# In[14]:


# Visual inspection of numerical features using box plots
features = ['Age', 'Fare']

# Plotting box plot for the selected features
sns.boxplot(data=data[features])


# In[15]:


Q1 = data['Age'].quantile(0.25)
Q3 = data['Age'].quantile(0.75)
IQR = Q3 - Q1
IQR


# In[16]:


#finding limits
lower_limit = Q1-1.5*IQR
upper_limit = Q3 +1.5*IQR


# In[17]:


# Handling outliers
# Replace the outliers with the upper or lower limits
data['Age'] = np.where(data['Age'] < lower_limit, lower_limit, data['Age'])
data['Age'] = np.where(data['Age'] > upper_limit, upper_limit, data['Age'])
# Plotting updated box plot
plt.boxplot(data['Age'])
plt.xlabel('Age')
plt.ylabel('Values')
plt.title('Box Plot of Age after handling Outliers')
plt.show()


# # Step 4: Normalize or standardize numerical features.

# In[18]:


# Selecting the numerical features
numerical_features = ['Age', 'Fare','Parch']

# Normalizing the features 
scaler = MinMaxScaler()
data_normalized = pd.DataFrame(scaler.fit_transform(data[numerical_features]), columns=numerical_features)


# In[19]:


data_normalized


# # Step 5: Encode categorical variables.

# In[20]:


# Performing one-hot encoding [because there is no inherent order in data features]
encoded_data = pd.get_dummies(data, columns=['Sex'],drop_first=True)
encoded_data

