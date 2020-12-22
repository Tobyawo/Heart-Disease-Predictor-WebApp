#!/usr/bin/env python
# coding: utf-8

# ### Context
# 
# This database contains 76 attributes, but all published 
# experiments refer to using a subset of 14 of them. 
# In particular, the Cleveland database is the only one that has been 
# used by ML researchers to this date. The "goal" field refers to the 
# presence of heart disease in the patient. It is integer valued from 0 (no presence) to 4.
# 
# Description
# Context
# This database contains 76 attributes, 
# but all published experiments refer to using a subset of 14 of them. 
# In particular, the Cleveland database is the only one that has been 
# used by ML researchers to this date. The "goal" field refers to the 
# presence of heart disease in the patient. It is integer valued from 0 (no presence) to 4.
# 
# Content
# 
# Attribute Information: 
# > 1. age 
# > 2. sex 
# > 3. chest pain type (4 values) 
# > 4. resting blood pressure 
# > 5. serum cholestoral in mg/dl 
# > 6. fasting blood sugar > 120 mg/dl
# > 7. resting electrocardiographic results (values 0,1,2)
# > 8. maximum heart rate achieved 
# > 9. exercise induced angina 
# > 10. oldpeak = ST depression induced by exercise relative to rest 
# > 11. the slope of the peak exercise ST segment 
# > 12. number of major vessels (0-3) colored by flourosopy 
# > 13. thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
# The names and social security numbers of the patients were recently
# removed from the database, replaced with dummy values.'''

# In[1]:


'''Importing Libraries'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
from sklearn.impute import SimpleImputer # for missing values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 

import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


'''Importing datasets '''
data = pd.read_csv(r"C:\Users\TOBI\Desktop\datascience\Machine Learning\databank\heart.csv")
#the datasets was splitted into train and testing 
training_data = data.iloc[:725, :]
Atesting_data = data.iloc[725:,:]
print(data.shape)
print(training_data.shape)
print(Atesting_data.shape)


# In[ ]:





# In[3]:


# extracting the target variable from the training dataset
training_Y = training_data['target']
print(len(training_Y))

Atesting_Y = Atesting_data['target']
print(len(Atesting_Y))

training_data = training_data.drop(['target'], axis=1) #droping the target column from training set
print(training_data.shape)

Atesting_data = Atesting_data.drop(['target'], axis=1)#droping the target column from training set
print(Atesting_data.shape)


# In[4]:


#checking the data information
training_data.info()


# In[5]:


#description
training_data.describe()


# In[6]:


'''checking features of various attributes'''
#SEX
male = len(training_data[training_data['sex']==1])
female = len(training_data[training_data['sex']==0])

plt.figure(figsize=(8,6))

#data to plot 
labels = 'Male', 'Female'
sizes = [male, female]
colors = ['Skyblue','yellowgreen']
explode = (0, 0) # explode 1st slice

#plot
plt.pie(sizes, explode=explode, labels= labels,colors=colors, 
autopct= '%1.1f%%', shadow=True, startangle=90)

plt.axis('equal')
plt.show()


# In[7]:


# chest pain type


plt.figure(figsize=(8,6))

#data to plot 
labels = 'Chest Pain Type: 0', 'Chest Pain Type: 1','Chest Pain Type: 2','Chest Pain Type: 3'
sizes = [len(training_data[training_data['cp']==0]), len(training_data[training_data['cp']==1]),
len(training_data[training_data['cp']==2]),len(training_data[training_data['cp']==3])]
colors = ['Skyblue','yellowgreen','orange','gold']
explode = (0, 0, 0, 0) # explode 1st slice

#plot
plt.pie(sizes, explode=explode, labels= labels,colors=colors, 
autopct= '%1.1f%%', shadow=True, startangle=90)

plt.axis('equal')
plt.show()


# In[8]:


#fbs: (Fasting Blood Sugar > 120 mg/dl) (1 = true, 0 = false)

plt.figure(figsize=(8,6))

#data to plot 
labels = 'Fasting Blood Sugar < 120 mg/dl','Fasting Blood Sugar > 120 mg/dl'
sizes = [len(training_data[training_data['fbs']==0]), len(training_data[training_data['cp']==1])]
colors = ['Skyblue','yellowgreen']
explode = (0.1, 0) # explode 1st slice

#plot
plt.pie(sizes, explode=explode, labels= labels,colors=colors, 
autopct= '%1.1f%%', shadow=True, startangle=90)

plt.axis('equal')
plt.show()


# In[9]:


plt.figure(figsize=(8,6))

#data to plot 
labels = 'No', 'Yes'
sizes = [len(training_data[training_data['exang']==0]), len(training_data[training_data['exang']==1])]
colors = ['Skyblue','yellowgreen']
explode = (0.1, 0) # explode 1st slice

#plot
plt.pie(sizes, explode=explode, labels= labels,colors=colors, 
autopct= '%1.1f%%', shadow=True, startangle=90)

plt.axis('equal')
plt.show()


# In[10]:


#heatmap correlations
plt.figure(figsize=(14,8))
sns.heatmap(training_data.corr(), annot =True,cmap='coolwarm', linewidths=1)
plt.show()


# In[11]:


#Number of people that has heart disease according to age
plt.figure(figsize=(15,6))
sns.countplot(x='age',hue=training_Y,data=training_data,palette='GnBu')
plt.show()


# In[12]:


#heart disease frequency for sex
pd.crosstab(training_data.sex, training_Y).plot(kind='bar', figsize=(15,6),color=['red','green'])
plt.title('heart disease frequency for sex')
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.xticks(rotation = 0)
plt.legend(['Have No Disease', 'Have Disease'])
plt.ylabel('Frequency')
plt.show()


# In[13]:


#to know the number of person with heart disease or not
training_Y.value_counts()


# In[14]:


#visualizing the target distribution
sns.countplot(x=training_Y, data=training_data,palette='bwr')
plt.show()


# In[15]:


''' #               MISSING DATA '''
#checking missing percentage
training_data_na = (training_data.isnull().sum() / len(training_data)) * 100
training_data_na = training_data_na.drop(training_data_na[training_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data1 = pd.DataFrame({'Missing Ratio' :training_data_na})
#print('training_data missing_data in percent: \n', missing_data1)


# In[16]:


Atesting_data_na = (Atesting_data.isnull().sum() / len(Atesting_data)) * 100
Atesting_data_na = Atesting_data_na.drop(Atesting_data_na[Atesting_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data2 = pd.DataFrame({'Missing Ratio' :Atesting_data_na})
#print('Atesting_data missing_data in percent: \n', missing_data2)


# In[17]:


#NORMALIZING (FIT_TRANSFORM) THE TRAINING DATASET WHILE
#THE VALIDATING DATASET IS ONLY TRANSFORMED
from sklearn.preprocessing import MinMaxScaler
normalizer = MinMaxScaler(feature_range=(0,1))

X_train_n = normalizer.fit_transform(training_data)
X_test_n = normalizer.transform(Atesting_data)

X_train_norm = pd.DataFrame(X_train_n, columns = training_data.columns)
X_test_norm = pd.DataFrame(X_test_n, columns = Atesting_data.columns)


# In[18]:



#FITTING THE MODEL WITH THE TRAIN DATA AND TARGET DATA'''
#MODELS

from sklearn.ensemble import RandomForestClassifier 

model = RandomForestClassifier(random_state =14)
model.fit(X_train_norm, training_Y)

#PRIDICTING THE TARGET OF THE VALIDATING DATA
model_pred = model.predict(X_test_norm)


# In[19]:


#CHECKING THE MODEL ACCURACY
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score,confusion_matrix, classification_report

model_score = round(model.score(X_train_norm, training_Y) * 100, 2)
compare_pred = pd.DataFrame({'Atesting_Y': Atesting_Y, 'model_pred': model_pred})
print('Accuracy_score on predicted values : \n', accuracy_score(Atesting_Y, model_pred))
print('log_loss on predicted values : \n', log_loss(Atesting_Y, model_pred))
print('roc_auc_score on predicted values : \n', roc_auc_score(Atesting_Y, model_pred))
print('confusion_matrix on predicted values : \n', confusion_matrix(Atesting_Y, model_pred))
report = classification_report(Atesting_Y, model_pred)
print('classification_reports on predicted values : \n', report)
print('model.score on train data and y train: \n',round(model_score,2,), "%")


# In[20]:


import pickle


# In[ ]:





# In[21]:


pickle.dump(model, open('heart_disease_model.pkl','wb'))
pickle.dump(normalizer, open('normalizer.pkl','wb'))


# In[22]:


#scaler = pickle.load(open('scaler.pkl', 'rb'))


# In[23]:


#model = pickle.load(open('heart_disease_model.pkl','rb'))


# In[ ]:




