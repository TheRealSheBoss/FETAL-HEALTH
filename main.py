#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Import data, balance the dataset, feature engineering, 


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea
from sklearn.preprocessing import LabelEncoder
from collections import Counter

fetal_health = pd.read_csv("C:/Users/sj21399/OneDrive - University of Bristol/Desktop/DATA ANALYTICS CW/fetal_health.csv")



#Check for null values
fetal_health.isnull().any

#split into target and input
a = fetal_health[fetal_health.columns.to_list()[:-1]]
b = fetal_health['fetal_health']

#assign label to target variable
b = LabelEncoder().fit_transform(b)


#See distribution of normal, pathological and suspected
def view(b):
    classifier = Counter(b)
    views = []
    for each, all in classifier.items():
        calc = all / len(b) * 100

    # plot the distribution
    fig = plt.bar(classifier.keys(), classifier.values())
    plt.savefig('Unbalanced dataset.png') 
    plt.show()
    #fig.(r"C:/Users/sj21399/OneDrive - University of Bristol/Desktop/DATA ANALYTICS CW/ Unbalanced dataset.png")
view(b)

#This chart shows that our dataset is very unbalanced


# In[2]:


# !pip install imblearn
from imblearn.over_sampling import RandomOverSampler

# define oversampling strategy
OS = RandomOverSampler(sampling_strategy= 'auto')
# fit and apply the transform
a,b = OS.fit_resample(a,b)

main_data = a
fetal_health1 = b

view (fetal_health1)


# # FEATURE ENGINEERING

# In[5]:


#!pip install pandas-profiling
from pandas_profiling import ProfileReport
import pandas_profiling as pdp
    
PROF = ProfileReport(fetal_health, title='Profiling Report of data', minimal=True,progress_bar=False,      
    missing_diagrams={
          'heatmap': False,
          'dendrogram': False,
      } )
PROF.to_file(output_file="Fetal Health Profile.html")
PROF


# In[36]:


correlation = fetal_health.corr()
fig, ax = plt.subplots(figsize=(15,15))
sea.heatmap(correlation, vmax=1.0, center=0, fmt='.2f', cmap="seismic",
                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})
plt.savefig('HeatMap.png') 
plt.show()



# 

# In[37]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize = (25,15))

for all, column in enumerate(fetal_health.columns):
    plt.subplot(4,6, all + 1)
    sea.histplot(data = fetal_health[column])
    plt.title(column)

plt.savefig('Histograms of all variables.png')     
plt.tight_layout()
plt.show()


# In[38]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize = (25,15))

for all, column in enumerate(fetal_health.columns):
    plt.subplot(4,6, all + 1)
    sea.boxplot(data = fetal_health[column])
    plt.title(column)

plt.savefig('Boxplots of all variables.png')     
plt.tight_layout()
plt.show()


# In[39]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize = (25,15))

plt.pie(fetal_health['fetal_health'].value_counts(), autopct = '%.2f%%', labels = ['Normal', 'Suspects', 'Pathological'],
        colors = sea.color_palette('seismic'))

plt.savefig('Piechart.png')     
plt.title('Distibution of cases')
plt.show()


# In[40]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize = (25,15))

for all, column in enumerate(fetal_health.columns):
    plt.subplot(4,6, all + 1)
    sea.violinplot(data = fetal_health[column])
    plt.title(column)

plt.savefig('Violinplots of all variables.png')     
plt.tight_layout()
plt.show()


# In[3]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

x = main_data.copy() #data
y = fetal_health1.copy() #target

def scaled ():
    x_train, x_test, y_train, y_test = train_test_split(main_data,fetal_health1, test_size = 0.2, shuffle = True, random_state = 10)
    #instantiate the scaler
    scaler = RobustScaler()
    scaler.fit(x_train)
    x_train = pd.DataFrame(scaler.transform(x_train), index = x_train.index, columns = x_train.columns)
    x_test = pd.DataFrame(scaler.transform(x_test), index = x_test.index, columns = x_test.columns)
    return x_train, x_test, y_train, y_test
    
def unscaled ():
    X_train, X_test, Y_train, Y_test = train_test_split(main_data,fetal_health1, test_size = 0.2, shuffle = True, random_state = 10)
    return X_train, X_test, Y_train, Y_test


# In[4]:


#Define training and test sets
x_train, x_test, y_train, y_test = scaled()
#we can see 80% of our training data
x_train
y_train
#we can see

X_train, X_test, Y_train, Y_test = unscaled()
X_train
Y_train


# In[5]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score

# #Scaled data
models = {
    'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=3),
    'GaussianNB': GaussianNB(),
    'DecisionTreeClassifier': DecisionTreeClassifier(max_depth= 10),
    'RandomForestClassifier': RandomForestClassifier()
}

for all, model in models.items():
    model.fit(x_train, y_train)
    #predict_y
    pred_y = model.predict(x_test)
    accuracy= accuracy_score(y_test, pred_y)
    precision= precision_score(y_test, pred_y, average = 'macro')
    recall = recall_score(y_test, pred_y , average = 'macro')
    print (f'This is the accuracy for scaled overbalanced data using {all}: {accuracy}')
    print (f'This is the precision for scaled overbalanced data using {all}: {precision}')
    print (f'This is the recall score for scaled overbalanced data using {all}: {recall}')


# In[6]:


#Unscaled data

for each, model in models.items():
    model.fit(X_train, Y_train)
    #predict_y
    pred_Y = model.predict(X_test)
    accuracy= accuracy_score(Y_test, pred_Y)
    precision= precision_score(Y_test, pred_Y, average = 'macro')
    recall = recall_score(Y_test, pred_Y , average = 'macro')
    print (f'This is the accuracy for unscaled overbalanced data using {each}: {accuracy}')
    print (f'This is the precision for unscaled overbalanced data using {each}: {precision}')
    print (f'This is the recall score for unscaled overbalanced data using {each}: {recall}')


# In[7]:


from sklearn import tree

model =DecisionTreeClassifier()
x = model.fit(x_train, y_train)

plt.figure(figsize = (60,20))
new = tree.plot_tree(x,
                   filled=True)
plt.savefig('Decision trees')
plt.show()


# 

# In[8]:


# let's create a dictionary of features and their importance values

# model =DecisionTreeClassifier()
# model.fit(x_train, y_train)

dictionary= {}

for columns, values in sorted (zip(x_train.columns, model.feature_importances_), key = lambda x: x[1], reverse = True):
    dictionary [columns] = values
    
dictionary


# In[9]:


#convert it to a dataframe
importance_dataframe = pd.DataFrame({'Feature':dictionary.keys(),'Importance':dictionary.values()})
importance_dataframe


# In[10]:


#visualize
#!pip install yellowbrick
from yellowbrick.features import FeatureImportances

fig , ax = plt.subplots(figsize=(10,8))
tree = FeatureImportances(model)
tree.fit(x_train, y_train)
fig.savefig('Important features.png',dpi=300)
plt.show()


# In[11]:


main_data

#select the five best features
drop = main_data[['histogram_mean', 'mean_value_of_short_term_variability', 'percentage_of_time_with_abnormal_long_term_variability', 
   'abnormal_short_term_variability', 'accelerations']]
drop


# In[12]:


#split new dataset into training and testing sets
drop_train, drop_test, drops_train, drops_test = train_test_split(drop,fetal_health1, test_size = 0.2, shuffle = True, random_state = 10)

  

#Ten features on these models to check for accuracy etc
models = {
    'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=3),
    'GaussianNB': GaussianNB(),
    'RandomForestClassifier': RandomForestClassifier()
}

for all, model in models.items():
    model.fit(drop_train, drops_train)
    #predict_y
    pred_y = model.predict(drop_test)
    accuracy= accuracy_score(drops_test, pred_y)
    precision= precision_score(drops_test, pred_y, average = 'macro')
    recall = recall_score(drops_test, pred_y , average = 'macro')
    print (f'This is the accuracy after feature selection using {all}: {accuracy}')
    print (f'This is the precision after feature selection using {all}: {precision}')
    print (f'This is the recall score after feature selection using {all}: {recall}')


# In[13]:


RF = RandomForestClassifier()
RF.fit(x_train, y_train)

RF.feature_importances_


# In[55]:


#plt.barh(main_data, RF.feature_importances_)
#convert it to a dataframe
importance_dataframes = pd.DataFrame({'Feature importance':RF.feature_importances_})
importance_dataframes


# In[14]:


#visualize feature importance using random forest
from yellowbrick.features import FeatureImportances

modela = RandomForestClassifier()
modela.fit(x_train, y_train)


fig , ax = plt.subplots(figsize=(10,8))
tree = FeatureImportances(modela)
tree.fit(x_train, y_train)
fig.savefig('Random forest.png',dpi=300)
plt.show()


# In[16]:


main_data

#select the five best features
dropss = main_data[['abnormal_short_term_variability', 'percentage_of_time_with_abnormal_long_term_variability', 
   'histogram_mean', 'accelerations', 'mean_value_of_short_term_variability']]
dropss


# In[17]:


#split new dataset into training and testing sets
dropss_train, dropss_test, dropsss_train, dropsss_test = train_test_split(dropss,fetal_health1, test_size = 0.2, shuffle = True, random_state = 10)

  

#Ten features on these models to check for accuracy etc
models = {
    'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=3),
    'GaussianNB': GaussianNB(),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
}

for all, model in models.items():
    model.fit(drop_train, drops_train)
    #predict_y
    pred_y = model.predict(drop_test)
    accuracy= accuracy_score(drops_test, pred_y)
    precision= precision_score(drops_test, pred_y, average = 'macro')
    recall = recall_score(drops_test, pred_y , average = 'macro')
    print (f'This is the accuracy after feature selection using {all}: {accuracy}')
    print (f'This is the precision after feature selection using {all}: {precision}')
    print (f'This is the recall score after feature selection using {all}: {recall}')


# In[ ]:




