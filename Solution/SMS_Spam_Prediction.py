
# coding: utf-8

# In[1]:

# Date: 03/August/2018
# @Author: Mohammad Noor Ul Hasan
# Title: SMS Spam Prediction
# Language used: Python 3.5
# Data Structure used: Python List & Dataframe
# Modules Used:  NLTK, NumPy, Pandas, Scikit-learn, Matplotlib & Seaborn
# Output Description: CSV File of Predicted Results 
# Last Edit: 06/August/2018 

import numpy as np
import pandas as pd
import csv
import nltk
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix

path = "../Problem/"                # Eg. "C:/Users/DELL/Downloads/"


# In[2]:

#        TASK - 1 :  Read CSV File ( ORIGINAL FILE )

train_data = pd.read_csv(open(path+'Training.csv'))
test_data = pd.read_csv(open(path + 'Testing.csv'))


# In[3]:

train_data.head(5)


# In[4]:

test_data[9:13]


# In[5]:

#        TASK - 2 :  CLEANING & LABEL ENCODING 

# Cleaning csv file, merging the splited columns message (Eg. Testing Set Row 11)

# Convert the testing data into list
test_data = csv.reader(open(path+'Testing.csv'))
lines = list(test_data)

newCSV = []
for row in lines:
    #new  temporary list to store a row containing only class and message   
    tempList= []
    #appending class which is the first element into temporary list
    tempList.append(row[0])
    #combining the coulmns which contains the splited message 
    i=1
    temp=row[1]
    while(i<len(row)-1):
        i+=1
        temp+=row[i]
    #set of class & message    
    tempList.append(temp)
    #making a new list with only class and single columned message
    newCSV.append(tempList)

    
# Creating cleaned  CSV Testing File
with open(path+"_Testing.csv", "w") as tempList:
    writer = csv.writer(tempList)
    writer.writerows(newCSV)
    
# open cleaned test file
test_data  = pd.read_csv(path+"_Testing.csv")

# drop none columns
test_data = test_data.dropna(axis=1)
train_data = train_data.dropna(axis=1)

# Label Encoding, converting class into numerics, spam = 1 & ham = 0
test_data['class'] = test_data['class'].apply(lambda x : 0 if x == 'ham' else 1)
train_data['class'] = train_data['class'].apply(lambda x : 0 if x == 'ham' else 1)

#Conclusion :  train_data &  test_data CONTAINS THE REQUIRED DATAFRAME--------------------------------------------


# In[6]:

train_data.head(5)


# In[7]:

test_data[13:18]


# In[10]:

#        TASK- 3 :  TOKENIZING & CREATING POSITION TAGS


# tokenize,  train_data using NLTK 
train_data['tokens'] = train_data['message'].apply(lambda x : nltk.word_tokenize(x))
train_data['position_tags'] = train_data['tokens'].apply(lambda x:[temp for z, temp in nltk.pos_tag(x)])
train_data['tags_sentence'] = train_data['position_tags'].apply(lambda x: ' '.join(x))

test_data['tokens'] = test_data['message'].apply(lambda x : nltk.word_tokenize(x))
test_data['position_tags'] = test_data['tokens'].apply(lambda x : [temp for z, temp  in nltk.pos_tag(x)])
test_data['tags_sentence'] = test_data['position_tags'].apply(lambda x : ' '.join(x))


# In[11]:

train_data.head()


# In[12]:

test_data.head()


# In[13]:

#        TASK- 4 :  DEFINING PARAMETERS

x_train = train_data['message'] 
x_test = test_data['message']
y_train = train_data['class']
y_test = test_data['class']


# In[14]:

#        TASK- 5 :  SELECT 'RFC' TO MAKE PREDICTIONS ON TESTING DATASET 

#using Random Forest Calassifier
classifier = RandomForestClassifier()

# Converting raw document to a matrix of TF-IDF features
vectorizer =TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
#max_df => When building the vocabulary ignore terms that have a frequency strictly higher than given threshold
#stop_words =>Terms that were ignored bcz they either:
#occurred in too many (max_df) or too few documents (min_df) or were cut off by feature selection (max_features)


X_train= vectorizer.fit_transform(x_train)

classifier.fit( X_train.todense(), y_train)

# Making predictions on Testing File
outputList = []
for index, row in test_data.iterrows():
    label = row[0]
    message = row[1]
    X_train = vectorizer.transform([message])
    predict = classifier.predict(X_train)[0]
    
    # If prediction is same as the actual label
    if predict == label:
        result = 'Correct'
    else:
        result = 'Wrong'
        
    # This list contains all details about prediction
    outputList.append([message, label, predict, result])


output = pd.DataFrame.from_records(outputList,columns=['Message', 'actualMessageType', 'predictedType', 'Result'])
# Converting data into it's original form
output['actualMessageType'] = output['actualMessageType'].apply(lambda x : 'ham' if x == 0 else 'spam')
output['predictedType'] = output['predictedType'].apply(lambda x : 'ham' if x == 0 else 'spam')

# Saving Prediction Results into a CSV File
output.to_csv("Prediction_Result.csv")
print("\t"*9,"--- * PREDICTED RESULTS * ---")
pd.read_csv("Prediction_Result.csv")


# In[15]:

#        TASK- 6 :  MAKE PREDICTION REPORT TO TEST PREDICTION 

X_train= vectorizer.fit_transform(x_train)
X_test = vectorizer.transform(x_test)

clf = RandomForestClassifier()
clf.fit(X_train.todense(), y_train)
y_pred = clf.predict(X_test.todense())
target_names = ['ham', 'spam']

cr = classification_report(y_test, y_pred, target_names = target_names)
print("Classification Report : \n\n",cr)
print("-"*70)
confusion_mat = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix : \n\n", confusion_mat)
print("\n","-"*70)
print("\nAccuracy Score : \n\n", accuracy_score(y_test, y_pred), "\n")


# In[30]:


# PAIR PLOT
sns.pairplot(train_data, hue='class', size=2.5);
plt.show()




# In[27]:

# CONFUSION MATRIX PLOT
plt.figure(figsize = (10,7))
sns.heatmap(confusion_mat, annot=True)
plt.show()
confusion_mat

