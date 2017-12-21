# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 18:31:48 2017

@author: prasa
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 16:30:22 2017

@author: prasa
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 15:57:07 2017

@author: prasa
"""

import pandas as pd

df = pd.read_csv("C:/Drive/FALL2017/NLP/PASSAGE_TFIDF/training_data_tfidf_only_160.csv",header=0, quoting=3)

df_Query1 = df['Query']

lis= list(df_Query1)
lis.sort()


def sort_max(query):
    df_query = query
    lis = list(df_query)
    lis.sort()
    return max(lis), min(lis)


def print_stat(string, maxim,minim):
    print("The maximum of {} {}".format(string,maxim))
    print("The minimum of {} {}".format(string,minim))
    
    
ma,mi = sort_max(df['Query'])
print_stat("Query",ma,mi)


# 0 - =0.15 Possibly relevant
# >0.15 to 1 defintely relevant


def label_race (row,string):
   if (0<= row[string] < 0.05) :
      return 'NOT'
   if (0.05 <= row[string] <= 0.075) :
      return 'POSSIBLY'
   if (0.075 < row[string] <= 1) :
      return 'DEFINITELY'
   return 'Other'



df['Q1_Relevant Judgment'] = df.apply (lambda row: label_race (row, "Query"),axis=1)


#df.to_csv("C:/Drive/FALL2017/NLP/Project/TFIDF_ALONE/training_data_tfidf_only_160_1.csv",index=False, quoting = 3)


###################################################################33

df_test = pd.read_csv("C:/Drive/FALL2017/NLP/PASSAGE_TFIDF/test_data_passage_163.csv",header=0, quoting=3)

df_Query1_test = df_test['Query']
df_doc = df_test['DocId']

from sklearn.naive_bayes import GaussianNB
X = df['Query'].reshape(-1,1)
y = df['Q1_Relevant Judgment']
clf_nb = GaussianNB()
clf_nb.fit(X, y)
X_prednb = df_Query1_test.reshape(-1,1)
out_nb = clf_nb.predict(X_prednb)
d = {'DocId': df_doc, 'Predicted_judgment': out_nb}
df_pred = pd.DataFrame(data=d)

df_pred.to_csv("C:/Drive/FALL2017/NLP/PASSAGE_TFIDF/predicted_data_163Q.csv",index=False, quoting = 3)

##################################################################################################3

df_1000 = pd.read_csv("C:/Drive/FALL2017/NLP/PASSAGE_TFIDF/predicted_data_163Q.csv",header=0, quoting=3)
df_trec = pd.read_csv("C:/Drive/FALL2017/NLP/PASSAGE_TFIDF/TREC_EXPECTED_163.csv",header=0, quoting=3)
#list(df_trec['Query']).sort()
#list(df_1000['Predicted_judgment']).sort()
from sklearn.metrics import confusion_matrix, precision_score, recall_score
cm_nb=confusion_matrix(df_trec['Query'],df_1000['Predicted_judgment'])
precision_nb=precision_score(df_trec['Query'], df_1000['Predicted_judgment'], average='weighted')
print("precision", precision_nb)

recall_nb = recall_score(df_trec['Query'], df_1000['Predicted_judgment'], average='weighted')
print("recall",recall_nb)

#################################################################################################33
from sklearn import tree
X_dt = df['Query'].reshape(-1,1)
y_dt = df['Q1_Relevant Judgment']
clf_dt = tree.DecisionTreeClassifier()
clf_dt = clf_dt.fit(X_dt, y_dt)
out_dt = clf_dt.predict(X_prednb)
d_dt= {'DocId': df_doc, 'Predicted_judgment': out_dt}
df_dt = pd.DataFrame(data=d_dt)

df_dt.to_csv("C:/Drive/FALL2017/NLP/PASSAGE_TFIDF/predicted_data_163Q_decisiontree.csv",index=False, quoting = 3)

####################################################3
df_1000_dt = pd.read_csv("C:/Drive/FALL2017/NLP/PASSAGE_TFIDF/predicted_data_163Q_decisiontree.csv",header=0, quoting=3)
#df_trec = pd.read_csv("C:/Drive/FALL2017/NLP/PASSAGE_TFIDF/TREC_EXPECTED_160.csv",header=0, quoting=3)

cm_dt=confusion_matrix(df_trec['Query'],df_1000_dt['Predicted_judgment'])
precision_dt=precision_score(df_trec['Query'], df_1000_dt['Predicted_judgment'], average='weighted')
print("precision for decision tree", precision_dt)
recall_dt = recall_score(df_trec['Query'], df_1000_dt['Predicted_judgment'], average='weighted')
print("recall for decision tree",recall_dt)
