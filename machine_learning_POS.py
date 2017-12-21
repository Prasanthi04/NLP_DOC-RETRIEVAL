# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 15:57:07 2017

@author: prasa
"""

import pandas as pd

df = pd.read_csv("C:/Drive/FALL2017/NLP/Project/training_data_with_POSTAG_162Q.csv",header=0, quoting=3)

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
   if (0<= row[string] <= 0.45)  :
      return 'NOT'
   if (0.45 < row[string] < 0.50) :
      return 'POSSIBLY'
   if (0.50 <= row[string] <= 1) :
      return 'DEFINITELY'
   return 'Other'

X_DF = df['Query']
df['Q1_Relevant Judgment'] = df.apply (lambda row: label_race (row, "Query"),axis=1)

#df.to_csv("training_data_with_POSTAG_163Q.csv",index=False, quoting = 3)


#########################MACHINE LEARNING MODELS ##########################################

df_test = pd.read_csv("C:/Drive/FALL2017/NLP/Project/test_data_with_POSTAG_162Q.csv",header=0, quoting=3)

df_Query1_test = df_test['Query']
df_doc = df_test['DocId']

from sklearn.naive_bayes import GaussianNB
X = X_DF.reshape(-1,1)
y = df['Q1_Relevant Judgment']
clf_nb = GaussianNB()
clf_nb.fit(X, y)
X_prednb = df_Query1_test.reshape(-1,1)
out_nb = clf_nb.predict(X_prednb)
d = {'DocId': df_doc, 'Predicted_judgment': out_nb}
df = pd.DataFrame(data=d)

df.to_csv("predicted_data_with_POSTAG_162Q.csv",index=False, quoting = 3)


##############YY####################################3
'''import pandas as pd
colnames = ['DocId', 'Predicted_judgement']
data_pred = pd.read_csv('C:/Drive/FALL2017/NLP/Project/predicted_data_with_POSTAG_163Q.csv', names=colnames)
colnames_trec = ['DocId', 'Query']
data_trec = pd.read_csv('TREC_EXPECTED_163.csv', names=colnames)


DocId_pred = data_pred.DocId.tolist()
Relv_jud_pred = data_pred.Predicted_judgement.tolist()


DocId_trec = data_trec.DocId.tolist()
DocId_pred.remove('DocId')
DocId_trec.remove('DOCID')
Relv_jud_pred.remove('Predicted_judgment')

flist = []

fNolist = []

DocId_Pred = []

for i in DocId_pred:
    DocId_Pred.append(i.rstrip('0').rstrip('.'))

pred_docid_rj_tuple = list(zip(DocId_Pred,Relv_jud_pred))
   
for qterm in DocId_trec:
    for dterm in DocId_Pred:
        if( dterm == qterm):
                flist.append(qterm)   
        
      

index = [elem for elem in DocId_Pred if elem not in DocId_trec ]
print(flist+index)

final = flist+index

d = {'DocId': final}
dfy = pd.DataFrame(data=d)

dfy.to_csv("C:/Drive/FALL2017/NLP/Project/predicted1000_data_with_POSTAG_163Q.csv",index=False, quoting = 3)

import csv
#data1 = {}
row2=[]
data1 = list(zip(DocId_Pred,Relv_jud_pred))
with open("predicted_data_with_POSTAG_161Q.csv", "r") as in_file1:
     reader1 = csv.reader(in_file1)
     for row1 in reader1:
         print(row1)
         data1[row1[0]] = row1[1]
with open("C:/Drive/FALL2017/NLP/Project/predicted1000_data_with_POSTAG_163Q.csv","r") as in_file2, open("C:/Drive/FALL2017/NLP/Project/predicted1000_data_with_POSTAG_163Q_with_judgement.csv","w",newline='') as out_file:
    reader2 = csv.reader(in_file2)
    writer = csv.writer(out_file)
    writer.writerow(['DOCID','Query'])
    for row2 in reader2:
        #print(row2)
        for i in range(0,len(data1)):
            if row2[0] in data1[i][0]:
                #print("hello")
                #print(row2[0])
                row2.append(data1[i][1])
                print(row2)
                writer.writerow(row2)'''




#################################################
'''from sklearn import tree
X_dt = df['Query'].reshape(-1,1)
y_dt = df['Q1_Relevant Judgment'].reshape(-1,1)
clf_dt = tree.DecisionTreeClassifier()
clf_dt = clf_dt.fit(X_dt, y_dt)
out_dt = clf_dt.predict(X_prednb)'''


#########CONFUSION MATRIX###############

df_1000 = pd.read_csv("C:/Drive/FALL2017/NLP/Project/predicted_data_with_POSTAG_162Q.csv",header=0, quoting=3)
df_trec = pd.read_csv("C:/Drive/FALL2017/NLP/Project/TREC_EXPECTED_162.csv",header=0, quoting=3)

from sklearn.metrics import confusion_matrix, precision_score, recall_score
cm_nb=confusion_matrix(df_trec['Query'],df_1000['Predicted_judgment'])
precision_nb=precision_score(df_trec['Query'], df_1000['Predicted_judgment'], average='weighted')
print("Precision of Q162 with NB", precision_nb)
recall_nb = recall_score(df_trec['Query'], df_1000['Predicted_judgment'], average='weighted')
print("Recall of Q162 with NB", recall_nb)

#DECISION TREE
cm_dt=confusion_matrix(df_trec['Query'],df_1000['Query'])
precision_dt=precision_score(df_trec['Query'], df_1000['Query'], average='weighted')
#recall_dt = recall_score(df['Q3_Relevant Judgment'], out_dt, average='weighted')



'''from sklearn import svm
X_svm = df['Query'].reshape(-1,1)
y_svm = df['Q1_Relevant Judgment'].reshape(-1,1)
clf = svm.SVC()
clf.fit(X_svm,y_svm)
X_pred = df_trec['Query'].reshape(-1,1)
out_svm=clf.predict(X_pred)

precision_svm=precision_score(df['Q3_Relevant Judgment'], out_svm, average='weighted')
recall_svm = recall_score(df['Q3_Relevant Judgment'], out_svm, average='weighted')

import numpy as np
from sklearn import metrics
y = np.array([1, 1, 2, 2])
scores = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)'''



