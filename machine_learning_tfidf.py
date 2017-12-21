import pandas as pd

df = pd.read_csv("C:/Drive/FALL2017/NLP/Project/train_data.csv",header=0, quoting=3)

df_Query1 = df['Query1']

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
    
    
ma,mi = sort_max(df['Query1'])
print_stat("Query1",ma,mi)

ma,mi = sort_max(df['Query2'])
print_stat("Query2",ma,mi)

ma,mi = sort_max(df['Query3'])
print_stat("Query3",ma,mi)

ma,mi = sort_max(df['Query4'])
print_stat("Query4",ma,mi)

ma,mi = sort_max(df['Query5'])
print_stat("Query5",ma,mi)

ma,mi = sort_max(df['Query6'])
print_stat("Query6",ma,mi)

ma,mi = sort_max(df['Query7'])
print_stat("Query7",ma,mi)

ma,mi = sort_max(df['Query8'])
print_stat("Query8",ma,mi)

ma,mi = sort_max(df['Query9'])
print_stat("Query9",ma,mi)

ma,mi = sort_max(df['Query10'])
print_stat("Query10",ma,mi)

ma,mi = sort_max(df['Query11'])
print_stat("Query11",ma,mi)

ma,mi = sort_max(df['Query12'])
print_stat("Query12",ma,mi)

ma,mi = sort_max(df['Query13'])
print_stat("Query13",ma,mi)

ma,mi = sort_max(df['Query14'])
print_stat("Query14",ma,mi)

ma,mi = sort_max(df['Query15'])
print_stat("Query15",ma,mi)

ma,mi = sort_max(df['Query16'])
print_stat("Query16",ma,mi)

ma,mi = sort_max(df['Query17'])
print_stat("Query17",ma,mi)

ma,mi = sort_max(df['Query18'])
print_stat("Query18",ma,mi)

ma,mi = sort_max(df['Query19'])
print_stat("Query19",ma,mi)

ma,mi = sort_max(df['Query20'])
print_stat("Query20",ma,mi)

ma,mi = sort_max(df['Query21'])
print_stat("Query21",ma,mi)

ma,mi = sort_max(df['Query22'])
print_stat("Query22",ma,mi)

ma,mi = sort_max(df['Query23'])
print_stat("Query23",ma,mi)

ma,mi = sort_max(df['Query24'])
print_stat("Query24",ma,mi)

ma,mi = sort_max(df['Query25'])
print_stat("Query25",ma,mi)

ma,mi = sort_max(df['Query26'])
print_stat("Query26",ma,mi)

ma,mi = sort_max(df['Query27'])
print_stat("Query27",ma,mi)

ma,mi = sort_max(df['Query28'])
print_stat("Query28",ma,mi)

x = [ 0.049120088,  0.054231845, 0.138793332, 0.163634496, 0.051342853,0.057616825, 0.30889995800000003, 
     0.211945307, 0.119095652, 0.219575035, 0.326107091, 0.125583258, 0.46994181, 0.22358432600000003, 
     0.11523827800000001, 0.094519484, 0.38666645899999996,0.135448567,0.361159392, 0.09932353,0.101938186,
     0.08129834400000001,0.069688678, 0.075721595, 0.12117628, 0.145570655, 0.074214895, 0.19196318399999998]  

# 0 - =0.15 Possibly relevant
# >0.15 to 1 defintely relevant


def label_race (row,string):
   if row[string] == 0 :
      return 'Not Relevant'
   if (0 < row[string] <= 0.15) :
      return 'Possibly Relevant'
   if (0.15 < row[string] <= 1) :
      return 'Definitely Relevant'
   return 'Other'


df.apply (lambda row: label_race (row,"Query1"),axis=1)

df['Q1_Relevant Judgment'] = df.apply (lambda row: label_race (row, "Query1"),axis=1)


df.apply (lambda row: label_race (row,"Query2"),axis=1)

df['Q2_Relevant Judgment'] = df.apply (lambda row: label_race (row, "Query2"),axis=1)


df.apply (lambda row: label_race (row,"Query3"),axis=1)

df['Q3_Relevant Judgment'] = df.apply (lambda row: label_race (row, "Query3"),axis=1)


df.apply (lambda row: label_race (row,"Query4"),axis=1)

df['Q4_Relevant Judgment'] = df.apply (lambda row: label_race (row, "Query4"),axis=1)


df.apply (lambda row: label_race (row,"Query5"),axis=1)

df['Q5_Relevant Judgment'] = df.apply (lambda row: label_race (row, "Query5"),axis=1)

df.apply (lambda row: label_race (row,"Query6"),axis=1)

df['Q6_Relevant Judgment'] = df.apply (lambda row: label_race (row, "Query6"),axis=1)

df.apply (lambda row: label_race (row,"Query7"),axis=1)

df['Q7_Relevant Judgment'] = df.apply (lambda row: label_race (row, "Query7"),axis=1)

df.apply (lambda row: label_race (row,"Query8"),axis=1)

df['Q8_Relevant Judgment'] = df.apply (lambda row: label_race (row, "Query8"),axis=1)

df.apply (lambda row: label_race (row,"Query9"),axis=1)

df['Q9_Relevant Judgment'] = df.apply (lambda row: label_race (row, "Query9"),axis=1)

df.apply (lambda row: label_race (row,"Query10"),axis=1)

df['Q10_Relevant Judgment'] = df.apply (lambda row: label_race (row, "Query10"),axis=1)

df.apply (lambda row: label_race (row,"Query11"),axis=1)

df['Q11_Relevant Judgment'] = df.apply (lambda row: label_race (row, "Query11"),axis=1)

df.apply (lambda row: label_race (row,"Query12"),axis=1)

df['Q12_Relevant Judgment'] = df.apply (lambda row: label_race (row, "Query12"),axis=1)

df.apply (lambda row: label_race (row,"Query13"),axis=1)

df['Q13_Relevant Judgment'] = df.apply (lambda row: label_race (row, "Query13"),axis=1)

df.apply (lambda row: label_race (row,"Query14"),axis=1)

df['Q14_Relevant Judgment'] = df.apply (lambda row: label_race (row, "Query14"),axis=1)

df.apply (lambda row: label_race (row,"Query15"),axis=1)

df['Q15_Relevant Judgment'] = df.apply (lambda row: label_race (row, "Query15"),axis=1)

df.apply (lambda row: label_race (row,"Query16"),axis=1)

df['Q16_Relevant Judgment'] = df.apply (lambda row: label_race (row, "Query16"),axis=1)

df.apply (lambda row: label_race (row,"Query17"),axis=1)

df['Q17_Relevant Judgment'] = df.apply (lambda row: label_race (row, "Query17"),axis=1)

df.apply (lambda row: label_race (row,"Query18"),axis=1)

df['Q18_Relevant Judgment'] = df.apply (lambda row: label_race (row, "Query18"),axis=1)

df.apply (lambda row: label_race (row,"Query19"),axis=1)

df['Q19_Relevant Judgment'] = df.apply (lambda row: label_race (row, "Query19"),axis=1)

df.apply (lambda row: label_race (row,"Query20"),axis=1)

df['Q20_Relevant Judgment'] = df.apply (lambda row: label_race (row, "Query20"),axis=1)

df.apply (lambda row: label_race (row,"Query21"),axis=1)

df['Q21_Relevant Judgment'] = df.apply (lambda row: label_race (row, "Query21"),axis=1)

df.apply (lambda row: label_race (row,"Query22"),axis=1)

df['Q22_Relevant Judgment'] = df.apply (lambda row: label_race (row, "Query22"),axis=1)

df.apply (lambda row: label_race (row,"Query23"),axis=1)

df['Q23_Relevant Judgment'] = df.apply (lambda row: label_race (row, "Query23"),axis=1)

df.apply (lambda row: label_race (row,"Query24"),axis=1)

df['Q24_Relevant Judgment'] = df.apply (lambda row: label_race (row, "Query24"),axis=1)

df.apply (lambda row: label_race (row,"Query25"),axis=1)

df['Q25_Relevant Judgment'] = df.apply (lambda row: label_race (row, "Query25"),axis=1)

df.apply (lambda row: label_race (row,"Query26"),axis=1)

df['Q26_Relevant Judgment'] = df.apply (lambda row: label_race (row, "Query26"),axis=1)

df.apply (lambda row: label_race (row,"Query27"),axis=1)

df['Q27_Relevant Judgment'] = df.apply (lambda row: label_race (row, "Query27"),axis=1)

df.apply (lambda row: label_race (row,"Query28"),axis=1)

df['Q28_Relevant Judgment'] = df.apply (lambda row: label_race (row, "Query28"),axis=1)

from sklearn.naive_bayes import GaussianNB
X = df['Query1'].reshape(-1,1)
y = df['Q1_Relevant Judgment'].reshape(-1,1)
clf_nb = GaussianNB()
clf_nb.fit(X, y)
X_prednb = df['Query3'].reshape(-1,1)
out_nb = clf_nb.predict(X_prednb)

from sklearn import tree
X_dt = df['Query1'].reshape(-1,1)
y_dt = df['Q1_Relevant Judgment'].reshape(-1,1)
clf_dt = tree.DecisionTreeClassifier()
clf_dt = clf_dt.fit(X_dt, y_dt)
out_dt = clf_dt.predict(X_prednb)

from sklearn.metrics import confusion_matrix, precision_score, recall_score
cm_nb=confusion_matrix(df['Q3_Relevant Judgment'], out_nb)
precision_nb=precision_score(df['Q3_Relevant Judgment'], out_nb, average='weighted')
recall_nb = recall_score(df['Q3_Relevant Judgment'], out_nb, average='weighted')

from sklearn import svm
X_svm = df['Query1'].reshape(-1,1)
y_svm = df['Q1_Relevant Judgment'].reshape(-1,1)
clf = svm.SVC()
clf.fit(X_svm,y_svm)
X_pred = df['Query3'].reshape(-1,1)
out_svm=clf.predict(X_pred)
precision_svm=precision_score(df['Q3_Relevant Judgment'], out_svm, average='weighted')
recall_svm = recall_score(df['Q3_Relevant Judgment'], out_svm, average='weighted')

import numpy as np
from sklearn import metrics
y = np.array([1, 1, 2, 2])
scores = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)

df.to_csv("training_data.csv",index=False, quoting = 3)
