# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 14:42:27 2017

@author: prasa
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 13:49:02 2017

@author: prasa
"""

import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import glob



from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import numpy.linalg as LA
import gc

gc.collect()

#read all the html files from directory
path = "C:/Drive/FALL2017/NLP/Project/data_all_weneed/*.html"   
files = glob.glob(path)
list_docs = []
doc_term_matrix = []
similarity=[]
filetext = []
stopWords = stopwords.words('english')

import os
ls=[]
# PROCESSING THE DOCUMENTS

rawtext = "I read a book about the history of America"
rawtext_token = nltk.word_tokenize(rawtext) 
rawtext_stop = [w for w in rawtext_token if not w in stopWords]  
rawtext_postag = nltk.pos_tag(rawtext_stop)
rawtext_bigram = nltk.bigrams(rawtext_postag)
rawtext_bigram_list = list(rawtext_bigram)
filetext.append(str(rawtext_bigram_list))

'''doc_postag_process=[]
for i in range(0,2):
    for a, b in filetext[i]:
        #print(a,b)
        doc_postag_process.append((str(a) + str(b)))'''
        
        
#PROCESSING THE QUERIES        
test_set = ["Julie loves me more than Linda loves me"] #Query
q_token = nltk.word_tokenize(str(test_set[0]))
q_stop = [w for w in q_token if not w in stopWords]  
q_postag = nltk.pos_tag(q_stop)
q_bigram = nltk.bigrams(q_postag) #BIGRAMS
q_bigram_list = list(q_bigram)
x = []
for i in range (0,len(q_bigram_list)):
    x.append(str(q_bigram_list[i]))
'''q_postag_process=[]
#concatinating word and tag together
for a, b in q_postag:
    #print(str(a) + '_' + str(b))
    q_postag_process.append((str(a) + str(b)))'''

#VECTORIZAITON AND TFIDF CALCULATION

#stopWords = stopwords.words('english')
vectorizer = CountVectorizer(lowercase=True)
trainVectorizerArray = vectorizer.fit_transform(filetext).toarray()
#print(trainVectorizerArray)
testVectorizerArray = vectorizer.transform(x).toarray()
#print(testVectorizerArray)

transformer = TfidfTransformer()

transformer.fit(trainVectorizerArray)

print(transformer.transform(trainVectorizerArray).toarray())
tfidf_train = transformer.transform(trainVectorizerArray)
transformer.fit(testVectorizerArray)
'''i = 1
while i < len(testVectorizerArray):
    print(i)
    if i == len(testVectorizerArray):
        break
    else:
        tv =+ (testVectorizerArray[i] + testVectorizerArray[i-1])
        i = i + 1'''
    
#tv = testVectorizerArray[0] + testVectorizerArray[1] + testVectorizerArray[2] + testVectorizerArray[3] + testVectorizerArray[4] + testVectorizerArray[5] + testVectorizerArray[6] + testVectorizerArray[7] + testVectorizerArray[8] + testVectorizerArray[9] + testVectorizerArray[10] + testVectorizerArray[11] + testVectorizerArray[12] + testVectorizerArray[13] + testVectorizerArray[14] + testVectorizerArray[15] + testVectorizerArray[16] + testVectorizerArray[17] + testVectorizerArray[18] + testVectorizerArray[19] + testVectorizerArray[20] + testVectorizerArray[21] + testVectorizerArray[22] + testVectorizerArray[23]        
tv = testVectorizerArray[0] + testVectorizerArray[1] + testVectorizerArray[2] #+ testVectorizerArray[3]
tfidf_test = transformer.transform(tv)
print(tfidf_test.todense())

cosine_similarity(tfidf_test, tfidf_train)
similarity.append(cosine_similarity(tfidf_test, tfidf_train))
similarity_stack = np.vstack(similarity)
print("Similarity score for the documents", similarity)
print("The relevant document is ID{} with similarity score {} ".format(np.argmax(similarity_stack),np.amax(similarity)))
print("The non-relevant document is ID{} with similarity score {}".format(np.argmin(similarity),np.amin(similarity)))

import pandas as pd
y = []

for i in range(0,len(similarity_stack[0])):
    y.append(similarity_stack[0][i])
    
#df = pd.DataFrame({'Query':[test_set,test_set,test_set]})
#df = pd.DataFrame({'Document':[1,994]})
d = {'DocId': ls, 'Query': y}
df = pd.DataFrame(data=d)
#df['Query']=y


#output_DT = pd.DataFrame(data={"id":test_set,"similarity_score":similarity_stack[0][i]})
#output_DT = pd.DataFrame(d, index=[])

df.to_csv("traning_data_160_postag_bigram.csv",index=False, quoting = 3)
        