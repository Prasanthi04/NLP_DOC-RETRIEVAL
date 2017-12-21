import os, zipfile
import shutil
import nltk
from nltk import word_tokenize, sent_tokenize, TweetTokenizer
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import glob
from collections import Counter
import re
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

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
ls=[]
for file in files:
    with open(file, encoding="utf-8", errors='ignore') as f:
        print(f.name)
        ls.append(f.name[46:54])
        data = f.read()
        soup = BeautifulSoup(data, "lxml")
        #remove script content
        for script in soup(["script", "style"]):
            script.extract()
        #extract raw text from html content
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        rawtext = ''.join(chunk for chunk in chunks if chunk)
        filetext.append(rawtext)
#for file in files:
    #with open(file, encoding="utf-8", errors='ignore') as f:

test_set = ["What is the role of PrnP in mad cow disease?"] #Query
stopWords = stopwords.words('english')
#tokenize = lambda f: filetext.split(" ")
vectorizer = CountVectorizer(stop_words = stopWords, lowercase=True)
#print(vectorizer)
transformer = TfidfTransformer()
#print transformer

trainVectorizerArray = vectorizer.fit_transform(filetext).toarray()
testVectorizerArray = vectorizer.transform(test_set).toarray()
print('Fit Vectorizer to train set', trainVectorizerArray)
print('Transform Vectorizer to test set', testVectorizerArray)

transformer.fit(trainVectorizerArray)

print(transformer.transform(trainVectorizerArray).toarray())
tfidf_train = transformer.transform(trainVectorizerArray)
transformer.fit(testVectorizerArray)
tfidf_test = transformer.transform(testVectorizerArray)
print(tfidf_test.todense())

cosine_similarity(tfidf_test, tfidf_train)
similarity.append(cosine_similarity(tfidf_test, tfidf_train))


import numpy as np
similarity_stack = np.vstack(similarity)



print("Similarity score for the documents", similarity)
print("The relevant document is ID{} with similarity score {} ".format(np.argmax(similarity_stack),np.amax(similarity)))
print("The non-relevant document is ID{} with similarity score {}".format(np.argmin(similarity),np.amin(similarity)))
y = []
import pandas as pd


for i in range(0,len(similarity_stack[0])):
    y.append(similarity_stack[0][i])
    
#df = pd.DataFrame({'Query':[test_set,test_set,test_set]})
#df = pd.DataFrame({'Document':[1,994]})
d = {'DocId': ls, 'Query': y}
df = pd.DataFrame(data=d)
#df['Query']=y


#output_DT = pd.DataFrame(data={"id":test_set,"similarity_score":similarity_stack[0][i]})
#output_DT = pd.DataFrame(d, index=[])

df.to_csv("training_data_tfidf_only_160.csv",index=False, quoting = 3)