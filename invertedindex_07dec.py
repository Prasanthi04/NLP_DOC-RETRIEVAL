# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 14:46:57 2017

@author: prasa
"""
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
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import glob
import re

#path = "C:/Users/Nandhini/Documents/CSI 5386 NLP/Project/data1/*.txt"
path = "C:/Drive/FALL2017/NLP/Project/temp1/*.html"
files = glob.glob(path)
tdictionary = {}
qdictionary = {}
similarity=[]
lno = 0

def low(z):
    return z.lower()

def first(z):
    return z[0]

def squish(x):
    s=set(x)
    return sorted([(i,x.count(i)) for i in s], key=first)

def clean(x):
    y=''
    s=[".",",","-","'"]
    for i in x:
        if i in s:
            continue
        else:
            y+=i
    return y

   
filetext = []      
l = []
for file in files:
    with open(file, encoding="utf-8", errors='ignore') as f:
        data = f.read()
        #print(data)
        soup = BeautifulSoup(data, "lxml")
        #remove script content
        for script in soup(["script", "style"]):
            script.extract()
        #extract raw text from html content
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        rawtext = ''.join(chunk for chunk in chunks if chunk).lower()
        filetext.append(rawtext)
        fr = filetext
        token = word_tokenize(str(data))
        print(token)
        words = re.compile("[A-Za-z0-9]")
        words_only = filter(words.match,token)
        words_list = list(words_only)
        stop_words = list(stopwords.words('english'))
        nostop_words = [i for i in words_list if i not in stop_words]
        print(nostop_words)
        #l.append(nostop_words)
        lno+=1
        for word in nostop_words:
            if word not in tdictionary:
                #print(word)
                tdictionary[word]=[]
                tdictionary[word].append(1)
                tdictionary[word].append([lno])
            else:
                #print("else term", word)
                tdictionary[word][0]+=1
                tdictionary[word][1].append(lno)
        
print("Term Dictionary list")
ld = list(tdictionary)
ld.sort(key=low)
for k in ld:
    print(k,tdictionary[k][0],squish(tdictionary[k][1]))
    
q = ["How can we implement the  google page ranking algorithm by using SVM ranking algorithm?"]
query = ["How can we implement the  google page ranking algorithm by using SVM ranking algorithm?"]
tokens_query = word_tokenize(str(q))
words_query = re.compile("[A-Za-z0-9]")
words_only_query = filter(words_query.match,tokens)
stop_words_query = list(stopwords.words('english'))
nostop_words_query = [i for i in words_only_query if i not in stop_words_query]
query = list(nostop_words_query)
lno+=1
for word1 in query:
    #print(word1)
    if word1 not in qdictionary:
        
        qdictionary[word1]=[]
        qdictionary[word1].append(1)
        qdictionary[word1].append([lno])
    else:
        
        qdictionary[word1][0]+=1
        qdictionary[word1][1].append(lno)
        
print("\nQuery Dictionary list")
lq = list(qdictionary)
lq.sort(key=low)
for q in lq:
    print(q,qdictionary[q][0],squish(qdictionary[q][1]))     
count = 0    
count_nr = 0
for qterm in lq:
    for dterm in ld:
        if(qterm == dterm):
            print(dterm,qterm)
            print("\nquery process:",dterm,squish(tdictionary[dterm][1]))
            
##TFIDF AFTER INVERTED INDEX
stopWords = stopwords.words('english')
vectorizer = CountVectorizer(stop_words = stopWords, lowercase=True)
#print(vectorizer)
transformer = TfidfTransformer()
#print transformer

trainVectorizerArray = vectorizer.fit_transform(filetext).toarray()
testVectorizerArray = vectorizer.transform(query).toarray()
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
