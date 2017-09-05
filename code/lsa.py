
# coding: utf-8

# In[95]:

import os
import re
import fnmatch
import string
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

def tfIdf(df):
    nFeatures=1000
    tf_vectorizer = TfidfVectorizer(max_df=0.85, min_df=0.1,
                                max_features=nFeatures,
                                stop_words='english', lowercase=True)
    tf = tf_vectorizer.fit_transform(df)
    features = tf_vectorizer.get_feature_names()
    return (tf, features)

def runLSA(n, iters, wordMat):
    print('Number of inputs:', np.shape(wordMat)[0], '\n')
    lsa = TruncatedSVD(n_components=n, n_iter=5,
                                random_state=0)
    model=lsa.fit(wordMat)
    lsaTransform=lsa.transform(wordMat)
    return (model, lsaTransform)
    
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx+1))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()
    
def print_topics(components, pred, features):
    pred=list(pred[0])
    topic_index=pred.index(max(pred))
    topic=components[topic_index]
    print(" ".join([features[i] for i in topic.argsort()[:-10 - 1:-1]]))
    print()
    
def predLDA(x, wordMat, features, model):
    print(x[['Name', 'Genre', 'Console']])
    r=wordMat[x.index.get_values()[0]]
    topics=lda.transform(r)
    print_topics(lda.components_, topics, features)

def getSimilarGames(gameIndex, df, wordMat, lsa): 
#     mat1=wordMat.todense()
#     mat2=np.linalg.pinv(lsa[0].components_)
#     x=mat1*mat2
#     cos=cosine_similarity(x)
    cos=cosine_similarity(lsa[1])
    game1=cos[gameIndex]
    probs_sorted=sorted(game1, reverse=True)

    ipGame=df.iloc[gameIndex]  
    i=0
    sim_games=[]
    lstGames=set()
    lstGames.add(ipGame[0])
    print('Input Game:')
    print(df.iloc[gameIndex], '\n')
    while(len(lstGames)<=5):
        index=int(np.where(game1==probs_sorted[i])[0][0])
        currentGame=df.iloc[index]
        if currentGame[0] not in lstGames:
            sim_games.append(df.iloc[index])
            lstGames.add(currentGame[0])
        i+=1
    
    print('Most similar games:')
    for game in sim_games:
        print(game)
        print()  
    


# In[2]:

df=pd.DataFrame(columns=['Name', 'Publisher', 'GameSpotScore', 'Review', 'Console', 'Genre'])

i=0
for dirpath, dirs, files in os.walk('dataset/reviews'):   
    for file in fnmatch.filter(files, '*.txt'):
        with open(os.path.join(dirpath, file), 'r') as ip:
            data=ip.read()
            name=re.findall(r':::Game Name:::(.*?)-----', data, re.DOTALL)[0].strip()
            review=re.findall(r':::Review:::(.*?)-----',data, re.DOTALL)[0].strip()
            scores=re.findall(r':::Scores:::(.*?)-----',data, re.DOTALL)[0]
            addition=re.findall(r':::Addition:::(.*?)-----',data, re.DOTALL)[0]
            gsScore=re.findall(r'GameSpot Score:(.*?)\n', scores)[0]
            review = review.lower()
            tVar = str.maketrans('', '', string.punctuation)
            review = review.translate(tVar)
            try:
                pub=re.findall(r'Publisher:(.*?)\n', addition)[0]
            except:
                pub=''
            try:
                genre=re.findall(r'Genre:(.*?)\n', addition)[0]
            except:
                genre=''
            console=dirpath.strip('dataset/reviews/')
            df.loc[i]=[name, pub, gsScore, review, console, genre]
            i+=1


# In[96]:

n=25
iters=5
nWords=10
wordMat, features=tfIdf(df['Review'])


# In[107]:

lsa=runLSA(n, iters, wordMat)
test=features.index('youll')
print_top_words(lsa[0], features, nWords)


# In[88]:

gameIndex=745
getSimilarGames(gameIndex, df, wordMat, lsa)


# In[89]:

gameIndex=2447
getSimilarGames(gameIndex, df, wordMat, lsa)


# In[90]:

gameIndex=1751
getSimilarGames(gameIndex, df, wordMat, lsa)


# In[91]:

gameIndex=4763
getSimilarGames(gameIndex, df, wordMat, lsa)


# In[92]:

gameIndex=7844
getSimilarGames(gameIndex, df, wordMat, lsa)


# In[ ]:



