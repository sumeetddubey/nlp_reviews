
# coding: utf-8

# In[1]:

import os
import re
import fnmatch
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import string

def vectorize(df):
    nFeatures=1000
    tf_vectorizer = CountVectorizer(max_df=0.85, min_df=0.2,
                                max_features=nFeatures,
                                stop_words='english')
    tf = tf_vectorizer.fit_transform(df)
    features = tf_vectorizer.get_feature_names()
    return (tf, features)

def runLDA(n, iters, wordMat):
    print('Number of inputs:', np.shape(wordMat)[0], '\n')
    lda = LatentDirichletAllocation(n_topics=n, max_iter=5,
                                learning_method='batch',
                                learning_offset=50.,
                                random_state=0)
    lda=lda.fit(wordMat)
    return lda
    
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

def getSimilarGames(gameIndex, df, wordMat, lda): 
    mat1=wordMat.todense()
    mat2=np.linalg.pinv(lda.components_)
    x=mat1*mat2
    cos=cosine_similarity(x)
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


# ### Creating a pandas dataframe out of our dataset of reviews 

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


# ###  Running LDA to get topics. Choosing number of topics = 10 and word per topic = 10

# In[3]:

n=25
iters=5
nWords=10
wordMat, features=vectorize(df['Review'])
lda=runLDA(n, iters, wordMat)
print_top_words(lda, features, nWords)


# ### Running lda only on top 10 publishers having maximum game releases

# In[4]:

df2=df[['Name', 'Publisher']].groupby(['Publisher']).count()
df2=df2.sort_values(['Name'], ascending=False).head(10)
topPubs= (list(df2.axes[0]))
topPubReviewsDf=df[df['Publisher'].isin(topPubs)]

n=10
iters=5
nWords=10
wordMat, features=vectorize(topPubReviewsDf['Review'])
lda=runLDA(n, iters, wordMat)
print_top_words(lda, features, nWords)


# ### Running for games that have a GameSpot Rating of 8 or above

# In[5]:

df['GameSpotScore'] =pd.to_numeric(df['GameSpotScore'])
topDf=df[df['GameSpotScore']>=8]

n=10
iters=5
nWords=10
wordMat, features=vectorize(topDf['Review'])
lda=runLDA(n, iters, wordMat)
print_top_words(lda, features, nWords)


# ### Running for games that have a GameSpot Rating of 4 or below

# In[14]:

df['GameSpotScore'] =pd.to_numeric(df['GameSpotScore'])
botDf=df[df['GameSpotScore']<=4]
n=10
iters=5
nWords=10
wordMat, features=vectorize(botDf['Review'])
lda=runLDA(n, iters, wordMat)
print_top_words(lda, features, nWords)


# ### Filtering dataset based on Genres and performing LDA on top 5 genres

# In[7]:

genres=df[['Name']].groupby(df['Genre']).count()
genres=genres.sort_values(['Name'], ascending=False).head(5)
genres=list(genres.axes[0])

for genre in genres:
    df_genre=df[(df['Genre'] == genre)]
    print(genre)
    n=5
    iters=5
    nWords=10
    wordMat, features=vectorize(df_genre['Review'])
    lda=runLDA(n, iters, wordMat)
    print_top_words(lda, features, nWords)


# ### Below we are trying to find similar games based on a given game. We have taken the cosine between vectors as a measure of similarity to find top 5 games with maximum similarity 

# In[8]:

n=25
iters=5
wordMat, features=vectorize(df['Review'])
lda=runLDA(n, iters, wordMat)


# In[9]:

gameIndex=745
getSimilarGames(gameIndex, df, wordMat, lda)


# In[10]:

gameIndex=2447
getSimilarGames(gameIndex, df, wordMat, lda)


# In[11]:

gameIndex=1751
getSimilarGames(gameIndex, df, wordMat, lda)


# In[12]:

gameIndex=4763
getSimilarGames(gameIndex, df, wordMat, lda)


# In[13]:

gameIndex=7844
getSimilarGames(gameIndex, df, wordMat, lda)


# In[ ]:



