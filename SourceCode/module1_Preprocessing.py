import pandas as pd 
import numpy as np
import gensim
import os
import spacy
import sys
import re
import copy

from gensim import corpora
from gensim.matutils import sparse2full
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer



spacy.prefer_gpu()
#Please download language model below in order to use this.
def load():
    print('loading spacy en_core_web_md')
    nlp = spacy.load("en_core_web_md")
    print('loading complete')

    
def basicpreprocessing(doc_DF,col_name):
    clean_DF=copy.copy(doc_DF)
    clean_DF[col_name] = clean_DF[col_name].apply(lambda x: x[:200000])
    clean_DF[col_name] = clean_DF[col_name].str.strip("'")
    clean_DF[col_name] = clean_DF[col_name].str.replace(r'https?:\/\/\S+\b|www\.(\w+\.)+\S*','') #remove URLs
    
    #Remove all MBTI related keywords in the text
    clean_DF[col_name] = clean_DF[col_name].str.replace(r'\b{}s?\b|\b{}s?\b|\b{}s?\b|\b{}s?\b|\b{}s?\b|\b{}s?\b|\b{}s?\b|\b{}s?\b|\b{}s?\b|\b{}s?\b|\b{}s?\b|\b{}s?\b|\b{}s?\b|\b{}s?\b|\b{}s?\b|\b{}s?\b'
                                                        .format('INTJ','INTP','INFP','INFJ','ISTJ','ISTP','ISFP','ISFJ','ENTJ','ENTP','ENFP','ENFJ','ESTJ','ESTP','ESFP','ESFJ'),'')
    clean_DF[col_name] = clean_DF[col_name].str.replace(r'\b{}s?\b|\b{}s?\b|\b{}s?\b|\b{}s?\b|\b{}s?\b|\b{}s?\b|\b{}s?\b|\b{}s?\b|\b{}s?\b|\b{}s?\b|\b{}s?\b|\b{}s?\b|\b{}s?\b|\b{}s?\b|\b{}s?\b|\b{}s?\b'
                                                        .format('intj','intp','infp','infj','istj','istp','isfp','isfj','entj','entp','enfp','enfj','estj','estp','esfp','esfj'),'')
    
    #Clean up reddit specific items
    clean_DF[col_name] = clean_DF[col_name].str.replace(r'<U.....>|�','') #remove encoding replacement characters
    clean_DF[col_name] = clean_DF[col_name].str.replace(r'/?\br/\w+','') #remove reference to subreddit
    clean_DF[col_name] = clean_DF[col_name].str.replace(r'/?\bu/\w+','') #remove reference to reddit user
    
    
#     #Clean up wassa dataset specific items
#     clean_DF[col_name] = clean_DF[col_name].str.replace(r"via @USER|@USER", "")
#     clean_DF[col_name] = clean_DF[col_name].str.replace(r"@HASHTAG", "")
#     clean_DF[col_name] = clean_DF[col_name].str.replace(r"@URL / +\S*", "@URL")
#     clean_DF[col_name] = clean_DF[col_name].str.replace(r"@URL","")
#     clean_DF[col_name] = clean_DF[col_name].str.replace(r"#\S+", '')

    
    #Emoticon conversion
    eyes = r"[8:=;]"
    nose = r"['`\-]?"
    clean_DF[col_name] = clean_DF[col_name].str.replace(r"\b{}{}[)pD]+\b|\b[(d]+{}{}\b".format(eyes, nose, nose, eyes), " SMILEFACE ")
    clean_DF[col_name] = clean_DF[col_name].str.replace(r"\b{}{}p+\b".format(eyes, nose), " LOLFACE ")
    clean_DF[col_name] = clean_DF[col_name].str.replace(r"\b{}{}\(+\b|\b\)+{}{}\b".format(eyes, nose, nose, eyes), " SADFACE ")
    clean_DF[col_name] = clean_DF[col_name].str.replace(r"\b{}{}[|]\b".format(eyes, nose), " NEUTRALFACE ")
    
    #Number conversion
    clean_DF[col_name] = clean_DF[col_name].str.replace(r"[-+]?[.\d]*[\d]+[:,.\d]*", " NUMBER123 ")
    
    return clean_DF

def createLabel(doc_df,MBTI_col):
    label=pd.DataFrame()
    label['MBTI']=doc_df[MBTI_col].str.upper()
    for n, pair in enumerate(['EI','SN','TF','JP']):
        temp={}
        label[pair]=pd.Series()
        for i,value in enumerate(label['MBTI']):
            temp[i]= (value[n]==pair[0])*1
        label[pair]=temp.values()
    return label



#Setup spacy to use gpu
def useGPU(TF):
    if TF:
        spacy.prefer_gpu()
        print('Using GPU for Spacy')
    else:
        pass
    
def loadSpacy():
    print('loading spacy en_core_web_md')
    nlp = spacy.load("en_core_web_md")
    print('finish loading') 
    return nlp
    
def dummy(doc):
    '''Define dummy function for skipping preprocessing and token pattern by sklearn'''
    return doc

conversion = dict(zip(['SMILEFACE','LOLFACE','SADFACE','NEUTRALFACE','HEARTEMO','NUMBER123'],['<smile>','<lolface>','sadface','<neutralface>','<heart>','<number>']))

def spacy_tokenizer(sentence, nlp, rm_stop=True, filterPOS=['PUNCT','NOUN']):
    """ Creating our token object, which is used to create documents with linguistic annotations."""
    tweets = nlp(sentence)
    mytokens=[]
    
    for word in tweets:
        try:
            # We replace certain items now since angle bracket cannot be tokenized correctly, we need this format for GloVe word embeddings 
            mytokens.append(conversion[word.text])
        except:        
            # Punctuation, NOUN and Spaces are removed. Stopword can be included, but is removed as default
            if (word.is_alpha and word.pos_ not in filterPOS and not(word.is_space) and not (rm_stop and word.is_stop)):
                if word.lemma_=='-PRON-':
                    mytokens.append(word.text)
                else:
                    mytokens.append(word.lemma_)
               
    return mytokens