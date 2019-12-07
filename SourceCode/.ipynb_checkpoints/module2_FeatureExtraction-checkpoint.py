import numpy as np
import pandas as pd
import spacy
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer,TfidfTransformer

# #Setup spacy to use gpu
# def useGPU(TF):
#     if TF:
#         spacy.prefer_gpu()
#         print('Using GPU for Spacy')
#     else:
#         pass
    
# def loadSpacy():
#     print('loading spacy en_core_web_md')
#     nlp = spacy.load("en_core_web_md")
#     print('finish loading') 
#     return nlp
    

# conversion = dict(zip(['SMILEFACE','LOLFACE','SADFACE','NEUTRALFACE','HEARTEMO'],['<smile>','<lolface>','sadface','<neutralface>','<heart>']))

# def spacy_tokenizer(sentence, nlp, rm_stop=True):
#     """ Creating our token object, which is used to create documents with linguistic annotations."""
#     tweets = nlp(sentence)
#     mytokens=[]
    
#     for word in tweets:
#         try:
#             # We replace certain items now since angle bracket cannot be tokenized correctly, we need this format for GloVe word embeddings 
#             mytokens.append(conversion[word.text])
#         except:        
#             # Punctuation, NOUN and Spaces are removed. Stopword can be included, but is removed as default
#             if (word.is_alpha and word.pos_ not in ['PUNCT','random'] and not(word.is_space) and not (rm_stop and word.is_stop)):
#                 if word.lemma_=='-PRON-':
#                     mytokens.append(word.text)
#                 else:
#                     mytokens.append(word.lemma_)
               
#     return mytokens





# def spacy_tokenizer(sentence, nlp, rm_stop=True):
#     """ Creating our token object, which is used to create documents with linguistic annotations."""
#     tweets = nlp(sentence)
#     mytokens=[]
    
#     def filter(word):
#         # Punctuation, NOUN and Spaces are removed. Stopword can be included, but is removed as default
#         if (word.is_alpha and word.pos_ not in ['PUNCT','NOUN'] and not(word.is_space) and not (rm_stop and word.is_stop)):
#             if word.lemma_=='-PRON-':
#                 return (word.text)
#             else:
#                 return (word.lemma_)
        
#     with multiprocessing.Pool(4) as Pool:
#         mytokens = Pool.map(filter,tweets)
        
#    # We replace certain items now since angle bracket cannot be tokenized correctly, we need this format for GloVe word embeddings 
#     try:
#         mytokens[mytokens.index('SMILEFACE')]='<smile>'
#         mytokens[mytokens.index('LOLFACE')]='<lolface>'
#         mytokens[mytokens.index('SADFACE')]='<sadface>'
#         mytokens[mytokens.index('NEUTRALFACE')]='<neutralface>'
#         mytokens[mytokens.index('HEARTEMO')]='<heart>'
#     except:
#         pass
    
#     # return preprocessed list of tokens
#     return mytokens



def dummy(doc):
    '''Define dummy function for skipping preprocessing and token pattern by sklearn'''
    return doc

def sum_array(ngram_mat):
    '''For iteration thru n-gram generator to get the count of words in array '''
    array=iter(ngram_mat)
    for i in range(ngram_mat.shape[0]):
        if i==0:
            sum_array=next(array).toarray()
        else:
            sum_array += next(array).toarray()
    return sum_array[0]

#Output dataframe and top features from sklearn vectorizer
def df2vector(data,vectorizer, num_features):
    '''For iteration thru n-gram generator to get the full array in DataFrame format, and to get top n features dataframe'''
    ngram_mat=vectorizer.fit_transform(data)
    if num_features> ngram_mat.shape[1]:
        num_features=ngram_mat.shape[1]
    array=iter(ngram_mat)
    pd_dict={}
    topFeatures=pd.DataFrame(zip(vectorizer.get_feature_names(), sum_array(ngram_mat))).sort_values(by=1, ascending=False)
    for i in range(ngram_mat.shape[0]):
        pd_dict[i]=next(array).toarray()[0][topFeatures.index[:num_features]]
    df=pd.DataFrame(pd_dict).T
    df.columns=topFeatures[0].values.tolist()[:num_features]
    return df, topFeatures


#Use Spacy tokenizer for sklearn vectorizer
tfidf_vectorizer =TfidfVectorizer(tokenizer = dummy, preprocessor=dummy, token_pattern=dummy, ngram_range=(1,3),min_df=40, max_df=0.95)
count_vectorizer =CountVectorizer(tokenizer = dummy, preprocessor=dummy, token_pattern=dummy, ngram_range=(1,3),min_df=40, max_df=0.95)


def TFIGM(data, y_class, count_vectorizer,TF_transformer, num_features, constant):
    y_class.reset_index(drop=True,inplace=True)
    class1_index=y_class[y_class==1].index.tolist()
    class0_index=y_class[y_class==0].index.tolist()
    
    count_DF, _=df2vector(data, count_vectorizer, num_features)
    TF_DF =pd.DataFrame(TF_transformer.fit_transform(count_DF).toarray())

    
    class1_count=count_DF.loc[class1_index]
    class0_count=count_DF.loc[class0_index]
    class1_docCount=class1_count.apply(lambda x: (x>0).astype('int')).sum(axis=0)
    class0_docCount=class0_count.apply(lambda x: (x>0).astype('int')).sum(axis=0)

    IGM=pd.Series([max(class1,class0)/(sorted([class1,class0],reverse=True)[0]
                                       +sorted([class1,class0],reverse=True)[1]*2) 
                   for class1,class0 in zip(class1_docCount,class0_docCount)])

    TF_IGM=pd.DataFrame()    
    RTF_IGM=pd.DataFrame()
    for i in range(len(IGM)):
        TF_IGM[i]=TF_DF.iloc[:,i].mul(1+constant*IGM[i])
        RTF_IGM[i]=np.sqrt(TF_DF.iloc[:,i])*(1+constant*IGM[i])
    return TF_IGM, RTF_IGM



def EmolexDF(wordvectorDF,lexiconPath):
    emolex = pd.read_excel(lexiconPath, encoding='utf-8') 
    
    # Drop rows without emotion assignment
    emolex=emolex[emolex.iloc[:,1:].any(axis=1)].set_index('Word')
    
    #get only words from dataset that appears in the lexicon
    relevant_cols=list(set(emolex.index) & set(wordvectorDF.columns))
    wordvectorDF=wordvectorDF.loc[:,relevant_cols]
    df_list=[]
    for i in range(len(wordvectorDF)):
        NonZero=wordvectorDF.iloc[i,:].index[wordvectorDF.iloc[i,:]>0.0]
        FoundInEmolex=NonZero[[word in emolex.index for word in NonZero]]
        emolex_row=emolex.loc[FoundInEmolex].mul(wordvectorDF.loc[i,FoundInEmolex],axis=0).sum()
        #normalize the row to total value of the row
        norm_emolex_row=emolex_row/emolex_row.max()
        
        
        df_list.append(norm_emolex_row.to_dict())
    return pd.DataFrame(df_list, columns=emolex.columns)

def LIWC(LIWC_path, filepath):
    fileLIWCpath=os.path.join(LIWC_path,'LIWC_'+os.path.split(filepath)[1].replace('.pickle','.csv'))
    fileLIWC=pd.read_csv(fileLIWCpath)
    
    #Divide the LIWC feature to the word count
    fileLIWC=fileLIWC.iloc[:,3:].div(fileLIWC.iloc[:,2].tolist(), axis=0)
    
    return fileLIWC