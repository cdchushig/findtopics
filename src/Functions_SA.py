#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np 
import pandas as pd 
import tensorflow as tf
import tensorflow.keras.backend as K
from transformers import * 
from tensorflow.keras.layers import *
from nltk.tokenize import TweetTokenizer
import re
import ktrain
from ktrain import text
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[18]:
import googletrans
from googletrans import Translator
from TweetNormalizer import normalizeTweet

import textattack
from textattack.augmentation import EasyDataAugmenter
from textattack.augmentation import WordNetAugmenter
from textattack.augmentation import EmbeddingAugmenter
# In[19]:


def prepare_df_for_analysis(df_in,input_text,label,ratio):
  ''' with the dataframe and the columns prep the dataframe, dataframe with input, where is the text to classify and where are the labels'''
  selected_cols = [input_text,label]
  df_out = df_in[selected_cols]
  df_out = df_out.dropna(subset=[label])
  m = dict(zip(*np.unique(df_out[label], return_counts=True)))
  df_out = df_out[df_out[label].map(m) > 10]
  df_out[label.upper()] = df_out[label].apply(lambda x :chr(ord('@')+int(x)+1))
  df_count_vars = df_out[label.upper()].value_counts()
  ratio_max = ratio
  print(df_out[label.upper()].value_counts())
  min_var = min(df_count_vars)
  max_var = max(df_count_vars)
  while (max_var/min_var)> ratio_max: 
    print(df_out[label.upper()].value_counts())
    max_var_len = ratio_max*min_var
    to_drop = abs(max_var - max_var_len) 
    max_label = [var_value for var_value, value in df_count_vars.items() if value == max_var][0]
    df_out = df_out.drop(df_out[df_out[label.upper()]==max_label].sample(to_drop).index)
    print(df_out[label.upper()].value_counts())
    df_count_vars = df_out[label.upper()].value_counts()
    min_var = min(df_count_vars)
    max_var = max(df_count_vars)

  return (df_out,input_text,label.upper())


# In[27]:


def train (df_out,var_label,field_to, ratio,path,label,lr=2e-5,batch=8,validation_size=0.2,augmentation='None'):
    MODEL_NAME = 'vinai/bertweet-base'
    t = text.Transformer(MODEL_NAME, maxlen=128)
    df_out, text_in, label = prepare_df_for_analysis(df_out,'cleaner_text',label,ratio)
    train, test = train_test_split(df_out, test_size=validation_size, random_state=42, shuffle=True, stratify=df_out[var_label])
    if augmentation != 'None':
        train_data,train_label= augmentation_function(train[field_to].values,train[var_label].values,var_label,method=augmentation)
        trn = t.preprocess_train(train_data, train_label)
    else:
        trn = t.preprocess_train(train[field_to].values, train[var_label].values)
    val = t.preprocess_test(test[field_to].values, test[var_label].values)
    model = t.get_classifier()
    learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=batch)
    learner.autofit(lr)

    p = ktrain.get_predictor(learner.model, t)
    p.save(path)
    validation = learner.validate()
    df_val = pd.DataFrame(validation)
    path_in = path+ '/validation_matrix.csv'
    df_val['var_predicted'] = var_label
    df_val.to_csv(path_in)

    return (df_val)


# In[28]:


def prediction(cat_class, dir_path, df_inp, label_t,numero=200):
  dic_mappings = { 'A':0,'B':1,
              'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'I':8, 'J':9, 'K':10, 'L':11, 'M':12}
  '''Prediction of the category'''

  predictor = ktrain.load_predictor(dir_path)
  list_all =[]
  # for element in df_validation['Tweet_english']:
  for row, element in enumerate(df_inp[label_t]):
    if row%numero ==0:
      print(row)
    try:
      temp_val = predictor.predict(element)
    except:
      temp_val=np.nan
    list_all.append(temp_val)

  df_pred = df_inp.copy()

  col_name = cat_class+'_prediction'
  df_pred [col_name] = list_all
  df_pred[col_name].replace(dic_mappings, inplace=True)
  df_pred.to_csv( dir_path+'/test_with_predictions.csv')
  pred =df_pred[col_name]
  real= df_pred[cat_class]
  pred.fillna(float(0), inplace=True)
  conf=classification_report(real,pred,output_dict=True)
  t1=pd.DataFrame(conf)
  t1.to_csv( dir_path+'/test_classification_report.csv')
  matrix=confusion_matrix(real, pred)
  t2=pd.DataFrame(matrix)
  t2.to_csv( dir_path+'/test_confussion_matrix.csv')
  print(t1)
  return df_pred

def prediction_final(cat_class, dir_path, df_inp, label_t,numero=200):
  dic_mappings = { 'A':0,'B':1,
              'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'I':8, 'J':9, 'K':10, 'L':11, 'M':12}
  '''Prediction of the category'''

  predictor = ktrain.load_predictor(dir_path)
  list_all =[]
  # for element in df_validation['Tweet_english']:
  for row, element in enumerate(df_inp[label_t]):
    if row%numero ==0:
      print(row)
    try:
      temp_val = predictor.predict(element)
    except:
      temp_val=np.nan
    list_all.append(temp_val)

  df_pred = df_inp.copy()

  col_name = cat_class+'_prediction'
  df_pred [col_name] = list_all
  df_pred[col_name].replace(dic_mappings, inplace=True)
  return df_pred
# In[29]:


def augmentation_function(data, target,label,method):
  aug_data = []
  aug_label = []
  df1=pd.DataFrame(target.tolist())
  df=pd.DataFrame(df1.value_counts(),columns=["1"])
  print(df.index.values)
  for e in df.index.values:
    print(e)
    if df["1"].max() != df["1"].loc[e]:
        p=int(df["1"].max()/df["1"].loc[e])
        for text, label in zip(data, target):
            if method =='Easy':
                embed_aug = EasyDataAugmenter(transformations_per_example=p)
            elif method =='Wordnet':
                embed_aug = WordNetAugmenter(transformations_per_example=p)
            elif method =='Embbeding':
                embed_aug = EmbeddingAugmenter(transformations_per_example=p)
          
            if label== e[0]:
                aug_list = embed_aug.augment(text)
                aug_data.append(text)
                aug_label.append(label)
                aug_data.extend(aug_list)
                aug_label.extend([label]*len(aug_list))
    else:
        for text, label in zip(data, target):
          if label== e[0]:
            aug_data.append(text)
            aug_label.append(label)
    
    df4=pd.DataFrame(aug_label)
    df2=pd.DataFrame(df4.value_counts(),columns=["1"])
    print(df2)
  return aug_data, aug_label


def cleaning_function(df,path,Trans=True):
  df=df.dropna(subset=['Tweet'])
  print(len(df))
  df['cleaner_text']=np.arange(0,len(df))
  timeout = httpx.Timeout(5) # 5 seconds timeout
  if Trans==True:
    translator = Translator(timeout=timeout)
    for p in np.arange(0,len(df['Tweet'])):
      if p%500 ==0:
        print(p)
      try:
        df['cleaner_text'].iloc[p]= translator.translate(df['Tweet'].iloc[p],dest='en').text 
      except:
        print('sleep')
        time.sleep(600)
        df['cleaner_text'].iloc[p]= translator.translate(df['Tweet'].iloc[p],dest='en').text 
        df.to_csv(path+'clean.csv', index=False)
    for o,z in enumerate(df['cleaner_text']):
      df['cleaner_text'].iloc[o]= normalizeTweet(z)
  else:
    for o,z in enumerate(df['Tweet']):
      df['cleaner_text'].iloc[o]= normalizeTweet(z)

  df.to_csv(path+'clean.csv', index=False)
  return df
