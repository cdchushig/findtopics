#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
# from gensim.models.coherencemodel import CoherenceModel
# from gensim.models.ldamodel import LdaModel
# from gensim.corpora.dictionary import Dictionary
# import pyLDAvis
# import pyLDAvis.gensim_models as gensimvis
# from bs4 import BeautifulSoup
# import string
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from dateutil import parser
# import nltk
# import operator
from itertools import chain 
from collections import defaultdict
# import seaborn as sns
# import pandas as pd
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')


def lowercase(input):
  """
  Returns lowercase text
  """
  return input.lower()

def remove_punctuation(input):
  """
  Returns text without punctuation
  """
  return input.translate(str.maketrans('','', string.punctuation))

def remove_whitespaces(input):
  """
  Returns text without extra whitespaces
  """
  return " ".join(input.split())
  
def remove_html_tags(input):
  """
  Returns text without HTML tags
  """
  soup = BeautifulSoup(input, "html.parser")
  stripped_input = soup.get_text(separator=" ")
  return stripped_input

def tokenize(input):
  """
  Returns tokenized version of text
  """
  return word_tokenize(input)

def remove_stop_words(input):
  """
  Returns text without stop words
  """
  input = word_tokenize(input)
  return [word for word in input if word not in stopwords.words('english') and len(word)>2]

def remove_stop_words2(input):
  """
  Returns text without stop words
  """
  input = word_tokenize(input)
  return [word for word in input if word not in stopwords.words('spanish') and len(word)>2]

def lemmatize(input):
  """
  Lemmatizes input using NLTK's WordNetLemmatizer
  """
  lemmatizer=WordNetLemmatizer()
  input_str=word_tokenize(input)
  new_words = []
  for word in input_str:
    new_words.append(lemmatizer.lemmatize(word))
  return ' '.join(new_words)


def nlp_pipeline(input,stopwords):
  """
  Function that calls all other functions together to perform NLP on a given text
  """
  tokens=tokenize(lemmatize(' '.join(remove_stop_words(remove_whitespaces(remove_punctuation(remove_html_tags(lowercase(input))))))))
  for e in stopwords:
    p=True
    while p==True:
        try :tokens.pop(tokens.index(e))
        except: p= False

  return tokens

def nlp_pipeline2(input,stopwords):
  """
  Function that calls all other functions together to perform NLP on a given text
  """
  tokens=tokenize(lemmatize(' '.join(remove_stop_words2(remove_whitespaces(remove_punctuation(remove_html_tags(lowercase(input))))))))
  for e in stopwords:
    p=True
    while p==True:
        try :tokens.pop(tokens.index(e))
        except: p= False

  return tokens



def preprocessing(Database,stopwords,Language):
    tokens=[]
    for e in Database:
        if Language== 'spanish':    
          mytexto=nlp_pipeline2(e,stopwords)
          tokens.append(mytexto)
        elif Language== 'english':    
          mytexto=nlp_pipeline(e,stopwords)
          tokens.append(mytexto)
    return tokens
def coherence_function(tokens,test_topics,cv):
    tw_dict_en=Dictionary(tokens)
    coherence2=[]
    corpus=[tw_dict_en.doc2bow(doc)for doc in tokens]
    for k in range(2,test_topics):
        print('Round: '+str(k))
        coherence=[]
        for e in range(cv):
            ldamodel = LdaModel(corpus, num_topics=k,                    id2word = tw_dict_en, passes=10)

            cm = CoherenceModel(             model=ldamodel, texts=tokens,             dictionary=tw_dict_en, coherence='c_v')   

            coherence.append(cm.get_coherence())
        coherence2.append((k,np.mean(coherence)-k*0.005))
    plt.plot(*zip(*coherence2))
    
    dictio = dict(coherence2)
    max_key = max(dictio, key = dictio.get)
    return coherence2,max_key
        
        
def topic_modelling(tokens,num_topics2,path_name):
    tw_dict_en=Dictionary(tokens)
    corpus=[tw_dict_en.doc2bow(doc)for doc in tokens]
    ldamodel = LdaModel(corpus, num_topics =num_topics2 ,id2word=tw_dict_en, passes=10)
    ldamodel.save(path_name+".model")
    vis_data=gensimvis.prepare(ldamodel,corpus,tw_dict_en)
    pyLDAvis.display(vis_data)
    return  vis_data ,ldamodel
    
def informacion(vis_data, ldamodel,topics):
    all_topics = {}
    lambd = 0.15  # Adjust this accordingly
    for i in range(1,topics+1): #Adjust number of topics in final model
        topic = vis_data.topic_info[vis_data.topic_info                .Category == 'Topic'+str(i)]
        topic['relevance'] = topic['loglift']*(1-lambd)                             +topic['logprob']*lambd
        all_topics['Topic '+str(i)] = topic.sort_values(by='relevance', ascending=False).Term[:30].values
    
    info= vis_data.topic_info
    
    distmat, annotations = ldamodel.diff(ldamodel, distance='hellinger', num_words=100)
    plt.imshow(distmat)
    plt.colorbar()
    plt.show()
    return info, all_topics
    

def best_documents_topic(Database_topic,topic,name,path):
    for e in range(topic):
        Data=Database_topic[Database_topic['Topic']==e].sort_values('Topic_values',ascending=False) 
        Data.to_csv(path +name+'_'+str(e) +'.csv')
    return 

def tweets_clasification(ldamodel,prepro_Database,Database):
    tw_dict_en=Dictionary(prepro_Database)
    corpus=[tw_dict_en.doc2bow(doc)for doc in prepro_Database]   
    dict3 = defaultdict(list)
    list_value=[]
    list_topics=[]
    for e in range(len(ldamodel.get_document_topics(corpus))):
        p2=dict(ldamodel.get_document_topics(corpus[e]))   
        for k, v in chain(p2.items()):
            dict3[k].append(v)
        list_value2=[]
        for e in dict3.keys():
            list_value2.append(dict3[e][-1])
        list_value.append(max(list_value2))
        list_topics.append(list_value2.index(max(list_value2)))
    Database['Topic']=list_topics
    Database['Topic_values']=list_value
    return Database

def load_function(tokens,vacuna,Surname):
    tw_dict_en=Dictionary(tokens)
    corpus=[tw_dict_en.doc2bow(doc)for doc in tokens]
    ldamodel = LdaModel.load('LDA/'+vacuna[0]+'_'+Surname+'.model')
    vis_data=gensimvis.prepare(ldamodel,corpus,tw_dict_en)
    return ldamodel, vis_data

def visualization_topic(ldamodel,topn,key,name,surname):
    
    fig, axes = plt.subplots(key,1,figsize=(10, 10*key), sharex=False) 
    for e in range(key):        
        df = pd.DataFrame(ldamodel.show_topic(e, topn),columns=['token','weight'])
        sns.barplot(x='weight', y='token', data=df, color='c', orient='h', ax=axes[e])
        axes[e].set_title('Topic ' + str(e))
        
    plt.savefig('Graficas/Topic/'+name+'_'+surname+'.png')    
    plt.show()
    
def load_function2(tokens,vacuna):
    tw_dict_en=Dictionary(tokens)
    corpus=[tw_dict_en.doc2bow(doc)for doc in tokens]
    ldamodel = LdaModel.load('LDA/'+vacuna+'.model')
    vis_data=gensimvis.prepare(ldamodel,corpus,tw_dict_en)
    return ldamodel, vis_data



def Topic_modelling_creation(Database_global,stopwords,Language,path,name,n_topics=False,cv=2,n_topics_max=6,n_words=20):
    prepro_Database=preprocessing(Database_global['cleaner_text'],stopwords,Language)
    if n_topics==False:
        coherence,key=coherence_function(prepro_Database,n_topics_max,cv)
        print(coherence)
        pd.DataFrame(coherence).to_excel(path+'_coherence.xlsx')
    else:
        key=n_topics
    vis_data,ldamodel=topic_modelling(prepro_Database,key,path)
    pyLDAvis.display(vis_data)
    
    info= informacion(vis_data, ldamodel,key)
    
    Database_topic= tweets_clasification(ldamodel,prepro_Database,Database_global)
    best_documents_topic(Database_topic,key,name,path)
    Database_topic.to_csv(path +name+'_all_topics.csv')
    return Database_topic,key



def Topic_modelling_load(Database_global,path,stopwords=['@'],Language='spa'):
    prepro_Database=preprocessing(Database_global['cleaner_text'],stopwords,Language)
    tw_dict_en=Dictionary(prepro_Database)
    corpus=[tw_dict_en.doc2bow(doc)for doc in prepro_Database]
    if Language == 'english':
        ldamodel = LdaModel.load(path+ 'en.model')
    else:
        ldamodel = LdaModel.load(path+ Language[0:3] +'.model')
    vis_data=gensimvis.prepare(ldamodel,corpus,tw_dict_en)
    return vis_data



def crear_años_2(database1,path,lang):
    
    
    groups = database1['Topic'].unique()
    new_dfs = {}
    for group in groups:
        name_s = "Topic" + str(group)
        new_dfs[name_s] = database1[database1['Topic'] == group]
        database = new_dfs[name_s]
        name = str(group)
        #lang = 'en'
        database.rename(columns={"UTC Date": "Date"}, inplace=True)
        list_07 = []
        list_08 = []
        list_09 = []
        list_10 = []
        list_11 = []
        list_12 = []
        list_13 = []
        list_14 = []
        list_15 = []
        list_16 = []
        list_17 = []
        list_18 = []
        list_19 = []
        list_20 = []
        list_21 = []
        list_22 = []
        list_23 = []

        fechas = ['2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022','2023']

        for e in database['Date']:
            if '2007' in e:
              list_07.append(database[database['Date']==e].index[0])

        for e in database['Date']:
            if '2008' in e:
              list_08.append(database[database['Date']==e].index[0])

        for e in database['Date']:
            if '2009' in e:
              list_09.append(database[database['Date']==e].index[0])

        for e in database['Date']:
            if '2009' in e:
              list_09.append(database[database['Date']==e].index[0])

        for e in database['Date']:
            if '2010' in e:
              list_10.append(database[database['Date']==e].index[0])

        for e in database['Date']:
            if '2011' in e:
              list_11.append(database[database['Date']==e].index[0])

        for e in database['Date']:
            if '2012' in e:
              list_12.append(database[database['Date']==e].index[0])

        for e in database['Date']:
            if '2013' in e:
              list_13.append(database[database['Date']==e].index[0])

        for e in database['Date']:
            if '2014' in e:
              list_14.append(database[database['Date']==e].index[0])

        for e in database['Date']:
            if '2015' in e:
              list_15.append(database[database['Date']==e].index[0])

        for e in database['Date']:
            if '2016' in e:
              list_16.append(database[database['Date']==e].index[0])

        for e in database['Date']:
            if '2017' in e:
              list_17.append(database[database['Date']==e].index[0])

        for e in database['Date']:
            if '2018' in e:
              list_18.append(database[database['Date']==e].index[0])


        for e in database['Date']:
            if '2019' in e:
              list_19.append(database[database['Date']==e].index[0])

        for e in database['Date']:
            if '2020' in e:
              list_20.append(database[database['Date']==e].index[0])

        for e in database['Date']:
            if '2021' in e:
              list_21.append(database[database['Date']==e].index[0])

        for e in database['Date']:
            if '2022' in e:
              list_22.append(database[database['Date']==e].index[0])

        for e in database['Date']:
            if '2023' in e:
              list_23.append(database[database['Date']==e].index[0])


        database_07 =database.iloc[list_07]
        database_08 =database.iloc[list_08]
        database_09 =database.iloc[list_09]
        database_10 =database.iloc[list_10]
        database_11 =database.iloc[list_11]
        database_12 =database.iloc[list_12]
        database_13 =database.iloc[list_13]
        database_14 =database.iloc[list_14]
        database_15 =database.iloc[list_15]
        database_16 =database.iloc[list_16]
        database_17 =database.iloc[list_17]
        database_18 =database.iloc[list_18]
        database_19 =database.iloc[list_19]
        database_20 =database.iloc[list_20]
        database_21 =database.iloc[list_21]
        database_22 =database.iloc[list_22]
        database_23 =database.iloc[list_23]

        list1 = [database_07, database_08, database_09, database_10, database_11, database_12, database_13, database_14, database_15, database_16, database_17, database_18, database_19, database_20, database_21, database_22, database_23]

        database_07.to_csv(path + name +'_'+lang+ '_database_07.csv')
        database_08.to_csv(path + name +'_'+lang+'_database_08.csv')
        database_09.to_csv(path + name +'_'+lang+'_database_09.csv')
        database_10.to_csv(path + name +'_'+lang+'_database_10.csv')
        database_11.to_csv(path + name +'_'+lang+'_database_11.csv')
        database_12.to_csv(path+ name +'_'+lang+'_database_12.csv')
        database_13.to_csv(path + name +'_'+lang+'_database_13.csv')
        database_14.to_csv(path + name +'_'+lang+'_database_14.csv')
        database_15.to_csv(path + name +'_'+lang+'_database_15.csv')
        database_16.to_csv(path + name +'_'+lang+'_database_16.csv')
        database_17.to_csv(path + name +'_'+lang+'_database_17.csv')
        database_18.to_csv(path + name +'_'+lang+'_database_18.csv')
        database_19.to_csv(path + name +'_'+lang+'_database_19.csv')
        database_20.to_csv(path + name +'_'+lang+'_database_20.csv')
        database_21.to_csv(path + name +'_'+lang+'_database_21.csv')  
        database_22.to_csv(path + name +'_'+lang+'_database_22.csv') 
        database_23.to_csv(path + name +'_'+lang+'_database_23.csv') 


def create_years(database, path, lang, name=''):

    database.rename(columns={"UTC Date": "Date"}, inplace=True)
    list_07 = []
    list_08 = []
    list_09 = []
    list_10 = []
    list_11 = []
    list_12 = []
    list_13 = []
    list_14 = []
    list_15 = []
    list_16 = []
    list_17 = []
    list_18 = []
    list_19 = []
    list_20 = []
    list_21 = []
    list_22 = []
    list_23 = []
    
    fechas = ['2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022','2023']

    for e in database['Date']:
        if '2007' in e:
          list_07.append(database[database['Date'] == e].index[0])

    for e in database['Date']:
        if '2008' in e:
          list_08.append(database[database['Date'] == e].index[0])

    for e in database['Date']:
        if '2009' in e:
          list_09.append(database[database['Date'] == e].index[0])

    for e in database['Date']:
        if '2009' in e:
          list_09.append(database[database['Date'] == e].index[0])

    for e in database['Date']:
        if '2010' in e:
          list_10.append(database[database['Date'] == e].index[0])

    for e in database['Date']:
        if '2011' in e:
          list_11.append(database[database['Date'] == e].index[0])

    for e in database['Date']:
        if '2012' in e:
          list_12.append(database[database['Date'] == e].index[0])

    for e in database['Date']:
        if '2013' in e:
          list_13.append(database[database['Date']==e].index[0])

    for e in database['Date']:
        if '2014' in e:
          list_14.append(database[database['Date']==e].index[0])

    for e in database['Date']:
        if '2015' in e:
          list_15.append(database[database['Date']==e].index[0])

    for e in database['Date']:
        if '2016' in e:
          list_16.append(database[database['Date']==e].index[0])

    for e in database['Date']:
        if '2017' in e:
          list_17.append(database[database['Date']==e].index[0])

    for e in database['Date']:
        if '2018' in e:
          list_18.append(database[database['Date']==e].index[0])

    for e in database['Date']:
        if '2019' in e:
          list_19.append(database[database['Date']==e].index[0])

    for e in database['Date']:
        if '2020' in e:
          list_20.append(database[database['Date']==e].index[0])

    for e in database['Date']:
        if '2021' in e:
          list_21.append(database[database['Date']==e].index[0])

    for e in database['Date']:
        if '2022' in e:
          list_22.append(database[database['Date']==e].index[0])

    for e in database['Date']:
        if '2023' in e:
          list_23.append(database[database['Date']==e].index[0])

    database_07 = database.iloc[list_07]
    database_08 = database.iloc[list_08]
    database_09 = database.iloc[list_09]
    database_10 = database.iloc[list_10]
    database_11 = database.iloc[list_11]
    database_12 = database.iloc[list_12]
    database_13 = database.iloc[list_13]
    database_14 = database.iloc[list_14]
    database_15 = database.iloc[list_15]
    database_16 = database.iloc[list_16]
    database_17 = database.iloc[list_17]
    database_18 = database.iloc[list_18]
    database_19 = database.iloc[list_19]
    database_20 = database.iloc[list_20]
    database_21 = database.iloc[list_21]
    database_22 = database.iloc[list_22]
    database_23 = database.iloc[list_23]
    
    list1 = [database_07, database_08, database_09, database_10, database_11, database_12, database_13, database_14, database_15, database_16, database_17, database_18, database_19, database_20, database_21, database_22, database_23]

    database_07.to_csv(path + name +'_'+lang+ '_database_07.csv')
    database_08.to_csv(path + name +'_'+lang+'_database_08.csv')
    database_09.to_csv(path + name +'_'+lang+'_database_09.csv')
    database_10.to_csv(path + name +'_'+lang+'_database_10.csv')
    database_11.to_csv(path + name +'_'+lang+'_database_11.csv')
    database_12.to_csv(path + name +'_'+lang+'_database_12.csv')
    database_13.to_csv(path + name +'_'+lang+'_database_13.csv')
    database_14.to_csv(path + name +'_'+lang+'_database_14.csv')
    database_15.to_csv(path + name +'_'+lang+'_database_15.csv')
    database_16.to_csv(path + name +'_'+lang+'_database_16.csv')
    database_17.to_csv(path + name +'_'+lang+'_database_17.csv')
    database_18.to_csv(path + name +'_'+lang+'_database_18.csv')
    database_19.to_csv(path + name +'_'+lang+'_database_19.csv')
    database_20.to_csv(path + name +'_'+lang+'_database_20.csv')
    database_21.to_csv(path + name +'_'+lang+'_database_21.csv')  
    database_22.to_csv(path + name +'_'+lang+'_database_22.csv') 
    database_23.to_csv(path + name +'_'+lang+'_database_23.csv')     
    
    
def load_años(path,lang,name=''):
    
    database_07=pd.read_csv(path + name +'_'+lang+ '_database_07.csv')
    database_08=pd.read_csv(path + name +'_'+lang+'_database_08.csv')
    database_09=pd.read_csv(path + name +'_'+lang+'_database_09.csv')
    database_10=pd.read_csv(path + name +'_'+lang+'_database_10.csv')
    database_11=pd.read_csv(path + name +'_'+lang+'_database_11.csv')
    database_12=pd.read_csv(path + name +'_'+lang+'_database_12.csv')
    database_13=pd.read_csv(path + name +'_'+lang+'_database_13.csv')
    database_14=pd.read_csv(path + name +'_'+lang+'_database_14.csv')
    database_15=pd.read_csv(path + name +'_'+lang+'_database_15.csv')
    database_16=pd.read_csv(path + name +'_'+lang+'_database_16.csv')
    database_17=pd.read_csv(path + name +'_'+lang+'_database_17.csv')
    database_18=pd.read_csv(path + name +'_'+lang+'_database_18.csv')
    database_19=pd.read_csv(path + name +'_'+lang+'_database_19.csv')
    database_20=pd.read_csv(path + name +'_'+lang+'_database_20.csv')
    database_21=pd.read_csv(path + name +'_'+lang+'_database_21.csv')  
    database_22=pd.read_csv(path + name +'_'+lang+'_database_22.csv') 
    database_23=pd.read_csv(path + name +'_'+lang+'_database_23.csv') 
    
    list1 = [database_07, database_08, database_09, database_10, database_11, database_12, database_13, database_14, database_15, database_16, database_17, database_18, database_19, database_20, database_21, database_22, database_23]
    
    return list1

