import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.pyplot as plt


def load_databases(my_list,path,list_colors,list_enfer):    

#     database_s = ['df_enfer_']*number
#     import itertools
#     example_dict = {x:[] for x in [x[0]+x[1] for x in itertools.product(database_s,my_list)]}
#     example_dict
#     my_list = list(example_dict)
#     list_t = []
    for e in range(len(my_list)): 
      df = pd.read_csv(path + my_list[e]+'.csv')
      list_t.append(df)
    
    for e in range(len(list_t)): 
      list_t[e]['size'] = 1

    for e in range(len(list_t)): 
      list_t[e]['color'] = list_colors[e]

    for e in range(len(list_t)): 
      list_t[e]['estabilizador'] = list_enfer[e]

    return list_t


def geolocation(list_t,list_enfer,list_colors):
    import plotly.express as px
    import pandas as pd
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    for e in range(len(list_t)):

    # Import data from USGS
      data = list_t[e][['Country code ', 'Latitude ', 'Longitude ', 'emotion', 'Bio Location','estabilizador','color','size']]


    # Drop rows with missing or invalid values in the 'mag' column
      data = data.dropna(subset=['emotion'])
      data = data.dropna(subset=['Latitude ', 'Longitude '])
      #data = data[data.mag >= 0]


    # Create scatter map
      fig = px.scatter_geo(data, lat='Latitude ', lon='Longitude ', color='estabilizador',    color_discrete_map=
      {list_enfer[i]: list_colors[i] for i in range(len(list_enfer))},
      hover_name='estabilizador', #size='mag',
      title='Tweets of '+ list_enfer[e]+' written around the World', projection="natural earth",opacity = 0.85, size = 'size',size_max = 4,
      basemap_visible=True)
      fig.update_layout(height=700,width = 1000)

      fig.show()


def number_tweets(list_t, list_enfer, list_colors, path): 


    def addlabels(x,y):
        for i in range(len(x)):
            plt.text(i,y[i],y[i], horizontalalignment='center')
 
    # create a dataset
    fig, ax =plt.subplots()

    height = []

    for e in range(len(list_t)): 
      height.append(len(list_t[e]['cleaner_text']))

    bars = list_enfer 
    x_pos = np.arange(len(bars))


    # Create bars with different colors
    plt.bar(x_pos, height, color=list_colors,edgecolor='grey')

    # calling the function to add value labels
    addlabels(bars, height)

    # Create names on the x-axis
    plt.xticks(x_pos, bars)
    fig.set_size_inches(20, 10)

    # add the legend
    plt.xlabel("Groups of databases", fontsize = 13, fontname = 'serif')
    plt.ylabel("Number of tweets", fontsize = 13, fontname = 'serif')
    plt.title("Number of tweets per database", fontsize = 21, fontname = 'serif')
    plt.legend()

    images_dir = path
    plt.savefig(f"{images_dir}/Number of tweets per database.png")

    plt.show()


def number_tweets_emo(list_t, list_enfer, list_colors, path, word): 

    fig, ax =plt.subplots()
  
    categories = ['joy', 'surprise', 'sadness', 'anger', 'fear', 'disgust']

    
    list_e = []

    for e in range(len(list_t)):
        list_e.append(pd.concat([list_t[e].emotion.value_counts().reindex(categories[::-1], fill_value=0),
        list_t[e].emotion.value_counts(normalize=True).reindex(categories[::-1], fill_value=0).mul(100).round(1).astype(str) +                     '%'],axis=1, keys=('Counts','Percentage')))
        
    list_emo = []

    for e in range(len(list_t)):
        list_emo.append(list_e[e]['Counts'])
    
#     list_emo = []
    
#     for e in range(len(list_t)): 
#         list_emo.append(list_t[e]['emotion'].value_counts(sort=False)[['joy', 'surprise', 'sadness', 'anger', 'fear', 'disgust']])
#         print(list_emo)
    
    categories.reverse()
    X_axis = np.arange(len(categories))
    width = (-0.08 * len(list_enfer))/2

    for e in range(len(list_enfer)): 
        plt.bar(X_axis+ width, list_emo[e], width = 0.08, label = list_enfer[e], color = list_colors[e], edgecolor ='grey')
        width+=0.08

  
    plt.xticks(X_axis, categories, fontname = 'serif')
    plt.xlabel("Groups of emotions", fontsize = 13, fontname = 'serif')
    plt.ylabel("Number of tweets", fontsize = 13, fontname = 'serif')
    plt.title("Tweets distribution by " + word +" and emotion", fontsize = 15, fontname = 'serif')
    fig.set_size_inches(20, 6)
    plt.legend()

    images_dir = path
    plt.savefig(f"{images_dir}/Tweets distribution by " + word +" and emotion.png")

    plt.show()


def number_tweets_sent(list_t, list_enfer, list_colors, path): 

    fig, ax =plt.subplots()
  
    categories = ['NEU', 'POS', 'NEG']

    list_s = []

    for e in range(len(list_t)):
        list_s.append(pd.concat([list_t[e].Sentiment.value_counts().reindex(categories[::-1], fill_value=0),
        list_t[e].Sentiment.value_counts(normalize=True).reindex(categories[::-1], fill_value=0).mul(100).round(1).astype(str) +                     '%'],axis=1, keys=('Counts','Percentage')))
        
    list_sents = []

    for e in range(len(list_t)):
        list_sents.append(list_s[e]['Counts'])
        
#     list_sents = []
#     for e in range(len(list_t)): 
#       list_sents.append(list_t[e]['Sentiment'].value_counts(sort=False)[['NEU', 'POS', 'NEG']])

    categories.reverse()
    X_axis = np.arange(len(categories))
    width = (-0.08 * len(list_enfer))/2

    for e in range(len(list_enfer)): 
        plt.bar(X_axis+ width, list_sents[e], width = 0.08, label = list_enfer[e], color = list_colors[e], edgecolor ='grey')
        width+=0.08
 
    plt.xticks(X_axis, categories, fontname = 'serif')
    plt.xlabel("Groups of sentiments", fontsize = 13, fontname = 'serif')
    plt.ylabel("Number of tweets", fontsize = 13, fontname = 'serif')
    plt.title("Distribution of sentiments per database", fontsize = 15, fontname = 'serif')
    fig.set_size_inches(20, 6)
    plt.legend(loc='upper left')

    images_dir = path
    plt.savefig(f"{images_dir}/Distribution of sentiments per database.png")

    plt.show()


def number_tweets_time(list_enfer, my_list, quarters, path, list_colors, times, path1, time, language):

    list_total=[]

    for e in range(len(list_enfer)):
      list_q = []
      for i in range(len(quarters)):
        df=pd.read_csv(path+list_enfer[e]+'/'+list_enfer[e]+'_'+language+'_database_'+quarters[i]+'.csv')
        list_q.append(df)
      list_total.append(list_q)

    list_length = []

    for e in range(len(list_total)): 
      list_l = []
      for i in list_total[e]: 
        list_l.append(len(i))
      list_length.append(list_l)


    #Obtenemos la posicion de cada etiqueta en el eje de X
    x = np.arange(len(times))
    width = 0.2

    fig, ax = plt.subplots()

    ax.set_title('Number of Tweets per ' + time,fontsize=20,fontfamily="serif")
    ax.set_xticks(x)
    ax.set_xticklabels(times)


    for e in range(len(list_length)): 
      ax.plot(times,list_length[e],'-o',linewidth=2,color=list_colors[e],label=my_list[e])

    ax.legend(fontsize=13)
    plt.xticks(rotation=0, fontsize = 12,fontname = 'serif')
    plt.xlabel(time, fontname = 'serif', fontsize = 18)
    plt.ylabel("Number of tweets", fontname = 'serif', fontsize = 18)
    fig.set_size_inches(20, 10)
    #fig.set_size_inches(10.5, 6.5)

    images_dir = path1
    plt.savefig(f"{images_dir}/Number of Tweets per " + time + ".png")


    #Mostramos la grafica con el metodo show()
    plt.show()


def number_tweets_positive(list_enfer, quarters, path, list_colors, times, path1,time, language):

    list_total=[]

    for e in range(len(list_enfer)):
      list_q = []
      for i in range(len(quarters)):
        df=pd.read_csv(path+list_enfer[e]+'/'+list_enfer[e]+'_' + language + '_database_'+quarters[i]+'.csv')
        list_q.append(df)
      list_total.append(list_q)

    list_Pos = []
    list_Neu = []
    list_Neg = []

    for e in range(len(list_total)): 
      list_neg = []
      list_p = []
      list_neu = []
      for i in list_total[e]: 
        list_p.append(len(i[i['Sentiment']=='POS']))
        list_neg.append(len(i[i['Sentiment']=='NEG']))
        list_neu.append(len(i[i['Sentiment']=='NEU']))
      list_Pos.append(list_p)
      list_Neu.append(list_neu)
      list_Neg.append(list_neg)


    #Obtenemos la posicion de cada etiqueta en el eje de X
    x = np.arange(len(times))
    width = 0.2
    fig, ax = plt.subplots()

    ax.set_title("Positive opinion's evolution over time",fontsize=20,fontfamily="serif")
    ax.set_xticks(x)
    ax.set_xticklabels(times)

    for e in range(len(list_Pos)): 
       ax.plot(times,list_Pos[e],'-o',linewidth=2,color=list_colors[e], label='Positive tweets '+ list_enfer[e])

    ax.legend()

    #ax.legend(fontsize=15)
    plt.xticks(rotation=0, fontsize = 12, fontname = 'serif')
    fig.set_size_inches(20, 10)
    plt.xlabel(time, fontname = 'serif', fontsize = 18)
    plt.ylabel("Number of tweets", fontname = 'serif', fontsize = 18)

    images_dir = path1
    plt.savefig(f"{images_dir}/Positive opinion's evolution over time.png")


    #Mostramos la grafica con el metodo show()
    plt.show()


def number_tweets_negative(list_enfer, quarters, path, list_colors, times, path1,time, language):

    list_total=[]

    for e in range(len(list_enfer)):
      list_q = []
      for i in range(len(quarters)):
        df=pd.read_csv(path+list_enfer[e]+'/'+list_enfer[e]+'_' + language + '_database_'+quarters[i]+'.csv')
        list_q.append(df)
      list_total.append(list_q)

    list_Pos = []
    list_Neu = []
    list_Neg = []

    for e in range(len(list_total)): 
      list_neg = []
      list_p = []
      list_neu = []
      for i in list_total[e]: 
        list_p.append(len(i[i['Sentiment']=='POS']))
        list_neg.append(len(i[i['Sentiment']=='NEG']))
        list_neu.append(len(i[i['Sentiment']=='NEU']))
      list_Pos.append(list_p)
      list_Neu.append(list_neu)
      list_Neg.append(list_neg)


    #Obtenemos la posicion de cada etiqueta en el eje de X
    x = np.arange(len(times))
    width = 0.2
    fig, ax = plt.subplots()

    ax.set_title("Negative opinion's evolution over time",fontsize=20,fontfamily="serif")
    ax.set_xticks(x)
    ax.set_xticklabels(times)

    for e in range(len(list_Neg)): 
       ax.plot(times,list_Neg[e],'-.o',linewidth=2,color=list_colors[e], label='Negative tweets '+ list_enfer[e])

    ax.legend()

    #ax.legend(fontsize=15)
    plt.xticks(rotation=0, fontsize = 12, fontname = 'serif')
    fig.set_size_inches(20, 10)
    plt.xlabel(time, fontname = 'serif', fontsize = 18)
    plt.ylabel("Number of tweets", fontname = 'serif', fontsize = 18)

    images_dir = path1
    plt.savefig(f"{images_dir}/Negative opinion's evolution over time.png")


    #Mostramos la grafica con el metodo show()
    plt.show()


def number_tweets_neutral(list_enfer, quarters, path, list_colors, times, path1,time, language):

    list_total=[]

    for e in range(len(list_enfer)):
      list_q = []
      for i in range(len(quarters)):
        df=pd.read_csv(path+list_enfer[e]+'/'+list_enfer[e]+'_' + language + '_database_'+quarters[i]+'.csv')
        list_q.append(df)
      list_total.append(list_q)

    list_Pos = []
    list_Neu = []
    list_Neg = []

    for e in range(len(list_total)): 
      list_neg = []
      list_p = []
      list_neu = []
      for i in list_total[e]: 
        list_p.append(len(i[i['Sentiment']=='POS']))
        list_neg.append(len(i[i['Sentiment']=='NEG']))
        list_neu.append(len(i[i['Sentiment']=='NEU']))
      list_Pos.append(list_p)
      list_Neu.append(list_neu)
      list_Neg.append(list_neg)


    #Obtenemos la posicion de cada etiqueta en el eje de X
    x = np.arange(len(times))
    width = 0.2
    fig, ax = plt.subplots()

    ax.set_title("Neutral opinion's evolution over time",fontsize=20,fontfamily="serif")
    ax.set_xticks(x)
    ax.set_xticklabels(times)

    for e in range(len(list_Neu)): 
       ax.plot(times,list_Neu[e],'--o',linewidth=2,color=list_colors[e], label='Neutral tweets '+ list_enfer[e])

    ax.legend()

    #ax.legend(fontsize=15)
    plt.xticks(rotation=0, fontsize = 12, fontname = 'serif')
    fig.set_size_inches(20,10)
    plt.xlabel(time, fontname = 'serif', fontsize = 18)
    plt.ylabel("Number of tweets", fontname = 'serif', fontsize = 18)

    images_dir = path1
    plt.savefig(f"{images_dir}/Neutral opinion's evolution over time.png")


    #Mostramos la grafica con el metodo show()
    plt.show()
    
    
def RT_per_item(list_t, list_enfer, list_colors, path, word):
    
    list_total_like = []
    list_total_RT = []
    emotions = ['joy', 'surprise', 'sadness', 'anger', 'fear', 'disgust']

    for e in list_t:
      list_like = []
      list_RT = []
      for i in emotions:
         l = e[e['emotion']== i]['Likes count'].dropna()
         r = e[e['emotion']== i]['Retweet count'].dropna()
         l1=[x for x in l if not pd.isnull(x)]
         r1=[x for x in r if not pd.isnull(x)]
         unwanted_num = {'Yes', 'No', 'JP'}
         l2 = [ele for ele in l1 if ele not in unwanted_num]
         r2 = [ele for ele in r1 if ele not in unwanted_num]
         l3 = [int(float(x)) for x in l2]
         r3 = [int(float(x)) for x in r2]
         if len(l3) == 0:
            l4= 0
         else: 
            l4 = sum(l3) / len(l3)
         l5 = round(l4, 1)
         if len(r3) == 0:
            r4= 0
         else: 
            r4 = sum(r3) / len(r3)
         r5 = round(r4, 1)
         list_like.append(l5)
         list_RT.append(r5)
      list_total_like.append(list_like)
      list_total_RT.append(list_RT)

    fig, ax =plt.subplots()

    X = ['joy', 'surprise', 'sadness', 'anger', 'fear', 'disgust']

    X_axis = np.arange(len(X))
    width = (-0.08 * len(list_enfer))/2 #antes 0.08

    for e in range(len(list_enfer)): 
      plt.bar(X_axis+ width, list_total_RT[e], width = 0.08, label = list_enfer[e], color = list_colors[e], edgecolor ='grey')
      width+=0.08
 

    plt.xticks(X_axis, X, fontname = 'serif')
    plt.xlabel("Groups of emotions", fontsize = 13, fontname = 'serif')
    plt.ylabel("Mean of RT per tweet", fontsize = 13, fontname = 'serif')
    plt.title("Tweets mean RT distribution by " + word + " and emotion", fontsize = 15, fontname = 'serif')
    fig.set_size_inches(20, 6)
    plt.legend()

    images_dir = path
    plt.savefig(f"{images_dir}/Tweets mean RT distribution by "+ word + " and emotion.png")


    #Mostramos la grafica con el metodo show()
    plt.show()
    
    
def LK_per_item(list_t, list_enfer, list_colors, path, word):
    
    list_total_like = []
    list_total_RT = []
    emotions = ['joy', 'surprise', 'sadness', 'anger', 'fear', 'disgust']

    for e in list_t:
      list_like = []
      list_RT = []
      for i in emotions:
         l = e[e['emotion']== i]['Likes count'].dropna()
         r = e[e['emotion']== i]['Retweet count'].dropna()
         l1=[x for x in l if not pd.isnull(x)]
         r1=[x for x in r if not pd.isnull(x)]
         unwanted_num = {'Yes', 'No','JP'}
         l2 = [ele for ele in l1 if ele not in unwanted_num]
         r2 = [ele for ele in r1 if ele not in unwanted_num]
         l3 = [int(float(x)) for x in l2]
         r3 = [int(float(x)) for x in r2]
         if len(l3) == 0:
            l4= 0
         else: 
            l4 = sum(l3) / len(l3)
         l5 = round(l4, 1)
         if len(r3) == 0:
            r4= 0
         else: 
            r4 = sum(r3) / len(r3)
         r5 = round(r4, 1)
         list_like.append(l5)
         list_RT.append(r5)
      list_total_like.append(list_like)
      list_total_RT.append(list_RT)

    fig, ax =plt.subplots()

    X = ['joy', 'surprise', 'sadness', 'anger', 'fear', 'disgust']

    X_axis = np.arange(len(X))
    width = (-0.08 * len(list_enfer))/2

    for e in range(len(list_enfer)): 
      plt.bar(X_axis+ width, list_total_like[e], width = 0.08, label = list_enfer[e], color = list_colors[e], edgecolor ='grey')
      width+=0.08
 

    plt.xticks(X_axis, X, fontname = 'serif')
    plt.xlabel("Groups of emotions", fontsize = 13, fontname = 'serif')
    plt.ylabel("Mean of LK per tweet", fontsize = 13, fontname = 'serif')
    plt.title("Tweets mean LK distribution by " + word + " and emotion", fontsize = 15, fontname = 'serif')
    fig.set_size_inches(20, 6) #ANTES 20
    plt.legend()

    images_dir = path
    plt.savefig(f"{images_dir}/Tweets mean LK distribution by " + word + " and emotion.png")


    #Mostramos la grafica con el metodo show()
    plt.show()
    
    
    
def RT_per_year(list_enfer, my_list, quarters, list_colors, language, path, years, path1):

    list_total=[]

    for e in range(len(list_enfer)):
      list_q = []
      for i in range(len(quarters)):
        df=pd.read_csv(path+list_enfer[e]+'/'+list_enfer[e]+'_' + language + '_database_' + quarters[i] + '.csv')
        list_q.append(df)
      list_total.append(list_q)

    list_total_like = []
    list_total_RT = []

    for e in range(len(list_total)): 
        list_like = []
        list_RT = []
        for i in list_total[e]:
            l = i['Likes count'].dropna()
            r = i['Retweet count'].dropna()
            l1=[x for x in l if not pd.isnull(x)]
            r1=[x for x in r if not pd.isnull(x)]
            unwanted_num = {'Yes', 'No', 'JP'}
            l2 = [ele for ele in l1 if ele not in unwanted_num]
            r2 = [ele for ele in r1 if ele not in unwanted_num]
            l3 = [int(float(x)) for x in l2]
            r3 = [int(float(x)) for x in r2]
            if len(l3) == 0:
                l4= 0
            else: 
                l4 = sum(l3) / len(l3)
            l5 = round(l4, 1)
            if len(r3) == 0:
                r4= 0
            else: 
                r4 = sum(r3) / len(r3)
            r5 = round(r4, 1)
            list_like.append(l5)
            list_RT.append(r5)
        list_total_like.append(list_like)
        list_total_RT.append(list_RT)
                
                
    #Obtenemos la posicion de cada etiqueta en el eje de X
    x = np.arange(len(years))
    width = 0.2

    fig, ax = plt.subplots()


    ax.set_title('Tweets mean RT distribution per year',fontsize=20,fontfamily="serif")
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    
    for e in range(len(list_enfer)):
        ax.plot(years,list_total_RT[e],'-o',linewidth=2,color=list_colors[e],label=my_list[e])


    ax.legend(fontsize=10)
    plt.xticks(rotation=0, fontsize = 12,fontname = 'serif')
    plt.xlabel("Years", fontname = 'serif', fontsize = 18)
    plt.ylabel("Mean RT distribution", fontname = 'serif', fontsize = 18)
    fig.set_size_inches(20, 10)
    
    images_dir = path1
    plt.savefig(f"{images_dir}/Tweets mean RT distribution by year.png")


    #Mostramos la grafica con el metodo show()
    plt.show()
    


def LK_per_year(list_enfer, my_list, quarters, list_colors, language, path, years, path1):

    list_total=[]

    for e in range(len(list_enfer)):
      list_q = []
      for i in range(len(quarters)):
        df=pd.read_csv(path+list_enfer[e]+'/'+list_enfer[e]+'_' + language + '_database_' + quarters[i] + '.csv')
        list_q.append(df)
      list_total.append(list_q)

    list_total_like = []
    list_total_RT = []

    for e in range(len(list_total)): 
        list_like = []
        list_RT = []
        for i in list_total[e]:
            l = i['Likes count'].dropna()
            r = i['Retweet count'].dropna()
            l1=[x for x in l if not pd.isnull(x)]
            r1=[x for x in r if not pd.isnull(x)]
            unwanted_num = {'Yes', 'No', 'JP'}
            l2 = [ele for ele in l1 if ele not in unwanted_num]
            r2 = [ele for ele in r1 if ele not in unwanted_num]
            l3 = [int(float(x)) for x in l2]
            r3 = [int(float(x)) for x in r2]
            if len(l3) == 0:
                l4= 0
            else: 
                l4 = sum(l3) / len(l3)
            l5 = round(l4, 1)
            if len(r3) == 0:
                r4= 0
            else: 
                r4 = sum(r3) / len(r3)
            r5 = round(r4, 1)
            list_like.append(l5)
            list_RT.append(r5)
        list_total_like.append(list_like)
        list_total_RT.append(list_RT)
                
                
    #Obtenemos la posicion de cada etiqueta en el eje de X
    x = np.arange(len(years))
    width = 0.2

    fig, ax = plt.subplots()


    ax.set_title('Tweets mean LK distribution per year',fontsize=20,fontfamily="serif")
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    
    for e in range(len(list_enfer)):
        ax.plot(years,list_total_like[e],'-o',linewidth=2,color=list_colors[e],label=my_list[e])


    ax.legend(fontsize=10)
    plt.xticks(rotation=0, fontsize = 12,fontname = 'serif')
    plt.xlabel("Years", fontname = 'serif', fontsize = 18)
    plt.ylabel("Mean LK distribution", fontname = 'serif', fontsize = 18)
    fig.set_size_inches(20, 10)
    
    images_dir = path1
    plt.savefig(f"{images_dir}/Tweets mean LK distribution by year.png")


    #Mostramos la grafica con el metodo show()
    plt.show()