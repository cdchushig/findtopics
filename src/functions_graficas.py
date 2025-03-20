#!/usr/bin/env python
# coding: utf-8

import matplotlib
from matplotlib import pyplot as plt
import numpy as np 
import pandas as pd 
def emotions_graficas(database,name,path,lang):    

    LK_N = []
    RT_N = []
    LK_S = [] 
    RT_S = []
    LK_F = []
    RT_F = []
    LK_SU = []
    RT_SU = []
    LK_A = []
    RT_A = []
    LK_J = []
    RT_J = []
    LK_D = []
    RT_D = []

    for i,e in enumerate(database['emotion']): 
      if e == 'others': 
        LK_N.append(database['Likes count'][i])
        RT_N.append(database['Retweet count'][i])
      if e == 'sadness': 
        LK_S.append(database['Likes count'][i])
        RT_S.append(database['Retweet count'][i])
      if e == 'fear': 
        LK_F.append(database['Likes count'][i])
        RT_F.append(database['Retweet count'][i])
      if e == 'surprise': 
        LK_SU.append(database['Likes count'][i])
        RT_SU.append(database['Retweet count'][i])
      if e == 'anger': 
        LK_A.append(database['Likes count'][i])
        RT_A.append(database['Retweet count'][i])
      if e == 'joy': 
        LK_J.append(database['Likes count'][i])
        RT_J.append(database['Retweet count'][i])
      if e == 'disgust': 
        LK_D.append(database['Likes count'][i])
        RT_D.append(database['Retweet count'][i])

    
    
    list_emotions = [LK_N, LK_S, LK_F, LK_SU, LK_A, LK_J, LK_D]
    list_RT_e = [RT_N, RT_S, RT_F, RT_SU, RT_A, RT_J, RT_D]

    from statistics import mean

    import math

    list_RT_e_1 = [[x for x in y if not pd.isnull(x)] for y in list_RT_e]

    unwanted_num = {'Yes', 'No'}
 
    list_RT = [[ele for ele in y if ele not in unwanted_num]for y in list_RT_e_1]
    new_list_RT = [[int(float(x)) for x in LK_int] for LK_int in list_RT]
    len(new_list_RT)
    row_average_RT = []
    for e in new_list_RT: 
        try: 
            row_average_RT.append(sum(e) / len(e))
        except: 
            row_average_RT.append(0)
    
    # row_average_RT = [sum(sub_list) / len(sub_list) for sub_list in new_list_RT]
    retweet_emotion = [round(num, 1) for num in row_average_RT]
    print(retweet_emotion)


    list_emotions_1 = [[x for x in y if not pd.isnull(x)] for y in list_emotions]
    unwanted_num = {'Yes', 'No'}
 
    list1 = [[ele for ele in y if ele not in unwanted_num]for y in list_emotions_1]
    new_list = [[int(float(x)) for x in LK_int] for LK_int in list1]
    len(new_list)
    row_average = []
    for e in new_list: 
        try: 
            row_average.append(sum(e) / len(e))
        except: 
            row_average.append(0)
    # row_average = [sum(sub_list) / len(sub_list) for sub_list in new_list]
    like_emotion = [round(num, 1) for num in row_average]
    print(like_emotion)
    emo=['others','sadness','fear','surprise','anger','joy','disgust']
    rt_lk=  pd.DataFrame([emo,retweet_emotion,like_emotion]).T
    rt_lk.columns=['names', 'RT','LK']

    import matplotlib
    import matplotlib.pyplot as plt
    bars = database['emotion'].value_counts().index.tolist()
    nam= database['emotion'].value_counts().values
    df2= pd.DataFrame([bars,nam]).T
    df2.columns=['names','values']
    Data=rt_lk.merge(df2, how='inner', on='names').sort_values(by=['values'])

    x = np.arange(len(bars))
    width = 0.2

    fig, ax = plt.subplots()

    #Generamos las barras para el conjunto de hombres
    #rects1 = ax.bar(x - width, num, width, label='Nº tweets', color = 'palevioletred')
    #Generamos las barras para el conjunto de mujeres
    rects2 = ax.bar(x , Data['RT'], width, label='RTs Mean per emotion', color =  'royalblue')
    rects3 = ax.bar(x + width,  Data['LK'], width, label='Likes Mean per emotion', color =  'mediumseagreen')

    ax.set_title('Engagement de los tweets por cada emoción '+lang,fontsize=21,fontfamily="serif")
    ax.set_xticks(x)
    ax.set_xticklabels(Data['names'])
    ax.legend(fontsize=16)
    plt.xticks(rotation=90, fontsize = 12)

    def autolabel(rects):
    
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width()/ 3, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    #autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    fig.set_size_inches(14, 10)
    fig.tight_layout()

    images_dir = path
    plt.savefig(f"{images_dir}/Engagement per emotion " + name + ' ' + lang+".png")
    #plt.savefig(path+'/Engagement de los tweets por cada emoción ' + name+'.png')

    #Mostramos la grafica con el metodo show()
    plt.show()


    from matplotlib.patches import Patch
    import matplotlib.pyplot as plt

    def addlabels(x,y):
        for i in range(len(x)):
            plt.text(i,y[i],y[i], horizontalalignment='center')
 
    fig, ax =plt.subplots()
    height = database['emotion'].value_counts()
    bars = database['emotion'].value_counts().index.tolist()
    x_pos = np.arange(len(bars))

    col = []
    for val in bars:
        if val == 'others':
            col.append('black')
        elif val == 'fear':
            col.append('darkorchid')
        elif val == 'sadness':
            col.append('royalblue')
        elif val == 'anger':
            col.append('crimson')
        elif val == 'joy':
            col.append('gold')
        elif val == 'surprise':
            col.append('palevioletred')
        else:
            col.append('olivedrab')

    colors = {'others':'black', 'fear':'darkorchid', 'sadness':'royalblue', 'anger':'crimson', 'joy':'gold', 'surprise':'palevioletred', 'disgust':'olivedrab'}

    patches = [Patch(color=v, label=k) for k, v in colors.items()]
    patches

    plt.bar(x_pos, height, color=col)
    ax.set_title('Emotion Tweet classification results '+lang,fontsize=21,fontfamily="serif")

    addlabels(bars, height)

    plt.xticks(x_pos, bars)
    fig.set_size_inches(12, 10)
    plt.legend(labels=['others','fear', 'sadness', 'anger', 'joy', 'surprise', 'disgust'], handles=patches, loc='best', borderaxespad=0, fontsize=14, frameon=True)

    images_dir = path
    plt.savefig(f"{images_dir}/Emotion Tweet classification results "+ name + ' ' +lang+".png")
    #plt.savefig(path+'/Emotion Tweet classification results '+ name + ' ' + lang +'.png')

    plt.show()


def n_tweets_year (database, name, path,lang):    

    years = ['2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022','2023']

    #Obtenemos la posicion de cada etiqueta en el eje de X
    x = np.arange(len(years))
    width = 0.2
    y=[]
    for e in database:
        y.append(len(e))
    fig, ax = plt.subplots()

    ax.set_title('Number of Tweets per year ' + lang,fontsize=20,fontfamily="serif")
    ax.set_xticks(x)
    ax.set_xticklabels(years)

    #Añadimos un legen() esto permite mostrar con colores a que pertence cada valor.
    for index in range(len(x)):
      ax.text(x[index]+0.3, y[index]+10.5, y[index], size=12,  horizontalalignment='right',
        verticalalignment='bottom')

    ax.plot(years,y,'-o',linewidth=3,color='mediumslateblue',label='Number of Tweets')
    ax.legend(fontsize=15)
    plt.xticks(rotation=90, fontsize = 10)
    fig.set_size_inches(17.5, 10.5)

    images_dir = path
    plt.savefig(f"{images_dir}/Nº de Tweets en cada año " + name + ' ' +lang+".png")

    #Mostramos la grafica con el metodo show()
    plt.show()
    
    
def emociones_years_graficas (database, name, path, lang):    

    others = []
    fear = []
    sadness = []
    joy = []
    anger = []
    surprise = []
    disgust = []

    for e in database:
        e.rename(columns={"emotion\r": "emotion"}, inplace=True)
        list1 = []
        for y in e['emotion']:
            list1.append(y.replace('\r',''))
        e['emotion']=list1
        others.append(len(e[e['emotion']=='others']))
        fear.append(len(e[e['emotion']=='fear']))
        sadness.append(len(e[e['emotion']=='sadness']))
        joy.append(len(e[e['emotion']=='joy']))
        anger.append(len(e[e['emotion']=='anger']))
        surprise.append(len(e[e['emotion']=='surprise']))
        disgust.append(len(e[e['emotion']=='disgust']))

    years = ['2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022','2023']
    x = np.arange(len(years))
    width = 0.2
    fig, ax = plt.subplots()

    ax.set_title('All the Tweets: Evolution of opinion over time ' + lang,fontsize=20,fontfamily="serif")
    ax.set_xticks(x)
    ax.set_xticklabels(years)

    for index in range(len(x)):
        ax.text(x[index]+0.3, joy[index]+0.5, joy[index], size=12,  horizontalalignment='right',
        verticalalignment='bottom')

    for index in range(len(x)):
        ax.text(x[index]+0.3, anger[index]+0.5, anger[index], size=12,  horizontalalignment='right',
        verticalalignment='bottom')
  
    for index in range(len(x)):
        ax.text(x[index]+0.3, surprise[index]+0.5, surprise[index], size=12,  horizontalalignment='right',
        verticalalignment='bottom')

    for index in range(len(x)):
        ax.text(x[index]+0.3, sadness[index]+0.5, sadness[index], size=12,  horizontalalignment='right',
        verticalalignment='bottom')
 
    for index in range(len(x)):
        ax.text(x[index]+0.3, fear[index]+0.5, fear[index], size=12,  horizontalalignment='right',
        verticalalignment='bottom')

    for index in range(len(x)):
        ax.text(x[index]+0.3, disgust[index]+0.5, disgust[index], size=12,  horizontalalignment='right',
        verticalalignment='bottom')
 
    #for index in range(len(x)):
     # ax.text(x[index]+0.3, others[index]+0.5, others[index], size=12,  horizontalalignment='right',
  #      verticalalignment='bottom')

    #ax.plot(years,others,'--o',linewidth=3,color='black', label='others')
    ax.plot(years,joy,'-o',linewidth=3,color='gold',label='joy')
    ax.plot(years,anger,'-o',linewidth=3, color='crimson',label='anger')
    ax.plot(years,sadness,'-o',linewidth=3,color='royalblue',label='sadness')
    ax.plot(years,surprise,'-o',linewidth=3, color='palevioletred',label='surprise')
    ax.plot(years,fear,'-o',linewidth=3,color='darkorchid',label='fear')
    ax.plot(years,disgust,'-o',linewidth=3, color='olivedrab',label='disgust')

    ax.legend(fontsize=15)
    plt.xticks(rotation=90, fontsize = 10)
    fig.set_size_inches(17.5, 10.5)

    images_dir = path
    plt.savefig(f"{images_dir}/Evolution of opinion over time (emotions) " + name + ' ' +lang+".png")

    plt.show()
    
def engagement_year (database, name, path, lang):    

    list_RT = []
    list_LK = []
    print(len(database))
    for e in database: 
        try: 
            list_RT.append(round(e['Retweet count'].mean(skipna=True)))
            list_LK.append(round(e['Likes count'].mean(skipna=True))) 
        except: 
            list_RT.append(0)
            list_LK.append(0)
            
    years = ['2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022','2023']
    print(list_RT)
    #Obtenemos la posicion de cada etiqueta en el eje de X
    x = np.arange(len(years))
    width = 0.2

    fig, ax = plt.subplots()
    
    ax.set_title('Tweets Engagement per year '+lang,fontsize=20,fontfamily="serif")
    ax.set_xticks(x)
    ax.set_xticklabels(years)

    #Añadimos un legen() esto permite mmostrar con colores a que pertence cada valor.
    for index in range(len(x)):
        ax.text(x[index]+0.3, list_RT[index]+0.05, list_RT[index], size=12,  horizontalalignment='right',
        verticalalignment='bottom')


    #Añadimos un legen() esto permite mmostrar con colores a que pertence cada valor.
    for index in range(len(x)):
        ax.text(x[index]+0.3, list_LK[index]+0.05, list_LK[index], size=12,  horizontalalignment='right',
        verticalalignment='bottom')

    ax.plot(years,list_RT,'-o',linewidth=3,color='royalblue',label='RTs Mean per year')
    # ax.plot(years,Neu,'-',linewidth=3,color='lightblue', label='Tweets Neutrales')
    ax.plot(years,list_LK,'-o',linewidth=3, color='mediumseagreen',label='LKs Mean per year')
    ax.legend(fontsize=15)
    plt.xticks(rotation=90, fontsize = 10)
    fig.set_size_inches(17.5, 10.5)

    z = np.polyfit(np.arange(0,17),np.asarray(list_RT), 1)
    y_hat = np.poly1d(z)(x)
    print('slope RT:',(y_hat[1]-y_hat[0]))
    plt.plot(np.arange(0,17), y_hat, "y--", lw=2, color = 'royalblue')

    z = np.polyfit(np.arange(0,17),np.asarray(list_LK), 1)
    y_hat = np.poly1d(z)(x)
    print('slope LK:',(y_hat[1]-y_hat[0]))
    plt.plot(np.arange(0,17), y_hat, "g--", lw=2)

    images_dir = path
    plt.savefig(f"{images_dir}/Tweets Engagement per year " + name + ' ' +lang+".png")
   
    #Mostramos la grafica con el metodo show()
    plt.show()
    

def geolocalization_tweets (database, name, path,lang):    
    
    database.rename(columns = {'Country code ':'Country'}, inplace = True)
    database.rename(columns = {'Latitude ':'Latitude'}, inplace = True)
    database.rename(columns = {'Longitude ':'Longitude'}, inplace = True)

    #function to convert to alpah2 country codes and continents
    import pycountry_convert as pc
    from pycountry_convert import country_alpha2_to_continent_code, country_name_to_country_alpha2
    def get_continent(col):
        try:
            cn_continent = pc.country_alpha2_to_continent_code(col)
        except:
            cn_continent = 'Unknown' 
        return (cn_continent)

    Continent = []

    for e in database['Country']:
        Continent.append(get_continent(e))

    database['Continent'] = Continent 

    print(database['Continent'].value_counts())

    database = database[database['Latitude'].notna()]
    database = database.reset_index()
    database = database[database['Continent'].notna()]
    database = database.reset_index()


    import descartes
    import geopandas as gpd
    from shapely.geometry import Point, Polygon
    import matplotlib.pyplot as plt

    crs = {'init': 'epsg:4326'}
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    geometry = [Point(xy) for xy in zip(database['Latitude'], database['Longitude'])]
    #geometry[:3]

    geo_df = gpd.GeoDataFrame(database,crs=crs,geometry=geometry)
    #print(geo_df.head())

    fig, ax = plt.subplots(figsize = (25,15))
    
    world.plot(ax = ax, alpha = 0.4)
    geo_df[geo_df['Continent']== 'EU'].plot(ax = ax, markersize = 5, color = 'blue', label = 'Europe')
    geo_df[geo_df['Continent']== 'NA'].plot(ax = ax, markersize = 5, color = 'red', label = 'North America' )
    geo_df[geo_df['Continent']== 'AS'].plot(ax = ax, markersize = 5, color = 'green', label = 'Asia' )
    geo_df[geo_df['Continent']== 'SA'].plot(ax = ax, markersize = 5, color = 'gold', label = 'South America' )
    geo_df[geo_df['Continent']== 'AF'].plot(ax = ax, markersize = 5, color = 'firebrick', label = 'Africa' )
    geo_df[geo_df['Continent']== 'OC'].plot(ax = ax, markersize = 5, color = 'darkmagenta', label = 'Oceania' )
    #geo_df[geo_df['Continent']== 'AN'].plot(ax = ax, markersize = 5, color = 'orangered', label = 'Antarctica' )

    plt.legend(prop={'size':15})
    ax.set_title('Geolocalization of Tweets '+ lang,fontsize=21,fontfamily="serif")

    images_dir = path
    plt.savefig(f"{images_dir}/Geolocalization of Tweets " + name + ' ' +lang+".png")
    
    #Mostramos la grafica con el metodo show()
    #plt.show()

    fig, ax = plt.subplots(figsize = (25,15))
    world.plot(ax = ax, alpha = 0.4)
    geo_df[geo_df['emotion']== 'others'].plot(ax = ax, markersize = 5, color = 'black', label = 'Others')
    geo_df[geo_df['emotion']== 'fear'].plot(ax = ax, markersize = 5, color = 'darkorchid', label = 'Fear' )
    geo_df[geo_df['emotion']== 'sadness'].plot(ax = ax, markersize = 5, color = 'royalblue', label = 'Sadness' )
    geo_df[geo_df['emotion']== 'surprise'].plot(ax = ax, markersize = 5, color = 'palevioletred', label = 'Surpirse' )
    geo_df[geo_df['emotion']== 'anger'].plot(ax = ax, markersize = 5, color = 'crimson', label = 'Anger' )
    geo_df[geo_df['emotion']== 'joy'].plot(ax = ax, markersize = 5, color = 'gold', label = 'Joy' )
    geo_df[geo_df['emotion']== 'disgust'].plot(ax = ax, markersize = 5, color = 'olivedrab', label = 'Disgust' )

    plt.legend(prop={'size':15})
    ax.set_title('Geolocalization of Tweets by emotions '+  lang,fontsize=21,fontfamily="serif")
    # name + ' ' +
    images_dir = path
    plt.savefig(f"{images_dir}/Geolocalization of Tweets by emotion "+ name + ' ' +lang+".png")
    
    #Mostramos la grafica con el metodo show()
    plt.show()
