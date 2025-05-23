
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
fechas = [ '2007','2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022']
list_colors=['#945B80','#D52756','#318061','#F28500','#FFC300','#0fa3b1','#fb6f92','#ff9770','#ffd670']

def count_per_name(list_names,topic_names, path, name, path_save):
    def autolabel(rects):
        """Funcion para agregar una etiqueta con el valor en cada barra"""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width()/ 2, height),
                        xytext=(0, 2),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    height=[]
    for e in list_names:
        df=pd.read_csv(e+'_'+path+'.csv')
        height.append(len(df))
     # create a dataset
    fig, ax =plt.subplots()
    bars = topic_names
    x_pos = np.arange(len(bars))

    # Color coded values 
    col = []
    for val in range(len(bars)):
        if val == 0:
            col.append('#945B80')
        elif val == 1:
            col.append('#D52756')
        elif val == 2:
            col.append('#318061')
        elif val == 3: 
            col.append('#F28500')
        elif val == 4:
            col.append('#FFC300')
        elif val == 5:
            col.append('#0fa3b1')
        elif val == 6:
            col.append('#fb6f92')
        elif val == 7:
            col.append('#ff9770')
        elif val == 8: 
            col.append('#ffd670')
        elif val == 9:
            col.append('#e9ff70')
        elif val == 10:
            col.append('#f07167')

    colors = {'0':'#945B80', '1':'#D52756', '2':'#318061','3':'#F28500', '4':'#FFC300', '5':'#0fa3b1',
    '6':'#fb6f92', '7':'#ff9770', '8':'#ffd670','9':'#e9ff70', '10':'#f07167'}
    # create the rectangles for the legend
    patches = [Patch(color=v, label=k) for k, v in colors.items()]
    # Create bars with different colors
    print(bars)
    box = plt.bar(x_pos, height, color=col)
    ax.set_title('Number of Tweets ' + name,fontsize=21,fontfamily="serif")
    print(box)
    # calling the function to add value labels
    autolabel(box)
    # Create names on the x-axis
    plt.xticks(x_pos, bars,rotation=15)
    fig.set_size_inches(12, 10)
    images_dir = path_save

    plt.savefig(f"{images_dir}/Number of tweets " +name+ ".png")

    # Show graph
    plt.show()
        
        
        

def topic_counts(database, name, language, path,rename_label=None):
    def autolabel(rects):
        """Funcion para agregar una etiqueta con el valor en cada barra"""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width()/ 2, height),
                        xytext=(0, 2),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    # create a dataset
    fig, ax =plt.subplots()
    database=database.drop(database[database['Topic']==-1].index)
    height = database['Topic'].value_counts()
    bars = database['Topic'].value_counts().index.tolist()
    x_pos = np.arange(len(bars))

    # Color coded values 
    col = []
    for val in bars:
        if val == 0:
            col.append('#945B80')
        elif val == 1:
            col.append('#D52756')
        elif val == 2:
            col.append('#318061')
        elif val == 3: 
            col.append('#F28500')
        elif val == 4:
            col.append('#FFC300')
        elif val == 5:
            col.append('#0fa3b1')
        elif val == 6:
            col.append('#fb6f92')
        elif val == 7:
            col.append('#ff9770')
        elif val == 8: 
            col.append('#ffd670')
        elif val == 9:
            col.append('#e9ff70')
        elif val == 10:
            col.append('#f07167')

    colors = {'0':'#945B80', '1':'#D52756', '2':'#318061','3':'#F28500', '4':'#FFC300', '5':'#0fa3b1',
    '6':'#fb6f92', '7':'#ff9770', '8':'#ffd670','9':'#e9ff70', '10':'#f07167'}
    # create the rectangles for the legend
    patches = [Patch(color=v, label=k) for k, v in colors.items()]
    patches
    if rename_label !=None:
        output=[]
        for e in bars:
            output.append(rename_label[e])
            
    else:
        output = x_pos
    # Create bars with different colors
    box = plt.bar(x_pos, height, color=col)
    ax.set_title('Number of Tweets per Topic in ' + name+ ' ' + language,fontsize=21,fontfamily="serif")

    # calling the function to add value labels
    autolabel(box)
    # Create names on the x-axis
    plt.xticks(x_pos,output,rotation=10)
    fig.set_size_inches(12, 10)
    images_dir = path

    plt.savefig(f"{images_dir}/Number of tweets per topic in " + name +' '+language+ ".png")
    plt.show()
def extract_evolution_name(list_names,Language):
    list_total=[]
    for e in list_names: 
        print(1)
        df=pd.read_csv(e + '_' + Language+'.csv')
        divided=division_year(df)
        list_total.append(divided)
    list_length = []
    for e in range(len(list_total)): 
      list_l = []
      for i in list_total[e]: 
        list_l.append(len(i))
      list_length.append(list_l)
    return list_length

def evolution_per_drugname(list_names,Language,path_save):
    
    list_length=extract_evolution_name(list_names,Language)
            
    #Obtenemos la posicion de cada etiqueta en el eje de X
    x = np.arange(len(fechas))
    fig, ax = plt.subplots()
    print(1)
    ax.set_title('Evolution of the number of Tweets '+ Language,fontsize=20,fontfamily="serif")
    
    ax.set_xticks(x)
    ax.set_xticklabels(fechas)

    for e in range(len(list_names)): 
        ax.plot(fechas,list_length[e],'-o',linewidth=3,color=list_colors[e],label=list_names[e])
    print(2)
    ax.legend(fontsize=16)
    plt.xticks(rotation=0, fontsize = 16,fontname = 'serif')
    plt.yticks(rotation=0, fontsize = 16,fontname = 'serif')
    # plt.xlabel(fechas, fontname = 'serif', fontsize = 19)
    plt.ylabel("Number of tweets", fontname = 'serif', fontsize = 19)
    fig.set_size_inches(15, 7.5)
    #fig.set_size_inches(10.5, 6.5)

    images_dir = path_save
    plt.savefig(f"{images_dir}/Number of Tweets per year " + Language+ ".png",bbox_inches='tight',dpi=1000)


    #Mostramos la grafica con el metodo show()
    plt.show()        


    
def emotions_graficas_LDA(n_topics, name, path_database, path, Language): 
    
    for e in range(n_topics):
        
        database = pd.read_csv(path_database + name +'_'+ str(e) +'.csv',error_bad_lines=False)
       
        database.rename(columns={"emotion\r": "Emotion"}, inplace=True)
        database['Emotion']=database.Emotion.replace({'others\r':'others','fear\r':'fear', 'sadness\r':'sadness','anger\r':'anger','joy\r':'joy','surprise\r':'surprise','disgust\r':'disgust'})
        print(len(database))
        def addlabels(x,y):
            for i in range(len(x)):
                plt.text(i,y[i],y[i], horizontalalignment='center')

        # create a dataset
        fig, ax =plt.subplots()
        height = database['Emotion'].value_counts()
        bars = database['Emotion'].value_counts().index.tolist()
        x_pos = np.arange(len(bars))

        # Color coded values 
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

        colors = {'others':'black', 'fear':'darkorchid', 'sadness':'royalblue', 'anger':'crimson', 'joy':'gold','surprise':'palevioletred', 'disgust':'olivedrab'}
        # create the rectangles for the legend
        patches = [Patch(color=v, label=k) for k, v in colors.items()]
        patches

        # Create bars with different colors
        plt.bar(x_pos, height, color=col)
        
#         if name=='spa':
#             name1='Spanish'
#         else:
#             name1='English'
        ax.set_title('Emotion Tweet classification results in '+ Language  + ' Topic ' + str(e),fontsize=21,fontfamily="serif")

        # calling the function to add value labels
        addlabels(bars, height)

        # Create names on the x-axis
        plt.xticks(x_pos, bars)
        fig.set_size_inches(12, 10)

        # add the legend
        # plt.legend(labels=['neutral','fear', 'sadness', 'anger', 'joy', 'surprise', 'disgust'], handles=patches, loc='best',   borderaxespad=0, fontsize=14, frameon=True)
        images_dir = path
        plt.savefig(f"{images_dir}/Emotion Tweet classification results in " + Language + ' Topic ' + str(e) +".png")

        # Show graph
        plt.show()
        
        
def numero_tweets_fechas (list1,n_topics, path, name, Language):    

    fechas = ['2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020','2021', '2022','2023']

    x = np.arange(len(fechas))
    width = 0.2
    y = []
    for e in list1:
        y.append(len(e))

    fig, ax = plt.subplots()
#     if name=='spa':
#         name1='Spanish'
#     else:
#         name1='English'
        

    ax.set_title('Nº Tweets per year in ' + Language + ' Topic '+ str(n_topics), fontsize=20, fontfamily="serif")
    ax.set_xticks(x)
    ax.set_xticklabels(fechas)

    for index in range(len(x)):
        ax.text(x[index]+0.3, y[index]+10.5, y[index], size=12,  horizontalalignment='right',
        verticalalignment='bottom')

    ax.plot(fechas,y,'-o',linewidth=3,color='mediumslateblue',label='Nº de tweets')
    ax.legend(fontsize=15)
    plt.xticks(rotation=90, fontsize = 10)
    fig.set_size_inches(17.5, 10.5)

 
    plt.savefig(path + name + str(n_topics) + 'in '+ Language + " lineas.png")

    plt.show()


def engagement_year_LDA (list1, n_topics, path, name, Language):    

    list_RT = []
    list_LK = []

    for e in list1: 
        try:
            list_RT.append(round(e['Retweet count'].mean(skipna=True),3))
            list_LK.append(round(e['Likes count'].mean(skipna=True),3))
        except:
            list_RT.append(0)
            list_LK.append(0)

    years = ['2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022','2023']
#     if name=='spa':
#         name1='Spanish'
#     else:
#         name1='English'

    #Obtenemos la posicion de cada etiqueta en el eje de X
    x = np.arange(len(years))
    width = 0.2

    fig, ax = plt.subplots()

    ax.set_title('Engagement of the tweets per year in '+ Language +' Topic '+ str(n_topics),fontsize=20,fontfamily="serif")
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

    ax.plot(years,list_RT,'--o',linewidth=3,color='royalblue',label='Mean of RTs per year')
    # ax.plot(years,Neu,'-',linewidth=3,color='lightblue', label='Tweets Neutrales')
    ax.plot(years,list_LK,'--o',linewidth=3, color='mediumseagreen',label='Mean of likes per year')
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
    plt.savefig(f"{images_dir}/Engagement de los tweets por año " + name + ' '+str(n_topics) + 'in '+ Language + ".png")
   
    #Mostramos la grafica con el metodo show()
    plt.show()
        

def number_tweets_time_LDA (list_enfer, quarters, path, list_colors, times, path1, time, language, language1,list_enfer2=[],path2=''):

    list_total=[]
    list_new_topic_names=[]
    for e in range(len(list_enfer)):
        if list_enfer[e] == 'delete':
            continue
        else:
            list_new_topic_names.append(list_enfer[e])
            list_q = []
            for i in range(len(quarters)):
                df=pd.read_csv(path+'/'+str(e)+'_'+language+'_database_'+quarters[i]+'.csv')
                list_q.append(df)
            list_total.append(list_q)

    list_length = []

    for e in range(len(list_total)): 
      list_l = []
      for i in list_total[e]: 
        list_l.append(len(i))
      list_length.append(list_l)

    
    if len(list_enfer2) !=0:
        if language== 'spa':
            language='en'
        else:
            language='spa'

            list_total=[]
        list_new_topic_names2=[]
        for e in range(len(list_enfer2)):
            if list_enfer2[e] == 'delete':
                continue
            else:
                list_new_topic_names2.append(list_enfer[e])
                list_q = []
                for i in range(len(quarters)):
                    df=pd.read_csv(path2+'/'+str(e)+'_'+language+'_database_'+quarters[i]+'.csv')
                    list_q.append(df)
                list_total.append(list_q)

        list_length2 = []

        for e in range(len(list_total)): 
          list_l = []
          for i in list_total[e]: 
            list_l.append(len(i))
          list_length2.append(list_l)

        
        
        
    #Obtenemos la posicion de cada etiqueta en el eje de X
    x = np.arange(len(times))
    fig, ax = plt.subplots()
    
    if len(list_enfer2) !=0:
        ax.set_title('Number of Tweets per ' + time  + ' in english and spanish',fontsize=20,fontfamily="serif")
    
    else:
        ax.set_title('Number of Tweets per ' + time  + ' in ' + language1,fontsize=20,fontfamily="serif")
    
    ax.set_xticks(x)
    ax.set_xticklabels(times)


    for e in range(len(list_length)): 
      ax.plot(times,list_length[e],'-o',linewidth=2,color=list_colors[e],label='Topic ' + list_new_topic_names[e])
    if len(list_enfer2) !=0:
        for e in range(len(list_length)): 
          ax.plot(times,list_length2[e],'-',linewidth=2,color=list_colors[e],label='Topic ' + list_new_topic_names2[e])
    
    ax.legend(fontsize=13)
    plt.xticks(rotation=0, fontsize = 12,fontname = 'serif')
    plt.xlabel(time, fontname = 'serif', fontsize = 18)
    plt.ylabel("Number of tweets", fontname = 'serif', fontsize = 18)
    fig.set_size_inches(20, 10)
    #fig.set_size_inches(10.5, 6.5)

    images_dir = path1
    if len(list_enfer2) !=0:
     plt.savefig(f"{images_dir}/Number of Tweets per " + time + " in english and spanish_corrected.png")
    
    else:
        plt.savefig(f"{images_dir}/Number of Tweets per " + time + ' in ' + language1 +"_corrected.png")


    #Mostramos la grafica con el metodo show()
    plt.show()        


    
    
def RT_LK_per_year_LDA(list_enfer, quarters, list_colors, language, path, years, path1, language1):

    list_total=[]
    list_new_topic_names=[]
    for e in range(len(list_enfer)):
        if list_enfer[e] == 'delete':
            continue
        else:
            list_new_topic_names.append(list_enfer[e])
            list_q = []
            for i in range(len(quarters)):
                df=pd.read_csv(path+'/'+str(e)+'_'+language+'_database_'+quarters[i]+'.csv')
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

    ax.set_title('Tweets mean RT and like distribution per year in ' + language1,fontsize=20,fontfamily="serif")
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    
    for e in range(len(list_enfer)):
        ax.plot(years,list_total_RT[e],'--',linewidth=2,color=list_colors[e],label='Topic ' + list_new_topic_names[e] + ' RT mean')
    
    for e in range(len(list_enfer)):
        ax.plot(years,list_total_like[e],'-o',linewidth=2,color=list_colors[e],label='Topic ' + list_new_topic_names[e] + ' LK mean')
        
    ax.legend(fontsize=10)
    plt.xticks(rotation=0, fontsize = 12,fontname = 'serif')
    plt.xlabel("Years", fontname = 'serif', fontsize = 18)
    plt.ylabel("Mean RT and like distribution", fontname = 'serif', fontsize = 18)
    fig.set_size_inches(20, 10)
    
    images_dir = path1
    plt.savefig(f"{images_dir}/Tweets mean RT and like distribution per year in " + language1 + ".png")


    #Mostramos la grafica con el metodo show()
    plt.show()    
    
    
    
def crear_años_LDA(n_topics, name, path_database, Language):
    df = pd.read_csv(name +'_topic_'+ Language +'.csv')
    for i in range(len(n_topics)):
        if n_topics=='delete':
            continue
        else:
            database=df[df['Topic']==i].reset_index(drop=True)
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

            numero_tweets_fechas(list1,n_topics[i], path_database, name, Language)
            engagement_year_LDA(list1, n_topics[i],path_database, name, Language)
        
def division_year(database):

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

        
        for e in database['Date']:
            if '2007' in e:
                list_07.append(database[database['Date']==e].index[0])
            elif '2008' in e:
                list_08.append(database[database['Date']==e].index[0])
            elif '2009' in e:
                list_09.append(database[database['Date']==e].index[0])
            elif '2009' in e:
                list_09.append(database[database['Date']==e].index[0])
            elif '2010' in e:
                list_10.append(database[database['Date']==e].index[0])
            elif '2011' in e:
                list_11.append(database[database['Date']==e].index[0])
            elif '2012' in e:
                list_12.append(database[database['Date']==e].index[0])
            elif '2013' in e:
                list_13.append(database[database['Date']==e].index[0])
            elif '2014' in e:
                list_14.append(database[database['Date']==e].index[0])
            elif '2015' in e:
                list_15.append(database[database['Date']==e].index[0])
            elif '2016' in e:
                list_16.append(database[database['Date']==e].index[0])
            elif '2017' in e:
                list_17.append(database[database['Date']==e].index[0])
            elif '2018' in e:
                list_18.append(database[database['Date']==e].index[0])
            elif '2019' in e:
                list_19.append(database[database['Date']==e].index[0])
            elif '2020' in e:
                list_20.append(database[database['Date']==e].index[0])
            elif '2021' in e:
                list_21.append(database[database['Date']==e].index[0])
            elif '2022' in e:
                list_22.append(database[database['Date']==e].index[0])



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

        list1 = [database_07,database_08, database_09, database_10, database_11, database_12, database_13, database_14, database_15, database_16, database_17, database_18, database_19, database_20, database_21, database_22]
        return list1

def extract_evolution_topic(df,list_topic):
    list_total=[]
    list_new_topic_names=[]
    for e in range(len(list_topic)):
        if list_topic[e] == 'delete':
            continue
        else:
            list_new_topic_names.append(list_topic[e])
            divided=division_year(df[df[Topic]==e])
            list_total.append(divided)
    list_length = []
    for e in range(len(list_total)): 
      list_l = []
      for i in list_total[e]: 
        list_l.append(len(i))
      list_length.append(list_l)
    return list_length


def evolution_topic_multilingual (drug_name,df,list_topic,path_save, df2=None,list_topic2=None):
    
    list_length=extract_evolution_topic(df,list_topic)
    if df2 != None:
        list_length2=extract_evolution_topic(df2,list_topic2)
            
    #Obtenemos la posicion de cada etiqueta en el eje de X
    x = np.arange(len(fechas))
    fig, ax = plt.subplots()
    
    ax.set_title('Evolution of the number of Tweets per topic in ' + drug_name,fontsize=20,fontfamily="serif")
    
    ax.set_xticks(x)
    ax.set_xticklabels(fechas)

    for e in range(len(list_length)): 
        ax.plot(fechas,list_length[e],'-o',linewidth=3,color=list_colors[e],label=list_topic[e])
    for e in range(len(list_length2)): 
        ax.plot(fechas,list_length2[e],'-',linewidth=2,color=list_colors[e],label=list_topi2[e])

    ax.legend(fontsize=16)
    plt.xticks(rotation=0, fontsize = 16,fontname = 'serif')
    plt.yticks(rotation=0, fontsize = 16,fontname = 'serif')
    plt.xlabel(fechas, fontname = 'serif', fontsize = 19)
    plt.ylabel("Number of tweets", fontname = 'serif', fontsize = 19)
    fig.set_size_inches(15, 7.5)
    #fig.set_size_inches(10.5, 6.5)

    images_dir = path_save
    plt.savefig(f"{images_dir}/Number of Tweets per year and topic in " + name+ ".png",bbox_inches='tight',dpi=1000)


    #Mostramos la grafica con el metodo show()
    plt.show()        


    