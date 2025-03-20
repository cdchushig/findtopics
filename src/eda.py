import os
import pandas as pd
import os
from pathlib import Path
import utils.consts as consts
from utils.plot_lda_corrected import LDA_topic_counts4, number_tweets_time_LDA4, plot_topic_comparison3
from utils.functions import create_years
from utils.loader import load_dataset_by_name


def getxxx(therapy_name: str, language: str):
    df_data = load_dataset_by_name(dataset_name=therapy_name, language=language)
    name = '{}_{}'.format(therapy_name, language)




# df = pd.read_csv("familycouple_therapy_merged_topic_es.csv")
# therapy = 'FamiliyandCouples'
# df.Topic.value_counts()

# lang = 'es'
# name = therapy + '_' + lang
# path2 = 'LDAPsicoterapias/' + name + '/'
# database = df
# path_file = 'LDAPsicoterapias/' + name
# path2_file = 'LDAPsicoterapias/GraficasLDAPsicoterapias/' + name
#
# path_reports_figures = str(Path.joinpath(consts.PATH_PROJECT_REPORTS_PSYCHOTHERAPY, '{}{}.csv'.format(therapy_name, language)))

# try:
#     os.mkdir(path2_file)
# except OSError:
#     print('file created')
#
# try:
#     os.mkdir(path_file)
# except OSError:
#     print('file created')
#
# grupos = df['Topic'].unique()
# for grupo in grupos:
#     grupo_df = df[df['Topic'] == grupo]
#     nombre_archivo = f"{path2}{therapy}_{lang}_{grupo}.csv"
#     path_filename = str(os.path.join(consts.PATH_PROJECT_REPORTS_PSYCHOTHERAPY, nombre_archivo))
#     grupo_df.to_csv(path_filename, index=False)

# names = ['0', '1', '2']
# for j, e in enumerate(names):
#     path_filename = str(os.path.join(consts.PATH_PROJECT_REPORTS_PSYCHOTHERAPY, path2 + therapy + '_' + lang + '_' + e + '.csv'))
#     df = pd.read_csv(path_filename)
#     df.drop_duplicates()
#     df = df.reset_index(drop=True)
#     pathx = str(os.path.join(consts.PATH_PROJECT_REPORTS_PSYCHOTHERAPY, path2))
#     create_years(df, pathx, lang, e)


# Definir las rutas y nombres de los archivos, así como las categorías y nombres de los temas
path_prefix = 'LDAPsicoterapias/' + name + '/'
file_names = [therapy + '_' + lang + '_0.csv', therapy + '_' + lang + '_1.csv', therapy + '_' + lang + '_2.csv']
categories = ['joy', 'surprise', 'sadness', 'anger', 'fear', 'disgust']
topic_names = ['Users requesting couples or family therapy', 'Professionals offering therapy',
               'Efficacy in sexual dysfunctions']
plot_title = 'Topic comparison for Familiy and Couples therapy in Spanish database'
# database = df
topic_colors = ['#ff6961', '#ffb480', '#f8f38d']  # ,'#9d94ff','#ff6961', '#ffb480','#f8f38d'
path1 = 'LDAPsicoterapias' + '/' + 'GraficasLDAPsicoterapias' + '/' + name
path = 'LDAPsicoterapias/' + name + '/'
quarters = ['07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22']
times = ['2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020',
         '2021', '2022']
time = 'year'
language = 'spanish'
language1 = 'es'
topics_to_display = [0, 1, 2]
fontsizex = 15

# Leer los DataFrames de los archivos CSV
topic_dfs = [pd.read_csv(str(os.path.join(consts.PATH_PROJECT_REPORTS_PSYCHOTHERAPY, path_prefix + file_name))) for file_name in file_names]

# Call functions
LDA_topic_counts4(database, therapy +' therapy', name, path, topic_names, topic_colors, topics_to_display, fontsizex, language)
number_tweets_time_LDA4(therapy +' therapy', topic_names, quarters, path, topic_colors, times, path1, time, language, language1, topics_to_display)
plot_topic_comparison3(therapy + ' therapy', categories, topic_dfs, topic_names, plot_title, path1,topic_colors)

