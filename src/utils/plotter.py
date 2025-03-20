import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series
import seaborn as sns
from matplotlib.patches import Patch
import utils.consts as consts
from pathlib import Path
from typing import List, Union
from utils.loader import load_dataset_by_name, load_dataset_addictions_by_name
from matplotlib.patches import Patch

import logging
import coloredlogs

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)



def plot_grouped_barplot(group: List[str], alg: List[str], results: List[Union[int, float]],
                         figsize=(10, 6), flag_save_figure=False, language='en') -> None:
    """
    Plots a bar plot with varying numbers of bars per group centered over the group without blank spaces.

    Parameters:
    - group: List of group names (categories) for each bar.
    - alg: List of algorithm names corresponding to each bar.
    - results: List of result values (int or float) corresponding to each bar.
    """

    # Define colors using a color map
    # colors = plt.cm.tab20.colors
    colors = sns.color_palette()
    alg_cat = pd.Categorical(alg)
    alg_colors = [colors[c] for c in alg_cat.codes]

    # Calculate positions
    dist_groups = 0.4  # Distance between successive groups
    pos = (np.array([0] + [g1 != g2 for g1, g2 in zip(group[:-1], group[1:])]) * dist_groups + 1).cumsum()
    labels = [g1 for g1, g2 in zip(group[:-1], group[1:]) if g1 != g2] + [group[-1]]
    label_pos = [sum([p for g, p in zip(group, pos) if g == label]) / len([1 for g in group if g == label])
                 for label in labels]

    # Create the bar plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(pos, results, color=alg_colors)

    for container in ax.containers:
        ax.bar_label(container, fontsize=15)

    # Set x-ticks and labels
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_xticks(label_pos, labels, fontsize=15)

    # ax.set(xlabel='Therapy', ylabel='Frequency')

    # Create legend
    handles = [Patch(color=colors[c], label=lab) for c, lab in enumerate(alg_cat.categories)]
    # sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    # ax.legend(handles=handles, title='Topic')
    ax.legend(handles=handles, fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    language_full = 'english' if language == 'en' else 'spanish'
    ax.set_title('Number of tweets per therapy in {}'.format(language_full), fontsize=15)

    plt.xlabel('Therapy', fontsize=15)
    plt.ylabel('Number of tweets', fontsize=15)

    # ax.get_legend().remove()

    plt.tight_layout()

    if flag_save_figure:
        plt.savefig(str(Path.joinpath(consts.PATH_PROJECT_REPORTS_FIGURES, 'number_tweets_topic_stacked_{}.png'.format(language))))
    else:
        plt.show()


def plot_number_tweets_stacked_language(df_data: pd.DataFrame, flag_save_figure: bool = False):
    # df_data = df_data[(df_data['emotion'] == 'others') & (df_data['language'] == language)]
    # df_data = df_data[(df_data['emotion'] == 'others')]

    df_data = df_data.sort_values(by=['therapy_fullname'], ascending=True)
    df_data['therapy_fullname_language'] = df_data[['therapy_fullname', 'language']].apply(lambda x: '{}_{}'.format(x[0], x[1]), axis=1)
    colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(12, 6))
    ax = sns.barplot(df_data, x="therapy_fullname_language", y="number_tweets_topic",
                     hue="topic_description", ax=ax, errorbar=None, palette=colors)
    for container in ax.containers:
        ax.bar_label(container, fontsize=15, rotation=90)

    # ax.set_box_aspect(5 / len(ax.patches))

    ax.legend(fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.title('Number tweets per therapy in english and spanish', fontsize=15)
    plt.xlabel('Therapy', fontsize=15)
    plt.ylabel('Number of tweets', fontsize=15)

    plt.tight_layout()

    if flag_save_figure:
        plt.savefig(str(Path.joinpath(consts.PATH_PROJECT_REPORTS_FIGURES,
                                      'number_tweets_topic_stacked_all_language.svg'.format())))
    else:
        plt.show()


def plot_number_tweets_stacked_per_topic(df_data: pd.DataFrame, language, figsize=(17, 8), flag_save_figure=False):

    df_data = df_data[(df_data['emotion'] == 'joy') & (df_data['language'] == language)]
    df_data = df_data.sort_values(by=['therapy_fullname'], ascending=True)
    print(df_data)

    plot_grouped_barplot(df_data['therapy_fullname'].tolist(),
                         df_data['topic_description'].tolist(),
                         df_data['number_tweets_topic'].tolist(),
                         figsize=figsize,
                         flag_save_figure=flag_save_figure,
                         language=language
    )


def perform_fusion_mapping_topics_addictions(dfx, dataset_name, language, type_data=''):
    df = dfx.copy()
    if dataset_name == 'prg' and language == 'spanish':
        df.loc[df["topics"] == 0, "topics"] = 100
        df.loc[df["topics"] == 5, "topics"] = 100
        df.loc[df["topics"] == 1, "topics"] = 20
        df.loc[df["topics"] == 2, "topics"] = 30
        df.loc[df["topics"] == 4, "topics"] = 40
        df.loc[df["topics"] == 6, "topics"] = 50
        df.loc[df["topics"] == 100, "topics"] = 1
        df.loc[df["topics"] == 20, "topics"] = 2
        df.loc[df["topics"] == 30, "topics"] = 3
        df.loc[df["topics"] == 40, "topics"] = 4
        df.loc[df["topics"] == 50, "topics"] = 5
        print(df['topics'].value_counts())
        df_filtered = df.loc[df["topics"].isin([1, 2, 3, 4, 5])].copy()
    elif dataset_name == 'prg' and language == 'english':
        df.loc[df["topics"] == 0, "topics"] = 10
        df.loc[df["topics"] == 1, "topics"] = 20
        df.loc[df["topics"] == 2, "topics"] = 30
        df.loc[df["topics"] == 3, "topics"] = 40
        df.loc[df["topics"] == 10, "topics"] = 1
        df.loc[df["topics"] == 20, "topics"] = 2
        df.loc[df["topics"] == 30, "topics"] = 3
        df.loc[df["topics"] == 40, "topics"] = 4
        df_filtered = df.loc[df["topics"].isin([1, 2, 3, 4])].copy()
    elif dataset_name == 'prg' and language == 'catalan':
        df.loc[df["topics"] == 0, "topics"] = 10
        df.loc[df["topics"] == 1, "topics"] = 20
        df.loc[df["topics"] == 4, "topics"] = 20
        df.loc[df["topics"] == 2, "topics"] = 30
        df.loc[df["topics"] == 3, "topics"] = 40
        df.loc[df["topics"] == 10, "topics"] = 1
        df.loc[df["topics"] == 20, "topics"] = 2
        df.loc[df["topics"] == 30, "topics"] = 3
        df.loc[df["topics"] == 40, "topics"] = 4
        df_filtered = df.loc[df["topics"].isin([1, 2, 3, 4])].copy()
    elif dataset_name == 'prg' and language == 'basque':
        df.loc[df["topics"] == 0, "topics"] = 10
        df.loc[df["topics"] == 1, "topics"] = 10
        df.loc[df["topics"] == 2, "topics"] = 10
        df.loc[df["topics"] == 4, "topics"] = 10
        df.loc[df["topics"] == 3, "topics"] = 20
        df.loc[df["topics"] == 5, "topics"] = 30
        df.loc[df["topics"] == 7, "topics"] = 30
        df.loc[df["topics"] == 6, "topics"] = 40
        df.loc[df["topics"] == 10, "topics"] = 1
        df.loc[df["topics"] == 20, "topics"] = 2
        df.loc[df["topics"] == 30, "topics"] = 3
        df.loc[df["topics"] == 40, "topics"] = 4
        df_filtered = df.loc[df["topics"].isin([1, 2, 3, 4])].copy()
    elif dataset_name == 'ldp' and language == 'spanish':
        if type_data == 'casas':
            df.loc[df["topics"] == 2, "topics"] = 3
            df.loc[df["topics"] == 1, "topics"] = 2
            df.loc[df["topics"] == 0, "topics"] = 1
            df_filtered = df.loc[df["topics"].isin([1, 2, 3])].copy()
        else:
            df.loc[df["topics"] == 2, "topics"] = 3
            df.loc[df["topics"] == 1, "topics"] = 2
            df.loc[df["topics"] == 0, "topics"] = 1
            df_filtered = df.loc[df["topics"].isin([1, 2, 3])].copy()
    elif dataset_name == 'ldp' and language == 'english':
        if type_data == 'casas':
            df.loc[df["topics"] == 1, "topics"] = 2
            df.loc[df["topics"] == 0, "topics"] = 1
            df_filtered = df.loc[df["topics"].isin([1, 2])].copy()
        else:
            df.loc[df["topics"] == 2, "topics"] = 20
            df.loc[df["topics"] == 1, "topics"] = 2
            df.loc[df["topics"] == 0, "topics"] = 1
            df_filtered = df.loc[df["topics"].isin([1, 2])].copy()
    elif dataset_name == 'ldp' and language == 'basque':
        print(dataset_name, language, type_data)
        if type_data == 'otros':
            df.loc[df["topics"] == 1, "topics"] = 2
            df.loc[df["topics"] == 0, "topics"] = 1
            df_filtered = df.loc[df["topics"].isin([1, 2])].copy()
    else:
        df_filtered = None
    return df_filtered


def plot_number_tweets_per_language(dataset_name: str,
                                    language: str,
                                    type_data: str,
                                    figsize=(15, 7),
                                    flag_save_figure=False
                                    ):

    if dataset_name == 'ldp':
        df_addiction = load_dataset_addictions_by_name(dataset_name, language, type_data)
    else: # prg
        df_addiction = load_dataset_addictions_by_name(dataset_name, language)

    df_addiction.rename(columns={"Topic": "topics"}, inplace=True)
    df_data = perform_fusion_mapping_topics_addictions(df_addiction, dataset_name, language, type_data)

    print(df_data)
    print(df_data['topics'].value_counts())

    dict_topic_names_id = '{}_{}'.format(dataset_name, language) if type_data == 'na' else '{}_{}_{}'.format(dataset_name, language, type_data)
    print('xxxxxxxxxxxx', dict_topic_names_id)
    dict_topic_names = consts.dict_addiction_language[dict_topic_names_id]
    df_data['topics_name'] = df_data['topics'].apply(lambda x: dict_topic_names[x])

    df_data_counts = df_data["topics_name"].value_counts().reset_index()

    print('xxx')
    print(df_data_counts)
    print('xxx')

    colors = sns.color_palette()

    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.barplot(df_data_counts, x="topics_name", y="count", legend=False, ax=ax, palette=colors)

    for container in ax.containers:
        ax.bar_label(container, fontsize=15)

    language_fullname = 'english' if language == 'en' else 'spanish'
    ax.set_title('Number of tweets per topic in {}'.format(language_fullname), fontsize=15)
    ax.set_ylabel('Number of tweets', fontsize=15)
    ax.set_xlabel('Topic', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='x', rotation=0)

    ax.legend().set_visible(False)
    plt.tight_layout()

    if flag_save_figure:
        plt.savefig(str(Path.joinpath(consts.PATH_PROJECT_REPORTS_FIGURES, 'number_tweets_{}_{}_{}.png'.format(dataset_name, language, type_data))))
    else:
        plt.show()


def plot_number_tweets_per_therapy(list_therapy_names: list, language, figsize=(15, 7), flag_save_figure=False):
    dict_therapy_fullnames = consts.DICT_THERAPY_FULL_NAME_EN
    list_size_datasets = []

    for therapy_name in list_therapy_names:
        df_therapy = load_dataset_by_name(therapy_name, language)
        list_size_datasets.append((therapy_name, df_therapy.shape[0]))

    df_data = pd.DataFrame(list_size_datasets)
    df_data = df_data.sort_values(0)
    df_data = df_data.rename(columns={0: 'therapy_id', 1: 'number_tweets'})
    df_data['therapy_fullname'] = df_data['therapy_id'].apply(lambda x: dict_therapy_fullnames[x])
    colors = sns.color_palette()

    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.barplot(df_data, x="therapy_fullname", y="number_tweets", hue="therapy_id", legend=False, ax=ax, palette=colors)

    for container in ax.containers:
        ax.bar_label(container, fontsize=15)

    language_fullname = 'english' if language == 'en' else 'spanish'
    ax.set_title('Number of tweets per therapy in {}'.format(language_fullname), fontsize=15)
    ax.set_ylabel('Number of tweets', fontsize=15)
    ax.set_xlabel('Therapy', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='x', rotation=0)

    ax.legend().set_visible(False)
    plt.tight_layout()

    if flag_save_figure:
        plt.savefig(str(Path.joinpath(consts.PATH_PROJECT_REPORTS_FIGURES, 'number_tweets_therapy_{}.png'.format(language))))
    else:
        plt.show()


def plot_number_tweets_per_dataset(list_x: list, list_y: list):

    def autolabel(rects):
        """Funcion para agregar una etiqueta con el valor en cada barra"""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 2),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    # create a dataset
    fig, ax = plt.subplots()
    # height = database['Topic'].value_counts()
    # bars = database['Topic'].value_counts().index.tolist()

    height = list_y
    bars = list_x

    x_pos = np.arange(len(bars))

    # Color coded values
    col = []
    for val in bars:
        if val == 1:
            col.append('#318061')  # 318061
        elif val == 2:
            col.append('#F28500')  # F28500
        elif val == 3:
            col.append('#D52756')  # D52756
        elif val == 4:
            col.append('#945B80')  # 945B80
        elif val == 5:
            col.append('#3467D4')
        elif val == 6:
            col.append('#0fa3b1')
        elif val == 7:
            col.append('#fb6f92')
        elif val == 8:
            col.append('#ff9770')
        elif val == 9:
            col.append('#ffd670')
        elif val == 10:
            col.append('#e9ff70')
        elif val == 11:
            col.append('#f07167')

    colors = {'1': '#318061', '2': '#F28500', '3': '#D52756', '4': '#945B80', '5': '#3467D4', '6': '#0fa3b1',
              '7': '#fb6f92', '8': '#ff9770', '9': '#ffd670', '10': '#e9ff70', '11': '#f07167'}

    # create the rectangles for the legend
    # patches = [Patch(color=v, label=k) for k, v in colors.items()]
    # patches

    print(bars)
    print(height)

    # Create bars with different colors
    box = plt.bar(bars, height, color=col)
    # box = plt.bar(x_pos, height, color=col)
    # ax.set_title('LDA results of ' + name, fontsize=21, fontfamily="serif")

    # calling the function to add value labels
    # autolabel(box)
    # Create names on the x-axis
    # plt.xticks(x_pos, bars)
    # fig.set_size_inches(12, 10)

    # add the legend
    # output = [str(x) for x in bars]
    # plt.legend(labels = output, handles=patches, loc='best', borderaxespad=0, fontsize=14, frameon=True)
    # images_dir = path
    # if name2 == 'spa':
    #     name1 = 'Spanish'
    # else:
    #     name1 = 'English'

    # plt.savefig(f"{images_dir}/Number of tweets per topic in " + name1 + ".png")

    plt.show()


def group_tweets_per_year(df_therapy: pd.DataFrame, therapy_id: str, timestamp_id: str = 'UTC Date'):
    # df_therapy['mydate'] = pd.to_datetime(df_therapy[timestamp_id], utc=True, format='mixed')
    df_therapy['mydate'] = pd.to_datetime(df_therapy[timestamp_id], errors='coerce')
    df_therapy['aux'] = 1
    dfx = df_therapy.groupby(df_therapy['mydate'].dt.year)['aux'].agg(['sum', 'mean', 'max'])
    dfx['year'] = dfx.index
    dfx['therapy_id'] = therapy_id
    dfx = dfx[['year', 'therapy_id', 'sum']]
    return dfx


def plot_radar_plot(dft: pd.DataFrame,
                    therapy_id: str,
                    language: str,
                    id_labels: str,
                    id_splits: str,
                    ax_external,
                    flag_save_figure: bool = False
                    ):

    if therapy_id is not None:
        dfx = dft[(dft['therapy_id'] == therapy_id) & (dft['language'] == language)]
    else:
        dfx = dft

    dfx = dfx.drop(dfx[dfx['emotion'] == 'others'].index)
    dfx = dfx.sort_values(['topic_id', 'emotion'], ascending=[True, True])

    print(dfx)

    # Each attribute we'll plot in the radar chart.
    labels_radar = dfx[id_labels].value_counts().index.tolist()
    # labels_radar.sort(key=str.lower)

    # Number of variables we're plotting.
    num_vars = len(labels_radar)

    # Split the circle into even parts and save the angles
    # so we know where to put each axis.
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # The plot is a circle, so we need to "complete the loop"
    # and append the start value to the end.
    # angles += angles[:1]

    if ax_external is None:
        fig, ax = plt.subplots(figsize=(12, 6), subplot_kw=dict(polar=True))
    else:
        ax = ax_external

    # Helper function to plot each car on the radar chart.
    def add_to_radar(topic_description, color):
        values = dfx[dfx[id_splits] == topic_description]['number_tweets_topic_emotion'].tolist()
        # values = dft.loc[topic_description].tolist()
        # values += values[:1]
        # label_description = 'english' if topic_description == 'en' else 'spanish'
        # ax.plot(angles, values, 'o-', color=color, linewidth=2, label=label_description)
        ax.plot(angles, values, 'o-', color=color, linewidth=2, label=topic_description)
        ax.fill(angles, values, color=color, alpha=0.25)

    # Define colors using a color map
    colors = plt.cm.tab10.colors

    # Add each car to the chart
    list_topic_description = dfx[id_splits].value_counts().index.values
    for idx, topic_description in enumerate(list_topic_description):
        add_to_radar(topic_description, colors[idx])

    # Fix axis to go in the right order and start at 12 o'clock.
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw axis lines for each angle and label.
    ax.set_thetagrids(np.degrees(angles), labels_radar, fontsize=14)

    # Go through labels and adjust alignment based on where it is in the circle
    for label, angle in zip(ax.get_xticklabels(), angles):
        if angle in (0, np.pi):
            label.set_horizontalalignment('center')
        elif 0 < angle < np.pi:
            label.set_horizontalalignment('left')
        else:
            label.set_horizontalalignment('right')

    # Ensure radar goes from 0 to 100.
    # ax.set_ylim(0, 100)
    # You can also set gridlines manually like this:
    # ax.set_rgrids([10, 440, 320, 210, 200])

    # Set position of y-labels (0-100) to be in the middle
    # of the first two axes.
    ax.set_rlabel_position(180 / num_vars)

    # Add some custom styling.
    # Change the color of the tick labels.
    ax.tick_params(colors='#222222')
    # Make the y-axis (0-100) labels smaller.
    ax.tick_params(axis='y', labelsize=10)
    # Change the color of the circular gridlines.
    ax.grid(color='#AAAAAA')
    # Change the color of the outermost gridline (the spine).
    ax.spines['polar'].set_color('#222222')
    # Change the background color inside the circle itself.
    ax.set_facecolor('#FAFAFA')

    dict_therapy_fullnames = consts.DICT_THERAPY_FULL_NAME_EN
    language_full = 'english' if language == 'en' else 'spanish'
    # Add title
    ax.set_title('Emotional analysis for tweets in english and spanish'.format(), y=1.08, fontsize=15)
    # ax.set_title('Emotional analysis for {} therapy in {}'.format(dict_therapy_fullnames[therapy_id], language_full), y=1.08, fontsize=15)
    # ax.set_title('Emotional analysis for \n {} therapy in {}'.format(dict_therapy_fullnames[therapy_id], language_full), fontsize=15)

    ax.legend(
        # loc='center',
        loc='upper center',
        # loc='upper right',
        # bbox_to_anchor=(1.69, 1.1),
        bbox_to_anchor=[0.5, -0.1],
        fontsize=14,
        # title='Topic',
    )

    if ax_external is None:

        if flag_save_figure:
            plt.tight_layout()
            plt.savefig(
                str(Path.joinpath(consts.PATH_PROJECT_REPORTS_FIGURES, 'radar_plot_{}_{}.png'.format(therapy_id, language))),
                bbox_inches='tight'
            )
        else:
            plt.show()


def plot_radar_plot_all(df_data: pd.DataFrame, type_image: str = 'png', flag_save_figure: bool = False):

    figw, figh = 14.5, 16.5
    fig, axs = plt.subplots(nrows=4, ncols=2, subplot_kw=dict(polar=True), figsize=(figw, figh))
    plt.rcParams['figure.constrained_layout.use'] = True
    plot_radar_plot(df_data, therapy_id='acceptance', id_labels='emotion', language='en', id_splits='topic_description',
                    ax_external=axs[0, 0], flag_save_figure=True)
    plot_radar_plot(df_data, therapy_id='cognitive', id_labels='emotion', language='en', id_splits='topic_description',
                    ax_external=axs[1, 0], flag_save_figure=True)
    # plot_radar_plot(df_data, therapy_id='family', id_labels='emotion', language='en', id_splits='topic_description',
    #                 ax_external=axs[2, 0], flag_save_figure=True)
    plot_radar_plot(df_data, therapy_id='narrative', id_labels='emotion', language='en', id_splits='topic_description',
                    ax_external=axs[2, 0], flag_save_figure=True)
    plot_radar_plot(df_data, therapy_id='psycho', id_labels='emotion', language='en', id_splits='topic_description',
                    ax_external=axs[3, 0], flag_save_figure=True)

    plot_radar_plot(df_data, therapy_id='acceptance', id_labels='emotion', language='es',
                    id_splits='topic_description', ax_external=axs[0, 1], flag_save_figure=True)
    plot_radar_plot(df_data, therapy_id='cognitive', id_labels='emotion', language='es',
                    id_splits='topic_description', ax_external=axs[1, 1], flag_save_figure=True)
    # plot_radar_plot(df_data, therapy_id='family', id_labels='emotion', language='es',
    #                 id_splits='topic_description', ax_external=axs[2, 1], flag_save_figure=True)
    plot_radar_plot(df_data, therapy_id='narrative', id_labels='emotion', language='es',
                    id_splits='topic_description', ax_external=axs[2, 1], flag_save_figure=True)
    plot_radar_plot(df_data, therapy_id='psycho', id_labels='emotion', language='es',
                    id_splits='topic_description', ax_external=axs[3, 1], flag_save_figure=True)

    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)
    plt.tight_layout()

    if flag_save_figure:
        plt.tight_layout()
        filename_fig = str(Path.joinpath(consts.PATH_PROJECT_REPORTS_FIGURES, 'radar_plot_all.{}'.format(type_image)))
        plt.savefig(filename_fig, bbox_inches='tight')
    else:
        plt.show()


def load_list_datasets_psycotherapy(list_therapy_names, language):
    list_stats = []
    for therapy_name in list_therapy_names:
        df_therapy = load_dataset_by_name(therapy_name, language)
        df_therapy_stats = group_tweets_per_year(df_therapy, therapy_name)
        list_stats.append(df_therapy_stats)

    dict_therapy_fullnames = consts.DICT_THERAPY_FULL_NAME_EN
    df_data = pd.concat(list_stats)
    df_data['therapy_fullname'] = df_data['therapy_id'].apply(lambda x: dict_therapy_fullnames[x])
    df_data = df_data.drop(df_data[df_data['year'] == 2023].index)

    return df_data


def load_list_datasets_addictions(dataset_name, language, type_data, unique_language):

    if unique_language:
        df = load_dataset_addictions_by_name(dataset_name, language, type_data)
        df.rename(columns={"Topic": "topics"}, inplace=True)
        df_data = perform_fusion_mapping_topics_addictions(df, dataset_name, language, type_data)
        dict_topic_names_id = '{}_{}'.format(dataset_name, language) if type_data == 'na' else '{}_{}_{}'.format(
            dataset_name, language, type_data)
        dict_topic_names = consts.dict_addiction_language[dict_topic_names_id]
        df_data['topics_name'] = df_data['topics'].apply(lambda x: dict_topic_names[x])
        list_unique_topics = df_data['topics_name'].unique()
        list_topics = []
        for topic in list_unique_topics:
            df_topic = df_data[df_data['topics_name'] == topic]
            df_topic_stats = group_tweets_per_year(df_topic, topic, 'createdAt')
            list_topics.append(df_topic_stats)
        df_topics = pd.concat(list_topics)
        print('000000', df_topics)
        return df_topics
    else:
        list_stats = []
        for language in ['english', 'basque', 'catalan', 'spanish']:
            df_addiction = load_dataset_addictions_by_name(dataset_name, language, type_data)
            logger.info('Dataset {} loaded'.format(language))
            df_therapy_stats = group_tweets_per_year(df_addiction, language, 'createdAt')
            list_stats.append(df_therapy_stats)
        df_data = pd.concat(list_stats)
        return df_data
    # df_data['therapy_fullname'] = df_data['therapy_id'].apply(lambda x: dict_therapy_fullnames[x])
    # df_data = df_data.drop(df_data[df_data['year'] == 2023].index)
    # return df_data


def plot_temporal_evolution_tweets(list_therapy_names: list,
                                   language: str,
                                   project: str,
                                   dataset_name: str,
                                   type_data: str,
                                   unique_language: bool,
                                   hue_var: str,
                                   flag_save_figure: bool,
                                   figsize=(12, 8),
                                   show_legend: bool = False
                                   ):

    if project == 'psyco':
        df_data = load_list_datasets_psycotherapy(list_therapy_names, language)
    else:
        df_data = load_list_datasets_addictions(dataset_name, language, type_data, unique_language)

    sns.set_style('ticks')
    plt.rcParams.update({'font.size': 15})

    fig, ax = plt.subplots(figsize=figsize)

    ax.grid(color='#EEEEEE', linestyle='--', linewidth=1.5)

    sns.lineplot(
        x="year", y="sum", hue=hue_var,
        data=df_data, ax=ax, markers=True, dashes=False, marker='o'
    )
    x = df_data['year'].values
    ax.tick_params(axis='both', which='major', labelsize=15)

    plt.xticks(np.arange(min(x), max(x) + 1, 1.0))

    # dict_therapy_fullnames = consts.DICT_THERAPY_FULL_NAME_EN
    # language_full = 'english' if language == 'en' else 'spanish'
    # plt.title('Number of tweets per year in {}'.format(language_full))
    plt.xlabel('Year', fontsize=15)
    plt.ylabel('Number of tweets', fontsize=15)

    if show_legend:
        ax.legend(fontsize=15)
    else:
        ax.get_legend().remove()

    plt.tight_layout()

    if flag_save_figure:
        plt.savefig(
            str(
                Path.joinpath(
                    consts.PATH_PROJECT_REPORTS_FIGURES,
                    'temporal_evolution_{}_{}_number_tweets_{}_{}.png'.format(project, dataset_name, language, type_data)
                )
            )
        )
        plt.close()
    else:
        plt.show()
