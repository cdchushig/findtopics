#!/bin/bash

python src/find_topics.py --language='english' --dataset='firearms' --type_data='firearms' --min_samples=500 --min_cluster_size=1000 --cluster_selection_epsilon=0.5 --stop_words='en' --keyword_list='twitter'
python src/find_topics.py --language='english' --dataset='firearms' --type_data='firearms' --min_samples=300 --min_cluster_size=1000 --cluster_selection_epsilon=0.2 --stop_words='en' --keyword_list='twitter'
python src/find_topics.py --language='english' --dataset='firearms' --type_data='firearms' --min_samples=100 --min_cluster_size=2000 --cluster_selection_epsilon=0.3 --stop_words='en' --keyword_list='twitter'
python src/find_topics.py --language='english' --dataset='firearms' --type_data='firearms' --min_samples=100 --min_cluster_size=2000 --cluster_selection_epsilon=0.5 --stop_words='en' --keyword_list='twitter'
python src/find_topics.py --language='english' --dataset='firearms' --type_data='firearms' --min_samples=300 --min_cluster_size=3000 --cluster_selection_epsilon=0.5 --stop_words='en' --keyword_list='twitter'

python src/find_topics.py --language='english' --dataset='firearms' --type_data='suicide' --min_samples=100 --min_cluster_size=200 --cluster_selection_epsilon=0.1 --stop_words='en' --keyword_list='twitter'
#python src/find_topics.py --language='english' --dataset='firearms' --type_data='suicide' --min_samples=200 --min_cluster_size=200 --cluster_selection_epsilon=0.1 --stop_words='en' --keyword_list='twitter'
#python src/find_topics.py --language='english' --dataset='firearms' --type_data='suicide' --min_samples=80 --min_cluster_size=300 --cluster_selection_epsilon=0.2 --stop_words='en' --keyword_list='twitter'
#python src/find_topics.py --language='english' --dataset='firearms' --type_data='suicide' --min_samples=40 --min_cluster_size=200 --cluster_selection_epsilon=0.1 --stop_words='en' --keyword_list='twitter'
#python src/find_topics.py --language='english' --dataset='firearms' --type_data='suicide' --min_samples=40 --min_cluster_size=500 --cluster_selection_epsilon=0.1 --stop_words='en' --keyword_list='twitter'
#
#python src/find_topics.py --language='english' --dataset='firearms' --type_data='suicide_firearms' --min_samples=100 --min_cluster_size=200 --cluster_selection_epsilon=0.1 --stop_words='en' --keyword_list='twitter'
#python src/find_topics.py --language='english' --dataset='firearms' --type_data='suicide_firearms' --min_samples=200 --min_cluster_size=200 --cluster_selection_epsilon=0.1 --stop_words='en' --keyword_list='twitter'
#python src/find_topics.py --language='english' --dataset='firearms' --type_data='suicide_firearms' --min_samples=80 --min_cluster_size=300 --cluster_selection_epsilon=0.2 --stop_words='en' --keyword_list='twitter'
#python src/find_topics.py --language='english' --dataset='firearms' --type_data='suicide_firearms' --min_samples=40 --min_cluster_size=200 --cluster_selection_epsilon=0.1 --stop_words='en' --keyword_list='twitter'
#python src/find_topics.py --language='english' --dataset='firearms' --type_data='suicide_firearms' --min_samples=40 --min_cluster_size=500 --cluster_selection_epsilon=0.1 --stop_words='en' --keyword_list='twitter'
