#!/bin/bash

python src/find_topics.py --language='english' --dataset='firearms' --type_data='firearms' --min_samples=500 --min_cluster_size=1000 --cluster_selection_epsilon=0.5 --stop_words='en' --keyword_list='twitter'
python src/find_topics.py --language='english' --dataset='firearms' --type_data='firearms' --min_samples=300 --min_cluster_size=1000 --cluster_selection_epsilon=0.2 --stop_words='en' --keyword_list='twitter'
python src/find_topics.py --language='english' --dataset='firearms' --type_data='firearms' --min_samples=100 --min_cluster_size=2000 --cluster_selection_epsilon=0.3 --stop_words='en' --keyword_list='twitter'
python src/find_topics.py --language='english' --dataset='firearms' --type_data='firearms' --min_samples=100 --min_cluster_size=2000 --cluster_selection_epsilon=0.5 --stop_words='en' --keyword_list='twitter'
python src/find_topics.py --language='english' --dataset='firearms' --type_data='firearms' --min_samples=300 --min_cluster_size=3000 --cluster_selection_epsilon=0.5 --stop_words='en' --keyword_list='twitter'
