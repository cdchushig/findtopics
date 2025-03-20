#!/bin/bash

python src/find_topics.py --language='english' --dataset='firearms' --min_samples=1000 --min_cluster_size=10000 --cluster_selection_epsilon=0.5 --stop_words='en'
python src/find_topics.py --language='english' --dataset='firearms' --min_samples=1000 --min_cluster_size=10000 --cluster_selection_epsilon=0.2 --stop_words='en'
python src/find_topics.py --language='english' --dataset='firearms' --min_samples=100 --min_cluster_size=2000 --cluster_selection_epsilon=0.3 --stop_words='en'
python src/find_topics.py --language='english' --dataset='firearms' --min_samples=100 --min_cluster_size=2000 --cluster_selection_epsilon=0.5 --stop_words='en'