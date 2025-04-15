findtopics
---

### Import vev-vars from file
```console
export $(cat .venv) 
```

### Train models for finding topics
```console
python src/find_topics.py --language='english' --dataset='firearms' --min_samples=1000 --min_cluster_size=10000 --cluster_selection_epsilon=0.5 --stop_words='en'
```

### Visualize results of topics
```console
python src/visualizer_topics.py 
```

