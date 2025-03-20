findtopics
---

### Import vev-vars from file
```console
export $(cat .venv) 
```

### Train topic models
```console
python src/topics.py --language='catalan' --dataset='prg' --min_samples=100 --min_cluster_size=2000 --cluster_selection_epsilon=0.5 --stop_words='ca'
```

### Visualize results of topics
```console
python src/visualizer_topics.py 
```

