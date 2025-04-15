findtopics
---

### Update pip
```bash
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11
```

### Install Python libraries
```bash
pip -r requirements.txt
```

### Import vev-vars from file
```bash
export $(cat .venv) 
```

### Train models for finding topics
```bash
python src/find_topics.py --language='english' --dataset='firearms' --min_samples=1000 --min_cluster_size=10000 --cluster_selection_epsilon=0.5 --stop_words='en'
```

### Visualize results of topics
```bash
python src/visualizer_topics.py 
```

