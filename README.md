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
python src/find_topics.py --dataset='firearms' --type_data='firearms' --min_samples=200 --min_cluster_size=40 --cluster_selection_epsilon=0.1 --n_jobs=64
python src/find_topics.py --dataset='firearms' --type_data='suicide' --min_samples=40 --min_cluster_size=200 --cluster_selection_epsilon=0.1 --n_jobs=64
python src/find_topics.py --dataset='firearms' --type_data='suicide_firearms' --min_samples=40 --min_cluster_size=200 --cluster_selection_epsilon=0.1 --n_jobs=32
```

### Visualize results of topics
```bash
python src/visualizer_topics.py 
```

