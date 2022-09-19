```
conda create -n workflow python=3.8
conda activate workflow
pip install -r requirements.txt
```

```
python src/sweep.py
```


```
├── LICENSE
├── Makefile           <- Makefile with commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── config         <- Config files
│   │   ├── data.py
│   │   ├── test.py
│   │   └── train.py
│   │
│   ├── data           <- Scripts to download or generate data
│   │   └── make_dataset.py
│   │
│   ├── features       <- Scripts to turn raw data into features for modeling
│   │   └── build_features.py
│   │
│   ├── model          <- Scripts to train models and test models
│   │   ├── test.py
│   │   └── train.py
│   │
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations
│       └── visualize.py
│
├── experiments        <- Configs and scripts for running experiments
│   ├── sweep.py       <- Run hyperparameter sweeps
```