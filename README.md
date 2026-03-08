# Latihan-ML: Indonesian ML Practice Datasets & Notebooks

Fork/practice repo for machine learning exercises ("Latihan" = practice). Forked from phgunawan/Latihan-ML (merged PR #1); my contributions: dataset curation, analysis notebooks. Permission: team agreement.

## Overview
Collection of 40+ datasets (CSV/XLSX/NPY) for regression, classification, NLP, clustering. UCI Air Quality, Iris, heart disease, COVID patients, book reviews, etc. Jupyter notebooks included.

## Quick Start
git clone https://github.com/amanjakunnel/Latihan-ML.git
cd Latihan-ML
pip install pandas numpy scikit-learn jupyter matplotlib seaborn
jupyter notebook LatihanML13.ipynb


## Key Datasets
| Category | Examples | Use Case |
|----------|----------|----------|
| Classification | Iris.csv, heart.csv, diabetes.csv, dataBreastCancer.csv | Logistic/SVM/trees |
| Regression | CarPrice_Assignment.csv, kuadratik.csv | Linear/non-linear |
| NLP/Text | wiki_sentences_v2.csv, amazon_baby.csv, Hotel-Review-train.csv | Sentiment analysis |
| Clustering | AirQualityUCI.csv, Non-linear-Circle-PHN.csv | K-means/anomalies |
| Other | PasienCov19.csv, Skin-Cancer-Dataset.csv | Domain ML |

## Featured Notebooks
- LatihanML13.ipynb, LatihanML14.ipynb, LatihanML15a.ipynb: Core exercises
- [Student_vers]_Statistics_and_Data_Analysis_week_9.ipynb: Stats pipeline

## Example: Heart Disease Classifier
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('heart.csv')
X_train, X_test, y_train, y_test = train_test_split(
    df.drop('target', axis=1), df['target'], test_size=0.2
)
model = RandomForestClassifier().fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test):.2f}")

Tech Stack
Python, pandas, scikit-learn, Jupyter, matplotlib/seaborn
