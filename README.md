# Machine Learning-Based Intrusion Detection for MANETs

This repository contains code and data for detecting Denial-of-Service attacks in Mobile Ad Hoc Networks using machine learning classifiers.

## Results Summary

We evaluated four classifiers using 5-fold stratified cross-validation on 4,207 network flow samples with 21 features.

**Multiclass Classification** (Smooth / Non-Malicious / Malicious):

| Model | Accuracy | F1-Score | Cohen's Kappa |
|-------|----------|----------|---------------|
| XGBoost | 94.7 ± 0.5% | 94.8% | 0.921 |
| Random Forest | 94.2 ± 0.3% | 94.2% | 0.913 |
| SVM | 93.9 ± 0.4% | 93.9% | 0.908 |
| KNN | 91.8 ± 0.9% | 91.8% | 0.878 |

**Binary Classification** (Attack vs No-Attack):

| Model | Accuracy | ROC-AUC | Avg Precision |
|-------|----------|---------|---------------|
| XGBoost | 96.7 ± 0.6% | 0.994 | 0.989 |
| Random Forest | 96.3 ± 0.3% | 0.994 | 0.987 |
| SVM | 96.1 ± 0.7% | 0.993 | 0.984 |

Statistical testing confirmed that XGBoost significantly outperforms Random Forest (paired t-test, p = 0.009).

## Dataset

The dataset contains 4,207 samples across three classes representing different network states in a simulated MANET environment with AODV routing and Random Waypoint mobility.

Features include queue metrics (length, buffer utilization), packet metrics (PDR, drop rate, forwarding consistency), timing metrics (response time, propagation delay), and behavioral indicators (route stability, trust values).

## Installation

```bash
git clone https://github.com/vssk18/manet-ids.git
cd manet-ids
pip install -r requirements.txt
```

## Usage

Run the complete experimental pipeline:

```bash
python src/run_experiments.py
```

This generates all figures in `figures/` and results in `results/`.

For quick testing:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('data/manet_dataset.csv')
X = df.drop('Class', axis=1).values
y = df['Class'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)
print(f"Accuracy: {accuracy_score(y_test, clf.predict(X_test))*100:.2f}%")
```

## Project Structure

```
manet-ids/
├── data/
│   └── manet_dataset.csv
├── src/
│   └── run_experiments.py
├── figures/
├── results/
├── requirements.txt
└── README.md
```

## Methodology Notes

All preprocessing (feature scaling) is performed within each cross-validation fold to prevent data leakage. We report mean accuracy with standard deviation across five folds, along with Cohen's Kappa and Matthews Correlation Coefficient for more robust performance assessment.

## Limitations

This work uses simulated network data rather than real-world traffic captures. The dataset focuses on a single attack type (DoS flooding) and does not include more sophisticated attacks like black hole or wormhole attacks. Future work should validate these results on real testbed deployments.

## Citation

```bibtex
@misc{karthik2025manet,
  author = {Varanasi, Sai Srinivasa Karthik and Ghantasala, Pravallika and 
            Reddy, Mitta Sreenidhi and Narra Rajeswari, and Arshad Ahmad Khan Mohammad},
  title = {Detection of Denial-of-Service Attacks in Mobile Ad Hoc Networks 
           Using Machine Learning Classifiers},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/vssk18/manet-ids}
}
```

## Authors

Department of CSE (Cybersecurity), GITAM School of Technology, Hyderabad Campus

Contact: amohamma2@gitam.edu

## License

MIT License
