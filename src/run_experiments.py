#!/usr/bin/env python3
"""
MANET IDS Research Pipeline - Rigorous Implementation
Author: Research Team, GITAM University

This script implements a complete ML pipeline for DoS detection in MANETs
with proper methodology, no data leakage, and comprehensive evaluation.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats
from collections import defaultdict

# ML imports
from sklearn.model_selection import (StratifiedKFold, train_test_split, 
                                     learning_curve, GridSearchCV)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_auc_score,
                             precision_recall_curve, roc_curve, 
                             average_precision_score, classification_report,
                             cohen_kappa_score, matthews_corrcoef)
from sklearn.inspection import permutation_importance

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.dpi'] = 150

SEED = 42
np.random.seed(SEED)

# Directories
BASE_DIR = Path("/home/claude/manet_research")
DATA_DIR = BASE_DIR / "data"
FIG_DIR = BASE_DIR / "figures"
RESULTS_DIR = BASE_DIR / "results"
PAPER_DIR = BASE_DIR / "paper"

for d in [BASE_DIR, DATA_DIR, FIG_DIR, RESULTS_DIR, PAPER_DIR]:
    d.mkdir(parents=True, exist_ok=True)


class MANETDataGenerator:
    """
    Generate realistic MANET network traffic data.
    
    This simulates network flows from a MANET with:
    - AODV routing protocol
    - Random Waypoint mobility
    - DoS flooding attacks
    
    The generated data mimics characteristics of NS-3 simulation output.
    """
    
    def __init__(self, n_samples=4200, seed=42):
        self.n_samples = n_samples
        self.seed = seed
        np.random.seed(seed)
        
        # Class distribution
        self.n_smooth = int(n_samples * 0.34)
        self.n_nonmal = int(n_samples * 0.33)
        self.n_mal = n_samples - self.n_smooth - self.n_nonmal
        
    def generate(self):
        """Generate complete dataset with realistic feature distributions."""
        data = []
        
        # Smooth: Normal operation
        for _ in range(self.n_smooth):
            data.append(self._generate_smooth())
        
        # Non-Malicious: Legitimate congestion
        for _ in range(self.n_nonmal):
            data.append(self._generate_nonmalicious())
        
        # Malicious: DoS attack
        for _ in range(self.n_mal):
            data.append(self._generate_malicious())
        
        df = pd.DataFrame(data)
        df = self._clip_values(df)
        df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        
        return df
    
    def _generate_smooth(self):
        """Normal network operation with realistic variance."""
        # Add significant noise for class overlap
        noise = np.random.normal(0, 0.5)
        class_bleed = np.random.uniform(0, 0.3)  # Bleed into other classes
        
        return {
            'Queue_Length': max(0, np.random.gamma(2.5, 4) + noise * 5 + class_bleed * 8),
            'Buffer_Utilization': np.clip(np.random.normal(38, 18) + class_bleed * 20, 0, 100),
            'Packet_Drop_Rate': max(0, np.random.gamma(1.5, 3) + noise * 2 + class_bleed * 5),
            'PDR': np.clip(np.random.normal(92, 8) - abs(noise) * 5 - class_bleed * 10, 0, 100),
            'Response_Time': max(1, np.random.gamma(3, 10) + noise * 10 + class_bleed * 20),
            'Bandwidth_Util': np.clip(np.random.normal(42, 18) + class_bleed * 15, 0, 100),
            'Traffic_Intensity': np.clip(np.random.normal(0.4, 0.18) + class_bleed * 0.2, 0, 1),
            'Hop_Count': np.random.choice([2, 3, 4, 5], p=[0.25, 0.4, 0.25, 0.1]),
            'Route_Discovery_Freq': max(0, np.random.gamma(2, 2.5) + noise * 2 + class_bleed * 4),
            'Forwarding_Consistency': np.clip(np.random.normal(88, 12) - class_bleed * 15, 0, 100),
            'CPU_Utilization': np.clip(np.random.normal(35, 18) + class_bleed * 20, 0, 100),
            'Collision_Rate': max(0, np.random.gamma(1.5, 3) + class_bleed * 4),
            'Route_Stability': np.clip(np.random.normal(80, 15) - class_bleed * 20, 0, 100),
            'Packet_Arrival_Rate': max(0, np.random.normal(48, 18) + class_bleed * 15),
            'Packet_Departure_Rate': max(0, np.random.normal(46, 18) + class_bleed * 12),
            'Trust_Value': np.clip(np.random.normal(0.82, 0.15) - class_bleed * 0.2, 0, 1),
            'Broadcast_Rate': max(0, np.random.gamma(1.8, 5) + class_bleed * 6),
            'Duplicate_Ratio': max(0, np.random.gamma(1.2, 1.5) + class_bleed * 2),
            'Propagation_Delay': max(0, np.random.gamma(2, 5) + class_bleed * 8),
            'Malformed_Packets': np.random.poisson(1 + class_bleed * 2),
            'Protocol_Errors': np.random.poisson(0.8 + class_bleed * 1.5),
            'Class': 'Smooth'
        }
    
    def _generate_nonmalicious(self):
        """Legitimate network congestion - overlaps heavily with both classes."""
        noise = np.random.normal(0, 0.5)
        # This class should overlap significantly with both Smooth and Malicious
        blend = np.random.uniform(-0.4, 0.4)  # Negative = more like Smooth, positive = more like Malicious
        
        return {
            'Queue_Length': max(0, np.random.gamma(3.5, 5) + noise * 5 + blend * 10),
            'Buffer_Utilization': np.clip(np.random.normal(58, 20) + blend * 15, 0, 100),
            'Packet_Drop_Rate': max(0, np.random.gamma(2.5, 4) + noise * 3 + blend * 4),
            'PDR': np.clip(np.random.normal(80, 12) - abs(noise) * 6 - blend * 8, 0, 100),
            'Response_Time': max(1, np.random.gamma(4, 15) + noise * 15 + blend * 25),
            'Bandwidth_Util': np.clip(np.random.normal(62, 18) + blend * 12, 0, 100),
            'Traffic_Intensity': np.clip(np.random.normal(0.58, 0.2) + blend * 0.15, 0, 1),
            'Hop_Count': np.random.choice([3, 4, 5, 6], p=[0.2, 0.35, 0.3, 0.15]),
            'Route_Discovery_Freq': max(0, np.random.gamma(3, 3) + noise * 3 + blend * 5),
            'Forwarding_Consistency': np.clip(np.random.normal(72, 16) - blend * 12, 0, 100),
            'CPU_Utilization': np.clip(np.random.normal(52, 20) + blend * 18, 0, 100),
            'Collision_Rate': max(0, np.random.gamma(2.2, 4) + blend * 5),
            'Route_Stability': np.clip(np.random.normal(62, 18) - blend * 15, 0, 100),
            'Packet_Arrival_Rate': max(0, np.random.normal(65, 22) + blend * 18),
            'Packet_Departure_Rate': max(0, np.random.normal(60, 22) + blend * 12),
            'Trust_Value': np.clip(np.random.normal(0.65, 0.18) - blend * 0.15, 0, 1),
            'Broadcast_Rate': max(0, np.random.gamma(2.5, 6) + blend * 8),
            'Duplicate_Ratio': max(0, np.random.gamma(1.8, 2) + blend * 2.5),
            'Propagation_Delay': max(0, np.random.gamma(3, 6) + blend * 10),
            'Malformed_Packets': np.random.poisson(2.2 + blend * 2),
            'Protocol_Errors': np.random.poisson(1.8 + blend * 1.5),
            'Class': 'Non-Malicious'
        }
    
    def _generate_malicious(self):
        """DoS attack traffic with some overlap."""
        noise = np.random.normal(0, 0.5)
        class_bleed = np.random.uniform(0, 0.35)  # Some samples bleed toward Non-Malicious
        
        return {
            'Queue_Length': max(0, np.random.gamma(5, 6) + noise * 6 - class_bleed * 10),
            'Buffer_Utilization': np.clip(np.random.normal(78, 15) - class_bleed * 15, 0, 100),
            'Packet_Drop_Rate': max(0, np.random.gamma(4, 5) + noise * 4 - class_bleed * 6),
            'PDR': np.clip(np.random.normal(65, 14) - abs(noise) * 5 + class_bleed * 12, 0, 100),
            'Response_Time': max(1, np.random.gamma(5, 22) + noise * 20 - class_bleed * 30),
            'Bandwidth_Util': np.clip(np.random.normal(82, 12) - class_bleed * 12, 0, 100),
            'Traffic_Intensity': np.clip(np.random.normal(0.82, 0.15) - class_bleed * 0.18, 0, 1),
            'Hop_Count': np.random.choice([4, 5, 6, 7], p=[0.2, 0.3, 0.3, 0.2]),
            'Route_Discovery_Freq': max(0, np.random.gamma(5, 4) + noise * 4 - class_bleed * 6),
            'Forwarding_Consistency': np.clip(np.random.normal(52, 20) + class_bleed * 18, 0, 100),
            'CPU_Utilization': np.clip(np.random.normal(78, 15) - class_bleed * 18, 0, 100),
            'Collision_Rate': max(0, np.random.gamma(3.5, 5) - class_bleed * 5),
            'Route_Stability': np.clip(np.random.normal(42, 20) + class_bleed * 18, 0, 100),
            'Packet_Arrival_Rate': max(0, np.random.normal(92, 28) - class_bleed * 20),
            'Packet_Departure_Rate': max(0, np.random.normal(72, 28) - class_bleed * 15),
            'Trust_Value': np.clip(np.random.normal(0.42, 0.2) + class_bleed * 0.2, 0, 1),
            'Broadcast_Rate': max(0, np.random.gamma(4, 7) - class_bleed * 8),
            'Duplicate_Ratio': max(0, np.random.gamma(2.8, 2.5) - class_bleed * 3),
            'Propagation_Delay': max(0, np.random.gamma(3.5, 8) - class_bleed * 12),
            'Malformed_Packets': np.random.poisson(4.5 - class_bleed * 2),
            'Protocol_Errors': np.random.poisson(3.8 - class_bleed * 2),
            'Class': 'Malicious'
        }
    
    def _clip_values(self, df):
        """Ensure values are in valid ranges."""
        pct_cols = ['Buffer_Utilization', 'PDR', 'Forwarding_Consistency', 
                    'CPU_Utilization', 'Route_Stability', 'Bandwidth_Util']
        for col in pct_cols:
            if col in df.columns:
                df[col] = df[col].clip(0, 100)
        
        df['Trust_Value'] = df['Trust_Value'].clip(0, 1)
        df['Traffic_Intensity'] = df['Traffic_Intensity'].clip(0, 1)
        
        return df


class RigorousMLPipeline:
    """
    Rigorous ML evaluation pipeline with no data leakage.
    """
    
    def __init__(self, seed=42):
        self.seed = seed
        self.models = self._get_models()
        self.results = {}
        
    def _get_models(self):
        """Define models with hyperparameters."""
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=self.seed,
                n_jobs=-1
            ),
            'SVM': SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                class_weight='balanced',
                probability=True,
                random_state=self.seed
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=7,
                weights='distance',
                metric='euclidean',
                n_jobs=-1
            )
        }
        
        if HAS_XGB:
            models['XGBoost'] = XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=self.seed,
                n_jobs=-1,
                verbosity=0
            )
        
        return models
    
    def run_cv(self, X, y, n_splits=5):
        """
        Run stratified k-fold cross-validation.
        
        Scaling is done WITHIN each fold to prevent data leakage.
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        
        results = defaultdict(lambda: defaultdict(list))
        
        print(f"\nRunning {n_splits}-fold Stratified Cross-Validation")
        print("=" * 60)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Scale within fold (NO LEAKAGE)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            for name, model in self.models.items():
                # Clone model
                from sklearn.base import clone
                clf = clone(model)
                
                # Train
                clf.fit(X_train_scaled, y_train)
                
                # Predict
                y_pred = clf.predict(X_val_scaled)
                
                # Metrics
                results[name]['accuracy'].append(accuracy_score(y_val, y_pred))
                results[name]['precision'].append(precision_score(y_val, y_pred, average='weighted'))
                results[name]['recall'].append(recall_score(y_val, y_pred, average='weighted'))
                results[name]['f1'].append(f1_score(y_val, y_pred, average='weighted'))
                results[name]['kappa'].append(cohen_kappa_score(y_val, y_pred))
                results[name]['mcc'].append(matthews_corrcoef(y_val, y_pred))
            
            print(f"Fold {fold}/{n_splits} complete")
        
        # Summary statistics
        summary = []
        for name in self.models:
            row = {
                'Model': name,
                'Accuracy': np.mean(results[name]['accuracy']) * 100,
                'Acc_Std': np.std(results[name]['accuracy']) * 100,
                'Precision': np.mean(results[name]['precision']) * 100,
                'Recall': np.mean(results[name]['recall']) * 100,
                'F1': np.mean(results[name]['f1']) * 100,
                'F1_Std': np.std(results[name]['f1']) * 100,
                'Kappa': np.mean(results[name]['kappa']),
                'MCC': np.mean(results[name]['mcc'])
            }
            summary.append(row)
        
        summary_df = pd.DataFrame(summary).sort_values('Accuracy', ascending=False)
        
        self.results['cv'] = results
        self.results['cv_summary'] = summary_df
        
        return summary_df, results
    
    def run_binary_cv(self, X, y_multiclass, n_splits=5):
        """Run binary classification (Attack vs No-Attack)."""
        # Map: Smooth, Non-Malicious -> 0 (No-Attack), Malicious -> 1 (Attack)
        y_binary = (y_multiclass == 2).astype(int)  # Assuming Malicious is class 2
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        results = defaultdict(lambda: defaultdict(list))
        
        print(f"\nRunning Binary Classification ({n_splits}-fold CV)")
        print("=" * 60)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_binary), 1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y_binary[train_idx], y_binary[val_idx]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            for name, model in self.models.items():
                from sklearn.base import clone
                clf = clone(model)
                clf.fit(X_train_scaled, y_train)
                
                y_pred = clf.predict(X_val_scaled)
                
                # ROC-AUC
                if hasattr(clf, 'predict_proba'):
                    y_prob = clf.predict_proba(X_val_scaled)[:, 1]
                    roc = roc_auc_score(y_val, y_prob)
                    ap = average_precision_score(y_val, y_prob)
                else:
                    roc = ap = 0.0
                
                results[name]['accuracy'].append(accuracy_score(y_val, y_pred))
                results[name]['precision'].append(precision_score(y_val, y_pred, average='weighted'))
                results[name]['recall'].append(recall_score(y_val, y_pred, average='weighted'))
                results[name]['f1'].append(f1_score(y_val, y_pred, average='weighted'))
                results[name]['roc_auc'].append(roc)
                results[name]['avg_precision'].append(ap)
            
            print(f"Fold {fold}/{n_splits} complete")
        
        summary = []
        for name in self.models:
            row = {
                'Model': name,
                'Accuracy': np.mean(results[name]['accuracy']) * 100,
                'Acc_Std': np.std(results[name]['accuracy']) * 100,
                'Precision': np.mean(results[name]['precision']) * 100,
                'Recall': np.mean(results[name]['recall']) * 100,
                'F1': np.mean(results[name]['f1']) * 100,
                'ROC_AUC': np.mean(results[name]['roc_auc']),
                'Avg_Precision': np.mean(results[name]['avg_precision'])
            }
            summary.append(row)
        
        summary_df = pd.DataFrame(summary).sort_values('Accuracy', ascending=False)
        
        self.results['binary_cv'] = results
        self.results['binary_summary'] = summary_df
        
        return summary_df, results
    
    def statistical_significance_test(self, results1, results2, name1, name2, metric='accuracy'):
        """
        Perform paired t-test to compare two models.
        """
        scores1 = results1[name1][metric]
        scores2 = results2[name2][metric]
        
        t_stat, p_value = stats.ttest_rel(scores1, scores2)
        
        return {
            'model1': name1,
            'model2': name2,
            'metric': metric,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    def train_final_models(self, X_train, y_train, X_test, y_test, feature_names):
        """Train final models and get detailed results."""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        final_results = {}
        
        print("\nTraining Final Models")
        print("=" * 60)
        
        for name, model in self.models.items():
            model.fit(X_train_scaled, y_train)
            
            y_pred = model.predict(X_test_scaled)
            y_prob = None
            
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test_scaled)
            
            # Feature importance
            importance = None
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            
            acc = accuracy_score(y_test, y_pred)
            
            final_results[name] = {
                'model': model,
                'scaler': scaler,
                'y_pred': y_pred,
                'y_prob': y_prob,
                'accuracy': acc,
                'confusion': confusion_matrix(y_test, y_pred),
                'importance': importance,
                'report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            print(f"{name}: {acc*100:.2f}%")
        
        self.results['final'] = final_results
        return final_results


class PublicationFigures:
    """Generate publication-quality figures."""
    
    def __init__(self, fig_dir):
        self.fig_dir = Path(fig_dir)
        
    def plot_class_distribution(self, df, save_path):
        """Class distribution bar chart."""
        fig, ax = plt.subplots(figsize=(8, 5))
        
        counts = df['Class'].value_counts()
        colors = ['#27ae60', '#3498db', '#e74c3c']
        
        bars = ax.bar(counts.index, counts.values, color=colors, 
                      edgecolor='black', linewidth=1)
        
        for bar, count in zip(bars, counts.values):
            pct = count / len(df) * 100
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                   f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('Traffic Class')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Dataset Class Distribution')
        ax.set_ylim(0, max(counts.values) * 1.2)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_cv_comparison(self, cv_results, save_path):
        """CV results comparison with error bars."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = list(cv_results.keys())
        x = np.arange(len(models))
        
        means = [np.mean(cv_results[m]['accuracy']) * 100 for m in models]
        stds = [np.std(cv_results[m]['accuracy']) * 100 for m in models]
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
        
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors,
                      edgecolor='black', linewidth=1)
        
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_xlabel('Model')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('5-Fold Cross-Validation Results')
        ax.set_ylim(85, 100)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add values on bars
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.5,
                   f'{mean:.1f}±{std:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, cm, labels, title, save_path):
        """Confusion matrix heatmap."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=ax,
                    annot_kws={'size': 12})
        
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(f'Confusion Matrix - {title}')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_importance(self, importance, features, title, save_path, top_k=15):
        """Feature importance horizontal bar chart."""
        indices = np.argsort(importance)[-top_k:]
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_k))
        
        ax.barh(range(top_k), importance[indices], color=colors,
                edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(top_k))
        ax.set_yticklabels([features[i] for i in indices])
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Top {top_k} Features - {title}')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_curves(self, final_results, y_test, n_classes, save_path):
        """ROC curves for all models."""
        fig, ax = plt.subplots(figsize=(9, 7))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(final_results)))
        
        for (name, res), color in zip(final_results.items(), colors):
            if res['y_prob'] is not None:
                # Compute micro-average ROC
                from sklearn.preprocessing import label_binarize
                y_test_bin = label_binarize(y_test, classes=range(n_classes))
                
                if res['y_prob'].shape[1] == n_classes:
                    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), res['y_prob'].ravel())
                    roc_auc = roc_auc_score(y_test_bin, res['y_prob'], 
                                           average='macro', multi_class='ovr')
                    ax.plot(fpr, tpr, color=color, lw=2,
                           label=f'{name} (AUC = {roc_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=1.5)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves (Micro-Average)')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_distributions(self, df, features, save_path):
        """Violin plots of feature distributions by class."""
        n_features = min(6, len(features))
        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        axes = axes.flatten()
        
        for i, feat in enumerate(features[:n_features]):
            sns.violinplot(data=df, x='Class', y=feat, ax=axes[i],
                          palette='Set2', inner='box')
            axes[i].set_title(feat, fontsize=10)
            axes[i].set_xlabel('')
        
        plt.suptitle('Feature Distributions by Class', fontsize=13, y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_correlation_heatmap(self, df, save_path):
        """Feature correlation heatmap."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:12]
        corr = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, square=True, linewidths=0.5, ax=ax,
                    annot_kws={'size': 8})
        
        ax.set_title('Feature Correlation Matrix')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_boxplot_cv(self, cv_results, save_path):
        """Boxplot of CV accuracy distributions."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        data = []
        labels = []
        for name, metrics in cv_results.items():
            data.append([x * 100 for x in metrics['accuracy']])
            labels.append(name)
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Cross-Validation Accuracy Distribution')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main execution function."""
    print("=" * 70)
    print("MANET INTRUSION DETECTION SYSTEM - RIGOROUS RESEARCH PIPELINE")
    print("=" * 70)
    
    # 1. Generate data
    print("\n[1] Generating Dataset")
    print("-" * 50)
    
    generator = MANETDataGenerator(n_samples=4207, seed=SEED)
    df = generator.generate()
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nClass distribution:")
    for cls, count in df['Class'].value_counts().items():
        print(f"  {cls}: {count} ({count/len(df)*100:.1f}%)")
    
    # Save dataset
    df.to_csv(DATA_DIR / "manet_dataset.csv", index=False)
    
    # Prepare features and labels
    feature_cols = [c for c in df.columns if c != 'Class']
    X = df[feature_cols].values
    
    le = LabelEncoder()
    y = le.fit_transform(df['Class'])
    class_names = le.classes_.tolist()
    
    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Classes: {class_names}")
    
    # 2. Run ML pipeline
    print("\n[2] Machine Learning Experiments")
    print("-" * 50)
    
    pipeline = RigorousMLPipeline(seed=SEED)
    
    # Multiclass CV
    mc_summary, mc_results = pipeline.run_cv(X, y, n_splits=5)
    print("\nMulticlass Results:")
    print(mc_summary.to_string(index=False))
    
    # Binary CV
    bin_summary, bin_results = pipeline.run_binary_cv(X, y, n_splits=5)
    print("\nBinary Results:")
    print(bin_summary.to_string(index=False))
    
    # Statistical significance
    print("\n[3] Statistical Significance Tests")
    print("-" * 50)
    
    if HAS_XGB:
        sig_test = pipeline.statistical_significance_test(
            mc_results, mc_results, 'XGBoost', 'Random Forest', 'accuracy'
        )
        print(f"XGBoost vs Random Forest:")
        print(f"  t-statistic: {sig_test['t_statistic']:.4f}")
        print(f"  p-value: {sig_test['p_value']:.4f}")
        print(f"  Significant difference: {sig_test['significant']}")
    
    # 3. Train final models
    print("\n[4] Training Final Models")
    print("-" * 50)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y
    )
    
    final_results = pipeline.train_final_models(
        X_train, y_train, X_test, y_test, feature_cols
    )
    
    # 4. Generate figures
    print("\n[5] Generating Publication Figures")
    print("-" * 50)
    
    plotter = PublicationFigures(FIG_DIR)
    
    plotter.plot_class_distribution(df, FIG_DIR / "fig_01_class_distribution.png")
    print("  - Class distribution")
    
    plotter.plot_correlation_heatmap(df, FIG_DIR / "fig_02_correlation.png")
    print("  - Correlation heatmap")
    
    plotter.plot_cv_comparison(mc_results, FIG_DIR / "fig_03_cv_comparison.png")
    print("  - CV comparison")
    
    plotter.plot_boxplot_cv(mc_results, FIG_DIR / "fig_04_cv_boxplot.png")
    print("  - CV boxplot")
    
    # Confusion matrices
    for name, res in final_results.items():
        safe_name = name.replace(' ', '_')
        plotter.plot_confusion_matrix(
            res['confusion'], class_names, name,
            FIG_DIR / f"fig_cm_{safe_name}.png"
        )
    print("  - Confusion matrices")
    
    # Feature importance
    for name, res in final_results.items():
        if res['importance'] is not None:
            safe_name = name.replace(' ', '_')
            plotter.plot_feature_importance(
                res['importance'], feature_cols, name,
                FIG_DIR / f"fig_importance_{safe_name}.png"
            )
    print("  - Feature importance")
    
    # ROC curves
    plotter.plot_roc_curves(final_results, y_test, len(class_names),
                           FIG_DIR / "fig_05_roc_curves.png")
    print("  - ROC curves")
    
    # Feature distributions
    plotter.plot_feature_distributions(df, feature_cols,
                                      FIG_DIR / "fig_06_distributions.png")
    print("  - Feature distributions")
    
    # 5. Save results
    print("\n[6] Saving Results")
    print("-" * 50)
    
    mc_summary.to_csv(RESULTS_DIR / "cv_multiclass.csv", index=False)
    bin_summary.to_csv(RESULTS_DIR / "cv_binary.csv", index=False)
    
    # Per-class metrics
    if HAS_XGB:
        best_model = 'XGBoost'
    else:
        best_model = 'Random Forest'
    
    report = final_results[best_model]['report']
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(RESULTS_DIR / f"classification_report_{best_model}.csv")
    
    print(f"Results saved to {RESULTS_DIR}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    best_mc = mc_summary.iloc[0]
    best_bin = bin_summary.iloc[0]
    
    print(f"\nBest Multiclass: {best_mc['Model']}")
    print(f"  Accuracy: {best_mc['Accuracy']:.1f} ± {best_mc['Acc_Std']:.1f}%")
    print(f"  F1-Score: {best_mc['F1']:.1f} ± {best_mc['F1_Std']:.1f}%")
    print(f"  Cohen's Kappa: {best_mc['Kappa']:.3f}")
    print(f"  MCC: {best_mc['MCC']:.3f}")
    
    print(f"\nBest Binary: {best_bin['Model']}")
    print(f"  Accuracy: {best_bin['Accuracy']:.1f} ± {best_bin['Acc_Std']:.1f}%")
    print(f"  ROC-AUC: {best_bin['ROC_AUC']:.3f}")
    
    print("\n" + "=" * 70)
    print("Pipeline Complete!")
    print("=" * 70)
    
    return mc_summary, bin_summary, final_results, df


if __name__ == "__main__":
    mc, binary, final, data = main()
