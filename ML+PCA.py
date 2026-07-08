#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from config import FEATURE_COLS

def train_ml_pipeline(df):
    print("--- Task 3 & 4: Applying PCA & Training Random Forest ---")

    X = df[FEATURE_COLS]
    y = df['target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=0.80)
    X_pca = pca.fit_transform(X_scaled)
    print(f"PCA reduced features from {X.shape[1]} to {X_pca.shape[1]} components.")

    split_idx = int(len(X_pca) * 0.8)
    X_train, X_test = X_pca[:split_idx], X_pca[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    model.fit(X_train, y_train)

    all_probs = model.predict_proba(X_pca)[:, 1]
    df['signal'] = np.where(all_probs > 0.60, 1, 0)

    # Optional: plot scree immediately if desired
    # plot_pca_variance(pca)

    return df, scaler, pca, model, split_idx

def plot_pca_variance(pca):
    exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)

    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, alpha=0.5, align='center', label='Individual Variance')
    plt.step(range(1, len(exp_var_cumul) + 1), exp_var_cumul, where='mid', label='Cumulative Variance', color='red')
    plt.axhline(y=0.80, color='g', linestyle='--', label='80% Threshold Requirement')

    plt.ylabel('Explained Variance Ratio')
    plt.xlabel('Principal Component Index')
    plt.title('PCA Explained Variance (Scree Plot)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

