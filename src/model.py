import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import itertools


df = pd.read_csv('90day.csv')


df_modified = df.drop(df.columns[[0, 1, 2, 3, -1, -5, -8, -7]], axis=1)
df_modified = df_modified.apply(pd.to_numeric, errors='coerce')
df_modified.fillna(df_modified.mean(), inplace=True)


original_cols = df.columns.tolist()
dropped_cols = [original_cols[i] for i in [0, 1, 2, 3, 4, 5, -1, -5, -8, -7]]
remaining_cols = [col for col in original_cols if col not in dropped_cols]
y_index = remaining_cols.index(original_cols[-6])

X = df_modified.drop(df_modified.columns[[y_index]], axis=1)
y = df_modified.iloc[:, y_index]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


best_rf_params = {
    'n_estimators': 200,
    'max_depth': 20,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'bootstrap': True,
    'class_weight': 'balanced'
}

best_lr_params = {
    'C': 1.0,
    'solver': 'lbfgs',
    'penalty': 'l2',
    'class_weight': 'balanced',
    'max_iter': 2000
}

best_mlp_params = {
    'hidden_layer_sizes': (100, 50, 50),
    'activation': 'relu',
    'solver': 'adam',
    'alpha': 0.001,
    'learning_rate': 'adaptive',
    'max_iter': 2000
}


rf = RandomForestClassifier(**best_rf_params)
rf.fit(X_train_scaled, y_train)

lr = LogisticRegression(**best_lr_params)
lr.fit(X_train_scaled, y_train)

mlp = MLPClassifier(**best_mlp_params)
mlp.fit(X_train_scaled, y_train)


y_test_prob_rf = rf.predict_proba(X_test_scaled)[:, 1]
y_test_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
y_test_prob_mlp = mlp.predict_proba(X_test_scaled)[:, 1]


weights = np.ones(3)
eta = 0.001


history = {
    'weights': [],
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1': [],
    'auc': []
}


predictions = np.array([y_test_prob_rf, y_test_prob_lr, y_test_prob_mlp])
final_prediction = np.zeros_like(y_test_prob_rf)

for t in range(1000):
    weighted_preds = np.dot(weights, predictions) / np.sum(weights)
    
    if np.isnan(weighted_preds).any():
        print("NaN detected in weighted predictions, breaking out of loop.")
        break
    
    final_prediction = np.where(weighted_preds > 0.5, 1, 0)

    errors = np.abs(final_prediction - y_test)
    for i in range(3):
        model_error = np.dot(errors, predictions[i])
        weights[i] *= np.exp(-eta * model_error)
    
    if np.isnan(weights).any():
        print("NaN detected in weights, breaking out of loop.")
        break
    
    weights /= np.sum(weights)

    accuracy = accuracy_score(y_test, final_prediction)
    precision = precision_score(y_test, final_prediction, average='macro')
    recall = recall_score(y_test, final_prediction, average='macro')
    f1 = f1_score(y_test, final_prediction, average='macro')
    auc = roc_auc_score(y_test, weighted_preds)
    
    history['weights'].append(weights.copy())
    history['accuracy'].append(accuracy)
    history['precision'].append(precision)
    history['recall'].append(recall)
    history['f1'].append(f1)
    history['auc'].append(auc)

print("Combined Model Evaluation")
print("Accuracy:", history['accuracy'][-1])
print("Precision:", history['precision'][-1])
print("Recall:", history['recall'][-1])
print("F1 Score:", history['f1'][-1])
print("AUC:", history['auc'][-1])


rf_feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_})
rf_feature_importances['Importance'] /= rf_feature_importances['Importance'].sum()
rf_feature_importances = rf_feature_importances.sort_values(by='Importance', ascending=False)

lr_hazard_ratios = pd.DataFrame({'Feature': X.columns, 'Hazard Ratio': np.exp(lr.coef_[0])})
lr_hazard_ratios = lr_hazard_ratios.sort_values(by='Hazard Ratio', ascending=False)

mlp_feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': np.abs(mlp.coefs_[0]).sum(axis=1)})
mlp_feature_importances['Importance'] /= mlp_feature_importances['Importance'].sum()
mlp_feature_importances = mlp_feature_importances.sort_values(by='Importance', ascending=False)

combined_importance = (rf_feature_importances['Importance'] * weights[0] +
                       lr_hazard_ratios['Hazard Ratio'] * weights[1] +
                       mlp_feature_importances['Importance'] * weights[2])
combined_feature_importances = pd.DataFrame({'Feature': rf_feature_importances['Feature'], 'Importance': combined_importance})
combined_feature_importances['Importance'] /= combined_feature_importances['Importance'].sum()  # Normalize to sum up to 1
combined_feature_importances = combined_feature_importances.sort_values(by='Importance', ascending=False)

print("\nCombined Feature Importances - MWU")
print(combined_feature_importances)


def plot_feature_importance(importances, title):
    fig, ax = plt.subplots(figsize=(10, 8))
    sorted_importances = importances.sort_values(by='Importance', ascending=False)
    ax.bar(sorted_importances['Feature'], sorted_importances['Importance'], color='red')
    ax.set_title(title)
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    plt.xticks(rotation=90)
    plt.show()

def plot_hazard_ratios(hazard_ratios, title):
    fig, ax = plt.subplots(figsize=(10, 8))
    sorted_ratios = hazard_ratios.sort_values(by='Hazard Ratio', ascending=False)
    ax.barh(sorted_ratios['Feature'], sorted_ratios['Hazard Ratio'], color='skyblue')
    ax.set_title(title)
    ax.set_xlabel('Hazard Ratio')
    ax.axvline(x=1, color='gray', linestyle='--')
    ax.set_xscale('log')
    plt.gca().invert_yaxis()
    plt.show()

plot_feature_importance(combined_feature_importances, 'Combined Feature Importances - MWU')
plot_hazard_ratios(lr_hazard_ratios, 'Combined Hazard Ratios - MWU')
