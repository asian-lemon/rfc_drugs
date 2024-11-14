import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'drug200.csv'  # Update the path as necessary
df = pd.read_csv(file_path)

# Encode categorical features
label_encoder = LabelEncoder()
df['Drug'] = label_encoder.fit_transform(df['Drug'])
df['Sex'] = label_encoder.fit_transform(df['Sex'])
df['BP'] = label_encoder.fit_transform(df['BP'])
df['Cholesterol'] = label_encoder.fit_transform(df['Cholesterol'])

# Separate features and target variable
X = df.drop(columns=['Drug']).values
y = df['Drug'].values


# Split the real data and combined data into train and test sets
X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train on Real Data
rf.fit(X_train_real, y_train_real)
feature_importance = rf.feature_importances_
y_pred_real = rf.predict(X_test_real)


# Define a function to calculate and display performance metrics in percentile format
def evaluate_model(y_true, y_pred, dataset_name):
    accuracy = accuracy_score(y_true, y_pred) * 100  # Convert to percentage
    precision = precision_score(y_true, y_pred, average='weighted') * 100
    recall = recall_score(y_true, y_pred, average='weighted') * 100
    f1 = f1_score(y_true, y_pred, average='weighted') * 100

    print(f"Performance Metrics for {dataset_name}:")
    print(f" - Accuracy: {accuracy:.2f}%")
    print(f" - Precision: {precision:.2f}%")
    print(f" - Recall: {recall:.2f}%")
    print(f" - F1 Score: {f1:.2f}%\n")

# Evaluate and compare results
evaluate_model(y_test_real, y_pred_real, "Real Data Only")

# Visualize synthetic data vs. real data
plt.figure(figsize=(10, 6))
sns.boxplot(x="Drug", y="Na_to_K", data=df, palette="Set3")
plt.title("Na_to_K Distribution by Drug Type (Synthetic + Real)")
plt.savefig("Na_to_K_Distribution_Synthetic_Real.jpg")

# Plot updated Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance, y=df.drop(columns=['Drug']).columns, palette="viridis")
plt.title("Feature Importance for Predicting Drug Type (Synthetic + Real)")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.savefig("feature_importance_synthetic_real.jpg")

# Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Features")
plt.savefig("correlation_heatmap.jpg")
plt.show()

# Count Plot for Categorical Features (Sex, BP, Cholesterol)
plt.figure(figsize=(18, 5))

# Sex Count Plot
plt.subplot(1, 3, 1)
sns.countplot(x='Sex', data=df, palette="Set2")
plt.title("Count of Sex Categories")
plt.savefig("count_plot_sex.jpg")

# BP Count Plot
plt.subplot(1, 3, 2)
sns.countplot(x='BP', data=df, palette="Set2")
plt.title("Count of BP Categories")
plt.savefig("count_plot_bp.jpg")

# Cholesterol Count Plot
plt.subplot(1, 3, 3)
sns.countplot(x='Cholesterol', data=df, palette="Set2")
plt.title("Count of Cholesterol Categories")
plt.savefig("count_plot_cholesterol.jpg")

plt.tight_layout()
plt.savefig("categorical_count_plots.jpg")
plt.show()

# Confusion Matrix Plot
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Generate the confusion matrix
cm = confusion_matrix(y_test_real, y_pred_real)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix for Real Data Only")
plt.savefig("confusion_matrix.jpg")
plt.show()

# Pair Plot to Visualize Feature Relationships by Drug Type
sns.pairplot(df, hue="Drug", palette="Set1", plot_kws={'alpha': 0.5})
plt.suptitle("Pair Plot of Features by Drug Type", y=1.02)
plt.savefig("pairplot_by_drug_type.jpg")
plt.show()