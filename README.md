# Drug Classification Model and Visualizations

This project involves using a **Random Forest Classifier** to predict drug types based on patient data and generating insightful visualizations to explore relationships between features. The dataset, `drug200.csv`, contains information on patient demographics and medical attributes.

---

## Dataset Overview

The dataset, `drug200.csv`, consists of the following columns:

### Features:
- **Age**: The patient's age.
- **Sex**: The gender of the patient.
- **BP**: Blood pressure level (low, normal, high).
- **Cholesterol**: Cholesterol level (normal, high).
- **Na_to_K**: Sodium-to-potassium ratio in the patient's blood.

### Target:
- **Drug**: The type of drug prescribed (categorical variable).

Categorical features (`Sex`, `BP`, `Cholesterol`, and `Drug`) are encoded using `LabelEncoder` for use in machine learning models.

---

## Random Forest Classifier

A **Random Forest Classifier** is trained to predict the type of drug prescribed based on the provided features:
- Data is split into 80% training and 20% testing sets.
- The classifier is initialized with 100 estimators and a fixed random state for reproducibility.
- Feature importance is calculated to determine which features contribute most to the predictions.

Performance metrics, including **accuracy**, **precision**, **recall**, and **F1 score**, are used to evaluate the model on the test set.

---

## Visualizations

This project includes several visualizations to enhance understanding of the dataset and model behavior:

### 1. **Feature Importance Barplot**
- Displays the relative importance of each feature in predicting the drug type.
- Helps identify the most influential features in the dataset.

### 2. **Na_to_K Distribution Boxplot**
- Shows the distribution of the sodium-to-potassium ratio (`Na_to_K`) for each drug type.
- Provides insights into how this feature varies across different drug categories.

### 3. **Correlation Heatmap**
- Visualizes the correlations between all features in the dataset.
- High or low correlations indicate relationships between features, helping refine model inputs.

### 4. **Categorical Feature Count Plots**
- **Sex**: Counts of male and female patients.
- **BP**: Distribution of blood pressure levels (low, normal, high).
- **Cholesterol**: Distribution of cholesterol levels (normal, high).
- These plots reveal the distribution of categorical variables across the dataset.

### 5. **Confusion Matrix**
- Displays the performance of the Random Forest Classifier in classifying drug types.
- Highlights misclassification patterns to evaluate the model's accuracy.

### 6. **Pair Plot**
- Visualizes pairwise relationships between features, with data points colored by drug type.
- Useful for identifying trends and clusters in the data.

---

## Output Files

- `feature_importance_synthetic_real.jpg`: Feature importance barplot.
- `Na_to_K_Distribution_Synthetic_Real.jpg`: Sodium-to-potassium distribution by drug type.
- `correlation_heatmap.jpg`: Heatmap showing correlations between features.
- `count_plot_sex.jpg`, `count_plot_bp.jpg`, `count_plot_cholesterol.jpg`: Categorical feature count plots.
- `confusion_matrix.jpg`: Confusion matrix for classifier performance.
- `pairplot_by_drug_type.jpg`: Pair plot of feature relationships by drug type.

---


