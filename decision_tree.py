# %%
# Loading library
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Load data as a dataframe
def sklearn_to_df(sklearn_dataset):
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df['target'] = pd.Series(sklearn_dataset.target)
    return df

wine = sklearn_to_df(datasets.load_wine())
wine.rename(columns={'target': 'class'}, inplace=True)

wine.describe().T

# %%
# Show the data
wine.head(10)
print(wine)

# %%
# Visualisasi data dengan Grafik pada data wine
sns.pairplot(wine, hue='class', palette='Set1')

# %%
# Split training and testing data
from sklearn.model_selection import train_test_split
x = wine.drop('class', axis=1)
y = wine['class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)

# %%
print(len(x_train))

# %%
# Train model (Decision Tree)
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

# %%
# Predict test data
y_pred = model.predict(x_test)

# %%
# Classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# %%
# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7, 7))

sns.set(font_scale=1.4)
sns.heatmap(cm, ax=ax, annot=True, annot_kws={"size": 16})

plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()

# %%
# Visualize tree
from sklearn import tree
features = x_train.columns.tolist()
fig, ax = plt.subplots(figsize=(25, 20))
tree.plot_tree(model, feature_names=features, class_names=True, filled=True)
plt.show()

# %%
# Print feature names (optional check)
print(x_train.columns.tolist())

# %%
# Create a sample wine data point for prediction (using real feature names)
wine_test_data = {
    'alcohol': 13.0,
    'malic_acid': 2.0,
    'ash': 2.5,
    'alcalinity_of_ash': 15.0,
    'magnesium': 100.0,
    'total_phenols': 2.5,
    'flavanoids': 2.0,
    'nonflavanoid_phenols': 0.3,
    'proanthocyanins': 1.5,
    'color_intensity': 5.0,
    'hue': 1.0,
    'od280/od315_of_diluted_wines': 3.0,
    'proline': 1000.0
}

# Ensure correct column order
feature_order = x_train.columns.tolist()
prediction_input_df = pd.DataFrame([wine_test_data])
prediction = model.predict(prediction_input_df[feature_order])
print("Prediksi kelas:", prediction)

# %%
