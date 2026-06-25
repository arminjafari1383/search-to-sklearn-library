import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from imblearn.over_sampling import SMOTE
import seaborn as sns


#Load and Explore the Dataset
data = pd.read_csv("./data/creditcard.csv")
data = data.dropna(subset=["Class"])

x = data.drop("Class", axis = 1)
y = data["Class"]

x = x.fillna(x.median)

plt.bar(y.value_counts().index, y.value_counts().values,color=['skyblue','salmon'])
plt.xticks([0,1],['Non-Fraud','Fraud'])
plt.ylabel("Count")
plt.title("Original Class Distribution")
plt.show()

