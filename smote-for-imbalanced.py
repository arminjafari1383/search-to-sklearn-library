import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTENC
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling  import ADASYN
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from collections import Counter

####Load and Explore the Dataset
data = pd.read_csv("./data/creditcard.csv")
data = data.dropna(subset=["Class"])

x = data.drop("Class", axis = 1)
y = data["Class"]

x = x.fillna(x.median)

# plt.bar(y.value_counts().index, y.value_counts().values,color=['skyblue','salmon'])
# plt.xticks([0,1],['Non-Fraud','Fraud'])
# plt.ylabel("Count")
# plt.title("Original Class Distribution")
# plt.show()


#### Apply SMOTE to Balance Classes
X_train,X_test,y_train,y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# plt.bar(y_train_sm.value_counts().index,y_train_sm.value_counts().values,color=['skyblue','salmon'])
# plt.xticks([0,1],['Non-Fraud','Fraud'])
# plt.ylabel("Count")
# plt.title("Class Distribution AFTER SMOTE")
# plt.show()


#### Train Logistic Regression After SMOTE
model = LogisticRegression(random_state=42,max_iter=1000)
model.fit(X_train_sm, y_train_sm)
y_pred_sm = model.predict(X_test)
# print("Accuracy AFTER SMOTE:",round(accuracy_score(y_test,y_pred_sm)*100,2),"%\n")
# print("Classification Report AFTER SMOTE:\n",classification_report(y_test,y_pred_sm))

# sns.heatmap(confusion_matrix(y_test,y_pred_sm),annot=True,fmt='d',cmap='Blues')
# plt.title("Confusion Matrix AFTER SMOTE")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.show()


#### ADASYN (ADSYN (adaptive synthetic sampling) stands for adaptive syntgetic sampling)
adasyn = ADASYN(sampling_strategy = 'minority', random_state = 42)
X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train,y_train)
# print("Class distribution AFTER ADASYN")
# print(y_train_adasyn.value_counts())


#### Borderline SMOTE is modified version of SMOTE that focues only on minority samples that lie near the boundary between classes
blsmote = BorderlineSMOTE(sampling_strategy='minority',kind = 'borderline-1',random_state=42)
X_train_blsmote,y_train_blsmote = blsmote.fit_resample(X_train, y_train)

# print("Class distribution AFTER Borderline-SMOTE:")
# print(y_train_blsmote.value_counts())


#### SMOTE-ENN (Edited Nearest Neighbors)
#### smote-enn combines two techniques i.e for oversampling and Edited Nearest Neighbors for cleaning
smote_enn = SMOTEENN(random_state=42)
X_train_smoteenn, y_train_smoteenn = smote_enn.fit_resample(X_train,y_train)

# print("Class distribution AFTER SMOTEENN:")
# print(y_train_smoteenn.value_counts())


#### SMOTE-TOMEK (Hybrid Method)
smote_tomek = SMOTETomek(random_state=42)
X_train_smotetomek,y_train_smotetomek = smote_tomek.fit_resample(X_train,y_train)
# print("Class distribution AFTER SMOTETomek:")
# print(y_train_smotetomek.value_counts())


##SMOTE-NC (Nominal Continuous)

X, y = make_classification(
    n_classes=2,
    class_sep=2,
    weights=[0.1, 0.9],
    n_informative=3,
    n_redundant=1,
    n_features=5,
    n_clusters_per_class=1,
    n_samples=100,
    random_state=42
)

categorical_features = [0, 3]

print("Before SMOTE-NC:", Counter(y))

smote_nc = SMOTENC(categorical_features=categorical_features, random_state=42)
X_res, y_res = smote_nc.fit_resample(X, y)

print("After SMOTE-NC:", Counter(y_res))


