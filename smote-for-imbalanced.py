import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from imblearn.over_sampling import SMOTE
import seaborn as sns
import imblearn.over_sampling import ADASYN



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
print("Accuracy AFTER SMOTE:",round(accuracy_score(y_test,y_pred_sm)*100,2),"%\n")
print("Classification Report AFTER SMOTE:\n",classification_report(y_test,y_pred_sm))

# sns.heatmap(confusion_matrix(y_test,y_pred_sm),annot=True,fmt='d',cmap='Blues')
# plt.title("Confusion Matrix AFTER SMOTE")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.show()


adasyn = ADASYN(sampling_strategy = 'minority', random_state = 42)
X_train_adasyn, y_train_adasyn = adasyn.fit_reasample(X_train,y_train)
print("Class distribution AFTER ADASYN")
print(y_train_adasyn.value_counts())



