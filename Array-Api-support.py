import os
os.environ["SCIPY_ARRAY_API"] = "1"
from sklearn.datasets import make_classification
from sklearn import config_context
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils._array_api import move_estimator_to
import torch
import numpy as np
# انتخاب خودکار دستگاه
device = "cuda" if torch.cuda.is_available() else "cpu"

# تولید داده
X_np, y_np = make_classification(random_state=0)

# تبدیل به Tensorهای PyTorch
X_torch = torch.asarray(X_np, device=device, dtype=torch.float32)
y_torch = torch.asarray(y_np, device=device, dtype=torch.int64)

# اجرای LDA با پشتیبانی Array API
with config_context(array_api_dispatch=True):
    lda = LinearDiscriminantAnalysis()
    X_trans = lda.fit_transform(X_torch, y_torch)

# print("Device:", device)
# print("Type of output:", type(X_trans))
# print("Shape:", X_trans.shape)


#Moving estimators between devices
#we provide move_estimator_to to transfer an estimator's
#array attributes to a dfferent namespace and device

lda_np = move_estimator_to(lda,np,device="cpu")
with config_context(array_api_dispatch=True):
    X_trans = lda_np.transform(X_np)
print(type(X_trans))
