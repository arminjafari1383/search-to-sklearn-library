from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_array
import numpy as np

np.random.seed(0)
X_dense = np.random.rand(100,100)
X_dense[:,2 * np.arange(50)] = 0
X = csr_array(X_dense)
svd = TruncatedSVD(n_components=5,n_iter=7,random_state=42)
y1 = svd.fit(X)
print(y1)

print(svd.explained_variance_ratio_)

print(svd.explained_variance_ratio_.sum())

print(svd.singular_values_)

