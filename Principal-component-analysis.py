import numpy as np
from sklearn.decomposition import PCA
X = np.array([[-1,-1],[-2,-1],[-3,-2],[1,1],[2,1],[3,2]])
pca = PCA(n_components=2)
y1 = pca.fit(X)
print(y1)
y2 = pca.explained_variance_ratio_
print(y2)
y3 = pca.singular_values_
print(y3)
pca = PCA(n_components=2,svd_solver='full')
y4 = pca.fit(X)
print(y4)
y5 = pca.explained_variance_ratio_
print(y5)
y6 = pca.singular_values_
print(y6)
pca = PCA(n_components=1,svd_solver='arpack')
y7 = pca.fit(X)
print(y7)
y8 = pca.explained_variance_ratio_
print(y8)
y9 = pca.singular_values_
print(y9)

