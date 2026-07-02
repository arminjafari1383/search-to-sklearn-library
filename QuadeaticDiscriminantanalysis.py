from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import numpy as np

X = np.array([[-1,-1],[-2,-1],[-3,-2],[1,1],[2,1],[3,2]])
y = np.array([1,1,1,2,2,2])
clf = QuadraticDiscriminantAnalysis()
y1 = clf.fit(X,y)
# print(y1)
print(clf.predict([[-0.8,-1]]))
