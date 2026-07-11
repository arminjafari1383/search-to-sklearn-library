# This estim is stateless and does not need to be fitted.
# However , we recommend to call fit_transform instead of transform,
# as parameter validation is only performed in fit.

from sklearn.feature_extraction import FeatureHasher
h = FeatureHasher(n_features=10)
D = [{'dog':1,'cat':2,'elephant':4},{'dog':2,'run':5}]
f = h.transform(D)
y1 = f.toarray()
# print(y1)


# with input_type="string", the input must be an iterable over iterables of strings

h = FeatureHasher(n_features=8,input_type="string")
raw_X = [["dog","cat","snake"],["snake","dog"],["cat","bird"]]
f = h.transform(raw_X)
y2 = f.toarray()
print(y2)

