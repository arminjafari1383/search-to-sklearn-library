from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, chi2


v = DictVectorizer(sparse=False)
D = [{'foo':1,'bar':2},{'foo':3,'baz':1}]
X = v.fit_transform(D)
# print(X)

x1 = v.inverse_transform(X) == [{'bar':2.0,'foo':1.0},
                                {'baz':1.0,'foo':3.0}]
# print(True)

x2 = v.transform({'foo':4,'unseen_feature':3})

# print(x2)

