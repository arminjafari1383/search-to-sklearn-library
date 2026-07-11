from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, chi2

v = DictVectorizer()
D = [{'foo':1,'bar':2},{'foo':3,'baz':1}]
X = v.fit_transform(D)
support = SelectKBest(chi2,k=2).fit(X,[0,1])
y1 = v.get_feature_names_out()
print(y1)

y2 = v.restrict(support.get_support())
print(y2)

y3 = v.get_feature_names_out()
print(y3)

