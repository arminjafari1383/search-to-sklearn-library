from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import seaborn as sns
import seaborn as sns
import matplotlib.pyplot as plt

iris = load_iris(as_frame=True)
print(iris.keys())

#Rename classes usingthe iris target names
iris.frame["target"] = iris.target_names[iris.target]
_ = sns.pairplot(iris.frame,hue = "target")

# unused but required import for doing 3d projections
# with matplotlib < 3.2

fig  = plt.figure(1,figsize=(8,6))
ax = fig.add_subplot(111,projection="3d",elev=-150,azim=110)
X_reduced = PCA(n_components=3).fit_transform(iris.data)
scatter = ax.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    X_reduced[:, 2],
    c=iris.target,
    s=40,
)

ax.set(
    title="First three principal components",
    xlabel="1st Principal Component",
    ylabel="2nd Principal Component",
    zlabel="3rd Principal Component",
)
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])

#Add a legand
legend1 = ax.legend(
    scatter.legend_elements()[0],
    iris.target_names.tolist(),
    loc="upper right",
    title="Classes",
)

ax.add_artist(legend1)

plt.show()


