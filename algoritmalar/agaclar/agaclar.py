from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.tree import export_text
from matplotlib import pyplot as plt

iris = load_iris()
X, y = iris.data, iris.target
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

tree.plot_tree(clf)
plt.show()

import graphviz 
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("algoritmalar/agaclar/cicekler.pdf")