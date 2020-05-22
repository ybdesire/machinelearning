from sklearn import tree
from sklearn.externals.six import StringIO
import pydotplus

X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)# train the model

result = clf.predict([[0.2, 0.1]])#predict ([0])
result_p = clf.predict_proba([[0.8, 0.1]])#predict probability ([[ 0.  1.]])

print(result)
print(result_p)


dot_data = StringIO() 
tree.export_graphviz(clf, out_file=dot_data) 
print(dot_data.getvalue())
graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
graph.write_pdf("dtree.pdf") 
