import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
import graphviz
import pydotplus
from sklearn.metrics import precision_score, f1_score

# Carregando o conjunto de dados "glass identification"
input_file = '0-Datasets/glassClear.data'
names = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']
data = pd.read_csv(input_file, names=names)

features = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
target = 'Type'

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(max_leaf_nodes=5)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("Acurácia do modelo:", accuracy)

class_names = [str(class_name) for class_name in model.classes_]

dot_data = export_graphviz(model, out_file=None, feature_names=features, class_names=class_names, filled=True, rounded=True)

# Converter o gráfico em um objeto graphviz
graph = pydotplus.graph_from_dot_data(dot_data)

# Exibir o gráfico
graphviz.Source(graph.to_string())

# Salvar o gráfico em um arquivo PDF
graph.write_pdf("arvore_decisao_glass.pdf")

# Plotar o gráfico da árvore de decisão
plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=features, class_names=class_names, filled=True, rounded=True)
plt.show()

# Realizar a previsão usando o conjunto de teste
y_pred = model.predict(X_test)

# Calcular a precisão
precision = precision_score(y_test, y_pred, average='macro')

# Calcular o F1-Score
f1 = f1_score(y_test, y_pred, average='macro')

# Imprimir os resultados
print("Precisão: {:.2f}%".format(precision))
print("F1-Score: {:.2f}%".format(f1))
