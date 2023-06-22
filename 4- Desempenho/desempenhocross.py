import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from tabulate import tabulate

# Carregar a base de dados
input_file = '0-Datasets/glassClear.data'
names = ['Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Tipo']
features = ['Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
target = 'Tipo'

data = pd.read_csv(input_file, names=names)

# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.25, random_state=0)

# Aplicar o SMOTE para balancear a base de dados
smote = SMOTE()
X_train, y_train = smote.fit_resample(X_train, y_train)

# Normalizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# KNN
knn_classifier = KNeighborsClassifier()
knn_grid = {'n_neighbors': [3, 5, 7],
            'metric': ['euclidean', 'manhattan']}
knn_grid_search = GridSearchCV(knn_classifier, knn_grid, scoring='accuracy', cv=5)
knn_grid_search.fit(X_train, y_train)
knn_classifier = knn_grid_search.best_estimator_

# SVM
svm_classifier = SVC(kernel='poly', C=1)

# Redes Neurais
rna_classifier = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)

# Árvore de Decisão
dt_classifier = DecisionTreeClassifier(max_depth=5)  # Definir uma profundidade máxima para a árvore

classifiers = {
    'KNN': knn_classifier,
    'SVM': svm_classifier,
    'RNA': rna_classifier,
    'Árvore de Decisão': dt_classifier
}

table_data = []
metrics = ['accuracy', 'f1_weighted', 'precision_macro']
for classifier_name, classifier in classifiers.items():
    cv_results = cross_validate(classifier, X_train, y_train, scoring=metrics, cv=10)
    accuracy_scores = cv_results['test_accuracy']
    f1_scores = cv_results['test_f1_weighted']
    precision_scores = cv_results['test_precision_macro']
    accuracy_mean = np.mean(accuracy_scores)
    f1_mean = np.mean(f1_scores)
    precision_mean = np.mean(precision_scores)
    classifier_data = [classifier_name, round(accuracy_mean, 2), round(f1_mean, 2), round(precision_mean, 2)]
    table_data.append(classifier_data)

# Printar tabela com os resultados
table_headers = ["Classificador", "Acurácia", "F1-Score", "Precision"]
print(tabulate(table_data, headers=table_headers, tablefmt="grid"))

# Printar matriz de confusão
for classifier_name, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    confusion_matrix_data = confusion_matrix(y_test, predictions)
    print(f"\nMatriz de Confusão - {classifier_name}:")
    print(tabulate(confusion_matrix_data, headers=np.unique(y_test), tablefmt="grid"))

# Gráfico comparativo de desempenho dos classificadores em 2D
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(classifiers))
width = 0.2

rects1 = ax.bar(x, [row[1] for row in table_data], width, label='Acurácia')
rects2 = ax.bar(x + width, [row[2] for row in table_data], width, label='F1-Score')
rects3 = ax.bar(x + 2 * width, [row[3] for row in table_data], width, label='Precision')

ax.set_title('Comparativo de Desempenho dos Classificadores')
ax.set_xlabel('Classificadores')
ax.set_ylabel('Desempenho')
ax.set_xticks(x + 1.5 * width)
ax.set_xticklabels([row[0] for row in table_data])
ax.legend()

# Adicionar legendas nas barras
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.tight_layout()
plt.show()

# Gráfico comparativo de desempenho dos classificadores em 2D (horizontal)
fig, ax = plt.subplots(figsize=(10, 6))

y = np.arange(len(classifiers))
height = 0.2

rects1 = ax.barh(y, [row[1] for row in table_data], height, label='Acurácia')
rects2 = ax.barh(y + height, [row[2] for row in table_data], height, label='F1-Score')
rects3 = ax.barh(y + 2 * height, [row[3] for row in table_data], height, label='Precision')

ax.set_title('Comparativo de Desempenho dos Classificadores')
ax.set_xlabel('Desempenho')
ax.set_ylabel('Classificadores')
ax.set_yticks(y + 1.5 * height)
ax.set_yticklabels([row[0] for row in table_data])
ax.legend()

# Adicionar legendas nas barras
def autolabel(rects):
    for rect in rects:
        width = rect.get_width()
        ax.annotate('{}'.format(width),
                    xy=(width, rect.get_y() + rect.get_height() / 2),
                    xytext=(3, 0),  # 3 points horizontal offset
                    textcoords="offset points",
                    ha='left', va='center')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.tight_layout()
plt.show()
