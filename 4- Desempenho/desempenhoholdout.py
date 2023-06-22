import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score
from keras.models import Sequential
from keras.layers import Dense
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
import lightgbm as lgb
import seaborn as sns
from tabulate import tabulate
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPClassifier


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
knn_predictions = knn_classifier.predict(X_test)

# Avaliação do KNN
knn_accuracy = accuracy_score(y_test, knn_predictions)
knn_f1 = f1_score(y_test, knn_predictions, average='weighted')
knn_precision = precision_score(y_test, knn_predictions, average='weighted')
knn_confusion_matrix = confusion_matrix(y_test, knn_predictions)

# SVM
svm_classifier = SVC(kernel='poly', C=1)
svm_classifier.fit(X_train, y_train)
svm_predictions = svm_classifier.predict(X_test)

# Avaliação do SVM
svm_accuracy = accuracy_score(y_test, svm_predictions)
svm_f1 = f1_score(y_test, svm_predictions, average='macro')
svm_precision = precision_score(y_test, svm_predictions, average='macro')
svm_confusion_matrix = confusion_matrix(y_test, svm_predictions)

# Redes Neurais
rna_classifier = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42, solver='adam', activation='relu')
rna_classifier.fit(X_train, y_train)
rna_predictions = rna_classifier.predict(X_test)

# Avaliação da RNA
rna_accuracy = accuracy_score(y_test, rna_predictions)
rna_f1 = f1_score(y_test, rna_predictions, average='weighted')
rna_precision = precision_score(y_test, rna_predictions, average='weighted')
rna_confusion_matrix = confusion_matrix(y_test, rna_predictions)

# Árvore de Decisão
dt_classifier = DecisionTreeClassifier(max_depth=5)  # Definir uma profundidade máxima para a árvore
dt_classifier.fit(X_train, y_train)
dt_predictions = dt_classifier.predict(X_test)

# Avaliação da Árvore de Decisão
dt_accuracy = accuracy_score(y_test, dt_predictions)
dt_f1 = f1_score(y_test, dt_predictions, average='weighted')
dt_precision = precision_score(y_test, dt_predictions, average='weighted')
dt_confusion_matrix = confusion_matrix(y_test, dt_predictions)

# Comparativo de desempenho dos classificadores
classifiers = ['KNN', 'SVM', 'RNA', 'Árvore de Decisão']
accuracy_scores = [knn_accuracy, svm_accuracy, rna_accuracy, dt_accuracy]
f1_scores = [knn_f1, svm_f1, rna_f1, dt_f1]
precision_scores = [knn_precision, svm_precision, rna_precision, dt_precision]
confusion_matrices = [knn_confusion_matrix, svm_confusion_matrix, rna_confusion_matrix, dt_confusion_matrix]

table_data = []
for i in range(len(classifiers)):
    classifier_data = [classifiers[i], round(accuracy_scores[i], 2), round(f1_scores[i], 2), round(precision_scores[i], 2)]
    table_data.append(classifier_data)

# Printar tabela com os resultados
table_headers = ["Classificador", "Acurácia", "F1-Score", "Precision"]
print(tabulate(table_data, headers=table_headers, tablefmt="grid"))

# Printar matriz de confusão
for i in range(len(classifiers)):
    print(f"\nMatriz de Confusão - {classifiers[i]}:")
    print(tabulate(confusion_matrices[i], headers=range(1, 8), tablefmt="grid"))

# Gráfico comparativo de desempenho dos classificadores em 2D
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(classifiers))
width = 0.2

rects1 = ax.bar(x, accuracy_scores, width, label='Acurácia')
rects2 = ax.bar(x + width, f1_scores, width, label='F1-Score')
rects3 = ax.bar(x + 2 * width, precision_scores, width, label='Precision')

ax.set_title('Comparativo de Desempenho dos Classificadores')
ax.set_xlabel('Classificadores')
ax.set_ylabel('Desempenho')
ax.set_xticks(x + 1.5 * width)
ax.set_xticklabels(classifiers)
ax.legend()

# Anotar as métricas no gráfico
for rect1, rect2, rect3 in zip(rects1, rects2, rects3):
    height1 = rect1.get_height()
    height2 = rect2.get_height()
    height3 = rect3.get_height()
    ax.annotate(f'{round(height1, 2)}', xy=(rect1.get_x() + rect1.get_width() / 2, height1),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom')
    ax.annotate(f'{round(height2, 2)}', xy=(rect2.get_x() + rect2.get_width() / 2, height2),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom')
    ax.annotate(f'{round(height3, 2)}', xy=(rect3.get_x() + rect3.get_width() / 2, height3),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Gráfico comparativo de desempenho dos classificadores em 2D (horizontal)
fig, ax = plt.subplots(figsize=(10, 6))

y = np.arange(len(classifiers))
height = 0.2

rects1 = ax.barh(y, accuracy_scores, height, label='Acurácia')
rects2 = ax.barh(y + height, f1_scores, height, label='F1-Score')
rects3 = ax.barh(y + 2 * height, precision_scores, height, label='Precision')

ax.set_title('Comparativo de Desempenho dos Classificadores')
ax.set_xlabel('Desempenho')
ax.set_ylabel('Classificadores')
ax.set_yticks(y + 1.5 * height)
ax.set_yticklabels(classifiers)
ax.legend()

# Anotar as métricas no gráfico
for rect1, rect2, rect3 in zip(rects1, rects2, rects3):
    width1 = rect1.get_width()
    width2 = rect2.get_width()
    width3 = rect3.get_width()
    ax.annotate(f'{round(width1, 2)}', xy=(width1, rect1.get_y() + rect1.get_height() / 2),
                xytext=(3, 0), textcoords="offset points",
                ha='left', va='center')
    ax.annotate(f'{round(width2, 2)}', xy=(width2, rect2.get_y() + rect2.get_height() / 2),
                xytext=(3, 0), textcoords="offset points",
                ha='left', va='center')
    ax.annotate(f'{round(width3, 2)}', xy=(width3, rect3.get_y() + rect3.get_height() / 2),
                xytext=(3, 0), textcoords="offset points",
                ha='left', va='center')

plt.tight_layout()
plt.show()