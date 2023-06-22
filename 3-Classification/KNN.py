import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score
import matplotlib.pyplot as plt
import itertools

# Função para calcular a distância de Minkowski entre dois pontos
def minkowski_distance(a, b, p=2):
    dim = len(a)
    distance = 0

    for d in range(dim):
        distance += abs(a[d] - b[d])**p

    distance = distance**(1/p)
    return distance

# Função para realizar a previsão usando o algoritmo k-NN
def knn_predict(X_train, X_test, y_train, k, p=2):
    y_hat_test = []

    for test_point in X_test:
        distances = []

        for train_point in X_train:
            distance = minkowski_distance(test_point, train_point, p)
            distances.append(distance)

        sorted_indices = np.argsort(distances)[:k]
        k_nearest_labels = y_train[sorted_indices]
        prediction = np.argmax(np.bincount(k_nearest_labels))
        y_hat_test.append(prediction)

    return y_hat_test

# Carregando os dados do arquivo "glass.data"
input_file = '0-Datasets/glassClear.data'
names = ['Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']
df = pd.read_csv(input_file, names=names)

# Separando os atributos e o alvo
X = df.drop('Type', axis=1).values
y = df['Type'].values

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Normalizando os dados usando o Min-Max Scaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Realizando a previsão usando o algoritmo k-NN
y_hat_test = knn_predict(X_train_scaled, X_test_scaled, y_train, k=5, p=2)

# Calculando a acurácia, F1 Score e Precisão
accuracy = accuracy_score(y_test, y_hat_test) * 100
f1 = f1_score(y_test, y_hat_test, average='macro') * 100
precision = precision_score(y_test, y_hat_test, average='macro') * 100

print("Acurácia: {:.2f}%".format(accuracy))
print("F1 Score: {:.2f}%".format(f1))
print("Precisão: {:.2f}%".format(precision))

# Calculando e plotando a matriz de confusão
cm = confusion_matrix(y_test, y_hat_test)

# Definindo os nomes das classes
class_names = df['Type'].unique()

# Função para plotar a matriz de confusão
def plot_confusion_matrix(cm, classes, normalize=False, title='Matriz de Confusão', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Rótulo Verdadeiro')
    plt.xlabel('Rótulo Previsto')

# Plotando a matriz de confusão
plt.figure()
plot_confusion_matrix(cm, classes=class_names, title='Matriz de Confusão - Glass Dataset')
plt.show()