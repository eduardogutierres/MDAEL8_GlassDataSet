# Initial imports
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.svm import SVC

# Carregando o conjunto de dados "glass identification"
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

# Criação do classificador SVM
svm = SVC(kernel='poly', C=1)  # polinomial com C=1
svm.fit(X_train_scaled, y_train)

# Realizando a previsão usando o conjunto de teste
y_hat_test = svm.predict(X_test_scaled)

# Calculando a acurácia, F1 Score e Precisão
accuracy = accuracy_score(y_test, y_hat_test) * 100
f1 = f1_score(y_test, y_hat_test, average='macro')
precision = accuracy  # A precisão é igual à acurácia no caso de classificação binária

print("Acurácia SVM: {:.2f}%".format(accuracy))
print("F1 Score SVM: {:.2f}".format(f1))
print("Precisão SVM: {:.2f}%".format(precision))

# Calculando e plotando a matriz de confusão
cm = confusion_matrix(y_test, y_hat_test)
class_names = df['Type'].unique()

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
plot_confusion_matrix(cm, classes=class_names, title='Matriz de Confusão - Glass Identification')
plt.show()
