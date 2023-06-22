import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# Carregar o conjunto de dados "glass"
glass = load_iris()
X = glass.data

# Lista para armazenar a variância explicada
variance = []

# Executar o algoritmo K-means para diferentes valores de k
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    variance.append(kmeans.inertia_)

# Plotar a variância explicada em função do número de clusters
plt.plot(range(1, 11), variance, marker='o')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Variância Explicada')
plt.title('Método do Cotovelo - Glass Dataset')

# Adicionar marcador no ponto de cotovelo
plt.plot(3, variance[2], marker='o', markersize=8, label='Ponto de Cotovelo', color='red')
plt.annotate('Ponto de Cotovelo', xy=(3, variance[2]), xytext=(4, variance[2]+100), arrowprops=dict(arrowstyle='->', color='black'))

plt.legend()
plt.show()
