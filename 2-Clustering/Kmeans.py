import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

#Carregar o conjunto de dados "glass dataset"
glass = fetch_openml(name='glass', version=4)

#Obter os dados e os rótulos do conjunto de dados
data = glass.data
labels = glass.target

#Pré-processamento dos dados
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

#Redução de dimensionalidade com PCA
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

#Criar o modelo K-Means com 3 clusters (número de classes no "glass dataset")
kmeans = KMeans(n_clusters=3, random_state=42)

#Treinar o modelo
kmeans.fit(data_pca)

#Obter os rótulos dos clusters
cluster_labels = kmeans.labels_

#Calcular o Silhouette Score
silhouette_avg = silhouette_score(data_pca, cluster_labels)

#Definir cores distintas para cada cluster
colors = ['red', 'green', 'blue']

#Criar mapa de cores personalizado
cmap = ListedColormap(colors)

#Exibir o gráfico de dispersão com legenda de barra para os clusters
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=cluster_labels, cmap=cmap)
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('K-Means Clustering - Glass Dataset\nSilhouette Score: {:.2f}'.format(silhouette_avg))
plt.colorbar(label='Cluster', ticks=[0, 1, 2], boundaries=np.arange(-0.5, 3, 1))
plt.xticks([])
plt.yticks([])
plt.show()