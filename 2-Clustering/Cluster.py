import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load dataset Glass
input_file = 'C:/Users/eduar/Downloads/DataMiningSamples-master/0-Datasets/glassClear.data'
col_names = ['ID', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']
glass_df = pd.read_csv(input_file, header=None, names=col_names, index_col='ID')
glass_df.dropna(inplace=True)

# Preparar os dados
X = glass_df.iloc[:, :-1].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determinar o número de clusters
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)

# Plotar o gráfico do método do cotovelo com legendas claras
plt.plot(range(1, 11), sse)
plt.xlabel('Número de clusters', fontsize=12)
plt.ylabel('SSE (Soma dos Erros Quadrados)', fontsize=12)
plt.title('Método do Cotovelo para Análise de Grupos', fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True)
plt.show()

# Executar o algoritmo de clusterização
k = int(input("Digite o número de clusters com base no gráfico do método do cotovelo: "))
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_scaled)

# Adicionar a coluna de rótulos de cluster ao DataFrame original
glass_df['cluster_label'] = kmeans.labels_

# Exibir os primeiros registros com os rótulos de cluster
print(glass_df.head())
