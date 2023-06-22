import pandas as pd
import matplotlib.pyplot as plt

# Carregar o arquivo de dados
input_file = '0-Datasets/glassClear.data'
names = ['Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']
df = pd.read_csv(input_file, names=names)

# Mapear os números das classes para seus rótulos correspondentes
class_labels = {
    1: 'Janela de construção - Processo flutuante',
    2: 'Janela de construção - Sem Processo flutuante',
    3: 'Janela de veículos - Processo flutuante',
    4: 'Janela de veículos - Sem Processo flutuante',
    5: 'Recipientes de vidro',
    6: 'Louças de vidro',
    7: 'Lanterna de cabeça'
}

# Substituir os números das classes pelos rótulos correspondentes
df['Type'] = df['Type'].map(class_labels)

# Contagem de instâncias em cada classe
class_counts = df['Type'].value_counts()

# Plotar o gráfico de barras
plt.figure(figsize=(8, 6))  # Ajustar o tamanho da figura
plt.bar(class_counts.index, class_counts.values)
plt.xlabel('Classe')
plt.ylabel('Número de instâncias')
plt.title('Distribuição das Classes no Conjunto de Dados "Glass"')
plt.xticks(rotation=45)  # Rotacionar os rótulos
plt.tight_layout()  # Melhorar a distribuição do espaço
plt.show()
