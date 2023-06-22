import pandas as pd

# Faz a leitura do arquivo
input_file = '0-Datasets/glassClear.data'
names = ['Id number','RI','Na','Mg','Al','Si','K','Ca','Ba','Fe','Type of glass']
df = pd.read_csv(input_file,    # Nome do arquivo com dados
                 names = names) # Nome das colunas

# Exibe um resumo estatístico dos dados
print("Resumo estatístico dos dados:\n", df.describe())

# Calcula a mediana de cada atributo
print("\nMediana de cada atributo:\n", df.median())

# Calcula o desvio padrão de cada atributo
print("\nDesvio padrão de cada atributo:\n", df.std())
