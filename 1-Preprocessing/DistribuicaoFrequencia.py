import pandas as pd
import matplotlib.pyplot as plt

# Definindo os limites inferiores das classes
def define_class_limits(min_value, max_value, n_classes):
    amplitude = (max_value - min_value) / n_classes
    class_limits = [min_value]
    for i in range(n_classes-1):
        class_limits.append(class_limits[i] + amplitude)
    class_limits.append(max_value)
    return class_limits

# Definindo os pontos médios das classes
def define_class_midpoints(class_limits):
    class_midpoints = []
    for i in range(len(class_limits)-1):
        midpoint = (class_limits[i] + class_limits[i+1])/2
        class_midpoints.append(midpoint)
    return class_midpoints

# Calculando as frequências absolutas, relativas e acumuladas
def calculate_frequencies(data, class_limits):
    frequencies = pd.cut(data, class_limits, include_lowest=True, right=False).value_counts(sort=False)
    abs_freq = frequencies.values.tolist()
    rel_freq = frequencies.apply(lambda x: x/len(data)).values.tolist()
    cum_freq = frequencies.cumsum().values.tolist()
    return abs_freq, rel_freq, cum_freq

# Lendo o dataset
data = pd.read_csv('C:/Users/eduar/Downloads/DataMiningSamples-master/0-Datasets/glassClear.data', 
                   names=['id', 'ri', 'na', 'mg', 'al', 'si', 'k', 'ca', 'ba', 'fe', 'glass_type'], skiprows=1)

# Definindo o número de classes desejado
n_classes = 6

# Definindo os limites inferiores das classes
class_limits = define_class_limits(data['ri'].min(), data['ri'].max(), n_classes)

# Definindo os pontos médios das classes
class_midpoints = define_class_midpoints(class_limits)

# Calculando as frequências absolutas, relativas e acumuladas
abs_freq, rel_freq, cum_freq = calculate_frequencies(data['ri'], class_limits)

# Criando o DataFrame com as informações da distribuição de frequências
df_freq = pd.DataFrame({
    'Limite Inferior': class_limits[:-1],
    'Limite Superior': class_limits[1:],
    'Ponto Médio': class_midpoints,
    'Frequência Absoluta': abs_freq,
    'Frequência Relativa': rel_freq,
    'Frequência Acumulada': cum_freq
})

print(df_freq)

plt.hist(data['ri'], bins=class_limits)
plt.title('Distribuição de Frequência')
plt.xlabel('Classe')
plt.ylabel('Frequência')
plt.show()
