import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from tabulate import tabulate

input_file = '0-Datasets/glassClear.data'
names = ['Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Tipo']
features = ['Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']

target = 'Tipo'
df = pd.read_csv(input_file, names=names)

X = df[features].values
y = df[target].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='micro')

metrics = [['Acurácia', accuracy], ['F1-Score', f1]]
print(tabulate(metrics, headers=['Métrica', 'Valor']))

precision = model.score(X_test, y_test)
print("Precisão: {:.2f}%".format(precision * 100))

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_title('Rede Neural')
ax.set_xlabel('Camada')
ax.set_ylabel('Neurônio')
ax.grid(True)

neuron_coords = []
input_coords = []
output_coords = []

for i, layer_sizes in enumerate(model.hidden_layer_sizes):
    layer_label = f'Camada Oculta {i+1}'
    neurons = layer_sizes
    x = i + 0.5

    for j in range(neurons):
        y = (neurons - 1) / 2 - j
        neuron_coords.append((x, y))

        if i == 0:
            input_coords.append((x-0.5, y))
        elif i == len(model.hidden_layer_sizes) - 1:
            output_coords.append((x+0.5, y))
            
        circle = plt.Circle((x, y), radius=0.1, facecolor='white', edgecolor='black')
        ax.add_patch(circle)
        ax.annotate(f'Neurônio {j+1}\n{layer_label}', (x, y), ha='center', va='center')

output_label = 'Camada de Saída'
neurons = model.n_outputs_
x = len(model.hidden_layer_sizes) + 1.5

for j in range(neurons):
    y = (neurons - 1) / 2 - j
    neuron_coords.append((x, y))
    output_coords.append((x, y))

    circle = plt.Circle((x, y), radius=0.1, facecolor='white', edgecolor='black')
    ax.add_patch(circle)
    ax.annotate(f'Neurônio {j+1}\n{output_label}', (x, y), ha='center', va='center')

for i, (x, y) in enumerate(neuron_coords):
    if i < len(neuron_coords) - neurons:
        outputs = neuron_coords[i+neurons:]
    else:
        outputs = output_coords

    for output in outputs:
        ax.arrow(x, y, output[0]-x, output[1]-y, head_width=0.05, head_length=0.1, fc='black', ec='black')

for coord in input_coords:
    ax.annotate('Entrada', coord, ha='center', va='center', xytext=(-1, 0), textcoords='offset points')

for coord in output_coords:
    ax.annotate('Saída', coord, ha='center', va='center', xytext=(1, 0), textcoords='offset points')

plt.xlim(-1, len(model.hidden_layer_sizes) + 2)
plt.ylim(-(max(neurons for neurons in model.hidden_layer_sizes) - 1) / 2 - 1, (max(neurons for neurons in model.hidden_layer_sizes) - 1) / 2 + 1)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
