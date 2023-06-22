import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    input_file = 'C:/Users/eduar/Downloads/DataMiningSamples-master/0-Datasets/glassClear.data'
    names = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']
    target = 'Type'
    df = pd.read_csv(input_file, names=names)

    corr_matrix = df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Matriz de correlação - Glass Dataset')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.savefig('correlation_matrix.png')
    plt.show()

if __name__ == "__main__":
    main()
