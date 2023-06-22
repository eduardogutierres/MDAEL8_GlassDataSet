import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def main():
    # Read in the data
    input_file = '0-Datasets/glassClear.data'
    names = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']
    features = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
    target = 'Type'
    df = pd.read_csv(input_file, names=names)

    ShowInformationDataFrame(df, "Original Dataframe")

    # Separating out the features
    x = df.loc[:, features].values

    # Separating out the target
    y = df.loc[:, [target]].values

    # Standardizing the features
    x = StandardScaler().fit_transform(x)
    normalizedDf = pd.DataFrame(data=x, columns=features)
    normalizedDf = pd.concat([normalizedDf, df[[target]]], axis=1)
    ShowInformationDataFrame(normalizedDf, "Normalized Dataframe")

    # PCA projection
    pca = PCA()
    principalComponents = pca.fit_transform(x)
    print("Explained variance per component:")
    print(pca.explained_variance_ratio_.tolist())
    print("\n\n")

    principalDf = pd.DataFrame(data=principalComponents[:, 0:2],
                               columns=['principal component 1',
                                        'principal component 2'])
    finalDf = pd.concat([principalDf, df[[target]]], axis=1)
    ShowInformationDataFrame(finalDf, "PCA Dataframe")

    VisualizePcaProjection(finalDf, target)


def ShowInformationDataFrame(df, message=""):
    print(message + "\n")
    print(df.info())
    print(df.describe())
    print(df.head(10))
    print("\n")


def VisualizePcaProjection(finalDf, targetColumn):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    targets = [1, 2, 3, 5, 6, 7]
    target_names = ['building_windows_float_processed',
                    'building_windows_non_float_processed',
                    'vehicle_windows_float_processed',
                    'containers',
                    'tableware',
                    'headlamps']
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    for target, color, target_name in zip(targets, colors, target_names):
        indicesToKeep = finalDf[targetColumn] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
                   finalDf.loc[indicesToKeep, 'principal component 2'],
                   c=color, s=50, label=target_name)
    ax.legend()
    ax.grid()
    plt.show()


if __name__ == "__main__":
    main()