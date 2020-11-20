from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

DATA_PATH = Path("simplevqdata.csv")



def plot_multi_n_prototypes(errors):

    plt.plot(N_PROTOTYPES, errors)

    plt.xlabel("K")
    plt.ylabel("Quantization error")
    plt.title("Quantization errors for different Ks")
    plt.show()


def show_prototypes_and_data(data, prototypes):
    data = data.transpose()
    prototypes = np.array(prototypes).transpose()
    plt.scatter(data[0], data[1], label="data points")
    plt.scatter(prototypes[0], prototypes[1], label="prototypes")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Data and prototypes")
    plt.legend()
    plt.show()


def plot_prototype_trajectories(all_prototypes, data, n_prototypes):

    data = data.transpose()

    # Plot data points as blue dots
    plt.scatter(data[0], data[1], label="data points")

    # Plot trajectories
    for j in range(n_prototypes):
        x_values = [prototypes[j][0] for prototypes in all_prototypes]
        y_values = [prototypes[j][1] for prototypes in all_prototypes]
        plt.plot(x_values, y_values, color='orange', linewidth=2)

    # To instantiate single label for protoype trajectories
    plt.plot(all_prototypes[0][0][0], all_prototypes[0][0]
             [0], color="orange", label="prototype trajectories")

    # Plot final prototypes as red dots
    final_prototypes = all_prototypes[-1]
    final_prototypes = np.array(final_prototypes).transpose()
    plt.scatter(final_prototypes[0], final_prototypes[1],
                label="final prototypes", color='r')

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Data and prototypes trajectories")
    plt.legend()
    plt.show()



def main():
    data_path = os.path.dirname(os.path.abspath(__file__)) / DATA_PATH

    df = pd.read_csv(data_path, header=None)
    data = df.values

    result = []


if __name__ == "__main__":
    main()
