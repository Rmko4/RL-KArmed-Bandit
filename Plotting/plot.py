from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

DATA_BERNOULLI = Path("Bernoulli")
DATA_GAUSSIAN = Path("Gaussian")
ALGORITHMS = ["Epsilon Greedy", "Pursuit Method",
              "Reinforcement Comparison", "Stochastic Gradient Descent"]


def open_data(data_path):
    data = []
    for file in os.listdir(data_path):
        if file.endswith(".csv"):
            df = pd.read_csv(data_path / file, header=None)
            data.append(df.values[:-1].transpose())
    return data


def plot_reward(data, title):
    plt.figure(figsize=(9, 5))
    for i in range(len(data)):
        plt.plot(data[i][0], label=ALGORITHMS[i])
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.title(title)
    plt.legend()
    plt.show()


def plot_optimality(data, title):
    for i in range(len(data)):
        plt.plot(100 * data[i][1], label=ALGORITHMS[i])
    plt.xlabel("Steps")
    plt.ylabel("% Optimal action")
    plt.yticks(np.linspace(0, 100, 6))
    plt.title(title)
    plt.legend()
    plt.show()


def main():
    data_path = Path(os.path.dirname(os.path.abspath(__file__))).parent

    gaussian = open_data(data_path / DATA_GAUSSIAN)
    bernoulli = open_data(data_path / DATA_BERNOULLI)

    plot_reward(
        gaussian, title="Average reward in Gaussian 10-armed bandit problem")
    plot_reward(
        bernoulli, title="Average reward in Bernoulli 10-armed bandit problem")

    plot_optimality(
        gaussian, title="Percentage of optimal actions in Gaussian 10-armed bandit problem")
    plot_optimality(
        bernoulli, title="Percentage of optimal actions in Bernoulli 10-armed bandit problem")


if __name__ == "__main__":
    main()
