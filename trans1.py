import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ADBib


def queue_expected_value(lambd, mu):
    rho = lambd / mu
    return rho / (mu * (1 - rho))


def plot_table(data):
    df = pd.DataFrame(data)

    _, ax = plt.subplots(figsize=(8, 8))  # Ajusta o tamanho da figura
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.show()


def main():
    lambd = 9
    mu = 10
    E = queue_expected_value(lambd, mu)
    r = 30
    data = { "means": [], "biases": [] }

    for _ in range(r):
        n = 10 ** 3
        wait_times = np.zeros(n, dtype=np.float32)

        for j in range(1, n):
            service_time = random.expovariate(mu) # Tempo de serviço de j-1
            arrival_time = random.expovariate(lambd) # Tempo de chegada de j

            curr_wait_time = wait_times[j-1] - arrival_time + service_time

            if curr_wait_time < 0:
                curr_wait_time = 0
            wait_times[j] = curr_wait_time

        mean = ADBib.arithmetic_mean(wait_times)
        data["means"].append(mean)
        data["biases"].append(mean - E)
    data["means"].append(ADBib.arithmetic_mean(data["means"]))
    data["biases"].append(ADBib.arithmetic_mean(data["biases"]))
    plot_table(data)


if __name__ == "__main__":
    main()