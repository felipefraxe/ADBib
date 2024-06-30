import random
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from bisect import bisect_right
import ADBib


def queue_expected_value(lambd, mu):
    rho = lambd / mu
    return rho / (mu * (1 - rho))


def vonNeumann_test(arr, B, s=0):
    K = len(arr) // B
    means = np.zeros(B)
    j = 0
    for i in range(0, len(arr), K):
        mean = ADBib.arithmetic_mean(arr[i:i+K-s])
        means[j] = mean
        j += 1
    
    sorted_means = np.sort(means)
    Rs = np.zeros(B)
    for i, mean in enumerate(means):
        Rs[i] = bisect_right(sorted_means, mean)

    R_mean = ADBib.arithmetic_mean(Rs)
    RVN_num = 0
    RVN_den = (Rs[-1] - R_mean) ** 2
    for i in range(len(Rs) - 1):
        RVN_num += (Rs[i] - Rs[i+1]) ** 2
        RVN_den += (Rs[i] - R_mean) ** 2

    RVN = RVN_num / RVN_den
    if B == 10:
        return RVN <= 1.23
    elif B == 20:
        return RVN <= 1.44
    return RVN <= 1.64


def generate_queue(n, mu, lambd):
    wait_times = np.zeros(n, dtype=np.float32)
    for i in range(0, n):
        service_time = random.expovariate(mu)
        arrival_time = random.expovariate(lambd)

        curr_wait_time = wait_times[i-1] - arrival_time + service_time

        if curr_wait_time < 0:
            curr_wait_time = 0
        wait_times[i] = curr_wait_time
    return wait_times


def mser5(arr):
    K = len(arr) // 5
    means = np.zeros(K)
    j = 0
    for i in range(0, len(arr), 5):
        mean = ADBib.arithmetic_mean(arr[i:i+5])
        means[j] = mean
        j += 1

    first_min_val = ADBib.standard_deviation(means) / sqrt(len(means))
    second_mean_val = first_min_val
    D = 0

    for d in range(1, K // 2):
        std_err = ADBib.standard_deviation(means[d:]) / sqrt(len(means) - d)
        if std_err <= first_min_val:
            first_min_val = std_err
            second_mean_val = first_min_val
            D = d

    if first_min_val < second_mean_val:
        return d * 5

    for d in range(K // 2, (K // 2) + (K // 4)):
        std_err = ADBib.standard_deviation(means[d:]) / sqrt(len(means) - d)
        if std_err < D:
            return len(arr) - 1
    return d * 5


def plot_expected_vs_confidence_intervals(results, expected_value):
    plt.figure(figsize=(10, 6))

    plt.boxplot(results, positions=np.arange(len(results)) * 3, widths=1.5)
    
    plt.axhline(y=expected_value, color='r', linestyle='-', label='Valor Esperado Teórico')
    
    plt.xlabel('Número de Clientes (log scale)')
    plt.ylabel('Tempo de Espera')
    plt.xticks(np.arange(len(results)) * 3, ['10^{}'.format(3 * (i + 1)) for i in range(len(results))])
    plt.title('Valor Esperado vs Intervalo de Confiança')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    mu = 10
    for lambd in [7, 8, 9, 9.5]:
        E = queue_expected_value(lambd, mu)
        for B in [10, 20, 50]:
            M = 100
            n = M * B

            wait_times = generate_queue(n, mu, lambd)
            while True:
                stable_index = mser5(wait_times)
                wait_times = np.concatenate((wait_times[stable_index+1:], generate_queue(stable_index+1, mu, lambd)))
                if stable_index != len(wait_times) - 1:
                    break

            while vonNeumann_test(wait_times, B) == False:
                wait_times = np.concatenate((wait_times, generate_queue(B * 50, mu, lambd)))


            mean = ADBib.arithmetic_mean(wait_times)
            ci = ADBib.confidence_interval(wait_times, mean, 0.95)
            H = ci[1] - mean
            gamma = 0.05
            while (H / mean) > gamma:
                wait_times = np.concatenate((wait_times, generate_queue(100, mu, lambd)))
                mean = ADBib.arithmetic_mean(wait_times)
                ci = ADBib.confidence_interval(wait_times, mean, 0.95)
                H = ci[1] - mean

            print(f"Tamanho da amostra {len(wait_times)}")
            print(f"Tempo Médio Estimado de Espera: {mean:.6f}")
            print(f"Intervalo de Confiança de 95%: {ci}")
            print(f"Tempo Médio Teórico de Espera: {E:.6f}", end="\n\n")
        print()


if __name__ == "__main__":
    main()