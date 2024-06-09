import numpy as np
import random
import matplotlib.pyplot as plt
import ADBib


def queue_expected_value(lambd, mu):
    rho = lambd / mu
    return rho / (mu * (1 - rho))


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
    lambd = 9
    mu = 10
    E = queue_expected_value(lambd, mu)
    results = list()

    for i in range(3, 7, 3):
        n = 10 ** i
        wait_times = np.zeros(n, dtype=np.float32)

        for j in range(1, n):
            service_time = random.expovariate(mu) # Tempo de serviço de j-1
            arrival_time = random.expovariate(lambd) # Tempo de chegada de j

            curr_wait_time = wait_times[j-1] - arrival_time + service_time

            if curr_wait_time < 0:
                curr_wait_time = 0
            wait_times[j] = curr_wait_time

        mean = ADBib.arithmetic_mean(wait_times)
        ci = ADBib.confidence_interval(wait_times, mean, 0.95)
        
        results.append(ci)

        print(f"\nPara n = 10^{i}:")
        print(f"Tempo Médio Estimado de Espera: {mean:.6f}")
        print(f"Intervalo de Confiança de 95%: {ci}")
        print(f"Tempo Médio Teórico de Espera: {E:.6f}")

    plot_expected_vs_confidence_intervals(results, E)


if __name__ == "__main__":
    main()