import numpy as np
import random
import ADBib


def main():
    lambd = 9
    mu = 10
    gamma = 0.05

    n = 10 ** 3 # Valor inicial

    # Valores iniciais para garantir a entrada no loop
    mean = 1
    H = mean
    while (H / mean) > gamma:
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
        H = ci[1] - mean
        if (H / mean) > gamma:
            n += 100

    print(f"\nPara n = {n}:")
    print(f"Para H / média = {H / mean:.6f}")
    print(f"Tempo Médio Estimado de Espera: {mean:.6f}")
    print(f"Intervalo de Confiança de 95%: {ci}")


if __name__ == "__main__":
    main()