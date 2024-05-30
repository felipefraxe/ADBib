import numpy as np
import ADBib


def queue_expected_value(lmbda, mu):
    rho = lmbda / mu
    return rho / (mu * (1 - rho))


def main():
    lmbda = 9
    mu = 10
    E = queue_expected_value(lmbda, mu)


    for i in range(3, 10, 3):
        n = 10 ** i
        clients = [0.0]
        arrival_time_clk = 0

        for j in range(1, n):
            service_time = np.random.exponential(1 / mu) # Tempo de serviço de j-1
            arrival_time = np.random.exponential(1 / lmbda) # Tempo de chegada de j
            diff_arrival_time = arrival_time_clk
            arrival_time_clk += arrival_time

            # Se tempo de chegada menor que tempo de serviço
            #if arrival_time < service_time:
            clk += arrival_time
            clients.append()
            #else:
            #    clients.append(0)
            
        print(clients)


if __name__ == "__main__":
    main()