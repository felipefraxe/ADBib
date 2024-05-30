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

        for j in range(1, n):
            service_time = np.random.exponential(1 / mu) # Tempo de serviço de j-1
            arrival_time = np.random.exponential(1 / lmbda) # Tempo de chegada de j

            curr_wait_time = clients[j-1] - arrival_time + service_time

            if curr_wait_time < 0:
                curr_wait_time = 0
            clients.append(curr_wait_time)
        print(ADBib.arithmetic_mean(clients), E)


if __name__ == "__main__":
    main()