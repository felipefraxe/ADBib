import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import ADBib

def queue_expected_value(lambd, mu):
    rho = lambd / mu
    return rho / (mu * (1 - rho))

def mser_5y(data, Y=5):
    n = len(data)
    mser_vals = []
    for i in range(n - Y):
        mser_vals.append(np.var(data[i:i+Y]) / np.mean(data[i:i+Y]))
    min_index = np.argmin(mser_vals)
    return min_index

def run_simulation(lambd, mu, horizon_inf=False, precision=0.05):
    n = 10 ** 3 if not horizon_inf else float('inf')
    E = queue_expected_value(lambd, mu)
    wait_times = []
    mean_times = []
    relative_error = float('inf')
    
    while relative_error > precision:
        service_time = random.expovariate(mu) # Tempo de serviço
        arrival_time = random.expovariate(lambd) # Tempo de chegada

        curr_wait_time = wait_times[-1] - arrival_time + service_time if wait_times else service_time

        if curr_wait_time < 0:
            curr_wait_time = 0

        wait_times.append(curr_wait_time)

        if len(wait_times) > n:
            stable_index = mser_5y(wait_times)
            stable_wait_times = wait_times[stable_index+1:]
            mean_time = np.mean(stable_wait_times)
            mean_times.append(mean_time)
            relative_error = stats.sem(stable_wait_times) / mean_time if mean_time != 0 else float('inf')
    
    ci_lower, ci_upper = stats.t.interval(0.95, len(mean_times)-1, loc=np.mean(mean_times), scale=stats.sem(mean_times))
    return np.mean(mean_times), (ci_lower, ci_upper), wait_times

def main():
    # Cenários
    scenarios = [(7, 10), (8, 10), (9, 10), (9.5, 10)]
    results = {"Scenario": [], "Mean": [], "CI Lower": [], "CI Upper": []}

    for lambd, mu in scenarios:
        mean, (ci_lower, ci_upper), wait_times = run_simulation(lambd, mu, horizon_inf=True)
        results["Scenario"].append(f"λ={lambd}, μ={mu}")
        results["Mean"].append(mean)
        results["CI Lower"].append(ci_lower)
        results["CI Upper"].append(ci_upper)

    # Salvar resultados em CSV
    df = pd.DataFrame(results)
    df.to_csv('parte3_simulacao.csv', index=False)
    
if __name__ == '__main__':
    main()
    