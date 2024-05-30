import matplotlib.pyplot as plt
import math
import scipy.stats as stats
import numpy as np


DECIMAL_PLACES = 2


def arithmetic_mean(data):
    """
    Calculates the arithmetic mean of a dataset.

    Args:
        data: A list of values.

    Returns:
        The arithmetic mean of the dataset.
    """
    return sum(data) / len(data)


def weighted_mean(data):
    """
    Calculates the weighted mean of a dataset.

    Args:
        data: A list of tuples (value, weight).

    Returns:
        The weighted mean of the dataset.
    """
    sum_product, sum_weights = 0,0

    for value, weight in data:
        sum_product += value * weight
        sum_weights += weight

    return round(sum_product / sum_weights, DECIMAL_PLACES)


def geometric_mean(data):
    """
    Calculates the geometric mean of a dataset using logarithms to avoid overflow.

    Args:
        data: A list of positive values.

    Returns:
        The geometric mean of the dataset.
    """
    log_sum = sum(math.log(num) for num in data)
    return round(math.exp(log_sum / len(data)), DECIMAL_PLACES)


def harmonic_mean(data):
    """
    Calculates the harmonic mean of a dataset.

    Args:
        dataset: A list of positive values.

    Returns:
        The harmonic mean of the dataset.
    """
    sum = 0
    for num in data:
        sum += (1 / num)
    return round(len(data) / sum, DECIMAL_PLACES)


def mean_rate_case1(data):
    """
    Calculates the mean rate for case 1, where the sum of numerators and the sum of denominators have physical meaning.

    Args:
        data: A list of tuples (numerator, denominator) representing the rates.

    Returns:
        The mean rate.
    """
    sum_numerators = sum(numerator for numerator, _ in data)
    sum_denominators = sum(denominator for _, denominator in data)
    return sum_numerators / sum_denominators


def mean_rate_case2(data):
    """
    Calculates the mean rate for case 2, where the denominator is constant and the sum of numerators has physical meaning.

    Args:
        data: A list of tuples (numerator, denominator) representing the rates.

    Returns:
        The mean rate.
    """
    if not all(denominator == data[0][1] for _, denominator in data):
        raise ValueError("Denominators must be constant for case 2.")

    sum_numerators = sum(numerator for numerator, _ in data)
    denominator = data[0][1]  # Get the constant denominator
    return sum_numerators / (len(data) * denominator)
    

def mean_rate_case3(data):
    """
    Calculates the mean rate for case 3, where the sum of denominators has physical meaning and the numerators are constant.

    Args:
        data: A list of tuples (numerator, denominator) representing the rates.

    Returns:
        The mean rate, rounded to DECIMAL_PLACES decimal places.
    """
    if not all(numerator == data[0][0] for numerator, _ in data):
        raise ValueError("Numerators must be constant for case 3.")

    sum_reciprocals = sum(denominator / numerator for numerator, denominator in data)
    return len(data) / sum_reciprocals


def select_appropriate_mean(data, physical=False, weight=False, rates=False):
    """
    Analyzes a dataset and selects the most appropriate mean to use (arithmetic, weighted, geometric, harmonic, or mean rate cases).

    Args:
        data: A list of values or tuples (if weights are provided).
        physical: If True, considers mean rate cases. Defaults to False.
        weight: If True and physical is False, uses weighted mean. Defaults to False.

    Returns:
        A tuple containing:
            - The calculated mean value
            - The type of mean used (e.g., "arithmetic", "weighted", etc.)
    """
    
    # Check for mean rate cases if physical is True
    if physical and all(isinstance(x, tuple) and len(x) == 2 for x in data):
        numerators, denominators = zip(*data)
        if all(x > 0 for x in denominators):  
            if all(x == numerators[0] for x in numerators):  
                return mean_rate_case3(data), "mean rate (case 3)"
            elif all(x == denominators[0] for x in denominators):  
                return mean_rate_case2(data), "mean rate (case 2)"
            else:  # Case 1
                return mean_rate_case1(data), "mean rate (case 1)"

    # Use weighted mean if weight is True and physical is False
    if weight and all(isinstance(x, tuple) and len(x) == 2 for x in data):
        if all(x[1] > 0 for x in data):  
            return weighted_mean(data), "weighted"

    # Check for positive values (geometric/harmonic mean)
    if all(x > 0 for x in data):
        if rates: # Use harmonic mean if rates is True
            return harmonic_mean(data), "harmonic"
        else: # Use geometric mean if rates is False
            return geometric_mean(data), "geometric"

    # Default to arithmetic mean if no other criteria are met
    return arithmetic_mean(data), "arithmetic"


def median(data):
    """
    Calculates the median of a dataset.

    Args:
        data: A list of values.

    Returns:
        The median of the dataset.
    """
    data = sorted(data)
    if len(data) % 2 == 0:
        return (data[len(data) // 2] + data[(len(data) // 2) - 1]) / 2
    return data[len(data) // 2]


def mode(data):
    """
    Calculates the mode of a dataset.

    Args:
        data: A list of values.

    Returns:
        The mode of the dataset.
    """
    historgram = build_histogram(data)
    return max(historgram, key=historgram.get)


def amplitude(data):
    """
    Calculates the amplitude of a dataset.

    Args:
        data: A list of numerical values.

    Returns:
        The amplitude of the dataset.
    """
    min_val, max_val = min(data), max(data)
    return max_val - min_val


def plot_amplitude(data):
    """
    Plots a bar chart showing the values in the dataset and highlights the min and max values.

    Args:
        data: A list of numerical values.
    """
    min_val, max_val = min(data), max(data)
    amp = amplitude(data)
    
    plt.bar(range(len(data)), data, width=0.4, align='center')
    
    plt.axhline(max_val, color='b', linestyle='-', label=f'Max value: {max_val}')
    plt.axhline(min_val, color='r', linestyle='-', label=f'Min value: {min_val}')
    
    plt.legend()
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.show()


def variance(data):
    """
    Calculates the sample variance of a dataset.

    Args:
        data: A list of numerical values.

    Returns:
        The sample variance of the dataset.
    """
    mean = arithmetic_mean(data)
    sum_squares = 0
    for num in data:
        sum_squares += (num - mean) ** 2
    return sum_squares / len(data)


def standard_deviation(data):
    """
    Calculates the sample standard deviation of a dataset.

    Args:
        data: A list of numerical values.

    Returns:
        The sample standard deviation of the dataset.
    """
    return math.sqrt(variance(data))


def coefficient_of_variation(data):
    """
    Calculates the variation coefficient of a dataset.

    Args:
        data: A list of numerical values.
        
    Returns:
        The variation coefficient of the dataset.
    """
    return round((standard_deviation(data) / arithmetic_mean(data)) * 100, DECIMAL_PLACES)


def calculate_quartile(data):
    """
    Calculates the quartiles of a dataset.
    
    Args:
        data: A list of numerical values.
        
    Returns:
        A tuple (q1, q2, q3) containing the first, second and third quartiles of the dataset.
    """
    data = sorted(data)
    q1 = data[len(data) // 4]
    q2 = median(data)
    q3 = data[(3 * len(data)) // 4]
    return q1, q2, q3


def interquartile_amp(data):
    """
    Calculates the interquartile amplitude of a dataset.

    Args:
        data: A list of numerical values.

    Returns:
        The interquartile amplitude of the dataset.
    """
    q1, _, q3 = calculate_quartile(data)
    return q3 - q1


def dispersion_measures(data,unit=""):
    """
    Calculates various dispersion measures of a dataset.

    Args:
        data: A list of numerical values.

    Returns:
        A dictionary containing the following dispersion measures:
            - Amplitude
            - Sample Variance
            - Sample Standard Deviation
            - Coefficient of Variation
            - Quartiles (Q1, Q2, Q3)
            - Interquartile Range 
    """

    results = {
        "Amplitude": f"{amplitude(data)} {unit}" if unit else amplitude(data),
        "Variância Amostral": f"{variance(data)} {unit}²" if unit else variance(data),
        "Desvio Padrão Amostral": f"{standard_deviation(data)} {unit}" if unit else standard_deviation(data),
        "Coeficiente de Variação (%)": coefficient_of_variation(data),
        "Quartis": calculate_quartile(data),
        "Amplitude Interquartil": f"{interquartile_amp(data)} {unit}" if unit else interquartile_amp(data),
    }

    for key, value in results.items():
        print(f'{key}: {value}')


def build_histogram(data):
    """
    Builds a histogram from a dataset.

    Args:
        data: A list of values.

    Returns:
        A dictionary containing the frequency of each value in the dataset.
    """
    histogram = dict()
    for item in data:
        if item not in histogram:
            histogram[item] = 0
        histogram[item] += 1
    return histogram


def confidence_interval(data, confidence_degree=0.99):
    """
    Calculates the confidence interval of a dataset.

    Args:
        data: A list of numerical values.
        confidence_degree: The confidence degree (e.g., 0.95 for 95% confidence).

    Returns:
        A tuple (min, max) containing the confidence interval of the dataset.
    """
    mean = arithmetic_mean(data)
    std_err = standard_deviation(data) / math.sqrt(len(data))
    if len(data) > 30:
        confidence_value = {
            0.9: 1.645,
            0.95: 1.960,
            0.99: 2.576
        }
        return (mean - (std_err * confidence_value[confidence_degree]),
                    mean + (std_err * confidence_value[confidence_degree]))

    t_value = stats.t.ppf((1 + confidence_degree) / 2, df=len(data) - 1)
    return (round(mean - (std_err * t_value), DECIMAL_PLACES), round(mean + (std_err * t_value), DECIMAL_PLACES))


def zero_mean_test(sample_a, sample_b, confidence_degree=0.99):
    """
    Performs a zero-mean test to determine if two samples are significantly similar.

    Args:
        sample_a: The first sample.
        sample_b: The second sample.
        confidence_degree: The confidence degree (e.g., 0.95 for 95% confidence).

    Returns:
        True if the samples are significantly similar, False otherwise.
    """
    if len(sample_a) == len(sample_b):
        diff = [(sample_a[i] - sample_b[i]) for i in range(len(sample_a))]
        min, max = confidence_interval(diff, confidence_degree)
        return min <= 0 and max >= 0


    var_a = variance(sample_a)
    var_b = variance(sample_b)

    numerator = ((var_a / len(sample_a)) + (var_b / len(sample_b))) ** 2
    denominator = (((var_a / len(sample_a)) ** 2) * (1 / (1 + len(sample_a)))) + (((var_b / len(sample_b)) ** 2) * (1 / (1 + len(sample_b))))
    v = (numerator / denominator) - 2

    mean_a = arithmetic_mean(sample_a)
    mean_b = arithmetic_mean(sample_b)
    mean_diff = mean_a - mean_b

    t_value = stats.t.ppf((1 + confidence_degree) / 2, df=v)
    min = round(mean_diff - t_value, DECIMAL_PLACES)
    max = round(mean_diff + t_value, DECIMAL_PLACES)
    return min < 0 and max > 0


def estimate_sample_size(confidence_degree, std_dev, mean=None, length=None, precision=None):
    """"
    Estimates the sample size required for a given confidence degree, standard deviation, mean, length, and precision.

    Args:
        confidence_degree: The confidence degree (e.g., 0.95 for 95% confidence).
        std_dev: The standard deviation of the population.
        mean: The mean of the population.
        length: The length of the population.
        precision: The desired precision of the estimate.

    Returns:
        The estimated sample size.
    """
    confidence_value = {
        0.9: 1.645,
        0.95: 1.960,
        0.99: 2.576
    }

    if length is None and precision is not None and mean is not None:
        return round(((100 * std_dev * confidence_value[confidence_degree]) / (precision * mean)) ** 2, DECIMAL_PLACES)

    return round((4 * ((std_dev * confidence_value[confidence_degree]) ** 2)) / (length ** 2), DECIMAL_PLACES)


def bernoulli_pmf(p, x):
    """Calculates the probability mass function (PMF) of a Bernoulli random variable.

    Args:
        p: The probability of success.
        x: The value at which to evaluate the PMF.
    
    Returns:
        The probability mass function (PMF) of a Bernoulli random variable.
    """
    if x == 1:
        return p
    return 1 - p


def bernoulli_expected_value(p):
    """Calculates the expected value of a Bernoulli random variable.

    Args:
        p: The probability of success.

    Returns:
        The expected value of a Bernoulli random variable.
    """
    return p


def bernoulli_variance(p):
    """Calculates the variance of a Bernoulli random variable.

    Args:
        p: The probability of success.

    Returns:
        The variance of a Bernoulli random variable.
    """
    return round(p * (1 - p), DECIMAL_PLACES)


def bernoulli_coefficient_of_variance(p):
    """Calculates the coefficient of variance of a Bernoulli random variable.

    Args:
        p: The probability of success.

    Returns:
        bernoulli_coefficient_of_variance: The coefficient of variance of a Bernoulli random variable.
    """
    return round(math.sqrt(bernoulli_variance(p)) / p, DECIMAL_PLACES)


def binomial_pmf(n, p, k):
    """Calculates the probability mass function (PMF) of a binomial random variable.

    Args:
        n: The number of trials.
        p: The probability of success.
        k: The number of successes.

    Returns:
        The probability mass function (PMF) of a binomial random variable.
    """
    binomial_coefficient = math.comb(n, k)
    return round(binomial_coefficient * (p ** k) * ((1 - p) ** (n - k)), DECIMAL_PLACES)


def binomail_expected_value(n, p):
    """" Calculates the expected value of a binomial random variable.

    Args:
        n: The number of trials.
        p: The probability of success.

    Returns:
        The expected value of a binomial random variable.
    """
    return n * p


def binomial_variance(n, p):
    """Calculates the variance of a binomial random variable.

    Args:
        n: The number of trials.
        p: The probability of success.

    Returns:
        The variance of a binomial random variable.
    """
    return round(n * p * (1 - p), DECIMAL_PLACES)


def binomial_coefficient_of_variance(n, p):
    """Calculates the coefficient of variance of a binomial random variable.

    Args:
        n: The number of trials.
        p: The probability of success.
    
    Returns:
        The coefficient of variance of a binomial random variable.
    """
    round(math.sqrt(binomial_variance(n, p)) / (n * p), DECIMAL_PLACES)


def geometric_pmf(i, p):
    """Calculates the probability mass function (PMF) of a geometric random variable.

    Args:
        i: The number of trials until the first success.
        p: The probability of success.

    Returns:
        The probability mass function (PMF) of a geometric random variable.
    """
    return round(p * ((1 - p) ** (i - 1)), 3)


def geometric_expected_value(p):
    """" Calculates the expected value of a geometric random variable.

    Args:
        p: The probability of success.

    Returns:
        The expected value of a geometric random variable.
    """
    return round(1 / p, DECIMAL_PLACES)


def geometric_variance(p):
    """Calculates the variance of a geometric random variable.

    Args:
        p: The probability of success.

    Returns:
        The variance of a geometric random variable.
    """
    return round((1 - p) / (p ** 2), DECIMAL_PLACES)


def geometric_coefficient_of_variance(p):
    """Calculates the coefficient of variance of a geometric random variable.

    Args:
        p: The probability of success.

    Returns:
        The coefficient of variance of a geometric random variable.
    """
    return round(math.sqrt(geometric_variance(p)) / (1 / p), DECIMAL_PLACES)


def poisson_pmf(lmbda, k):
    """Calculates the probability mass function (PMF) of a Poisson random variable.

    Args:
        lmbda: The average number of events per interval.
        k: The number of events.

    Returns:
        The probability mass function (PMF) of a Poisson random variable.
    """
    return round((lmbda ** k * math.exp(-lmbda)) / math.factorial(k), DECIMAL_PLACES)


def poisson_expected_value(lmbda):
    """Calculates the expected value of a Poisson random variable.

    Args:
        lmbda: The average number of events per interval.

    Returns:
        The expected value of a Poisson random variable.
    """
    return lmbda


def poisson_variance(lmbda):
    """Calculates the variance of a Poisson random variable.
    
    Args:
        lmbda: The average number of events per interval.

    Returns:
        The variance of a Poisson random variable.
    """
    return lmbda


def poisson_coefficient_of_variance(lmbda):
    """Calculates the coefficient of variance of a Poisson random variable.

    Args:
        lmbda: The average number of events per interval.

    Returns:
        The coefficient of variance of a Poisson random variable.
    """
    return round(math.sqrt(lmbda) / lmbda, DECIMAL_PLACES)


def plot_histogram(data, y_label='Frequency', x_label='Values', title='Histogram'):
    """
    Plots a histogram from a dataset.

    Args:
        data: A list of values.
        y_label: The label of the y-axis.
        x_label: The label of the x-axis.
        title: The title of the plot.
    """
    histogram = build_histogram(data)
    plt.bar(histogram.keys(), histogram.values())
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


def plot_histogram_avengers(data, bins='auto'):
    """
    Plota um histograma para os dados fornecidos.

    Args:
        data: Lista de valores.
        bins: Número de bins (intervalos) para o histograma. Pode ser um inteiro ou 'auto' para definir automaticamente.
    """
    plt.figure(figsize=(10, 6))  # Tamanho da figura
    plt.hist(data, bins=bins, edgecolor='black')
    plt.xlabel('Appearances')
    plt.ylabel('Frequency')
    plt.title('Histogram')
    plt.grid(axis='y', alpha=0.75)  # Adiciona grade no eixo y para melhor visualização
    plt.show()


def plot_boxplot(data):
    """
    Plots a box plot from a dataset.

    Args:
        data: A list of numerical values.
    """
    plt.figure(figsize=(8, 6))
    plt.boxplot(data)
    plt.title('Box Plot')
    plt.ylabel('Value')
    plt.show()


def plot_values_with_mean(data, mean):
    """
    Plots a bar chart showing the values in the dataset and highlights the mean value.

    Args:
        data: A list of numerical values.
        mean: The mean value of the dataset.
    """
    # Verifica se 'data' é uma lista de tuplas
    if isinstance(data[0], tuple):
        values = [value for value, _ in data]
    else:
        values = data

    plt.bar(range(len(values)), values, width=0.4, align='center')
    plt.axhline(mean, color='r', linestyle='-', label=f'Média: {mean}')
    
    plt.legend()
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.show()


def calcular_estatisticas_e_plotar_fdp(valores, probabilidades):
    """
    Calcula estatísticas (valor esperado, valor esperado quadrado, variância) e plota a função de distribuição
    de probabilidade (FDP) de uma variável aleatória discreta.

    Args:
        valores: Lista de valores que a variável aleatória pode assumir.
        probabilidades: Lista de probabilidades correspondentes a cada valor.
    """

    if len(valores) != len(probabilidades):
        raise ValueError("As listas de valores e probabilidades devem ter o mesmo tamanho.")

    # Cálculo das estatísticas
    valor_esperado = sum(x * p for x, p in zip(valores, probabilidades))
    valor_esperado_quadrado = sum(x**2 * p for x, p in zip(valores, probabilidades))
    variancia = valor_esperado_quadrado - valor_esperado**2

    # Cálculo da FDP 
    fdp = [0]  # Começa em 0 para x < valor mínimo
    for p in probabilidades:
        fdp.append(fdp[-1] + p)

    plt.figure(figsize=(8, 5))

    # Adicionar valores extras para o plot (começar e terminar fora dos valores)
    extended_values = [valores[0] - 1] + valores + [valores[-1] + 1]
    extended_fdp = [0] + fdp

    for i in range(1, len(extended_values)):
        plt.hlines(extended_fdp[i], extended_values[i-1], extended_values[i], colors='red', linewidth=2)
    
    plt.hlines(1, valores[-1], valores[-1] + 1, colors='red', linewidth=2)
    plt.axvline(x=0, color='black', linestyle='-')

    # Ajustar o eixo x para incluir 0
    all_x_values = sorted(set(valores + [0]))
    plt.xticks(all_x_values)

    plt.xlabel('x')
    plt.ylabel('F(x)')
    plt.title('Função de Distribuição de Probabilidade (FDP)')
    plt.xticks(valores)
    plt.ylim(-0.01, 1.01)
    plt.grid(axis='y', alpha=0.75)
    plt.show()

    print(f"Valor Esperado (Média): {valor_esperado:.3f}")
    print(f'Valor Esperado Quadrado: {valor_esperado_quadrado:.3f}')
    print(f"Variância: {variancia:.3f}")


def uniform_continuous(a, b):
    """
    Calculates statistics (expected value, variance) and plots the probability density function (PDF)
    of a uniform continuous random variable.

    Args:
        a: Lower bound of the uniform distribution.
        b: Upper bound of the uniform distribution.
    """
    def f(x):
        if a <= x <= b:
            return 1 / (b - a)
        else:
            return 0
    
    valor_esperado = (a + b) / 2
    variancia = (b - a) ** 2 / 12

    x = np.linspace(a - 1, b + 1, 1000)
    y = [f(xi) for xi in x]
    
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label='PDF')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Probability Density Function - Uniform Continuous')
    plt.grid(True)
    plt.legend()
    plt.show()

    print(f"Valor Esperado (Média): {round(valor_esperado, DECIMAL_PLACES)}")
    print(f"Variância: {round(variancia, DECIMAL_PLACES)}")


def exponential_continuous(lambd):
    """
    Calculates statistics (expected value, variance) and plots the probability density function (PDF)
    of an exponential continuous random variable.

    Args:
        lambd: Rate parameter of the exponential distribution.
    """
    def f(x):
        if x >= 0:
            return lambd * np.exp(-lambd * x)
        else:
            return 0
    
    valor_esperado = 1 / lambd
    variancia = 1 / lambd ** 2

    x = np.linspace(0, 5 / lambd, 1000)
    y = [f(xi) for xi in x]
    
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label='PDF')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Probability Density Function - Exponential')
    plt.grid(True)
    plt.legend()
    plt.show()

    print(f"Valor Esperado (Média): {round(valor_esperado, DECIMAL_PLACES)}")
    print(f"Variância: {round(variancia, DECIMAL_PLACES)}")


def normal_continuous(mu, sigma):
    """
    Calculates statistics (expected value, variance) and plots the probability density function (PDF)
    of a normal continuous random variable.

    Args:
        mu: Mean of the normal distribution.
        sigma: Standard deviation of the normal distribution.
    """
    def f(x):
        return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    
    valor_esperado = mu
    variancia = sigma ** 2

    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)
    y = [f(xi) for xi in x]
    
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label='PDF')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Probability Density Function - Normal')
    plt.grid(True)
    plt.legend()
    plt.show()

    print(f"Valor Esperado (Média): {valor_esperado:.3f}")
    print(f"Variância: {variancia:.3f}")


def model_random_variable(distribution, *params):
    """
    Models a random variable based on the specified distribution and parameters.

    Args:
        distribution (str): The type of distribution ('uniform', 'exponential', 'normal').
        *params: The parameters for the specified distribution.
    """
    if distribution == 'uniform':
        if len(params) != 2:
            raise ValueError("Uniform distribution requires 2 parameters (a, b).")
        uniform_continuous(*params)
    elif distribution == 'exponential':
        if len(params) != 1:
            raise ValueError("Exponential distribution requires 1 parameter (lambda).")
        exponential_continuous(*params)
    elif distribution == 'normal':
        if len(params) != 2:
            raise ValueError("Normal distribution requires 2 parameters (mu, sigma).")
        normal_continuous(*params)
    else:
        raise ValueError("Unsupported distribution type. Choose from 'uniform', 'exponential', or 'normal'.")
