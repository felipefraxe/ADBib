import matplotlib.pyplot as plt
import math
import scipy.stats as stats


DECIMAL_PLACES = 2


def arithmetic_mean(data):
    """
    Calculates the arithmetic mean of a dataset.

    Args:
        data: A list of values.

    Returns:
        The arithmetic mean of the dataset.
    """
    return round(sum(data) / len(data), DECIMAL_PLACES)


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
    return round(sum_squares / (len(data)), DECIMAL_PLACES)


def standard_deviation(data):
    """
    Calculates the sample standard deviation of a dataset.

    Args:
        data: A list of numerical values.

    Returns:
        The sample standard deviation of the dataset.
    """
    return round(math.sqrt(variance(data)), DECIMAL_PLACES)


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

    return results


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
    mean = arithmetic_mean(data)
    std_err = standard_deviation(data) / math.sqrt(len(data))

    if len(data) > 30:
        confidence_value = {
            0.9: 1.645,
            0.95: 1.960,
            0.99: 2.576
        }
        return (round(mean - (std_err * confidence_value[confidence_degree]), DECIMAL_PLACES),
                    round(mean + (std_err * confidence_value[confidence_degree]), DECIMAL_PLACES))

    t_value = stats.t.ppf((1 + confidence_degree) / 2, df=len(data) - 1)
    return (round(mean - (std_err * t_value), DECIMAL_PLACES), round(mean + (std_err * t_value), DECIMAL_PLACES))


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