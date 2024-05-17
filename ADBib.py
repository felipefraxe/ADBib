import matplotlib.pyplot as plt
import math

DECIMAL_PLACES = 2

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
    Calculates the geometric mean of a dataset.

    Args:
        data: A list of positive values.

    Returns:
        The geometric mean of the dataset.
    """
    prod = 1
    for num in data:
        prod *= num
    return round(prod ** (1 / len(data)), DECIMAL_PLACES)


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


def amplitude(data):
    min_val, max_val = min(data), max(data)
    return max_val - min_val


def variance(data):
    mean = arithmetic_mean(data)
    sum = 0
    for num in data:
        sum += ((num - mean) ** 2)
    return sum / len(data)


def standard_deviation(data):
    return math.sqrt(variance(data))


def variation_coefficient(data):
    return (standard_deviation(data) / arithmetic_mean(data)) * 100


def build_histogram(data):
    histogram = dict()
    for item in data:
        if item not in histogram:
            histogram[item] = 0
        histogram[item] += 1
    return histogram


def plot_histogram(data, y_label='Frequency', x_label='Values', title='Histogram'):
    histogram = build_histogram(data)
    plt.bar(histogram.keys(), histogram.values())
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


def plot_boxplot(data):
    plt.figure(figsize=(8, 6))
    plt.boxplot(data)
    plt.title('Box Plot')
    plt.ylabel('Value')
    plt.show()


def plot_values_with_mean(data, mean):
    # Verifica se 'data' é uma lista de tuplas
    if isinstance(data[0], tuple):
        values = [value for value, _ in data]
    else:
        values = data

    plt.bar(range(len(values)), values, width=0.4, align='center')
    plt.axhline(mean, color='r', linestyle='-', label=f'Média: {mean}')
    plt.legend()
    plt.show()


def calculate_quartile(data):
    data = sorted(data)
    q1 = data[len(data) // 4]
    q2 = data[len(data) // 2]
    q3 = data[(2 * len(data)) // 3]
    return q1, q2, q3


def interquartile_amp(data):
    q1, _, q3 = calculate_quartile(data)
    return q3 - q1


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