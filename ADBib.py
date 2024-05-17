import matplotlib.pyplot as plt
import math

DECIMAL_PLACES = 2

def mode(data):
    historgram = build_histogram(data)
    return max(historgram, key=historgram.get)


def median(data):
    data = sorted(data)
    if len(data) % 2 == 0:
        return (data[len(data) // 2] + data[(len(data) // 2) - 1]) / 2
    return data[len(data) // 2]


def arithmetic_mean(data):
    return round(sum(data) / len(data), DECIMAL_PLACES)


def weighted_mean(data):
    sum_product, sum_weights = 0,0

    for value, weight in data:
        sum_product += value * weight
        sum_weights += weight

    return round(sum_product / sum_weights, DECIMAL_PLACES)


def geometric_mean(data):
    prod = 1
    for num in data:
        prod *= num
    return round(prod ** (1 / len(data)), DECIMAL_PLACES)


def harmonic_mean(data):
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