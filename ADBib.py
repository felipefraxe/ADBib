import matplotlib.pyplot as plt
import math


def mode(data):
    historgram = build_histogram(data)
    return max(historgram, key=historgram.get)


def median(data):
    data = sorted(data)
    if len(data) % 2 == 0:
        return (data[len(data) // 2] + data[(len(data) // 2) - 1]) / 2
    return data[len(data) // 2]


def arithmetic_mean(data):
    return sum(data) / len(data)


def weighted_mean(data):
    sum, weight = 0, 0
    for i in range(0, len(data), 2):
        sum += (data[i] * data[i+1])
        weight += data[i+1]

    return sum / weight


def geometric_mean(data):
    prod = 1
    for num in data:
        prod *= num
    return prod ** (1 / len(data))


def harmonic_mean(data):
    sum = 0
    for num in data:
        sum += (1 / num)
    return len(data) / sum


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


def plot_histogram(data):
    histogram = build_histogram(data)
    plt.bar(histogram.keys(), histogram.values())
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Histogram')
    plt.show()