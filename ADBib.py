import matplotlib.pyplot as plt


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
    sum = 0
    weight = 0
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