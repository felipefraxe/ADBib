import matplotlib.pyplot as plt


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