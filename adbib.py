import matplotlib.pyplot as plt


def plot_boxplot(data):
    plt.figure(figsize=(8, 6))
    plt.boxplot(data)
    plt.title('Box Plot')
    plt.ylabel('Value')
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


def main():
    # data = [16, 46, 88, 11, 44, 91, 53, 71, 25]
    # print(calculate_quartile(data))
    # Example usage:
    data = [16, 46, 88, 11, 44, 91, 53, 71, 25, 200]
    plot_boxplot(data)

if __name__ == "__main__":
    main()