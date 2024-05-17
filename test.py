from ADBib import *

def main():
    confidence_degree = [0.9, 0.95, 0.99]
    sys_a = [5.3, 16, 0.6, 1.4, 0.6, 7.7, 3.6, 2.4, 12, 6, 57, 2, 1, 4, 6, 4, 8, 1]
    sys_b = [19, 3.5, 3.3, 2.5, 3.6, 1.7, 12, 2, 8, 1, 1, 4]
    for confidence in confidence_degree:
        print(f"Amostras são significativamente similares: {zero_mean_test(sys_a, sys_b, confidence)}")

if __name__ == "__main__":
    main()