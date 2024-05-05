import ADBib

def main():
    data = [59, 61, 74, 84, 86, 75, 96, 92, 53, 66, 58, 49, 71, 72, 73, 66, 91, 68, 79, 79, 64, 84,
            86, 79, 88, 59, 98, 82, 69, 75, 84, 61, 92, 84, 82, 62, 61, 88, 74, 58, 60, 62, 75, 86,
                88, 72, 90, 96, 51, 64]
    print(ADBib.build_histogram(data))
    ADBib.plot_histogram(data)

if __name__ == "__main__":
    main()