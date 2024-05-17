from ADBib import *

def main():
    data = [177, 122, 128, 191, 180, 142, 197, 196, 67, 160, 167, 138, 107, 188, 102, 116,
                138, 114, 188, 176, 148, 175, 169, 203, 135, 142, 168, 181, 168, 150, 132,
                196, 88, 177, 164, 118, 178, 102, 156, 114]
    confidence_levels = [0.9, 0.95, 0.99]

    for level in confidence_levels:
        print(confidence_interval(data, level))

if __name__ == "__main__":
    main()