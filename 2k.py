from itertools import product, combinations
import pandas as pd
import matplotlib.pyplot as plt
import ADBib


def plot_table(df):
    _, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

    plt.show()


def main():
    """PARA TESTAR COM ENTRADA DIRETO DO ARQUIVO, SE QUISER COM INPUT DE USUÁRIO, COMENTE ESTA
    PARTE E TROQUE PELA PARTE COMENTADA NO FINAL DA FUNÇÃO MAIN
    """
    values = None
    with open("input.txt", "r") as file:
        values = file.readlines()
    k = int(values[0].strip())
    factors = [0] * 3
    for i in range(3):
        vals = values[i+1].strip().split(',')
        vals[0], vals[1] = int(vals[0]), int(vals[1])
        factors[i] = tuple(vals)

    num_executions = int(values[4])
    factors_combinations = product(*factors)

    results = [0] * (2 ** k)
    for i in range(5, len(values)):
        results[i-5] += int(values[i])

    for i in range(len(results)):
        results[i] /= num_executions
    """
        TROQUE ATÉ AQUI
    """

    data = []
    for i, combo in enumerate(factors_combinations):
        row = [1]
        for j, factor in enumerate(factors):
            if combo[j] == factor[0]:
                row.append(1)
            else:
                row.append(-1)

        for n in range(2, k+1):
            for combination in combinations(range(0, k), n):
                res = 1
                for index in combination:
                    res *= row[index+1]
                row.append(res)
        data.append(row + [results[i]])

    totals = [0] * (len(data[0]))
    for row in data:
        val = row[-1]
        for i, col in enumerate(row[:-1]):
            totals[i] += (col * val)
    totals[-1] = "Total"
    data.append(totals)

    totals_div = [num / (2 ** k) for num in totals[:-1]]
    totals_div.append(f"Total/{2 ** k}")
    data.append(totals_div)

    columns = ['I'] + [chr(65 + i) for i in range(k)]
    tmp = columns[1:]
    for i in range(2, k+1):
        for combination in combinations(tmp, i):
            columns.append(''.join(combination))
    columns += ['Y']

    plot_table(pd.DataFrame(data, columns=columns))


    Q_terms = [(2 ** k) * num ** 2 for num in data[-1][1:-1]]
    SST = sum(Q_terms)

    plot_table(pd.DataFrame([[f"{round((q * 100) / SST)}%" for q in Q_terms]], columns=columns[1:-1]))

    """ k = -1
    while not 5 >= k >= 2:
        k = int(input("Valor de k: "))
    
    factors = [0] * k
    for i in range(k):
        print(f"Escolha do fator {i+1}")
        max, min = int(input("Maior valor: ")), int(input("Menor valor: "))
        factors[i] = (max, min)
    combinations = list(product(*factors))

    pilot_executions = int(input("Vezes de execução piloto: "))
    results = [0] * (2**k)
    for i in range(pilot_executions):
        print("Resultado para:", end=' ')
        for combination in combinations:
            print(f"Resultado para: {combination}")
            results[i] += int(input(""))

    for i in range(len(results)):
        results[i] /= num_executions"""

if __name__ == "__main__":
    main()