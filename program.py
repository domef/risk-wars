import argparse
import re
import math
import copy
import numpy as np
import matplotlib.pyplot as plt


#TODO
# fix plotting (axes start from 0, colorbar not always shows 0 and 1)
# expexted values and variance + matrices for heatmap
# normal distribution?
# plot 3d
# plot histograms
# calculate matrices only if --plot?
# get statistics from matrices?
# input loop?
# doc
# axes should have integers
# BUG if attacker and defender >= 77


def read():
    attacker = 0
    defender = 0
    pattern = re.compile('^[0-9]+$')

    while attacker == 0 or defender == 0:
        a = input('Attacker armies: ')
        d = input('Defender armies: ')

        if pattern.match(a) and pattern.match(d) and  int(a) >= 1 and int(d) >= 1:
            # attacker = int(a) - 1
            attacker = int(a)
            defender = int(d)
        else:
            print ('These armies cannot fight!\n')

    return attacker, defender


def get_probabilities():
    # dim 0: attacker number
    # dim 1: defender number
    # dim 2: attacker losses
    probabilities = [
        [
            [15/36, 21/36],
            [55/216, 161/216],
            [225/1296, 1071/1296]
        ],
        [
            [125/216, 91/216],
            [295/1296, 420/1296, 581/1296],
            [979/7776, 1981/7776, 4816/7776]
        ],
        [
            [855/1296, 441/1296],
            [2890/7776, 2611/7776, 2275/7776],
            [6420/46656, 10017/46656, 12348/46656, 17871/46656]
        ]
    ]

    return probabilities


def solve(attacker, defender):
    probabilities = get_probabilities()

    matrixQ = np.zeros((attacker * defender, attacker * defender))

    if defender > 1:
        matrixQ[1][0] = probabilities[0][1][0]
    if defender > 2:
        matrixQ[2][1] = probabilities[0][2][0]
    if attacker > 1:
        matrixQ[defender][0] = probabilities[1][0][1]
    if attacker > 1 and defender > 1:
        matrixQ[defender + 1][0] = probabilities[1][1][1]
    if attacker > 1 and defender > 2:
        matrixQ[defender + 2][defender] = probabilities[1][2][0]
    if attacker > 1 and defender > 2:
        matrixQ[defender + 2][1] = probabilities[1][2][1]
    if attacker > 2:
        matrixQ[2 * defender][defender] = probabilities[2][0][1]
    if attacker > 2 and defender > 1:
        matrixQ[2 * defender + 1][defender] = probabilities[2][1][1]
    if attacker > 2 and defender > 1:
        matrixQ[2 * defender + 1][1] = probabilities[2][1][2]
    if attacker > 2 and defender > 2:
        matrixQ[2 * defender + 2][defender] = probabilities[2][2][1]
    if attacker > 2 and defender > 2:
        matrixQ[2 * defender + 2][1] = probabilities[2][2][2]

    for i in range(attacker - 1, -1, -1):
        for j in range(defender - 1, -1, -1):
            if i >= 3 and j >= 3:
                matrixQ[i * defender + j][i * defender + j - 3] = probabilities[2][2][0]
            if i >= 3 and j >= 3:
                matrixQ[i * defender + j][(i - 1) * defender + j - 2] = probabilities[2][2][1]
            if i >= 3 and j >= 3:
                matrixQ[i * defender + j][(i - 2) * defender + j - 1] = probabilities[2][2][2]
            if i >= 3 and j >= 3:
                matrixQ[i * defender + j][(i - 3) * defender + j] = probabilities[2][2][3]
            if i >= 3 and j == 0:
                matrixQ[i * defender + j][(i - 1) * defender + j] = probabilities[2][0][1]
            if i >= 3 and j == 1:
                matrixQ[i * defender + j][(i - 2) * defender + j] = probabilities[2][1][2]
            if i >= 3 and j == 1:
                matrixQ[i * defender + j][(i - 1) * defender + j - 1] = probabilities[2][1][1]
            if i >= 3 and j == 2:
                matrixQ[i * defender + j][(i - 3) * defender + j] = probabilities[2][2][3]
            if i >= 3 and j == 2:
                matrixQ[i * defender + j][(i - 2) * defender + j - 1] = probabilities[2][2][2]
            if i >= 3 and j == 2:
                matrixQ[i * defender + j][(i - 1) * defender + j - 2] = probabilities[2][2][1]
            if j >= 3 and i == 0:
                matrixQ[i * defender + j][i * defender + j - 1] = probabilities[0][2][0]
            if j >= 3 and i == 1:
                matrixQ[i * defender + j][i * defender + j - 2] = probabilities[1][2][0]
            if j >= 3 and i == 1:
                matrixQ[i * defender + j][(i - 1) * defender + j - 1] = probabilities[1][2][1]
            if j >= 3 and i == 2:
                matrixQ[i * defender + j][i * defender + j - 3] = probabilities[2][2][0]
            if j >= 3 and i == 2:
                matrixQ[i * defender + j][(i - 1) * defender + j - 2] = probabilities[2][2][2]
            if j >= 3 and i == 2:
                matrixQ[i * defender + j][(i - 2) * defender + j - 1] = probabilities[2][2][1]

    matrixR = np.zeros((attacker * defender, attacker + defender))
    
    matrixR[0][0] = probabilities[0][0][1]
    matrixR[0][defender] = probabilities[0][0][0]

    if defender > 1:
        matrixR[1][1] = probabilities[0][1][1]
    if defender > 2:
        matrixR[2][2] = probabilities[0][2][1]
    if attacker > 1:
        matrixR[defender][defender + 1] = probabilities[1][0][0]
    if attacker > 1 and defender > 1:
        matrixR[defender + 1][defender + 1] = probabilities[1][1][0]
    if attacker > 1 and defender > 1:
        matrixR[defender + 1][1] = probabilities[1][1][2]
    if attacker > 1 and defender > 2:
        matrixR[defender + 2][2] = probabilities[1][2][2]
    if attacker > 2:
        matrixR[2 * defender][defender + 2] = probabilities[2][0][0]
    if attacker > 2 and defender > 1:
        matrixR[2 * defender + 1][defender + 2] = probabilities[2][1][0]
    if attacker > 2 and defender > 2:
        matrixR[2 * defender + 2][defender + 2] = probabilities[2][2][0]
    if attacker > 2 and defender > 2:
        matrixR[2 * defender + 2][2] = probabilities[2][2][3]

    for i in range(attacker - 1, -1, -1):
        for j in range(defender - 1, -1, -1):
            if i >= 3 and j == 0:
                matrixR[i * defender + j][defender + i] = probabilities[2][0][0]
            if i >= 3 and j == 1:
                matrixR[i * defender + j][defender + i] = probabilities[2][1][0]
            if i >= 3 and j == 2:
                matrixR[i * defender + j][defender + i] = probabilities[2][2][0]
            if j >= 3 and i == 0:
                matrixR[i * defender + j][j] = probabilities[0][2][1]
            if j >= 3 and i == 1:
                matrixR[i * defender + j][j] = probabilities[1][2][2]
            if j >= 3 and i == 2:
                matrixR[i * defender + j][j] = probabilities[2][2][3]

    matrixF = np.linalg.inv(np.identity(attacker * defender) - matrixQ) @ matrixR

    # test if matrixF is well-formed
    # print(all(math.isclose(sum(row), 1) for row in matrixF))

    # sparsity ratios
    # print(len(matrixQ[matrixQ == 0]) / len(matrixQ.flatten()))
    # print(len(matrixR[matrixR == 0]) / len(matrixR.flatten()))
    # print(len(matrixF[matrixF == 0]) / len(matrixF.flatten()))

    # sparsity matrices
    # sparsityQ = copy.deepcopy(matrixQ)
    # sparsityQ[sparsityQ != 0] = 1
    # plot_heatmap(sparsityQ)

    # sparsityR = copy.deepcopy(matrixR)
    # sparsityR[sparsityR != 0] = 1
    # plot_heatmap(sparsityR)

    # sparsityF = copy.deepcopy(matrixF)
    # sparsityF[sparsityF != 0] = 1
    # plot_heatmap(sparsityF)

    winA = sum(matrixF[attacker * defender - 1][defender + i] for i in range(attacker))
    # winD = 1 - winA

    pm = np.array([sum(matrixF[attacker * defender - 1 - j][defender + i] for i in range(attacker)) for j in range(attacker * defender)]).reshape((attacker, defender))
    pm = np.rot90(pm, k=2)

    # winA = pm[attacker - 1][defender - 1]

    # losses_evs = list(reversed([1 - matrixF[attacker * defender - 1][defender + i] for i in range(attacker)]))
    # print(losses_evs)
    # plt.hist(losses_evs, bins=losses_evs, density=True)
    # plt.fill_between(list(range(attacker)), losses_evs, step='post')
    # plt.show()

    losses = sum(1 - (i + 1) * matrixF[attacker * defender - 1][defender + i] for i in range(attacker)) # can sum losses_evs

    # evm = np.array([attacker - sum((i + 1) * matrixF[attacker * defender - 1 - j][defender + i] for i in range(attacker)) for j in range(attacker * defender)]).reshape((attacker, defender))
    # evm = np.rot90(evm, k=2)

    # evm_norm = evm / attacker

    return winA, losses, pm


def plot_heatmap(heatmap):
    plt.imshow(heatmap, cmap='hot', origin='lower')
    # plt.gcf().canvas.set_window_title('win probability')
    plt.xlabel('defender')
    plt.ylabel('attacker')
    plt.colorbar().set_label('win probability')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A program that, given the armies size of the attacker and the defender, calculates the probability of winning the battle and the expected losses of the attacker using a Markov stochastic process.')
    parser.add_argument('-p', '--plot', default=False, action='store_true', help='plot the heatmaps')
    args = parser.parse_args()

    # print ('WARNING: 1 attacking tank is always considered to stay still\n')
    attacker, defender = read()
    p, ev, pm = solve(attacker, defender)
    print ('Win probability: %.2f' % (p * 100) + '%')
    print ('Attacker expected losses: %.2f' % ev)

    if args.plot:
        plot_heatmap(pm)
