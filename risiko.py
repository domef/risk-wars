import numpy as np
import re
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import torch
import math

#TODO
# plot 3d
# fix gui graph
# line args
# input loop?
# README.md

def read():
    """
    """

    attacker = 0
    defender = 0
    pattern = re.compile('^[0-9]+$')

    while attacker == 0 or defender == 0:
        a = input('Attacker armies: ')
        d = input('Defender armies: ')

        # if pattern.match(a) and pattern.match(d) and  int(a) >= 2 and int(d) >= 1:
        if pattern.match(a) and pattern.match(d) and  int(a) >= 1 and int(d) >= 1:
            # attacker = int(a) - 1
            attacker = int(a)
            defender = int(d)
        else:
            print ('These armies cannot fight!\n')

    return attacker, defender

def get_probabilities():
    """
    """
    # dim 0: attackers number
    # dim 1: defenders num
    # dim 2: attackers losses

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
    """
    """

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

    # dev = torch.device('cuda')

    # pytorch
    # tensorQ = torch.from_numpy(matrixQ).to(dev)
    # tensorR = torch.from_numpy(matrixR).to(dev)
    # temp1 = torch.eye(attacker * defender, dtype=float).to(dev) - tensorQ
    # temp2 = torch.inverse(temp1).to(dev)
    # tensorF = torch.mm(temp2, tensorR).cpu  ()
    # matrixF = tensorF.numpy()

    #cupy
    # matrixQ = cp.array(matrixQ)
    # matrixR = cp.array(matrixR)
    # matrixF = cp.matmul(cp.linalg.inv(cp.identity(attacker * defender) - matrixQ), matrixR)
    # matrixF = cp.asnumpy(matrixF)

    #numpy
    matrixF = np.linalg.inv(np.identity(attacker * defender) - matrixQ) @ matrixR

    # test if matrixF is well-formed
    print(all(math.isclose(sum(row), 1) for row in matrixF))

    winA = sum(matrixF[attacker * defender - 1][i] for i in range(attacker, attacker + defender))
    # winD = (1 - winA) * 100

    pm = np.array([1 - sum(matrixF[attacker * defender - 1 - j][i] for i in range(defender)) for j in range(attacker * defender)]).reshape((attacker, defender))
    pm = np.rot90(pm, k=2)

    remaining = sum((i + 1) * matrixF[attacker * defender - 1][i + defender] for i in range(attacker))
    losses = attacker - remaining

    evm = np.array([attacker - sum((i + 1) * matrixF[attacker * defender - 1 - j][i + defender] for i in range(attacker)) for j in range(attacker * defender)]).reshape((attacker, defender))
    evm = np.rot90(evm, k=2)

    # evm_norm = evm / attacker

    return winA, losses, pm, evm

if __name__ == '__main__':
    # print ('WARNING: 1 attacking tank is always considered to stay still\n')
    attacker, defender = read()
    p, ev, pm, evm = solve(attacker, defender)
    print ('Win probability: %.2f' % (p * 100) + '%')
    print ('Attacker expected losses: %.2f' % ev)

    plt.imshow(pm, cmap='hot', origin='lower')
    plt.gcf().canvas.set_window_title('win probability')
    # plt.title('win probability')
    plt.xlabel('defenders')
    plt.ylabel('attackers')
    clb = plt.colorbar()
    clb.set_label('win probability')    
    plt.show()

    plt.imshow(evm, cmap='hot')
    plt.gcf().canvas.set_window_title('losses expected value')
    # plt.title('losses expected value')
    plt.xlabel('defenders')
    plt.ylabel('attackers')
    clb = plt.colorbar()
    clb.set_label('losses expected value')
    plt.show()

    # n = 10

    # def fun(x, y):
    #     p, ev = solve(x, y)
    #     return p
    #     # return ev

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # x = y = np.array(list(range(1, n)))
    # X, Y = np.meshgrid(x, y)

    # for x, y in zip(np.ravel(X), np.ravel(Y)):
    #     print(str(x) + ' ' + str(y))

    # zs = np.array([fun(x, y) / 100 for x, y in zip(np.ravel(X), np.ravel(Y))])
    # zs = np.array(fun(np.ravel(X), np.ravel(Y)))
    # Z = 

    # ax.plot_surface(X, Y, Z)

    # ax.set_xlabel('attackers')
    # ax.set_ylabel('defenders')
    # ax.set_zlabel('win probability')
    # ax.set_zlabel('expected value')

    # plt.show()

    # for interp in ['none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']:
    #     print(interp)
    #     plt.imshow(Z, cmap='hot', interpolation=interp)
    #     plt.show()