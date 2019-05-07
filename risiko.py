import numpy as np
import re

def read():
    attacker = 0
    defender = 0
    pattern = re.compile('^[0-9]+$')

    while attacker == 0 or defender == 0:
        a = input('Attacker armies: ')
        d = input('Defender armies: ')

        if pattern.match(a) and pattern.match(d) and  int(a) >= 2 and int(d) >= 1:
            attacker = int(a) - 1
            defender = int(d)
        else:
            print ('These armies cannot fight!\n')

    return attacker, defender

def solve(attacker, defender):
    matrixQ = np.zeros((attacker * defender, attacker * defender))

    if defender > 1:
        matrixQ[1][0] = 0.25462962962962965
    if defender > 2:
        matrixQ[2][1] = 0.1736111111111111
    if attacker > 1:
        matrixQ[defender][0] = 0.4212962962962963
    if attacker > 1 and defender > 1:
        matrixQ[defender + 1][0] = 0.32407407407407407
    if attacker > 1 and defender > 2:
        matrixQ[defender + 2][defender] = 0.12590020576131689
    if attacker > 1 and defender > 2:
        matrixQ[defender + 2][1] = 0.25475823045267487
    if attacker > 2:
        matrixQ[2 * defender][defender] = 0.3402777777777778
    if attacker > 2 and defender > 1:
        matrixQ[2 * defender + 1][defender] = 0.3357767489711934
    if attacker > 2 and defender > 1:
        matrixQ[2 * defender + 1][1] = 0.2925668724279835
    if attacker > 2 and defender > 2:
        matrixQ[2 * defender + 2][defender] = 0.21469907407407407
    if attacker > 2 and defender > 2:
        matrixQ[2 * defender + 2][1] = 0.2646604938271605

    for i in range(attacker - 1, -1, -1):
        for j in range(defender - 1, -1, -1):
            if i >= 3 and j >= 3:
                matrixQ[i * defender + j][i * defender + j - 3] = 0.1376028806584362
            if i >= 3 and j >= 3:
                matrixQ[i * defender + j][(i - 1) * defender + j - 2] = 0.21469907407407407
            if i >= 3 and j >= 3:
                matrixQ[i * defender + j][(i - 2) * defender + j - 1] = 0.2646604938271605
            if i >= 3 and j >= 3:
                matrixQ[i * defender + j][(i - 3) * defender + j] = 0.3830375514403292
            if i >= 3 and j == 0:
                matrixQ[i * defender + j][(i - 1) * defender + j] = 0.3402777777777778
            if i >= 3 and j == 1:
                matrixQ[i * defender + j][(i - 2) * defender + j] = 0.2925668724279835
            if i >= 3 and j == 1:
                matrixQ[i * defender + j][(i - 1) * defender + j - 1] = 0.3357767489711934
            if i >= 3 and j == 2:
                matrixQ[i * defender + j][(i - 3) * defender + j] = 0.3830375514403292
            if i >= 3 and j == 2:
                matrixQ[i * defender + j][(i - 2) * defender + j - 1] = 0.2646604938271605
            if i >= 3 and j == 2:
                matrixQ[i * defender + j][(i - 1) * defender + j - 2] = 0.21469907407407407
            if j >= 3 and i == 0:
                matrixQ[i * defender + j][i * defender + j - 1] = 0.1736111111111111
            if j >= 3 and i == 1:
                matrixQ[i * defender + j][i * defender + j - 2] = 0.12590020576131689
            if j >= 3 and i == 1:
                matrixQ[i * defender + j][(i - 1) * defender + j - 1] = 0.25475823045267487
            if j >= 3 and i == 2:
                matrixQ[i * defender + j][i * defender + j - 3] = 0.1376028806584362
            if j >= 3 and i == 2:
                matrixQ[i * defender + j][(i - 1) * defender + j - 2] = 0.2646604938271605
            if j >= 3 and i == 2:
                matrixQ[i * defender + j][(i - 2) * defender + j - 1] = 0.21469907407407407

    matrixR = np.zeros((attacker * defender, attacker + defender))
    
    matrixR[0][0] = 0.5833333333333334
    matrixR[0][defender] = 0.4166666666666667

    if defender > 1:
        matrixR[1][1] = 0.7453703703703703
    if defender > 2:
        matrixR[2][2] = 0.8263888888888888
    if attacker > 1:
        matrixR[defender][defender + 1] = 0.5787037037037037
    if attacker > 1 and defender > 1:
        matrixR[defender + 1][defender + 1] = 0.22762345679012347
    if attacker > 1 and defender > 1:
        matrixR[defender + 1][1] = 0.44830246913580246
    if attacker > 1 and defender > 2:
        matrixR[defender + 2][2] = 0.6193415637860082
    if attacker > 2:
        matrixR[2 * defender][defender + 2] = 0.6597222222222222
    if attacker > 2 and defender > 1:
        matrixR[2 * defender + 1][defender + 2] = 0.37165637860082307
    if attacker > 2 and defender > 2:
        matrixR[2 * defender + 2][defender + 2] = 0.1376028806584362
    if attacker > 2 and defender > 2:
        matrixR[2 * defender + 2][2] = 0.3830375514403292

    for i in range(attacker - 1, -1, -1):
        for j in range(defender - 1, -1, -1):
            if i >= 3 and j == 0:
                matrixR[i * defender + j][defender + i] = 0.6597222222222222
            if i >= 3 and j == 1:
                matrixR[i * defender + j][defender + i] = 0.37165637860082307
            if i >= 3 and j == 2:
                matrixR[i * defender + j][defender + i] = 0.1376028806584362
            if j >= 3 and i == 0:
                matrixR[i * defender + j][j] = 0.8263888888888888
            if j >= 3 and i == 1:
                matrixR[i * defender + j][j] = 0.6193415637860082
            if j >= 3 and i == 2:
                matrixR[i * defender + j][j] = 0.3830375514403292
    
    matrixF = np.linalg.inv(np.subtract(np.identity(attacker * defender), matrixQ)) @ matrixR

    winD = sum(matrixF[attacker * defender - 1][i] for i in range(0, defender))
    winA = (1 - winD) * 100

    remaining = sum((i + 1) * matrixF[attacker * defender - 1][i + defender] for i in range(0, attacker))
    losses = attacker - remaining

    return winA, losses

if __name__ == '__main__':
    print ('WARNING: 1 attacking tank is always considered to stay still\n')
    attacker, defender = read()
    p, ev = solve(attacker, defender)
    print ('Win probability: %.2f' % p + '%')
    print ('Attacker expected losses: %.2f' % ev)