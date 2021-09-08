# from functools import cached_property
from typing import Dict
import numpy as np


class RISIKO:

    # @cached_property
    @classmethod
    def P(cls) -> np.ndarray:
        P = np.zeros((3, 3, 4))
        P[0, 0, 0] = 15 / 36
        P[0, 0, 1] = 21 / 36
        P[0, 1, 0] = 55 / 216
        P[0, 1, 1] = 161 / 216
        P[0, 2, 0] = 225 / 1296
        P[0, 2, 1] = 1071 / 1296
        P[1, 0, 0] = 125 / 216
        P[1, 0, 1] = 91 / 216
        P[1, 1, 0] = 295 / 1296
        P[1, 1, 1] = 420 / 1296
        P[1, 1, 2] = 581 / 1296
        P[1, 2, 0] = 979 / 7776
        P[1, 2, 1] = 1981 / 7776
        P[1, 2, 2] = 4816 / 7776
        P[2, 0, 0] = 855 / 1296
        P[2, 0, 1] = 441 / 1296
        P[2, 1, 0] = 2890 / 7776
        P[2, 1, 1] = 2611 / 7776
        P[2, 1, 2] = 2275 / 7776
        P[2, 2, 0] = 6420 / 46656
        P[2, 2, 1] = 10017 / 46656
        P[2, 2, 2] = 12348 / 46656
        P[2, 2, 3] = 17871 / 46656
        return P

    @classmethod
    def Q(cls, attacker: int, defender: int) -> np.ndarray:
        P = cls.P()
        Q = np.zeros((attacker * defender, attacker * defender))

        if defender > 1:
            Q[1, 0] = P[0, 1, 0]
        if defender > 2:
            Q[2, 1] = P[0, 2, 0]
        if attacker > 1:
            Q[defender, 0] = P[1, 0, 1]
        if attacker > 1 and defender > 1:
            Q[defender + 1, 0] = P[1, 1, 1]
        if attacker > 1 and defender > 2:
            Q[defender + 2, defender] = P[1, 2, 0]
        if attacker > 1 and defender > 2:
            Q[defender + 2, 1] = P[1, 2, 1]
        if attacker > 2:
            Q[2 * defender, defender] = P[2, 0, 1]
        if attacker > 2 and defender > 1:
            Q[2 * defender + 1, defender] = P[2, 1, 1]
        if attacker > 2 and defender > 1:
            Q[2 * defender + 1, 1] = P[2, 1, 2]
        if attacker > 2 and defender > 2:
            Q[2 * defender + 2, defender] = P[2, 2, 1]
        if attacker > 2 and defender > 2:
            Q[2 * defender + 2, 1] = P[2, 2, 2]

        for i in range(attacker - 1, -1, -1):
            for j in range(defender - 1, -1, -1):
                if i >= 3 and j >= 3:
                    Q[i * defender + j, i * defender + j - 3] = P[2, 2, 0]
                if i >= 3 and j >= 3:
                    Q[i * defender + j, (i - 1) * defender + j - 2] = P[2, 2, 1]
                if i >= 3 and j >= 3:
                    Q[i * defender + j, (i - 2) * defender + j - 1] = P[2, 2, 2]
                if i >= 3 and j >= 3:
                    Q[i * defender + j, (i - 3) * defender + j] = P[2, 2, 3]
                if i >= 3 and j == 0:
                    Q[i * defender + j, (i - 1) * defender + j] = P[2, 0, 1]
                if i >= 3 and j == 1:
                    Q[i * defender + j, (i - 2) * defender + j] = P[2, 1, 2]
                if i >= 3 and j == 1:
                    Q[i * defender + j, (i - 1) * defender + j - 1] = P[2, 1, 1]
                if i >= 3 and j == 2:
                    Q[i * defender + j, (i - 3) * defender + j] = P[2, 2, 3]
                if i >= 3 and j == 2:
                    Q[i * defender + j, (i - 2) * defender + j - 1] = P[2, 2, 2]
                if i >= 3 and j == 2:
                    Q[i * defender + j, (i - 1) * defender + j - 2] = P[2, 2, 1]
                if j >= 3 and i == 0:
                    Q[i * defender + j, i * defender + j - 1] = P[0, 2, 0]
                if j >= 3 and i == 1:
                    Q[i * defender + j, i * defender + j - 2] = P[1, 2, 0]
                if j >= 3 and i == 1:
                    Q[i * defender + j, (i - 1) * defender + j - 1] = P[1, 2, 1]
                if j >= 3 and i == 2:
                    Q[i * defender + j, i * defender + j - 3] = P[2, 2, 0]
                if j >= 3 and i == 2:
                    Q[i * defender + j, (i - 1) * defender + j - 2] = P[2, 2, 2]
                if j >= 3 and i == 2:
                    Q[i * defender + j, (i - 2) * defender + j - 1] = P[2, 2, 1]

        return Q

    @classmethod
    def R(cls, attacker: int, defender: int) -> np.ndarray:
        P = cls.P()
        R = np.zeros((attacker * defender, attacker + defender))

        R[0, 0] = P[0, 0, 1]
        R[0, defender] = P[0, 0, 0]

        if defender > 1:
            R[1, 1] = P[0, 1, 1]
        if defender > 2:
            R[2, 2] = P[0, 2, 1]
        if attacker > 1:
            R[defender, defender + 1] = P[1, 0, 0]
        if attacker > 1 and defender > 1:
            R[defender + 1, defender + 1] = P[1, 1, 0]
        if attacker > 1 and defender > 1:
            R[defender + 1, 1] = P[1, 1, 2]
        if attacker > 1 and defender > 2:
            R[defender + 2, 2] = P[1, 2, 2]
        if attacker > 2:
            R[2 * defender, defender + 2] = P[2, 0, 0]
        if attacker > 2 and defender > 1:
            R[2 * defender + 1, defender + 2] = P[2, 1, 0]
        if attacker > 2 and defender > 2:
            R[2 * defender + 2, defender + 2] = P[2, 2, 0]
        if attacker > 2 and defender > 2:
            R[2 * defender + 2, 2] = P[2, 2, 3]

        for i in range(attacker - 1, -1, -1):
            for j in range(defender - 1, -1, -1):
                if i >= 3 and j == 0:
                    R[i * defender + j, defender + i] = P[2, 0, 0]
                if i >= 3 and j == 1:
                    R[i * defender + j, defender + i] = P[2, 1, 0]
                if i >= 3 and j == 2:
                    R[i * defender + j, defender + i] = P[2, 2, 0]
                if j >= 3 and i == 0:
                    R[i * defender + j, j] = P[0, 2, 1]
                if j >= 3 and i == 1:
                    R[i * defender + j, j] = P[1, 2, 2]
                if j >= 3 and i == 2:
                    R[i * defender + j, j] = P[2, 2, 3]

        return R

    @classmethod
    def F(cls, attacker: int, defender: int) -> np.ndarray:
        Q = cls.Q(attacker, defender)
        R = cls.R(attacker, defender)
        F = np.linalg.inv(np.eye(attacker * defender) - Q) @ R
        return F

    @classmethod
    def solve(cls, attacker: int, defender: int) -> Dict[str, float]:
        F = cls.F(attacker, defender)

        win_probability = np.sum(F[-1, defender:])
        remaining = np.concatenate(
            [np.ones(defender) * attacker, np.arange(attacker, 0, -1)], axis=0
        )
        expected_losses = np.sum(F[-1] * remaining)
        var_losses = np.sum(F[-1] * remaining ** 2) - expected_losses ** 2
        std_losses = np.sqrt(var_losses)
        win_probability_matrix = np.sum(F[:, defender:], axis=1).reshape(
            attacker, defender
        )

        results = {
            "win_probability": win_probability,
            "expected_losses": expected_losses,
            "std_losses": std_losses,
            "win_probability_matrix": win_probability_matrix,
            'F': F
        }

        return results
