import numpy as np
import cvxpy as cp


n = 12


def apply(w, t):
    x, y, rot, mirr = t
    if mirr:
        w = w[::-1, :]
    w = np.rot90(w, rot)
    return translate_array(w, x, y)


# chatgpt generated, except for zeroed_region
# prompt: i'd like to translate an array by (x,y) integer coordinates,
# such that values that exit the original area are dropped.
# the resulting array should have the same shape as the original.
def translate_array(arr, x, y):
    # Get the shape of the original array
    rows, cols = arr.shape

    # Calculate the new starting indices
    start_row = max(0, y)
    start_col = max(0, x)

    # Calculate the new ending indices
    end_row = min(rows, rows + y)
    end_col = min(cols, cols + x)

    # Translate the array and extract the desired region
    translated = arr[start_row:end_row, start_col:end_col]

    # this is not chatgpt. zeroed_region == 0 will become a constraint:
    zeroed_region = np.sum(arr) - np.sum(translated)

    # Create a new array with the same shape as the original
    translated_array = np.zeros_like(arr)

    # Calculate the indices for placing the translated region in the new array
    translated_start_row = max(0, -y)
    translated_end_row = translated_start_row + (end_row - start_row)
    translated_start_col = max(0, -x)
    translated_end_col = translated_start_col + (end_col - start_col)

    # Place the translated region in the new array
    translated_array[translated_start_row:translated_end_row,
                     translated_start_col:translated_end_col] = translated

    return translated_array, zeroed_region


def test_translate_array():
    arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
    translated_arr = translate_array(arr, 1, -2)
    print(translated_arr)


def pretty(w):
    for i in range(n):
        for j in range(n):
            if w[i, j] is cp.Variable:
                print(w[i, j].name, end="\t")
            else:
                print(w[i, j], end="\t")
        print()


def standard_translations():
    assert n % 3 == 0
    t1 = (0, 0, 0, False)
    t2 = (0, n // 3, 0, False)
    t3 = (0, 2 * n // 3, 0, False)
    ts = [t1, t2, t3]
    return ts


def random_congruence(use_rotation=True, use_reflection=True):
    x = np.random.randint(-n , n)
    y = np.random.randint(-n , n)
    if use_rotation:
        rot = np.random.randint(4)
    else:
        rot = 0
    if use_reflection:
        mirror = bool(np.random.randint(2))
    else:
        mirror = False
    return x, y, rot, mirror


w = [[cp.Variable(name=f"w[{i},{j}]") for j in range(n)] for i in range(n)]
w = np.array(w, dtype=object)


# takes a list of congruences, return a pair (density, solution)
# where density is the ratio of the square that can be covered by (soft) disjoint congruent
# versions of some (soft) subset of the square.
# the congruent versions must stay inside the square.
def evaluate_congruences(ts):
    constraints =  [w[i, j] >= 0 for i in range(n) for j in range(n)]
    constraints += [w[i, j] <= 1 for i in range(n) for j in range(n)]

    translates_and_zero_sums = [apply(w, t) for t in ts]
    translates = np.array([translates_and_zero_sum[0] for translates_and_zero_sum in translates_and_zero_sums])
    zero_sums =  np.array([translates_and_zero_sum[1] for translates_and_zero_sum in translates_and_zero_sums])
    s = translates.sum(axis=0)
    constraints += [s[i, j] <= 1 for i in range(n) for j in range(n)]
    constraints += [zero_sum == 0 for zero_sum in zero_sums]

    lp = cp.Problem(
        cp.Maximize(np.sum(s)),
        constraints=constraints
    )
    lp.solve(verbose=False)
    solution = np.array([[w[i, j].value for j in range(n)] for i in range(n)])
    return lp.value / n ** 2, solution


np.random.seed(2)

best = 0
for _ in range(10000):
    ts = [random_congruence(use_reflection=False) for _ in range(3)]
    # ts = standard_translations()
    density, solution = evaluate_congruences(ts)
    print(f"best {best:0.3f} current {density:0.2f}")
    if density > best:
        best = density
    if density > 0.6:
        break

print("the disjoint congruent versions can cover at most this ratio of the square:", density)

import matplotlib.pyplot as plt
plt.imshow(solution)
plt.show()

translates_and_zero_sums = [apply(solution, t) for t in ts]
translates = np.array([translates_and_zero_sum[0] for translates_and_zero_sum in translates_and_zero_sums])

s = np.array(translates).sum(axis=0)

plt.imshow(s)
plt.show()
