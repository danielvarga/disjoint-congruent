import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt



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


def pretty_congruence(t):
    x, y, rot, mirr = t
    return f"x={x} y={y} rot={90 * rot} refl={'yes' if mirr else 'no'}"


def standard_translations():
    return [(0, 0, 0, False),
        (0*1, 2, 2, False),
        (0*1, 0, 2, False)]

    assert n % 3 == 0
    t1 = (0, 0, 0, False)
    t2 = (0, n // 3, 0, False)
    t3 = (0, 2 * n // 3, 0, False)
    ts = [t1, t2, t3]
    return ts


def random_congruence(n, use_rotation=True, use_reflection=True):
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


def prefilter_congruence(n, t, k):
    w1 = np.ones((n, n), dtype=int)
    translate, zero_sum = apply(w1, t)
    survives = translate.sum() * k >= w1.sum()
    return survives


# takes a list of congruences, return a pair (density, solution)
# where density is the ratio of the square that can be covered by (soft) disjoint congruent
# versions of some (soft) subset of the square.
# the congruent versions must stay inside the square.
def evaluate_congruences(w, ts):
    n = len(w)
    constraints =  [w[i, j] >= 0 for i in range(n) for j in range(n)]
    constraints += [w[i, j] <= 1 for i in range(n) for j in range(n)]

    translates_and_zero_sums = [apply(w, t) for t in ts]
    translates = np.array([translates_and_zero_sum[0] for translates_and_zero_sum in translates_and_zero_sums])
    zero_sums =  np.array([translates_and_zero_sum[1] for translates_and_zero_sum in translates_and_zero_sums])
    s = translates.sum(axis=0)
    constraints += [zero_sum == 0 for zero_sum in zero_sums]
    constraints += [s[i, j] <= 1 for i in range(n) for j in range(n)]
    lp = cp.Problem(
        cp.Maximize(np.sum(s)),
        constraints=constraints
    )
    lp.solve(verbose=False)
    solution = np.array([[w[i, j].value for j in range(n)] for i in range(n)])
    return lp.value / n ** 2, solution


def random_congruences(n, k):
    ts = [(0, 0, 0, False)] # the identity is always in there
    while len(ts) < k:
        t = random_congruence(n, use_reflection=False)
        if prefilter_congruence(n, t, k):
            ts.append(t)
    return ts



def visualize_solution(w, ts):
    density, solution = evaluate_congruences(w, ts)

    print("the disjoint congruent versions can cover at most this ratio of the square:", density)
    for t in ts:
        print(pretty_congruence(t))

    n = len(w)
    k = len(ts)
    fig, axs = plt.subplots(2, k, figsize=(4 * k, 8))
    fig.suptitle(f"Density {density:.3f} partial cover\nWarning: the bottom row gradients are not parts of the set.")
    axs[0, 0].imshow(solution, vmin=0)
    axs[0, 0].set_title("Optimal solution")

    translates_and_zero_sums = [apply(solution, t) for t in ts]
    translates = np.array([translates_and_zero_sum[0] for translates_and_zero_sum in translates_and_zero_sums])

    s = np.array(translates).sum(axis=0)

    axs[0, 1].imshow(s, vmin=0)
    axs[0, 1].set_title("Partial cover")


    lin = np.linspace(0, 0.5, n)
    xa, xb = np.meshgrid(lin, lin)
    w_rainbow = xa + xb

    for column, t in enumerate(ts):
        translate_rainbow, zero_sum = apply(w_rainbow, t)
        translate_solution, zero_sum = apply(solution, t)
        ax = axs[1, column]
        ax.imshow(translate_solution + translate_rainbow, vmin=0)
        ax.set_title(pretty_congruence(t))

    plt.show()


def main():
    np.random.seed(2)

    # the size of the grid
    n = 12
    # the number of congruences
    k = 3

    w = [[cp.Variable(name=f"w[{i},{j}]") for j in range(n)] for i in range(n)]
    w = np.array(w, dtype=object)

    best_density = 0
    best_ts = None
    iteration = 0
    while True:
        ts = random_congruences(n, k)
        iteration += 1

        density, solution = evaluate_congruences(w, ts)
        if density > best_density:
            best_density = density
            best_ts = ts
        if iteration % 10 == 0:
            print(f"attempt {iteration} current best density {best_density}")
        if iteration >= 20:
            break

    print(f"finishing after {iteration} tries.")

    visualize_solution(w, best_ts)


if __name__ == "__main__":
    main()
