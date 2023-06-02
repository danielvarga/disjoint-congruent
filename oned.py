import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt



def apply(w, t):
    w = w.copy()
    x, mirr = t
    if mirr:
        w = w[::-1]
    if x == 0:
        return w, 0
    if x < 0:
        p = -x
        zeroed_region = w[: p].sum()
        w[: -p] = w[p: ]
        w[-p :] = 0
    else:
        p = x
        zeroed_region = w[-p :].sum()
        w[p :] = w[: -p]
        w[: p] = 0
    return w, zeroed_region


def pretty(w):
    n = len(w)
    for i in range(n):
        if w[i] is cp.Variable:
            print(w[i].name, end="\t")
        else:
            print(w[i], end="\t")
    print()


def pretty_congruence(t):
    x, mirr = t
    return f"x={x} refl={'yes' if mirr else 'no'}"


def random_congruence(n, use_reflection=True):
    x = np.random.randint(-n , n)
    if use_reflection:
        mirror = bool(np.random.randint(2))
    else:
        mirror = False
    return x, mirror


def prefilter_congruence(n, t, k):
    w1 = np.ones((n, ), dtype=int)
    translate, zero_sum = apply(w1, t)
    survives = translate.sum() * k >= w1.sum()
    return survives


# takes a list of congruences, return a pair (density, solution)
# where density is the ratio of the square that can be covered by (soft) disjoint congruent
# versions of some (soft) subset of the square.
# the congruent versions must stay inside the square.
def evaluate_congruences(w, ts):
    n = len(w)
    constraints =  [w[i] >= 0 for i in range(n)]
    constraints += [w[i] <= 1 for i in range(n)]

    translates_and_zero_sums = [apply(w, t) for t in ts]
    translates = np.array([translates_and_zero_sum[0] for translates_and_zero_sum in translates_and_zero_sums])
    zero_sums =  np.array([translates_and_zero_sum[1] for translates_and_zero_sum in translates_and_zero_sums])

    s = translates.sum(axis=0)
    constraints += [zero_sum == 0 for zero_sum in zero_sums]
    constraints += [s[i] <= 1 for i in range(n)]
    lp = cp.Problem(
        cp.Maximize(np.sum(s)),
        constraints=constraints
    )
    lp.solve(verbose=False)
    solution = np.array([w[i].value for i in range(n)])
    return lp.value / n, solution


def random_congruences(n, k):
    ts = [(0, False)] # the identity is always in there
    while len(ts) < k:
        t = random_congruence(n, use_reflection=False)
        if prefilter_congruence(n, t, k):
            ts.append(t)
    return ts


def two_d(w):
    width = len(w) // 5 + 1
    return np.tile(w.reshape(-1, 1), (1, width)).T


def visualize_solution(w, ts):
    density, solution = evaluate_congruences(w, ts)

    print("the disjoint congruent versions can cover at most this ratio of the square:", density)
    for t in ts:
        print(pretty_congruence(t))

    n = len(w)
    k = len(ts)
    fig, axs = plt.subplots(2, k, figsize=(4 * k, 8))
    fig.suptitle(f"Density {density:.3f} partial cover\nWarning: the bottom row gradients are not parts of the set.")
    axs[0, 0].imshow(two_d(solution), vmin=0)
    axs[0, 0].set_title("Optimal solution")

    translates_and_zero_sums = [apply(solution, t) for t in ts]
    translates = np.array([translates_and_zero_sum[0] for translates_and_zero_sum in translates_and_zero_sums])

    s = np.array(translates).sum(axis=0)

    axs[0, 1].imshow(two_d(s), vmin=0)
    axs[0, 1].set_title("Partial cover")

    w_rainbow = np.linspace(0, 0.3, n)

    for column, t in enumerate(ts):
        translate_rainbow, zero_sum = apply(w_rainbow, t)
        translate_solution, zero_sum = apply(solution, t)
        ax = axs[1, column]
        vis = translate_solution + translate_rainbow
        ax.imshow(two_d(vis), vmin=0)
        ax.set_title(pretty_congruence(t))

    plt.show()


def main():
    np.random.seed(2)

    # the size of the grid
    n = 120
    # the number of congruences
    k = 3

    w = [cp.Variable(name=f"w[{i}]") for i in range(n)]
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


def grid_1d():
    n = 120
    w = [cp.Variable(name=f"w[{i}]") for i in range(n)]
    w = np.array(w, dtype=object)
    xs = np.arange(n) + 1
    ys = []
    for x in xs:
        ts = [(0, False), (x, False)]
        density, solution = evaluate_congruences(w, ts)
        ys.append(density)
        if x % 10 == 0:
            print(x)
    plt.plot(xs, ys)
    plt.scatter(xs, ys, marker='.')
    plt.show()


def grid_2d():
    n = 24
    w = [cp.Variable(name=f"w[{i}]") for i in range(n)]
    w = np.array(w, dtype=object)
    results = np.empty((n, n))
    results[:, :] = np.nan
    for x in range(1, n):
        for y in range(1, n):
            ts = [(0, False), (x, False), (y, False)]
            density, solution = evaluate_congruences(w, ts)
            results[x, y] = density
            if np.isclose(density, 1) and x < y:
                print(x, y)
        print(x)
    plt.imshow(results)
    plt.show()


if __name__ == "__main__":
    grid_2d() ; exit()
    grid_1d() ; exit()
    main() ; exit()
