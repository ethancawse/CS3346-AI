# Movement actions
ACTIONS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}

# Slip model: with prob 0.8 go intended, 0.1 left, 0.1 right
SLIP_LEFT_RIGHT = {
    "U": ("L", "R"),
    "D": ("R", "L"),
    "L": ("D", "U"),
    "R": ("U", "D"),
}

TERMINAL_REWARD = 10.0  # terminal cells always have reward 10


def is_wall(cell):
    return cell == "##"


def is_numeric(cell):
    return isinstance(cell, (int, float))


def is_terminal(cell):
    return is_numeric(cell) and float(cell) == TERMINAL_REWARD


def in_bounds(grid, i, j):
    return 0 <= i < len(grid) and 0 <= j < len(grid[0])

# Transition
def next_state(grid, i, j, action):
    di, dj = ACTIONS[action]
    # Apply tansition
    ni, nj = i + di, j + dj
    # Stay in place if hit or oob
    if not in_bounds(grid, ni, nj) or is_wall(grid[ni][nj]):
        return i, j
    return ni, nj


def get_reward(cell):
    if is_numeric(cell):
        return float(cell)
    return 0.0


def compute_q_value(grid, V, i, j, action, gamma, p_forward=0.8, p_left=0.1, p_right=0.1):
    cell = grid[i][j]
    # No actions from terminal or wall
    if is_wall(cell) or is_terminal(cell):
        return V[i][j]
    # R(s)
    R = get_reward(cell)
    # Get possible transition states
    left_a, right_a = SLIP_LEFT_RIGHT[action]
    s_f = next_state(grid, i, j, action)
    s_l = next_state(grid, i, j, left_a)
    s_r = next_state(grid, i, j, right_a)
    # Sum[T(s,a,s') * Vk(s')]
    expected_value = (
        p_forward * V[s_f[0]][s_f[1]] +
        p_left    * V[s_l[0]][s_l[1]] +
        p_right   * V[s_r[0]][s_r[1]]
    )

    return R + gamma * expected_value


def value_iteration(grid, gamma=0.9, theta=0.01, p_forward=0.8, p_left=0.1, p_right=0.1, max_iterations=10000):
    rows, cols = len(grid), len(grid[0])

    # Initialize value function
    V = [[0.0 for i in range(cols)] for i in range(rows)]
    for i in range(rows):
        for j in range(cols):
            cell = grid[i][j]
            if is_wall(cell):
                V[i][j] = None  # just a marker for walls
            elif is_terminal(cell):
                V[i][j] = get_reward(cell)  # 10
            else:
                V[i][j] = 0.0  # initial guess

    # Value iteration loop
    for i in range(max_iterations):
        delta = 0.0
        new_V = [row[:] for row in V]

        for i in range(rows):
            for j in range(cols):
                cell = grid[i][j]

                if is_wall(cell) or is_terminal(cell):
                    continue
                # Compute q values
                best_q = float("-inf")
                for a in ACTIONS:
                    q = compute_q_value(grid, V, i, j, a, gamma, p_forward, p_left, p_right)
                    if q > best_q:
                        best_q = q

                new_V[i][j] = best_q
                # Convergence
                delta = max(delta, abs(best_q - V[i][j]))
        V = new_V
        if delta < theta:
            break

    # Extract greedy policy
    policy = [[None for i in range(cols)] for i in range(rows)]
    for i in range(rows):
        for j in range(cols):
            cell = grid[i][j]

            if is_wall(cell):
                policy[i][j] = "Wall"
                continue

            if is_terminal(cell):
                policy[i][j] = "Goal"
                continue

            best_a = None
            best_q = float("-inf")
            # Find largest q value
            for a in ACTIONS:
                q = compute_q_value(grid, V, i, j, a, gamma, p_forward, p_left, p_right)
                if q > best_q:
                    best_q = q
                    best_a = a

            policy[i][j] = best_a
    # Return convegece v values and policy
    return V, policy


def arrows_from_policy(policy):
    mapping = {
        "U": "↑",
        "D": "↓",
        "L": "←",
        "R": "→",
        "Wall": "#",
        "Goal": "G"
    }
    return [[mapping[a] for a in row] for row in policy]


# ---------------- Usage testing----------------
if __name__ == "__main__":
    grid = [
        ["##", -0.0000000001, -0.01, "##"],
        [0.0, -0.04, "##", 10.0],
        ["##", -0.04, -0.04, -0.04],
    ]
    # One from assign 2
    grid1 = [
        [0, -0.04, 10],
        [-0.04, "##", -0.04], 
        [-0.04, -0.04, -10],      

    ]
    big_grid = [
    # row 0 (top)
    [-0.20, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, 10.0],
    # row 1
    [-0.20,  "##", -1.00,  "##", -0.20, -1.00, -0.20,  "##", -0.20, -0.10],
    # row 2
    [-0.10, -0.01, -0.01, -0.01, -0.01, -0.01, -0.20, -0.01, -0.01, -0.01],
    # row 3
    [-0.10, -0.20, -0.20,  "##", -0.20, -1.00, -0.20,  "##", -0.20, -0.04],
    # row 4
    [-0.10, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.04],
    # row 5
    [-0.10, -0.01,  "##", -0.01, -0.10, -0.10, -0.04, -1.00, -0.20, -0.04],
    # row 6
    [-0.01, -0.01,  "##", -0.01, -0.20,  "##", -0.04, -1.00, -0.20, -0.04],
    # row 7 (bottom)
    [ 0.0 , -0.01, -0.01, -0.01, -0.20, -0.20, -0.04, -0.04, -0.04, -0.04],
    ]
    grid_bad_shortcut = [
    [0.0,  -50.0, -50.0, 10.0],
    [-0.1,  "##",  "##", -0.1],
    [-0.1,  "##",  "##", -0.1],
    [-0.1, -0.1, -0.1, -0.1],
    ]


    convergence, policy = value_iteration(big_grid)

    print("Convergence Values:")
    for row in convergence:
        print(["##" if v is None else round(v, 3) for v in row])
    print("\nPolicy:")
    print(policy)
    print()
    for row in arrows_from_policy(policy):
        print(row)

