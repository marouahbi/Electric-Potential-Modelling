import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Grid setup and parameters
def initialize_grid(N, boundary_value_long=10, boundary_value_short=0):
    """
    Initialize the grid with boundary conditions:
    - Long sides have a fixed potential (e.g., 10V)
    - Short sides have a zero potential
    """
    V = np.zeros((N, N))
    V[:, 0] = boundary_value_long  # Left boundary
    V[:, -1] = boundary_value_long  # Right boundary
    return V


def place_charge(V, x, y, charge_value):
    """
    Place a charge at a specified location in the grid.
    - This creates a non-zero electric potential that influences the entire field.
    """
    V[x, y] = charge_value
    return V


# Jacobi Method
def jacobi_method(V, tolerance, max_iterations=10000):
    N = V.shape[0]
    for iteration in range(max_iterations):
        V_new = V.copy()
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                V_new[i, j] = 0.25 * (V[i + 1, j] + V[i - 1, j] + V[i, j + 1] + V[i, j - 1])

        # Convergence check: sum of absolute differences in grid values
        if np.sum(np.abs(V_new - V)) < tolerance:
            break
        V = V_new
    return V


# Gauss-Seidel Method
def gauss_seidel_method(V, tolerance, max_iterations=10000):
    N = V.shape[0]
    for iteration in range(max_iterations):
        max_diff = 0
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                old_value = V[i, j]
                V[i, j] = 0.25 * (V[i + 1, j] + V[i - 1, j] + V[i, j + 1] + V[i, j - 1])
                max_diff = max(max_diff, abs(V[i, j] - old_value))

        # Convergence check based on max change
        if max_diff < tolerance:
            break
    return V


# Successive Over-Relaxation (SOR) Method
def sor_method(V, tolerance, omega=1.5, max_iterations=10000):
    N = V.shape[0]
    for iteration in range(max_iterations):
        max_diff = 0
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                old_value = V[i, j]
                V[i, j] = (1 - omega) * V[i, j] + omega * 0.25 * (V[i + 1, j] + V[i - 1, j] + V[i, j + 1] + V[i, j - 1])
                max_diff = max(max_diff, abs(V[i, j] - old_value))

        # Convergence based on max difference threshold
        if max_diff < tolerance:
            break
    return V


# Function to plot the potential distribution
def plot_potential(V, method_name):
    """
    Plot the electric potential distribution.
    - In electrostatics, potential maps give insights into the electric field.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(range(V.shape[0]), range(V.shape[1]))
    ax.plot_surface(X, Y, V, cmap='viridis')
    ax.set_title(f"Potential Distribution ({method_name})")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Potential (V)')
    plt.show()


# Main script
N = 50  # Grid size
tolerance = 1e-5
charge_value = 100  # Arbitrary potential representing a point charge in the center

# Initialize grid and place charge
V = initialize_grid(N)
V = place_charge(V, N // 2, N // 2, charge_value)

# Run Jacobi Method
V_jacobi = jacobi_method(V.copy(), tolerance)
plot_potential(V_jacobi, "Jacobi Method")

# Run Gauss-Seidel Method
V_gauss_seidel = gauss_seidel_method(V.copy(), tolerance)
plot_potential(V_gauss_seidel, "Gauss-Seidel Method")

# Run SOR Method
V_sor = sor_method(V.copy(), tolerance, omega=1.5)
plot_potential(V_sor, "SOR Method")

# Analysis of grid spacing effect
for h in [0.1, 0.05, 0.025]:  # Smaller h increases grid resolution
    N = int(1 / h)
    V = initialize_grid(N)
    V = place_charge(V, N // 2, N // 2, charge_value)
    V_jacobi_h = jacobi_method(V.copy(), tolerance)
    plot_potential(V_jacobi_h, f"Jacobi Method with h={h}")
    # Add similar analysis for Gauss-Seidel and SOR if desired
