import matplotlib.pyplot as plt
import numpy as np


def generate_trajectory(ocp_solver, with_obstacles=False):
    # Solve the optimization problem
    status = ocp_solver.solve()

    if status != 0:
        print(f"Error in solving the optimization problem. Status: {status}")
        return None

    # Get the optimal solution
    opt_trajectory = ocp_solver.get_optimal_dynamics()

    if opt_trajectory is not None:
        # Extract the trajectory
        x_opt = opt_trajectory['x']
        u_opt = opt_trajectory['u']

        # Time vector
        t_grid = np.linspace(0, ocp_solver.ocp.dims.N, ocp_solver.ocp.dims.N + 1)

        # Plot trajectory
        plt.figure(figsize=(10, 6))
        plt.plot(t_grid, x_opt[:, 0], label='x')
        plt.plot(t_grid, x_opt[:, 1], label='y')
        plt.xlabel('Time')
        plt.ylabel('Position')
        plt.title('Optimal Trajectory')

        if with_obstacles:
            # Plot obstacle locations
            plt.scatter([3.5, 1.1], [3.5, 0.6], color='red', marker='X', label='Obstacles')

        plt.legend()
        plt.grid(True)
        plt.show()
