from mobile_robot.optimization import create_ocp_solver
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver


# # Create Acados solver
# acados_solver = create_ocp_solver()
#
# # Solve the optimal control problem
# status = acados_solver.solve()
#
# # Check the solver status
# if status != 0:
#     print("Acados solver failed with status:", status)
# else:
#     print("Acados solver successfully solved the optimal control problem.")
#
# # Get the optimal solution
# u_opt = acados_solver.get(0, "u")
#
# # Print the optimal control input
# print("Optimal control input:")
# print(u_opt)


# Define the initial conditions
x0 = np.array([0, 0, 0, 0])

# Create Acados solver
ocp, acados_solver = create_ocp_solver(x0=x0)

acados_solver.print_statistics()

# Access the dimensions from the AcadosOcp object
N = acados_solver.N
nx = ocp.dims.nx
nu = ocp.dims.nu

# Simulate the system
simX = np.zeros((N + 1, nx))
simU = np.zeros((N, nu))

xcurrent = x0
simX[0, :] = x0

# Initialize solver
for i in range(N):
    acados_solver.set(i,'x', simX[i, :])
    status = acados_solver.solve()

    if status != 0:
        print("Solver failed!")

    simU[i, :] = acados_solver.get(0, 'u')
    simX[i + 1, :] = acados_solver.get(1, 'x')


# Extract values
time_grid = np.linspace(0, 1, N + 1)
v_values = simX[:, 2]
theta_values = simX[:, 3]
a_values = simU[:, 0]
w_values = simU[:, 1]

# Plotting
plt.figure(figsize=(12, 8))
plt.subplot(4, 1, 1)
plt.plot(time_grid, v_values, label='v')
plt.legend()
plt.subplot(4, 1, 2)
plt.plot(time_grid, theta_values, label='theta')
plt.legend()
plt.subplot(4, 1, 3)
plt.plot(time_grid[:-1], a_values, label='a')
plt.legend()
plt.subplot(4, 1, 4)
plt.plot(time_grid[:-1], w_values, label='w')
plt.legend()

plt.xlabel('Time (s)')
plt.tight_layout()
plt.show()
