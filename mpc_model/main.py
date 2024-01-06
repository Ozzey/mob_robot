from mobile_robot.create_solver import create_ocp_solver, desired_trajectory
import numpy as np
import matplotlib.pyplot as plt
import timeit
import matplotlib
matplotlib.use("TkAgg")

# --------------PARAMETERS-----------
# Define target and obstacle positions
x_target, y_target = 5, 5
x_obst1, y_obst1 = 3.5, 3.5
x_obst2, y_obst2 = 1.1, 0.6

# Create optimal control problem solver
ocp, solver = create_ocp_solver()

nx = ocp.model.x.size()[0]
nu = ocp.model.u.size()[0]
N = ocp.dims.N

# ----------------INITIALIZE-------------------
# Set initial state and parameter values
x0 = np.array([0, 0, 0, 0])  # initial position and orientation
p = np.array([])  # target and obstacle positions

solver.set(0, "x", x0)
solver.set(0, "p", p)

# Get desired trajectory
P, R = desired_trajectory(N)

# Initialize Optimal Trajectory
x_opt = np.zeros((ocp.dims.N + 1, nx))
u_opt = np.zeros((ocp.dims.N, nu))

# Set yref for each stage in the prediction horizon
for i in range(N):
    # Concatenate state and control references
    yref = np.concatenate((P[i], R[i]))
    solver.set(i, "yref", yref)

# Set yref for the terminal stage
ocp.cost.yref_e = P[N]

start = timeit.default_timer()

# ---------------------SOLVE-----------------

# Solve optimal control problem
status = solver.solve()
if status != 0:
    raise Exception("Solver failed!")
time_record = timeit.default_timer() - start

solver.print_statistics()

for i in range(ocp.dims.N+1):
    x_opt[i, :] = solver.get(i, "x")
    print(i, "State:", x_opt[i])
    if i < ocp.dims.N:
        u_opt[i, :] = solver.get(i, "u")

# ----------------OUTPUTS------------------

print("---------------------------")
print("Cost : ", solver.get_cost())
print("Time: ", format(time_record))
print("---------------------------")


# ---------------------PLOT---------------------------
# Plot reference path, obstacle, and optimal trajectory
plt.figure()
plt.plot([0, x_target], [0, y_target], "k--", label="Reference path")
plt.plot(x_obst1, y_obst1, "ro", label="Obstacle 1")
plt.plot(x_obst2, y_obst2, "ro", label="Obstacle 2")
plt.plot(x_opt[:, 0], x_opt[:, 1], "b-", label="Optimal trajectory")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()

# New code for additional graphs
time_values = np.arange(0, N+1)

plt.figure()

plt.subplot(2, 2, 1)
plt.plot(time_values, x_opt[:, 2], "g-", label="Velocity")
plt.xlabel("Time")
plt.ylabel("Velocity")
plt.legend()
plt.grid()

plt.subplot(2, 2, 2)
plt.plot(time_values, x_opt[:, 3], "m-", label="Angle")
plt.xlabel("Time")
plt.ylabel("Theta")
plt.legend()
plt.grid()

plt.subplot(2, 2, 3)
plt.plot(time_values[:N], u_opt[:, 0], "c-", label="Acceleration")
plt.xlabel("Time")
plt.ylabel("Acceleration")
plt.legend()
plt.grid()

plt.subplot(2, 2, 4)
plt.plot(time_values[:N], u_opt[:, 1], "y-", label="Angular Velocity")
plt.xlabel("Time")
plt.ylabel("Angular Velocity")
plt.legend()
plt.grid()

plt.tight_layout()  # Adjust the layout to prevent overlap
plt.show()
