import casadi as ca
import numpy as np
from .robot_model import mobile_robot_model
from acados_template import AcadosOcp, AcadosOcpSolver


def desired_trajectory(N):
    # Define parameters
    initial_position = (0, 0)
    final_position = (5, 5)

    # Linear interpolation for positions
    x = np.linspace(initial_position[0], final_position[0], N+1)
    y = np.linspace(initial_position[1], final_position[1], N+1)

    # Velocity and orientation
    v = np.zeros(len(x))
    theta = np.zeros(len(x))

    # Calculate v and theta for each pair of x and y
    for i in range(len(x)-1):
        theta[i] = np.arctan2(y[i], x[i])
        v[i] = np.sqrt(x[i] ** 2 + y[i] ** 2)
        if v[i] >= 0.3:
            v[i] = 0.3

    v[N] = 0
    theta[N] = 0
    # Combine into p
    p = np.vstack((x, y, v, theta)).T

    return p


def create_ocp_solver():
    """
    Create Acados solver for trajectory optimization.
    """

    # Create AcadosOcp object
    ocp = AcadosOcp()

    # Set up the optimization problem
    model = mobile_robot_model()
    ocp.model = model

    # --------------------PARAMETERS--------------
    # constants
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    N = 30  # Prediction horizon (works for N =200)
    T = 30  # Total time for trajectory (assuming a constant velocity of 0.3 m/s)

    # Setting initial conditions
    ocp.dims.N = N
    ocp.dims.nx = nx
    ocp.dims.nu = nu
    ocp.solver_options.tf = T

    # Set initial condition for the robot
    ocp.constraints.x0 = np.array([0, 0, 0, 0])

    # ---------------------CONSTRAINTS------------------
    # Define constraints on states and control inputs
    ocp.constraints.lbu = np.array([-0.1, -0.3])  # Lower bounds on control inputs
    ocp.constraints.ubu = np.array([0.1, 0.3])  # Upper bounds on control inputs
    ocp.constraints.ux = np.array([100, 100, 1, 10])  # Upper bounds on states
    ocp.constraints.lx = np.array([-100, -100, 0, -10])  # Lower bounds on states
    ocp.constraints.idxbu = np.array([0, 1])  # for indices 0 & 1 of u

    # ---------------------COSTS--------------------------
    # Set up the cost function
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"

    X = ocp.model.x

    ocp.model.cost_y_expr = X
    ocp.model.cost_y_expr_e = X

    ocp.cost.yref = np.zeros(nx)
    ocp.cost.yref_e = np.zeros(nx)

    ocp.cost.W = 10 * np.eye(nx)  # State weights
    ocp.cost.W_e = 100 * np.eye(nx)  # Terminal state weights

    # ---------------------SOLVER-------------------------
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'EXACT'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    # Set up Acados solver
    acados_solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')
    return ocp, acados_solver


# need to debug
# need to add obstacle cost
