import casadi as ca
import numpy as np
import scipy.linalg
from .robot_model import mobile_robot_model
from acados_template import AcadosOcp, AcadosOcpSolver


def trajectory_cost(Q, U):
    J2 = 0
    x_ref = ca.vertcat(5, 5, 0.3, 0)
    u_ref = ca.vertcat(0, 0)

    e_x = Q - x_ref
    e_u = U - u_ref

    # Add quadratic cost terms
    J2 += e_x.T @ e_x + e_u.T @ e_u

    return J2


def obstacle_cost(X, w=1.0):
    # Add the custom obstacle avoidance cost
    b = 0.4
    x_obst1, y_obst1 = 6, 6
    x_obst2, y_obst2 = 6, 6

    # Define the cost function using CasADi functions
    dist_obst1 = (X[0] - x_obst1) ** 2 + (X[1] - y_obst1) ** 2
    dist_obst2 = (X[0] - x_obst2) ** 2 + (X[1] - y_obst2) ** 2
    J2 = 0
    J2 += ca.arctan(w - (dist_obst1 / b) ** 2) + ca.pi / 2
    J2 += ca.arctan(w - (dist_obst2 / b) ** 2) + ca.pi / 2

    return J2


def create_ocp_solver():
    """
    Create Acados solver for trajectory optimization.

    Parameters:

    Returns:
    - ocp: AcadosOcp object representing the optimal control problem.
    - solver: AcadosOcpSolver object representing the solver for the optimal control problem.
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
    T = 30
    N = 30

    # Setting initial conditions
    ocp.dims.N = N
    ocp.dims.nx = nx
    ocp.dims.nu = nu
    ocp.solver_options.tf = T

    # Set initial condition for the robot
    ocp.constraints.x0 = np.zeros(4)

    # ---------------------CONSTRAINTS------------------
    # Define constraints on states and control inputs
    ocp.constraints.lbu = np.array([-0.1, -0.3])  # Lower bounds on control inputs
    ocp.constraints.ubu = np.array([0.1, 0.3])  # Upper bounds on control inputs
    ocp.constraints.lu = np.array([100, 100, 1, 10])  # Upper bounds on states
    ocp.constraints.idxbu = np.array([0, 1])  # for indices 0 & 1

    # ---------------------COSTS--------------------------
    # Set up the cost function
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    X = ocp.model.x
    U = ocp.model.u

    # Evaluate the cost functions with inputs 'a' and 'w'
    J1 = trajectory_cost(X, U)

    # Define the obstacle cost function J_2
    J2 = obstacle_cost(X)

    # Initial Cost
    J = J1 + J2
    ocp.model.cost_expr_ext_cost = J

    # Terminal Cost (Since Terminal cost cannot be dependent on U
    e_x_N = X - ca.vertcat(5, 5, 0, 0)
    J_e = e_x_N.T @ e_x_N
    ocp.model.cost_expr_ext_cost_e = J_e + J2

    # ---------------------SOLVER-------------------------
    ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'EXACT'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'

    # Set up Acados solver
    acados_solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')
    return ocp, acados_solver


# Need to debug
# Need to check optimal path
