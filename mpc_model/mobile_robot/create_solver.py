import casadi as ca
import numpy as np
import scipy.linalg
from .robot_model import mobile_robot_model
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver


def trajectory_cost():
    cost = 0
    return cost


def obstacle_cost(X, w=1.0):
    # Add the custom obstacle avoidance cost
    b = 0.4
    x_obst1, y_obst1 = 3.5, 3.5
    x_obst2, y_obst2 = 1.1, 0.6

    # Define the cost function using CasADi functions
    dist_obst1 = (X[0] - x_obst1) ** 2 + (X[1] - y_obst1) ** 2
    dist_obst2 = (X[0] - x_obst2) ** 2 + (X[1] - y_obst2) ** 2

    J2 = ca.pi / 2 + ca.arctan(w - (dist_obst1 / b) ** 2) + ca.pi / 2 + ca.arctan(w - (dist_obst2 / b) ** 2)

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
    N = 100
    n_params = len(model.p)

    # Setting initial conditions
    ocp.dims.N = N
    ocp.dims.nx = nx
    ocp.dims.nu = nu
    ocp.solver_options.tf = T

    # initial state
    x_ref = np.zeros(nx)

    # Set initial condition for the robot
    ocp.constraints.x0 = x_ref

    # initialize parameters
    ocp.dims.np = n_params
    ocp.parameter_values = np.zeros(n_params)

    # ---------------------CONSTRAINTS------------------
    # Define constraints on states and control inputs
    ocp.constraints.lbu = np.array([-0.1, -0.3])  # Lower bounds on control inputs
    ocp.constraints.ubu = np.array([0.1, 0.3])    # Upper bounds on control inputs
    ocp.constraints.lu = np.array([100, 100, 1, 10])  # Upper bounds on states
    ocp.constraints.idxbu = np.array([0, 1])  # for indices 0 & 1

    # ---------------------COSTS--------------------------

    # Define J1 and J2
    # Define cost_y_expr, cost_y_expr_e
    # Define W and W_e
    # Define yref and yref_e

    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'

    X = ocp.model.x
    U = ocp.model.u

    # Cost weights
    w1 = 1.0  # adjust as needed
    w2 = 1.0  # adjust as needed
    w3 = 1.0  # adjust as needed
    w = 1.0

    J2 = obstacle_cost(X, w)

    # Parameters:
    ocp.model.cost_y_expr = J2
    ocp.model.cost_y_expr_e = J2

    ocp.cost.W = np.eye(1)
    ocp.cost.W_e = np.eye(1)

    ocp.cost.yref = np.array([0])
    ocp.cost.yref_e = np.array([0])

    # ---------------------SOLVER-------------------------
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'

    # Set up Acados solver
    acados_solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')
    acados_integrator = AcadosSimSolver(ocp, json_file='acados_ocp.json')

    return ocp, acados_solver, acados_integrator


# Need to find cost
# Need to set desired Linear Velocity = 0.3
# Need to Visualize
