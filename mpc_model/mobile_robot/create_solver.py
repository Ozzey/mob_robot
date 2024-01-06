import casadi as ca
import numpy as np
from .robot_model import mobile_robot_model
from acados_template import AcadosOcp, AcadosOcpSolver


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
    N = 30  # Prediction horizon
    T = 5  # Total time for trajectory (assuming a constant velocity of 0.3 m/s)
    dt = T / N  # Time step

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
    ocp.constraints.uu = np.array([100, 100, 1, 10])  # Upper bounds on states
    ocp.constraints.uu = np.array([-100, -100, -1, -10])  # Upper bounds on states
    ocp.constraints.idxbu = np.array([0, 1])  # for indices 0 & 1

    # ---------------------COSTS--------------------------
    # Set up the cost function
    # Set up the cost function
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"

    X = ocp.model.x
    U = ocp.model.u

    # Reference State & Control
    x_ref = ca.vertcat(5, 5, 0.3, 0)
    u_ref = ca.vertcat(0, 0)

    # Difference between Initial and Real State & Control
    e_x = X - x_ref
    e_u = U - u_ref

    # Difference in Terminal Case
    e_x_N = X - ca.vertcat(5, 5, 0, 0)

    ocp.model.cost_y_expr = e_x.T @ e_x + e_u.T @ e_u
    ocp.model.cost_y_expr_e = e_x_N.T @ e_x_N

    ocp.cost.W = np.eye(1)  # State weights
    ocp.cost.W_e = np.eye(1)  # Terminal state weights

    ocp.cost.yref = np.zeros(1)
    ocp.cost.yref_e = np.zeros(1)

    # ---------------------SOLVER-------------------------
    ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'

    # Set up Acados solver
    acados_solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')
    return ocp, acados_solver


# Need to debug
# Need to check optimal path

