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

    # v = np.full(N+1, 0.3)
    # theta = np.full(N+1, 0)

    v[N] = 0
    theta[N] = 0

    # Combine into p
    p = np.vstack((x, y, v, theta)).T

    a = np.zeros(len(x))  # Acceleration
    w = np.zeros(len(x))  # Angular velocity

    # Combine control variables into matrix r
    r = np.vstack((a, w)).T

    return p, r


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

    # Setting initial conditions
    ocp.dims.N = N
    ocp.dims.nx = nx
    ocp.dims.nu = nu

    # Set initial condition for the robot
    ocp.constraints.x0 = np.array([0, 0, 0, 0])

    # ---------------------CONSTRAINTS------------------
    # Define constraints on states and control inputs
    ocp.constraints.idxbu = np.array([0, 1])  # indices 0 & 1 of u
    ocp.constraints.idxbx = np.array([0, 1, 2, 3])  # indices 0...3 of x
    ocp.constraints.lbu = np.array([-0.1, -0.3])  # Lower bounds on control inputs
    ocp.constraints.ubu = np.array([0.1, 0.3])  # Upper bounds on control inputs
    ocp.constraints.lbx = np.array([-100, -100, 0, -10])  # Lower bounds on states
    ocp.constraints.ubx = np.array([100, 100, 1, 10])  # Upper bounds on states
    # ---------------------COSTS--------------------------
    # Set up the cost function
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"

    X = ocp.model.x
    U = ocp.model.u

    ocp.model.cost_y_expr = ca.vertcat(X, U)
    ocp.model.cost_y_expr_e = X

    ocp.cost.yref = np.zeros(nx + nu)
    ocp.cost.yref_e = np.zeros(nx)

    W_x = np.array([5, 5, 5, 1, 0.1, 0.1])
    W = np.diag(W_x)
    W_xe = np.array([50, 50, 10, 10])
    W_e = np.diag(W_xe)

    ocp.cost.W = W  # State weights
    ocp.cost.W_e = W_e  # Terminal state weights

    # ---------------------SOLVER-------------------------
    ocp.solver_options.tf = 50
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.qp_solver_cond_N = 1
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'IRK'
    ocp.solver_options.levenberg_marquardt = 1e-4
    ocp.solver_options.nlp_solver_max_iter = 30
    ocp.solver_options.qp_solver_iter_max = 100
    ocp.solver_options.nlp_solver_tol_stat = 1e2
    ocp.solver_options.nlp_solver_tol_eq = 1e-1

    # Set up Acados solver
    acados_solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')
    return ocp, acados_solver


# need to debug
# need to add obstacle cost
