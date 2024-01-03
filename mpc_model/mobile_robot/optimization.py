import casadi as ca
import numpy as np
import scipy.linalg
from .model import mobile_robot_model
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver


def trajectory_cost(q, u, p, r, weights):
    """
    Compute the cost function for trajectory optimization.

    Parameters:
    - q: Output vector of the current trajectory
    - u: Input vector of the current trajectory
    - p: Desired output vector (reference trajectory)
    - r: Desired input vector (reference input)
    - weights: List of weights [w1, w2, w3] for each term in the cost function

    Returns:
    - J: Cost value
    """

    # Extracting individual components of the state and control vectors
    q_N = q[-1]
    p_N = p[-1]

    # Define the cost function components
    J_a = weights[0] * (q_N - p_N) ** 2
    J_b = ca.sum1(weights[1] * (q[:-1] - p[:-1]) ** 2)
    J_c = ca.sum1(weights[2] * (u - r) ** 2)

    # Total cost function
    J_1 = J_a + J_b + J_c

    return J_1


def obstacle_avoidance_cost(x, y, x_obst, y_obst, w):
    """
    Compute the obstacle avoidance cost function.

    Parameters:
    - x: Current x-coordinate of the robot
    - y: Current y-coordinate of the robot
    - x_obst: x-coordinate of the obstacle
    - y_obst: y-coordinate of the obstacle
    - w: Weight parameter

    Returns:
    - J_2: Obstacle avoidance cost value
    """

    # Define obstacle parameters
    b = 0.4

    # Calculate distance term 'd' in the obstacle avoidance cost
    d = ((x - x_obst) ** 2 / b ** 2) + ((y - y_obst) ** 2 / b ** 2)

    # Compute the obstacle avoidance cost function component
    J_2 = (ca.pi / 2) + ca.atan(w - d * w)

    return J_2


def create_ocp_solver():
    """
    Create Acados solver for trajectory optimization.

    Parameters:
    - model: AcadosModel object representing the robot model.
    - x0: Initial Conditions (x,y,v,theta).
    - a: Control vector (acceleration)
    - w: Control vector (angular velocity)
    - weights: Cost weight of obstacles

    Returns:
    - ocp: AcadosOcp object representing the optimal control problem.
    - solver: AcadosOcpSolver object representing the solver for the optimal control problem.
    """

    # Create AcadosOcp object
    ocp = AcadosOcp()

    # Set up the optimization problem
    model = mobile_robot_model()
    ocp.model = model

    # constants
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ny = nx + nu
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
    u_ref = np.zeros(nu)

    # Set initial condition for the robot
    ocp.constraints.x0 = x_ref

    # initialize parameters
    ocp.dims.np = n_params
    ocp.parameter_values = np.zeros(n_params)

    # Set Cost

    ocp.cost.yref = np.concatenate((x_ref, u_ref))
    ocp.cost.yref_e = x_ref

    # Q & R are diagonal cost matrices
    diagonal_Q = np.array([0, 0, 0, 0])
    diagonal_R = np.array([0.0, 0.0])
    Q = np.diag(diagonal_Q)  # dim same as nx
    R = np.diag(diagonal_R)  # dim same as nu

    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'
    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = Q
    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx, :nx] = np.eye(nx)
    ocp.cost.Vu = np.zeros((ny, nu))
    ocp.cost.Vu[-nu:, -nu:] = np.eye(nu)
    ocp.cost.Vx_e = np.eye(nx)

    # Define constraints on states and control inputs
    ocp.constraints.lbu = np.array([-0.1, -0.3])  # Lower bounds on control inputs
    ocp.constraints.ubu = np.array([0.1, 0.3])    # Upper bounds on control inputs
    ocp.constraints.lu = np.array([100, 100, 1, 10])  # Upper bounds on states
    ocp.constraints.idxbu = np.array([0, 1])  # need to fix

    # solver options
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    # explicit Runge-Kutta integrator
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.print_level = 0
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'

    # Set up Acados solver
    acados_solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')
    acados_integrator = AcadosSimSolver(ocp, json_file='acados_ocp.json')

    return ocp, acados_solver, acados_integrator


# Need to find cost
# Need to fix trajectory when no obstacles
