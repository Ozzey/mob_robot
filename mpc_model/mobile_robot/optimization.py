import casadi as ca
import numpy as np
from .model import mobile_robot_model
from acados_template import AcadosOcp, AcadosOcpSolver


def cost_function(q, u, p, r, weights):
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


def create_acados_solver(x0=ca.vertcat(0.0, 0.0, 0.0, 0.0),
                         a=0,
                         w=0,
                         weights=np.array([1.0, 1.0, 1.0])):
    """
    Create Acados solver for trajectory optimization.

    Parameters:
    - model: AcadosModel object representing the robot model.
    - x0: Initial Conditions (x,y,v,theta).
    - a: Control vector (acceleration)
    - w: Control vector (angular velocity)

    Returns:
    - ocp: AcadosOcp object representing the optimal control problem.
    - solver: AcadosOcpSolver object representing the solver for the optimal control problem.
    """

    # Create AcadosOcp object
    ocp = AcadosOcp()

    # Set up the optimization problem
    model = mobile_robot_model()
    ocp.model = model

    # Setting initial conditions
    ocp.solver_options.tf = 1.0
    ocp.dims.N = 30
    ocp.dims.nx = model.x.size()[0]
    ocp.dims.nu = model.u.size()[0]

    # Set Cost
    J_1 = 0
    J_2 = 0

    # Define the total cost function
    ocp.cost.cost_type = 'EXTERNAL'
    ocp.cost.cost_ext_cost = ca.vertcat(J_1, J_2)

    # Define constraints on states and control inputs
    ocp.constraints.lbu = np.array([-0.1, -0.3])  # Lower bounds on control inputs
    ocp.constraints.ubu = np.array([0.1, 0.3])    # Upper bounds on control inputs
    ocp.constraints.lu = np.array([100, 100, 1, 10])  # Upper bounds on states
    ocp.constraints.idxbu = np.array([0, 1]) # need to fix

    # Set initial condition for the robot
    ocp.constraints.x0 = np.array([0, 0, 0, 0])

    # Define the dynamics function using the continuous dynamics model
    #ocp.dyn_expr = model.continous_dynamics(x0, ca.vertcat(a, w))

    # Set up Acados solver
    acados_solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')

    # set options
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"  # 'GAUSS_NEWTON', 'EXACT'
    ocp.solver_options.integrator_type = "IRK"
    ocp.solver_options.nlp_solver_type = "SQP"  # SQP_RTI, SQP
    ocp.solver_options.nlp_solver_max_iter = 400

    return ocp, acados_solver


## Need to separate solver
## Need to fix error
## Need to find cost
