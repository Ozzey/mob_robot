import casadi as ca
import numpy as np
import scipy.linalg
from .model import mobile_robot_model
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver


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

    # Define constraints on states and control inputs
    ocp.constraints.lbu = np.array([-0.1, -0.3])  # Lower bounds on control inputs
    ocp.constraints.ubu = np.array([0.1, 0.3])    # Upper bounds on control inputs
    ocp.constraints.lu = np.array([100, 100, 1, 10])  # Upper bounds on states
    ocp.constraints.idxbu = np.array([0, 1])  # for indices 0 & 1

    # Set Cost
    ocp.cost.yref = np.concatenate((x_ref, u_ref))
    ocp.cost.yref_e = x_ref

    # Q & R are diagonal cost matrices
    diagonal_Q = np.array([1, 1, 1, 1])
    diagonal_R = np.array([0.5, 0.5])
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

    # solver options
    # ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    # ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    # ocp.solver_options.integrator_type = 'ERK'
    # ocp.solver_options.nlp_solver_type = 'SQP_RTI'

    # Set up Acados solver
    acados_solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')
    acados_integrator = AcadosSimSolver(ocp, json_file='acados_ocp.json')

    return ocp, acados_solver, acados_integrator


# Need to find cost
# Need to fix trajectory when no obstacles
# Need to set desired Linear Velocity = 0.3
