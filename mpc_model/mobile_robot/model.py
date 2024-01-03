import casadi as ca
from acados_template import AcadosModel


def mobile_robot_model():
    """
    Define a simple mobile robot model.

    Returns:
        x_vec (MX): Symbolic vector representing the state [x, y, v, theta].
        u (MX): Symbolic vector representing the control input [a, w].
        continuous_dynamics (Function): CasADi function for the continuous-time dynamics.
            Takes state vector and control input vectors as input and returns
            the vector representing the rates of change of the state variables.
    """

    model_name = 'mobile_robot'

    # Define symbolic variables (states)
    x = ca.MX.sym('x')
    y = ca.MX.sym('y')
    v = ca.MX.sym('v')
    theta = ca.MX.sym('theta')

    # Control
    a = ca.MX.sym('a')  # acceleration
    w = ca.MX.sym('w')  # angular velocity

    # Define state and control vectors
    states = ca.vertcat(x, y, v, theta)
    controls = ca.vertcat(a, w)
    rhs = [v*ca.cos(theta), v*ca.sin(theta), a, w]
    x_dot = ca.MX.sym('x_dot', len(rhs))

    # Create a CasADi function for the continuous-time dynamics
    continuous_dynamics = ca.Function(
        'continuous_dynamics',
        [states, controls],
        [ca.vcat(rhs)],
        ["state", "control_input"],
        ["rhs"]
    )

    f_impl = x_dot - continuous_dynamics(states, controls)

    model = AcadosModel()

    model.f_expl_expr = continuous_dynamics(states, controls)
    model.f_impl_expr = f_impl
    model.x = states
    model.xdot = x_dot
    model.u = controls
    model.p = []
    model.name = model_name

    return model
