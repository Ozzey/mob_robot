import casadi as ca
from acados_template import AcadosModel


def mobile_robot_model():
    """
    Define a simple mobile robot model.
    """

    model_name = 'mobile_robot'

    # Define symbolic variables (states)
    x = ca.MX.sym('x')
    y = ca.MX.sym('y')
    v = ca.MX.sym('v')
    theta = ca.MX.sym('theta')

    x_dot = ca.MX.sym('x_dot')
    y_dot = ca.MX.sym('y_dot')
    theta_dot = ca.MX.sym('theta_dot')
    v_dot = ca.MX.sym('v_dot')

    # Control
    a = ca.MX.sym('a')  # acceleration
    w = ca.MX.sym('w')  # angular velocity

    # Define state and control vectors
    states = ca.vertcat(x, y, v, theta)
    controls = ca.vertcat(a, w)

    rhs = ca.vertcat(states[2] * ca.cos(states[3]),
                     states[2] * ca.sin(states[3]),
                     controls[0],
                     controls[1])

    x_dot = ca.vertcat(x_dot, y_dot, v_dot, theta_dot)

    f_impl = x_dot - rhs

    model = AcadosModel()

    model.f_expl_expr = rhs
    model.f_impl_expr = f_impl
    model.x = states
    model.xdot = x_dot
    model.u = controls
    model.p = []
    model.name = model_name

    return model
