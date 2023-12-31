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


    a = ca.MX.sym('a')  # acceleration
    w = ca.MX.sym('w')  # angular velocity

    # Define dynamics (xdot)
    x_dot = ca.MX.sym("x_dot")
    y_dot = ca.MX.sym("y_dot")
    v_dot = ca.MX.sym("v_dot")
    theta_dot = ca.MX.sym("theta_dot")

    # Define state and control vectors
    x = ca.vertcat(x, y, v, theta)
    xdot = ca.vertcat(x_dot, y_dot, v_dot, theta_dot)
    u = ca.vertcat(a, w)

    # Initialize Dynamics
    x_dot = v * ca.cos(theta)
    y_dot = v * ca.sin(theta)
    v_dot = a
    theta_dot = w

    # Define the continuous-time dynamics
    f_expl = ca.vertcat(x_dot, y_dot, v_dot, theta_dot)
    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_expl_expr = f_expl
    model.f_impl_expr = f_impl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name

    return model
