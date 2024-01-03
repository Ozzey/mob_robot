import numpy as np
import timeit
from mobile_robot.optimization import create_ocp_solver
from draw import Draw_MPC_point_stabilization_v1


def simulation(model):
    N = 100
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    x0 = np.zeros(nx)
    xs = np.array([5, 5, 0, 0])

    simX = np.zeros((N + 1, nx))
    simU = np.zeros((N, nu))
    x_current = x0
    simX[0, :] = x0.reshape(1, -1)
    xs_between = np.concatenate((xs, np.zeros(2)))
    time_record = np.zeros(N)

    ocp, solver, integrator = create_ocp_solver()

    # closed loop
    solver.set(N, 'yref', xs)
    for i in range(N):
        solver.set(i, 'yref', xs_between)

    for i in range(N):
        # solve ocp
        start = timeit.default_timer()
        # set inertial (stage 0)
        solver.set(0, 'lbx', x_current)
        solver.set(0, 'ubx', x_current)
        status = solver.solve()

        if status != 0:
            raise Exception('acados acados_ocp_solver returned status {}. Exiting.'.format(status))

        simU[i, :] = solver.get(0, 'u')
        time_record[i] = timeit.default_timer() - start
        # simulate system
        integrator.set('x', x_current)
        integrator.set('u', simU[i, :])

        status_s = integrator.solve()
        if status_s != 0:
            raise Exception('acados integrator returned status {}. Exiting.'.format(status))

        # update
        x_current = integrator.get('x')
        simX[i + 1, :] = x_current

    print("-----------------------------------------------------")
    print("average estimation time :", format(time_record.mean()))
    print("Resulting Cost : ", solver.get_cost())
    print("-----------------------------------------------------")
    Draw_MPC_point_stabilization_v1(rob_diam=0.3, init_state=x0, target_state=xs, robot_states=simX, )


