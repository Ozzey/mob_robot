from mobile_robot.create_solver import create_ocp_solver
from mobile_robot.robot_model import mobile_robot_model
from visualization.utils import simulationV1

import numpy as np
import matplotlib.pyplot as plt


def main():
    # Create Acados solver
    ocp, solver, integrator = create_ocp_solver()

    model = mobile_robot_model()

    simulationV1(model)


if __name__ == "__main__":
    main()




