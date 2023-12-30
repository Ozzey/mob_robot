from mobile_robot.model import mobile_robot_model
import casadi as ca
from mobile_robot.optimization import create_acados_solver, cost_function, obstacle_avoidance_cost
from acados_template import AcadosOcp, AcadosOcpSolver


def main():

    # Define problem parameters

    # Create and solve the Acados optimal control problem
    ocp, solver = create_acados_solver()
    status = solver.solve()

    if status != 0:
        print(f"Solver failed with status {status}")
        return

    # Retrieve and print the optimal solution
    optimal_trajectory = solver.get("x")
    print("Optimal trajectory:")
    print(optimal_trajectory)


if __name__ == "__main__":
    main()
