from mobile_robot.optimization import create_ocp_solver
from mobile_robot.model import mobile_robot_model
from utils import simulation

model = mobile_robot_model()
simulation(model)



