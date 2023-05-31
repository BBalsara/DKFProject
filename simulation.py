# code for simulation and running tests
# import utilities as util
import numpy as np
from utilities import Movement, Sensor, World
import matplotlib.pyplot as plt
# Simulation class



# Cameras (Position, FOV)
fov = np.radians(120)
bearing1 = np.radians(45) 
bearing2 = np.radians(225)
bearing3 = np.radians(135)
bearing4 = np.radians(180) 
position1 = np.array([0,0])
position2 = np.array([10,10])
position3 = np.array([15,0])
position4 = np.array([30,5])
angle_noise = 0.1
sensor1 = Sensor(position1, fov, bearing1, angle_noise)
sensor2 = Sensor(position2, fov, bearing2, angle_noise)
sensor3 = Sensor(position3, fov, bearing3, angle_noise)
sensor4 = Sensor(position4, fov, bearing4, angle_noise)
world = World(np.array([sensor1, sensor2, sensor3, sensor4]))
render, ax = world.FOVplot()


# EKF prediction
def predict(prev_mu, prev_cov, target: Movement, u):
    pred_mu = target.f_function(mu = prev_mu, u = u)
    A = target.jacobian_A(mu = prev_mu, u = u)
    pred_cov = A @ prev_cov @ A.T + target.Q
    return pred_mu, pred_cov

# EKF update
def update(target: Movement, pred_mu, pred_cov, world: World):
    # pull out sensor data
    sensors = [sensor for sensor in world.sensors if sensor.is_visible(target.state)]
    sensor_count = len(sensors)
    true_state = target.state
    sensor_noises = np.zeros(sensor_count) # for storing noise values of sensors with target in FOV
    C = np.zeros((sensor_count, len(true_state))) # for storing jacobian values of sensors with target in FOV
    y = np.zeros(sensor_count) # for storing true measurements of sensors with target in FOV
    g = np.zeros(sensor_count) # for storing predicted measurements of sensors with target in FOV
    for idx, sensor in enumerate(sensors):
        C[idx,:] = sensor.jacobian_C(pred_mu)
        sensor_noises[idx] = sensor.angle_noise
        y[idx] = sensor.angle_meas(true_state)
        g[idx] = sensor.g(pred_mu)
    if sensor_count == 0:
        return pred_mu, pred_cov
    else:
        Kt = pred_cov @ C.T @ np.linalg.inv(C @ pred_cov @ C.T + np.diag(sensor_noises))
        mu = pred_mu + Kt @ (y-g)
        cov = pred_cov - Kt @ C @ pred_cov 

    return mu, cov


Tmax = 40
d_t = 0.01
mu = np.zeros((3, int(Tmax/d_t)))
cov = np.zeros((3, 3, int(Tmax/d_t)))
mu0 = np.array([4.5,0,0])
cov0 = 0.01*np.eye(3)
mu[:, 0] = mu0
cov[:, :, 0] = cov0

trueInit = np.array([5,0,0])
trueState = np.zeros((3, int(Tmax/d_t)))
trueState[:, 0] = trueInit

# Initialize the object
Object_a = Movement(Movement_pattern='diff_drive', del_t=d_t, initial_location=trueInit)


for t in range(1, int(Tmax/d_t)):
    # Simulation
    prev_mu = mu[:, t-1]
    prev_cov = cov[:, :, t-1]

    # Calculate u if needed
    u = np.array([0.8, np.sin(t*d_t)])

    # Update true State
    x = Object_a.one_step(u)
    # print("x: ", x)
    trueState[:, t] = x
    
    # Predict step
    pred_mu, pred_cov = predict(prev_mu, prev_cov, Object_a, u)

    # Update step
    updated_mu, updated_cov = update(Object_a, pred_mu, pred_cov, world)

    # Store values
    mu[:, t] = updated_mu
    cov[:, :, t] = updated_cov
    
# Plotting
timeArray = np.arange(0,Tmax,d_t)
plt.figure()
plt.plot(timeArray,trueState[0,:], label='true px')
plt.plot(timeArray,mu[0,:], label='estimated px')
plt.legend()
plt.title('EKF Output for px')
plt.xlabel('time')
plt.ylabel('px')
# plt.show()

plt.figure()
plt.plot(timeArray,trueState[1,:], label='true py')
plt.plot(timeArray,mu[1,:], label='estimated py')
plt.legend()
plt.title('EKF Output for py')
plt.xlabel('time')
plt.ylabel('py')
# plt.show()

ax.plot()
ax.plot(trueState[0,:],trueState[1,:], label='true path')
ax.plot(mu[0,:],mu[1,:], label='estimated path')
ax.legend()
ax.set_title('EKF Output for path')
ax.set_xlabel('px')
ax.set_ylabel('py')

plt.figure()
plt.plot(sensor2.history, label='sensor1')
plt.legend()
plt.title('Sensor1 history')
plt.xlabel('time')
plt.ylabel('angle')
plt.show()