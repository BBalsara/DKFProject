# code for simulation and running tests
# import utilities as util
import numpy as np
from utilities import Movement, Sensor, World
import matplotlib.pyplot as plt

n = 3 # toggles state dimension
np.random.seed(11)
# Simulation class


# Cameras (Position, FOV)
fov = np.radians(30)
bearing1 = np.radians(45) 
bearing2 = np.radians(225)
bearing3 = np.radians(135)
bearing4 = np.radians(180) 
position1 = np.array([0,0])
position2 = np.array([10,10])
position3 = np.array([15,0])
position4 = np.array([30,5])
angle_noise = 0.1
sensor1 = Sensor(position1, fov, bearing1, angle_noise,n)
sensor2 = Sensor(position2, fov, bearing2, angle_noise,n)
sensor3 = Sensor(position3, fov, bearing3, angle_noise,n)
sensor4 = Sensor(position4, fov, bearing4, angle_noise,n)
world = World(np.array([sensor1, sensor3, sensor4]))
render, ax = world.FOVplot()
measurement_difference = np.array([0,0])

# EKF prediction
def predict(prev_mu, prev_cov, target: Movement, u):
    pred_mu = target.f_function(mu = prev_mu, u = u)
    A = target.jacobian_A(mu = prev_mu, u = u)
    pred_cov = A @ prev_cov @ A.T + target.Q
    return pred_mu, pred_cov

# EKF update
def update(target: Movement, pred_mu, pred_cov, world: World):
    measurement_difference = np.array([])
    # pull out sensor data
    sensors = [sensor for sensor in world.sensors if sensor.is_visible(target.state)]
    sensor_count = len(sensors)
    true_state = target.state
    sensor_noises = np.zeros(sensor_count) # for storing noise values of sensors with target in FOV
    C = np.zeros((sensor_count, len(true_state))) # for storing jacobian values of sensors with target in FOV
    y = np.zeros(sensor_count) # for storing true measurements of sensors with target in FOV
    g = np.zeros(sensor_count) # for storing predicted measurements of sensors with target in FOV
    for idx, sensor in enumerate(sensors):
        C[idx,:] = sensor.jacobian_C(pred_mu, n)
        sensor_noises[idx] = sensor.angle_noise
        y[idx] = sensor.angle_meas(true_state)
        g[idx] = sensor.g(pred_mu)
    if sensor_count == 0:
        return pred_mu, pred_cov
    else:
        Kt = pred_cov @ C.T @ np.linalg.inv(C @ pred_cov @ C.T + np.diag(sensor_noises))
        measurement_difference = y-g
        

        mu = pred_mu + Kt @ (measurement_difference)
        measurement_difference = np.append(measurement_difference, np.array(y-g).flatten())
        cov = pred_cov - Kt @ C @ pred_cov 

    return mu, cov, measurement_difference


Tmax = 40
d_t = 0.01
mu = np.zeros((n, int(Tmax/d_t)))
cov = np.zeros((n, n, int(Tmax/d_t)))
if n == 2:
    mu0 = np.array([4.5,0])
    trueInit = np.array([5,0])
elif n == 3:
    mu0 = np.array([4.5,0,0])
    trueInit = np.array([5,0,0])
else:
    print("Invalid state dimension")
cov0 = 0.01*np.eye(n)
mu[:, 0] = mu0
cov[:, :, 0] = cov0

trueState = np.zeros((n, int(Tmax/d_t)))
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
    updated_mu, updated_cov, mdiff = update(Object_a, pred_mu, pred_cov, world)
    measurement_difference = np.append(measurement_difference, mdiff)
    # Store values
    mu[:, t] = updated_mu
    cov[:, :, t] = updated_cov

# Performance Calculation
error = np.linalg.norm(trueState-mu)
print('Error:', error)


# Plotting
ax.plot()
ax.plot(trueState[0,:],trueState[1,:], label='True Path')
ax.plot(mu[0,:],mu[1,:], label='Estimated Path')
ax.legend()
ax.set_title('EKF Estimated Trajectory')
ax.set_xlabel('x')
ax.set_ylabel('y')
# adjust window size based on the trajectory
xlow = np.minimum(np.min(trueState[0,:])-5, (np.min([sensor.position[0] for sensor in world.sensors])-5))
xhigh = np.maximum(np.max(trueState[0,:])+5, (np.max([sensor.position[0] for sensor in world.sensors])+5))
ylow = np.minimum(np.min(trueState[1,:])-5, (np.min([sensor.position[1] for sensor in world.sensors])-5))
yhigh = np.maximum(np.max(trueState[1,:])+5, (np.max([sensor.position[1] for sensor in world.sensors])+5))
ax.set_xlim(xlow, xhigh)
ax.set_ylim(ylow, yhigh)
plt.show()

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



plt.figure()
plt.plot(np.degrees(sensor1.history))
plt.title('Sensor1 history')
plt.xlabel('time')
plt.ylabel('angle')

plt.figure()
plt.plot((measurement_difference))
plt.title('measurement difference history')
plt.xlabel('time')
plt.ylabel('angle')
# plt.show()

