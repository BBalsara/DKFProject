# code for simulation and running tests
# import utilities as util
import numpy as np
from utilities import Movement, Sensor, World
import matplotlib.pyplot as plt

n = 2 # toggles state dimension
# np.random.seed(11)
# Simulation class


# Cameras (Position, FOV)
fov = np.radians(100)
bearing1 = np.radians(45) 
bearing2 = np.radians(225)
bearing3 = np.radians(135)
bearing4 = np.radians(180) 
position1 = np.array([0,0])
position2 = np.array([10,10])
position3 = np.array([16,0])
position4 = np.array([30,5])
angle_noise = 0.1
sensor1 = Sensor(position1, fov, bearing1, angle_noise,n)
sensor2 = Sensor(position2, fov, bearing2, angle_noise,n)
sensor3 = Sensor(position3, fov, bearing3, angle_noise,n)
sensor4 = Sensor(position4, fov, bearing4, angle_noise,n)
world = World(np.array([sensor1, sensor2, sensor3]))
render, ax = world.FOVplot()
measurement_difference = np.array([0,0])

radius = 12
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
Object_a = Movement(Movement_pattern='linear', del_t=d_t, initial_location=trueInit)


for t in range(1, int(Tmax/d_t)):
    # Simulation
    # prev_mu = sensors.mean[:, t-1]
    # prev_cov = cov[:, :, t-1]

    # Calculate u if needed for movement
    u = np.array([0.8, np.sin(t*d_t)])
    
    # Update true State
    x = Object_a.one_step(u)
    trueState[:, t] = x
        
    for sensor in (world.sensors):
        # Predict step
        sensor.predict(Object_a, u)
        # Update step
        sensor.update(Object_a, world)

    # Consensus step
    world.consensus(radius)
    


# Performance Calculation
error = np.linalg.norm(trueState-mu)
print('Error:', error)

# Plotting
ax.plot()
ax.plot(trueState[0,:],trueState[1,:], label='True Path')
ax.plot(sensor1.mu_history[:,0],sensor1.mu_history[:,1], label='Estimated Path 1')
ax.plot(sensor2.mu_history[:,0],sensor2.mu_history[:,1], label='Estimated Path 2')
ax.plot(sensor3.mu_history[:,0],sensor3.mu_history[:,1], label='Estimated Path 3')
ax.legend()
ax.set_title('EKF Estimated Trajectory')
ax.set_xlabel('x')
ax.set_ylabel('y')
# adjust window size based on the trajectory
xlow = np.minimum(np.min(trueState[0,:])-10, (np.min([sensor.position[0] for sensor in world.sensors])-10))
xhigh = np.maximum(np.max(trueState[0,:])+10, (np.max([sensor.position[0] for sensor in world.sensors])+10))
ylow = np.minimum(np.min(trueState[1,:])-10, (np.min([sensor.position[1] for sensor in world.sensors])-10))
yhigh = np.maximum(np.max(trueState[1,:])+10, (np.max([sensor.position[1] for sensor in world.sensors])+10))
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

