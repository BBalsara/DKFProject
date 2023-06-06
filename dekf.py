# code for simulation and running tests
# import utilities as util
import numpy as np
from utilities import Movement, Sensor, World
import matplotlib.pyplot as plt

n = 3 # toggles state dimension
# np.random.seed(11)
# Simulation class

def is_nearby(sensor1: Sensor, sensor2: Sensor , neighborhood_threshold):
    pos1 = sensor1.position
    pos2 = sensor2.position
    dist = np.linalg.norm(pos1 - pos2) # Euclidian distance between 2 sensors

    return dist < neighborhood_threshold

# Cameras (Position, FOV)
eps = 1e-5
R = np.array([[0.1]])
fov = np.radians(70)
bearing1 = np.radians(45) 
bearing2 = np.radians(225)
bearing3 = np.radians(135)
bearing4 = np.radians(180) 
bearing5 = np.radians(80) 
bearing6 = np.radians(0)
bearing7 = np.radians(300)
bearing8 = np.radians(220) 
position1 = np.array([0,0])
position2 = np.array([10,10])
position3 = np.array([16,0])
position4 = np.array([30,5])
position5 = np.array([20,-10])
position6 = np.array([10,15])
position7 = np.array([16,26])
position8 = np.array([30,30])
angle_noise = 0.1
sensor1 = Sensor(position1, fov, bearing1, angle_noise,n)
sensor2 = Sensor(position2, fov, bearing2, angle_noise,n)
sensor3 = Sensor(position3, fov, bearing3, angle_noise,n)
sensor4 = Sensor(position4, fov, bearing4, angle_noise,n)
sensor5 = Sensor(position5, fov, bearing5, angle_noise,n)
sensor6 = Sensor(position6, fov, bearing6, angle_noise,n)
sensor7 = Sensor(position7, fov, bearing7, angle_noise,n)
sensor8 = Sensor(position8, fov, bearing8, angle_noise,n)

world = World(np.array([sensor1, sensor2, sensor3, sensor4, sensor5, sensor6, sensor7, sensor8]))




render, ax = world.FOVplot()
measurement_difference = np.array([0,0])

radius = 15
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

xbar = [mu0 for k in world.sensors]
P = [cov0 for k in world.sensors]
xhat = []
for i, sensor111 in enumerate(world.sensors):
    sensor111.mu_history = np.array([0,0]).copy()
    
for t in range(1, int(Tmax/d_t)):
    # Simulation
    # prev_mu = sensors.mean[:, t-1]
    # prev_cov = cov[:, :, t-1]

    # Calculate u if needed for movement
    u = np.array([0.8, np.sin(t*d_t)])
    
    # Update true State
    x = Object_a.one_step(u)
    trueState[:, t] = x

    little_u = []
    big_U = []

    for i, sensor in enumerate(world.sensors):
            y = sensor.angle_meas(x)
            C = sensor.jacobian_C(xbar[i], n)
            u_i = sensor.little_u(C, y, R, xbar[i], n, Object_a)
            U_i = sensor.big_U(C, R, n, Object_a)
            # xbar_i = sensor.xbar_i()
            # np.append(xbar, xbar_i)
            little_u.append(u_i)
            big_U.append(U_i)

    # Calculating G value        
    g_val = []
    S_val = []
    
    for i, sensor in enumerate(world.sensors):
        g = np.zeros(n)
        S = np.zeros((n, n))
        diff_xbar = np.zeros(n)
        for j, sensor11111 in enumerate(world.sensors):
            if is_nearby(world.sensors[i], world.sensors[j], radius) and not i == j:
                g += little_u[j] - big_U[j] @ (xbar[i] - xbar[j])
                S += big_U[j]
                diff_xbar += (xbar[j] - xbar[i])


        g_val.append(g)
        S_val.append(S)
        M_i = np.linalg.inv(np.linalg.inv(P[i]) + S_val[i])
        x_hat_i = xbar[i] + M_i @ g_val[i] + eps * M_i @ diff_xbar
        # print(x_hat_i)

        sensor.addHistory(x_hat_i)
        # if i == 0 or i == 1:
            # print(x_hat_i)
            # print(sensor)
            # print('sensor1', sensor.mu_history.shape)
        # sensor111.mu_history = np.vstack((sensor111.mu_history, x_hat_i))
        xbar[i] = Object_a.f_function(x_hat_i, u)
        A = Object_a.jacobian_A(x_hat_i, u) 
        P[i] = A @ M_i @ A.T + Object_a.Q
    

    


# Performance Calculation
error1 = np.linalg.norm(trueState[:2].T-sensor1.mu_history)
error2 = np.linalg.norm(trueState[:2].T-sensor2.mu_history)
error3 = np.linalg.norm(trueState[:2].T-sensor3.mu_history)
error4 = np.linalg.norm(trueState[:2].T-sensor4.mu_history)
error5 = np.linalg.norm(trueState[:2].T-sensor5.mu_history)
error6 = np.linalg.norm(trueState[:2].T-sensor6.mu_history)
error7 = np.linalg.norm(trueState[:2].T-sensor7.mu_history)
error8 = np.linalg.norm(trueState[:2].T-sensor8.mu_history)



error = 1/8 * (error1 + error2 + error3 + error4 + error5 + error6 + error7 + error8)
print('Error:', error)

# Plotting
ax.plot()
plotted = False
for sensor in world.sensors:
    for neighbor in world.neighborhood(radius, sensor):
        if not plotted:
            ax.plot([sensor.position[0], neighbor.position[0]], [sensor.position[1], neighbor.position[1]], 'r--', linewidth=0.75, label = 'Communication')
            plotted = True
        else:
            ax.plot([sensor.position[0], neighbor.position[0]], [sensor.position[1], neighbor.position[1]], 'r--', linewidth=0.75)
ax.plot(sensor1.mu_history[:,0],sensor1.mu_history[:,1], label='Estimated Path 1')
ax.plot(sensor2.mu_history[:,0],sensor2.mu_history[:,1], label='Estimated Path 2')
ax.plot(sensor3.mu_history[:,0],sensor3.mu_history[:,1], label='Estimated Path 3')
ax.plot(sensor4.mu_history[:,0],sensor4.mu_history[:,1], label='Estimated Path 4')
ax.plot(sensor5.mu_history[:,0],sensor5.mu_history[:,1], label='Estimated Path 5')
ax.plot(sensor6.mu_history[:,0],sensor6.mu_history[:,1], label='Estimated Path 6')
ax.plot(sensor7.mu_history[:,0],sensor7.mu_history[:,1], label='Estimated Path 7')
ax.plot(sensor8.mu_history[:,0],sensor8.mu_history[:,1], label='Estimated Path 8')
ax.plot(trueState[0,:],trueState[1,:], label='True Path', color='black')
# ax.plot(sensor3.mu_history, label='Estimated Path 3')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_title('CEKF Estimated Trajectory')
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


