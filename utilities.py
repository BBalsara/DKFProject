# code for utility functions
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Sensor():
    def __init__(self, position, fov, bearing, angle_noise):
        self.position = position # position of the sensor in global frame
        self.bearing = bearing # angle definining center of FOV from global frame x-axis
        self.fov = fov # total angle of view for the camera (split in two across the bearing)
        # self.dist_noise = dist_noise # should be larger
        self.angle_noise = angle_noise # should be smaller
        self.history = np.array([])
    
    # returns true if the target is visible to the sensor
    def is_visible(self, target_pos):
        # Relative position
        rel_pos = target_pos[0:2] - self.position

        # Angle between sensor and target
        meas_angle = self.g(target_pos)

        return (-self.fov/2 <= meas_angle and meas_angle <= self.fov/2)

    # returns the measurement of the object's distance in the sensor frame
    # def dist_meas(self, target_pos):
    #     dist = np.linalg.norm(self.position - target_pos)
    #     # dist += np.random.normal(0, self.dist_noise)

    #     return dist
    
    # returns the measurement of the object's angle in the sensor frame
    def angle_meas(self, target_pos):
        meas = self.g(target_pos) + np.random.normal(0, self.angle_noise)
        if meas > np.pi:
            meas -= 2*np.pi
        elif meas < -np.pi:
            meas += 2*np.pi
        self.history = np.append(self.history, meas)
        return meas   
    
    # sensor measurement model return (dist, theta)
    def g(self, target_pos):
        # Angle between sensor and target
        rel_pos = target_pos[0:2] - self.position
        angle = ((np.arctan2(rel_pos[1], rel_pos[0]))) - self.bearing
        if angle > np.pi:
            angle -= 2*np.pi
        elif angle < -np.pi:
            angle += 2*np.pi
        return angle
    
    # sensor measurement model with noise
    # def measurment(self, target_pos):
    #     return self.g(target_pos) + np.array([np.random.normal(0, self.angle_noise)])

    # returns the measurement of the object in the global frame (Not needed)
    # def globalTransform(self, target_pos):
    #     if self.isVisible(target_pos):
    #         angle = self.angle_meas(target_pos)
    #         # dist = self.dist_meas(target_pos)
            
    #         xm = dist*np.cos(angle+self.bearing) + self.position[0]
    #         ym = dist*np.sin(angle+self.bearing) + self.position[1]

    #         return np.array([xm, ym])
        
    def jacobian_C(self, target_pos, n):
        Px = target_pos[0]
        Py = target_pos[1]
        Sx = self.position[0]
        Sy = self.position[1]
        # firsWt row
        # dist = self.dist_meas(target_pos)
        # C11 = (Px - Sx)/dist
        # C12 = (Py - Sy)/dist
        # second row
        C21 = (Sy - Py)/(np.linalg.norm(self.position - target_pos[0:2])**2)
        C22 = (Px - Sx)/(np.linalg.norm(self.position - target_pos[0:2])**2)
        if n == 3:
            return np.array([C21,C22,0])
        else:
            return np.array([C21,C22])


class World():
    def __init__(self, sensors):
        self.sensors = sensors # list of sensors in the world

    # function for adding sensors to the world
    def addSensor(self, position, FOV, bearing, angle_noise):
        self.sensors.append(Sensor(position, FOV, bearing, angle_noise))        
        
    # function to plot location of sensors and their respective FOVs
    def FOVplot(self):
        k = 5 # length of arrow scale factor
        fig, ax = plt.subplots()
        plt.grid()
        for sensor in self.sensors:   
            plt.scatter(sensor.position[0], sensor.position[1], color='k')
            plt.arrow(sensor.position[0], sensor.position[1], k*np.cos(sensor.bearing), k*np.sin(sensor.bearing), color='b', head_width=0.1, linestyle = 'dotted')
            plt.arrow(sensor.position[0], sensor.position[1], k*np.cos(sensor.bearing+sensor.fov/2), k*np.sin(sensor.bearing+sensor.fov/2), color='k', head_width=0.1)
            plt.arrow(sensor.position[0], sensor.position[1], k*np.cos(sensor.bearing-sensor.fov/2), k*np.sin(sensor.bearing-sensor.fov/2), color='k', head_width=0.1)
            arc = patches.Arc((sensor.position[0], sensor.position[1]), 2*k, 2*k, theta1=np.degrees(sensor.bearing-sensor.fov/2), theta2=np.degrees(sensor.bearing+sensor.fov/2), color='k')
            ax.add_patch(arc)
            triangle = patches.Polygon([[sensor.position[0], sensor.position[1]], [sensor.position[0]+k*np.cos(sensor.bearing+sensor.fov/2), sensor.position[1]+k*np.sin(sensor.bearing+sensor.fov/2)], 
                                        [sensor.position[0]+k*np.cos(sensor.bearing-sensor.fov/2), sensor.position[1]+k*np.sin(sensor.bearing-sensor.fov/2)]], color='k', alpha=0.2)
            ax.add_patch(triangle)
            theta = np.linspace(sensor.bearing-sensor.fov/2, sensor.bearing+sensor.fov/2, 100)
            r = k
            x = r*np.cos(theta) + sensor.position[0]
            y = r*np.sin(theta) + sensor.position[1]
            arcPart = patches.Polygon(np.column_stack((x,y)), color='k', alpha=0.2)
            ax.add_patch(arcPart)
        plt.gca().set_aspect('equal')
        return fig, ax

class Movement():
    def __init__(self, Movement_pattern='linear', initial_location=np.array([0, 0]), del_t=0.01, speed=.5, angle=0):
        self.pattern = Movement_pattern.lower()  # can be 'Linear', 'sine', 'diff_drive' 
        self.state = initial_location  # Location in the world frame
        self.initial_location = initial_location
        self.del_t = del_t
        self.speed = speed
        self.angle = angle

        if self.pattern == 'linear':
            self.state = initial_location
            self.Q = 0.001* np.eye(2)
        elif self.pattern == 'sine' or self.pattern == 'sin':
            self.pattern = 'sine'
            self.state = initial_location
            self.Q = 0.08* np.eye(2)
        elif self.pattern == 'diff_drive':
            self.Q = 0.02* np.eye(3)
            if initial_location.shape[0] == 2:
                self.state = np.hstack((initial_location, angle))
                self.Q = 0.02* np.eye(3)


    def f_function(self, mu, u=np.array([0, 0])):
        # Linear movement
        if self.pattern == 'linear':
            # x_dot = vel * cos angle
            state = np.array([np.cos(self.angle) * self.speed * self.del_t,
                                np.sin(self.angle) * self.speed * self.del_t]) + mu
            return state

        # Sine movement
        elif self.pattern == 'sine':
            del_xt = self.speed * self.del_t
            state = np.array([del_xt + mu[0],
                                np.sin(del_xt + mu[0] - self.initial_location[0]) + self.initial_location[1]])
            return state
        
        # Diff drive
        elif self.pattern == 'diff_drive':
            px = mu[0]
            py = mu[1]
            theta = mu[2]
            v = u[0]
            phi = u[1]
            state = (np.array([px + self.del_t * v * np.cos(theta),
                            py + self.del_t * v * np.sin(theta),
                            theta + self.del_t * phi]))
            return state
        
        print('No Movement Pattern defined')
        return None
    
    def jacobian_A(self, mu, u=np.array([0, 0])):
        # Linear movement
        if self.pattern == 'linear':
            return np.eye(2)
        
        # Sine movement
        elif self.pattern == 'sine':
            return np.array([[1, 0],
                            [np.cos(self.speed * self.del_t + mu[0] - self.initial_location[0]), 1]])
        
        # Diff drive
        elif self.pattern == 'diff_drive':
            theta = mu[2]
            v = u[0]
            out = np.array([[1, 0, -self.del_t * v * np.sin(theta)],
                        [0, 1, self.del_t * v * np.cos(theta)],
                        [0, 0, 1]])
        return out
        
    # use this to update the internal state values
    def one_step(self, u=np.array([0, 0])):
        dims = self.state.shape[0]
        self.state = self.f_function(self.state, u) + np.random.multivariate_normal(np.zeros(dims), self.del_t*self.Q)
        return self.state
