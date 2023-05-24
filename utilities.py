# code for utility functions
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Sensor():
    def __init__(self, position, fov, bearing, dist_noise, angle_noise):
        self.position = position # position of the sensor in global frame
        self.bearing = bearing # angle definining center of FOV from global frame x-axis
        self.fov = fov # total angle of view for the camera (split in two across the bearing)
        self.dist_noise = dist_noise # should be larger
        self.angle_noise = angle_noise # should be smaller
    
    # returns true if the target is visible to the sensor
    def is_visible(self, target_pos):
        # Relative position
        rel_pos = target_pos - self.position

        # Angle between sensor and target
        meas_angle = np.arctan2(rel_pos[1], rel_pos[0]) - self.bearing

        return (-self.fov/2 <= meas_angle and meas_angle <= self.fov/2)

    # returns the measurement of the object's distance in the sensor frame
    def dist_meas(self, target_pos):
        dist = np.linalg.norm(self.position - target_pos)
        # dist += np.random.normal(0, self.dist_noise)

        return dist
    
    # returns the measurement of the object's angle in the sensor frame
    def angle_meas(self, target_pos):
        # Angle between sensor and target
        rel_pos = target_pos - self.position
        angle = np.arctan2(rel_pos[1], rel_pos[0]) - self.bearing
        # angle += np.random.normal(0, self.angle_noise)

        return angle   
    
    # sensor measurement model return (dist, theta)
    def g(self, target_pos):
        return np.array([self.dist_meas(target_pos), self.angle_meas(target_pos)])
    
    # sensor measurement model with noise
    def measurment(self, target_pos):
        return self.g(target_pos) + np.array([np.random.normal(0, self.dist_noise), np.random.normal(0, self.angle_noise)])

    # returns the measurement of the object in the global frame
    def globalTransform(self, target_pos):
        if self.isVisible(target_pos):
            angle = self.angle_meas(target_pos)
            dist = self.dist_meas(target_pos)
            
            xm = dist*np.cos(angle+self.bearing) + self.position[0]
            ym = dist*np.sin(angle+self.bearing) + self.position[1]

            return np.array([xm, ym])
        
    def jacobian(self, target_pos):
        Px = target_pos[0]
        Py = target_pos[1]
        Sx = self.position[0]
        Sy = self.position[1]
        # first row
        dist = self.dist_meas(target_pos)
        C11 = (Px - Sx)/dist
        C12 = (Py - Sy)/dist
        # second row
        C21 = (Sy - Py)/(dist**2)
        C22 = (Px - Sx)/(dist**2)
        return np.array([C11, C12],[C21,C22])

class World():
    def __init__(self, sensors):
        self.sensors = sensors # list of sensors in the world

    # function for adding sensors to the world
    def addSensor(self, position, FOV, bearing, dist_noise, angle_noise):
        self.sensors.append(Sensor(position, FOV, bearing, dist_noise, angle_noise))        
        
    # function to plot location of sensors and their respective FOVs
    def FOVplot(self):
        k = 5 # length of arrow scale factor
        fig, ax = plt.subplots()
        for sensor in self.sensors:   
            plt.scatter(sensor.position[0], sensor.position[1], color='k')
            plt.arrow(sensor.position[0], sensor.position[1], k*np.cos(sensor.bearing), k*np.sin(sensor.bearing), color='b', head_width=0.1, linestyle = 'dotted')
            plt.arrow(sensor.position[0], sensor.position[1], k*np.cos(sensor.bearing+sensor.fov/2), k*np.sin(sensor.bearing+sensor.fov/2), color='k', head_width=0.1)
            plt.arrow(sensor.position[0], sensor.position[1], k*np.cos(sensor.bearing-sensor.fov/2), k*np.sin(sensor.bearing-sensor.fov/2), color='k', head_width=0.1)
            arc = patches.Arc((sensor.position[0], sensor.position[1]), 2*k, 2*k, theta1=np.degrees(sensor.bearing-sensor.fov/2), theta2=np.degrees(sensor.bearing+sensor.fov/2), color='k')
            ax.add_patch(arc)
            triangle = patches.Polygon([[sensor.position[0], sensor.position[1]], [sensor.position[0]+k*np.cos(sensor.bearing+sensor.fov/2), sensor.position[1]+k*np.sin(sensor.bearing+sensor.fov/2)], [sensor.position[0]+k*np.cos(sensor.bearing-sensor.fov/2), sensor.position[1]+k*np.sin(sensor.bearing-sensor.fov/2)]], color='k', alpha=0.2)
            ax.add_patch(triangle)
            theta = np.linspace(sensor.bearing-sensor.fov/2, sensor.bearing+sensor.fov/2, 100)
            r = k
            x = r*np.cos(theta) + sensor.position[0]
            y = r*np.sin(theta) + sensor.position[1]
            arcPart = patches.Polygon(np.column_stack((x,y)), color='k', alpha=0.2)
            ax.add_patch(arcPart)
        plt.grid()
        plt.gca().set_aspect('equal')
        plt.show()


# Objects (different movement pattern)
# class Target():
#     def _init_(self, pos_0, dynamics):
#         self.pos = pos_0
#         self.dynamics = dynamics
    
#     def f(self, t):
        


# Linear movement
# Sine movement
