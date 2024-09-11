import numpy as np


if __name__ == '__main__':
    mass = 0.5 # kg
    gravity = 9.83 # m/s^2
    angle = 0.5 # rad
    applied_force = 20.0 # N
    friction_coefficient = 0.2
    
    gf = mass * gravity * np.sin(angle)
    gn = mass * gravity * np.cos(angle)
    Na = applied_force
    Nb = applied_force + gn
    
    evaluation = (friction_coefficient**2) * ((Na**2)/(Nb**2) - 1) + (gf**2) / (Nb**2)
    print(evaluation)