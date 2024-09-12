import casadi as cs
import numpy as np

def Rz(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])

class DualLimitSurfaceParams:
    def __init__(self, mu_A, r_A, N_A, mu_B, r_B, N_B):
        self.mu_A = mu_A
        self.r_A = r_A
        self.N_A = N_A
        self.mu_B = mu_B
        self.r_B = r_B
        self.N_B = N_B
        self.c = 0.6
    def get_A(self):
        return np.diag([1/(self.mu_A*self.N_A)**2, 1/(self.mu_A*self.N_A)**2, 1/(self.mu_A*self.N_A*self.c*self.r_A)**2])
    def get_B(self):
        return np.diag([1/(self.mu_B*self.N_B)**2, 1/(self.mu_B*self.N_B)**2, 1/(self.mu_B*self.N_B*self.c*self.r_B)**2])

def inhand_planner(obj2left_se2: np.ndarray, obj2right_se2: np.ndarray, desired_obj2left_se2: np.ndarray, desired_obj2right_se2: np.ndarray, dual_limit_surface_params: DualLimitSurfaceParams, obj_mass = 0.5, angle = 30 * np.pi/180, palm_radius = 0.04, steps = 10):
    # assert mu_A == mu_B
    assert dual_limit_surface_params.mu_A == dual_limit_surface_params.mu_B
    obj2world_se2 = np.array([0,0,0]) # +y is towards ground
    
    qlefts  = cs.SX.sym('qleft', 3, steps)
    qrights = cs.SX.sym('qright', 3, steps)
    qobjs   = cs.SX.sym('qobj', 3, steps)
    vs      = cs.SX.sym('v', 3, steps-1)
    
    
    Qmatvec = np.array([100, 100, 100])
    cost = cs.sum2(cs.sum1(Qmatvec * (desired_obj2left_se2 - qlefts)**2)) + cs.sum2(cs.sum1(Qmatvec * (desired_obj2right_se2 - qrights)**2))
    
    ineq    = []
    ineq_lb = []
    ineq_ub = []
    
    for t in range(steps):
        if t == 0:
            ineq.append(qlefts[:,0] - obj2left_se2)
            ineq_lb = ineq_lb + [0,0,0]
            ineq_ub = ineq_ub + [0,0,0]
            
            ineq.append(qrights[:,0] - obj2right_se2)
            ineq_lb = ineq_lb + [0,0,0]
            ineq_ub = ineq_ub + [0,0,0]
            
            ineq.append(qobjs[:,0] - obj2world_se2)
            ineq_lb = ineq_lb + [0,0,0]
            ineq_ub = ineq_ub + [0,0,0]
            continue    
        
        if t % 2 == 0:
            # right is moving, left is still
            ineq.append(qlefts[:,t] - (qlefts[:,t-1] + vs[:,t-1]) )
            ineq_lb = ineq_lb + [0.0,0.0,0.0]
            ineq_ub = ineq_ub + [0.0,0.0,0.0]
            
            ineq.append(qrights[:,t] - (Rz(vs[2,t-1]) @ qrights[:,t-1]))
            ineq_lb = ineq_lb + [0.0,0.0,0.0]
            ineq_ub = ineq_ub + [0.0,0.0,0.0]
            

        else:
            # left is moving, right is still
            ineq.append(qlefts[:,t] - (Rz(vs[2,t-1]) @ qlefts[:,t-1]))
            ineq_lb = ineq_lb + [0.0,0.0,0.0]
            ineq_ub = ineq_ub + [0.0,0.0,0.0]
            
            ineq.append(qrights[:,t] - (qrights[:,t-1] + vs[:,t-1]))
            ineq_lb = ineq_lb + [0.0,0.0,0.0]
            ineq_ub = ineq_ub + [0.0,0.0,0.0]
        
        ineq.append(qobjs[:,t] - (qobjs[:,t-1] + vs[:,t-1]))
        ineq_lb = ineq_lb + [0.0,0.0,0.0]
        ineq_ub = ineq_ub + [0.0,0.0,0.0]
        
        # ensure obj2left and obj2right are in circle
        # \| qleft \|^2 <= palm_radius^2
        ineq.append(cs.sum1(qlefts[:2,t]**2) - palm_radius**2)
        ineq_lb.append(-np.inf)
        ineq_ub.append(0)
        
        # \| qright \|^2 <= palm_radius^2
        ineq.append(cs.sum1(qrights[:2,t]**2) - palm_radius**2)
        ineq_lb.append(-np.inf)
        ineq_ub.append(0)
    
    # code to reshape code for optimization
    opt_x = cs.vertcat(qlefts.reshape((-1,1)), qrights.reshape((-1,1)), qobjs.reshape((-1,1)), vs.reshape((-1,1)))
    qL_lb = [-np.inf, -np.inf, -np.pi]*steps
    qL_ub = [np.inf, np.inf, np.pi]*steps
    qR_lb = [-np.inf, -np.inf, -np.pi]*steps
    qR_ub = [np.inf, np.inf, np.pi]*steps
    qO_lb = [-np.inf, -np.inf, -np.pi]*steps
    qO_ub = [np.inf, np.inf, np.pi]*steps
    vs_lb = [-np.inf]*3*(steps-1)
    vs_ub = [np.inf]*3*(steps-1)
    lb_x = qL_lb + qR_lb + qO_lb + vs_lb
    ub_x = qL_ub + qR_ub + qO_ub + vs_ub
    
    options = {'ipopt.print_level':0}
    solver = cs.nlpsol('solver', 'ipopt', {'f': cost, 'x': opt_x, 'g': cs.vertcat(*ineq)}, options)
    solution = solver(lbx = lb_x, ubx = ub_x, lbg = ineq_lb, ubg = ineq_ub)
    
    solution_x = np.array(solution['x'])
    obj2left = solution_x[:3*steps].reshape((-1,3)).T
    obj2right = solution_x[3*steps:6*steps].reshape((-1,3)).T
    obj2world = solution_x[6*steps:9*steps].reshape((-1,3)).T
    vs = solution_x[9*steps:].reshape((-1,3)).T
    
    return obj2left, obj2right, vs

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    palm_radius = 0.04
    obj2left_se2  = np.zeros(3)
    obj2right_se2 = np.zeros(3)
    
    desired_obj2left_se2  = np.array([0.1, 0.1, 0])
    desired_obj2right_se2 = np.array([0.1, 0.1, 0])
    
    dls_params = DualLimitSurfaceParams(1, 0.1, 0.1, 1, 0.1, 0.1)
    obj2left, obj2right, vs = inhand_planner(obj2left_se2, obj2right_se2, desired_obj2left_se2, desired_obj2right_se2, dls_params, steps = 4)
    
    # plot (2,1) subplots, draw xy and yaw vector
    fig, axs = plt.subplots(2,1)
    axs[0].plot(obj2left[0,:], obj2left[1,:], 'r')
    axs[0].quiver(obj2left[0,:], obj2left[1,:], np.cos(obj2left[2,:]), np.sin(obj2left[2,:]), color='b', scale=20)
    axs[0].set_title('Left Object')
    
    axs[1].plot(obj2right[0,:], obj2right[1,:], 'r')
    axs[1].quiver(obj2right[0,:], obj2right[1,:], np.cos(obj2right[2,:]), np.sin(obj2right[2,:]), color='b', scale=20)
    axs[1].set_title('Right Object')
    plt.show()