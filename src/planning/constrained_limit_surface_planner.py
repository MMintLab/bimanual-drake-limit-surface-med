import casadi as cs
import matplotlib.pyplot as plt
import numpy as np

def limit_surface_plan(start_pose: np.ndarray, end_pose: np.ndarray, kv: float, C1: float, C2: float, radius=1.0, steps:int = 10, inverse:bool = False):
    qs = cs.SX.sym('q', 3, steps)
    qhats = cs.SX(np.linspace(start_pose, end_pose, steps).squeeze().T)
    
    cost = C1*cs.sum2(cs.sum1(((qs-qhats)**2)))
    
    P = np.array([[1,-2,1]])
    P = cs.SX(np.kron(P.T @ P, np.eye(3))) # (9x9)
    qstack_3 = cs.vertcat(qs[:,0:steps-2], qs[:,1:steps-1], qs[:,2:steps]) # (9xN)
    cost += C2*cs.sum2(cs.sum1((qstack_3.T @ P).T * qstack_3))
    
    K = np.array([[1,-1],
                  [-1,1]])
    K = np.kron(K,np.diag([kv,kv,-1]))
    K = cs.SX(K)
    
    qstack_2 = cs.vertcat(qs[:,0:steps-1], qs[:,1:steps]) #6xN
    inverse_multiplier = (1 if inverse else -1)
    ineq = inverse_multiplier*cs.sum1((qstack_2.T @ K).T * qstack_2) # 1xN-1
    ubg = [0]*qstack_2.shape[1]
    lbg = [-np.inf]*qstack_2.shape[1]
    
    ineq2 = cs.sum1((qs[:2,:])**2)
    ub2 = [radius**2]*steps
    lb2 = [0]*steps
    
    ineq = cs.horzcat(ineq, ineq2)
    ubg.extend(ub2)
    lbg.extend(lb2)
    
    ubx = [np.inf]*(3*steps)
    lbx = [-np.inf]*(3*steps)
    ubx[:3]  = start_pose.flatten().tolist()
    lbx[:3]  = start_pose.flatten().tolist()
    ubx[-3:] = end_pose.flatten().tolist()
    lbx[-3:] = end_pose.flatten().tolist()
    
    options = {'ipopt.print_level':0}
    solver = cs.nlpsol('solver','ipopt', {'f': cost, 'x': qs.reshape((-1,1)), 'g': ineq}, options)
    qs0 = np.linspace(start_pose, end_pose, steps).squeeze().T.flatten().tolist()
    solution = solver(x0=qs0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    q_sol  = np.array(solution['x'].reshape((3,steps)))
    g_sol = np.array(solution['g']).flatten()
    
    #cost
    cost = solution['f']
    return q_sol, g_sol, cost

if __name__ == '__main__':
    initial_pose = np.array([[0.0,0,0]]).T
    goal_pose    = np.array([[0.0, 0.03, np.pi/2]]).T
    kv = 0.5
    C1 = 1 #path length penalty scale
    C2 = 100 #k_a
    steps = 30
    inverse = False
    radius = 0.05
    q_sol, g_sol, cost = limit_surface_plan(initial_pose, goal_pose, kv, C1, C2, radius=radius, steps=steps, inverse=inverse)

    ts = np.linspace(0,q_sol.shape[1],q_sol.shape[1])
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(q_sol[0,:], q_sol[1,:], 'b')
    ax.quiver(*q_sol[:2,:], np.cos(q_sol[-1,:]), np.sin(q_sol[-1,:]), color='r', scale=20)

    #plot circle of radius
    thetas = np.linspace(0,2*np.pi,100)
    x = radius*np.cos(thetas)
    y = radius*np.sin(thetas)
    ax.plot(x,y,'g--')


    ax.set_xlabel('x/m')
    ax.set_ylabel('y/m')
    ax.set_title('DLS Planner Result')
    #equal scale
    ax.set_aspect('equal', 'box')
    plt.show()