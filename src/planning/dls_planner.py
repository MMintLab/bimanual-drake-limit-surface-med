#===============================================
# limit_surface.py
# 
# limit surface implementation of open-loop
# planner. attempts to generate trajectories
# for arm to follow.
#===============================================

import casadi as cs
import numpy as np

def dual_limit_surface_cost(C1, C2, start_pose, end_pose, steps):
    '''
    Parameters:
    ===========
    @param C1         : (float) scalar representing cost scale for deviation of path from straight line from start to finish.
    @param C2         : (float) scalar representing cost scale on second order behavior of path.
    @param start_pose : (3x1 ndarray) [x,y,theta] 2d starting pose of object for interpolation
    @param end_pose   : (3x1 ndarray) [x,y,theta] 2d ending   pose of object for interpolation
    @param steps      : (int)   number of points to plan for from start to finish.

    Description:
    ============
    creates objective cost for optimization.

    Cost function will be seen as:
        J(q) = C1*(q-qhat).T @ (q-qhat) + C2*sum([qi-2,qi-1,qi].T @ P @ [qi-2,qi-1,qi])

    The C1 term is the deviation from start to finish
    The C2 term is the penalty scale on the second orderness of the path created.

    Returns:
    ========
    @return casadi.Function: qs -> cost.
    '''
    

    qs = cs.SX.sym('q', 3, steps)
    qhats = cs.SX(np.linspace(start_pose, end_pose, steps).squeeze().T)

    # deviation cost from line
    J = C1*cs.sum2(cs.sum1(((qs-qhats)**2)))

    #kronecker product applied on 3x3 matrix and I_{3x3} to make P.
    # P (x) I = [p11, p12, p13]     [1, 0, 0]   [p11 I, p12 I, p13 I]
    #           [p21, p22, p23] (x) [0, 1, 0] = [p21 I, p22 I, p23 I]
    #           [p31, p32, p33]     [0, 0, 1]   [p31 I, p32 I, p33 I]

    # second order penalty cost
    P = np.array([[1,-2,1]])
    P = cs.SX(np.kron(P.T @ P, np.eye(3))) # (9x9)
    qstack = cs.vertcat(qs[:,0:steps-2], qs[:,1:steps-1], qs[:,2:steps]) # (9xN)
    
    # for i in range(qstack.shape[1]):
        # J += C2*(qstack[:,i].T @ P @ qstack[:,i])
    J += C2*cs.sum2(cs.sum1((qstack.T @ P).T * qstack))

    #return as Function
    Jcost = cs.Function('cost_fn', [qs], [J])
    return Jcost

def dls_constraint_ineq(kv, inverse, steps):
    '''
    Parameters:
    ===========
    @param kv     
    @param inverse
    @param steps

    Description:
    ============
    returns calculations for left side of inequality constraint g(x) <= 0.

    Explicitly this is mainly a quadratic constraint which represents
    the ellipsoid constraint approximated for the dual limit contact model.

    Returns:
    ========
    @return casadi.Function: qs -> inequality LHS.
    '''

    #create K matrix derived from contact formulation in Dual Limit Surface
    K = np.array([[1,-1],
                  [-1,1]])
    K = np.kron(K,np.diag([kv,kv,-1]))
    K = cs.SX(K)
    qs = cs.SX.sym('x', 3, steps)

    qstack = cs.vertcat(qs[:,0:steps-1], qs[:,1:steps]) #6xN

    #apply quadratic
    
    inverse_multiplier = (1 if inverse else -1)
    # g_ineq = cs.SX.zeros(qstack.shape[1])
    # for i in range(qstack.shape[1]):
    #     g_ineq[i] = inverse_multiplier*(qstack[:,i].T @ K @ qstack[:,i])
    g_ineq = inverse_multiplier*cs.sum1((qstack.T @ K).T * qstack) # 1xN-1

    contact_ineq_fn = cs.Function('contact_fn', [qs], [g_ineq])

    ubg = [0]*qstack.shape[1]
    lbg = [-np.inf]*qstack.shape[1]
    return contact_ineq_fn, ubg, lbg

def dls_decision_constr(start_pose, end_pose, steps):
    ubx = [np.inf]*(3*steps)
    lbx = [-np.inf]*(3*steps)

    ubx[:3]  = start_pose.flatten().tolist()
    lbx[:3]  = start_pose.flatten().tolist()
    ubx[-3:] = end_pose.flatten().tolist()
    lbx[-3:] = end_pose.flatten().tolist()
    #the only constraint is making sure we don't change start and end pose
    return ubx, lbx

def limit_surface_plan(start_pose: np.ndarray, end_pose: np.ndarray, kv: float, C1: float, C2: float, steps:int = 10, inverse:bool = False):
    '''
    Parameters:
    ===========
    @param start_pose: (3x1 ndarray) [x,y,theta] 2d starting pose of object
    @param end_pose  : (3x1 ndarray) [x,y,theta] 2d ending   pose of object
    @param kv        : (float) double cone constant
    @param ka        : (float) scalar representing smoothness of path
    @param steps     : (int)   number of points to plan for from start to finish.
    @param inverse   : (bool)  whether kv^2 < w^2 or kv^2 > w^2 for double cone constraint

    Description:
    ============
    open-loop planner that performs optimization to solve a best path given contact
    dynamics.

    Returns:
    ========
    @return (3xN ndarray) each column is [x y theta]. a set of waypoints representing solved path
                          between start and end pose. N = number of waypoints = step param
    '''

    '''
        We aim to solve the quadratic constrained QP formulated as.

        min. q.T@Q@q + f@q + c
         q
        s.t.
             q.T@H@q <= bineq
             q       >= beq
             q       <= aeq
    '''
    qs = cs.SX.sym('q', 3, steps)
    cost_fn = dual_limit_surface_cost(C1, C2, start_pose, end_pose, steps)
    ineq_fn, ubg, lbg  = dls_constraint_ineq(kv, inverse, steps)
    ubq, lbq = dls_decision_constr(start_pose,end_pose, steps)
    
    options = {'ipopt.print_level':0}
    solver = cs.nlpsol('solver','ipopt', {'f': cost_fn(qs), 'x': qs.reshape((-1,1)), 'g': ineq_fn(qs)}, options)
    qs0 = np.linspace(start_pose, end_pose, steps).squeeze().T.flatten().tolist()
    
    solution = solver(x0=qs0, lbx=lbq, ubx=ubq, lbg=lbg, ubg=ubg)
    q_sol  = np.array(solution['x'].reshape((3,steps)))
    g_sol = np.array(solution['g']).flatten()
    
    #cost
    cost = solution['f']
    return q_sol, g_sol, cost


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    initial_pose = np.array([[0.2,0,0]]).T
    goal_pose    = np.array([[0.0, 0.7, np.pi/2]]).T
    kv = 0.5
    C1 = 10 #path length penalty scale
    C2 = 100 #k_a
    steps = 30
    inverse = False
    q_sol, g_sol, cost = limit_surface_plan(initial_pose, goal_pose, kv, C1, C2,steps=steps, inverse=inverse)

    ts = np.linspace(0,q_sol.shape[1],q_sol.shape[1])
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(q_sol[0,:], q_sol[1,:], 'b')
    ax.quiver(*q_sol[:2,:], np.cos(q_sol[-1,:]), np.sin(q_sol[-1,:]), color='r', scale=20)
    ax.set_xlabel('x/m')
    ax.set_ylabel('y/m')
    ax.set_xlim(-1,1)
    ax.set_title('DLS Planner Result')
    
    
    fig = plt.figure()
    K = np.array([[1, -1],
             [-1, 1]])
    K = np.kron(K, np.diag([kv, kv, -1]))

    K00 = K[:3, :3]
    K01 = K[:3, 3:6]
    K10 = K[3:6, :3]
    K11 = K[3:6, 3:6]
    
    g_evals = []
    inverse_multiplier = 1 if inverse else -1
    for i in range(1, steps):
        g_eval = q_sol[:,i-1].T.dot(K00.dot(q_sol[:,i-1])) + q_sol[:,i-1].T.dot(K01.dot(q_sol[:,i])) \
            + q_sol[:,i].T.dot(K10.dot(q_sol[:,i-1])) + q_sol[:,i].T.dot(K11.dot(q_sol[:,i]))
            
        g_eval = inverse_multiplier*g_eval
        g_evals.append(g_eval)
    g_evals = np.array(g_evals)
    plt.plot(g_evals)
    plt.xlabel("Time")
    plt.ylabel("Constraint Value")
    plt.title("yolo")
    
    fig = plt.figure()
    # evaluate constraint kv
    K = np.array([[1,-1],
                  [-1,1]])
    K = np.kron(K,np.diag([kv,kv,-1]))
    qstack = np.vstack((q_sol[:,:-1], q_sol[:,1:]))
    inverse_mult = 1 if inverse else -1
    g_ineq = np.sum((qstack.T @ K).T * qstack, axis=0)*inverse_mult
    plt.plot(g_ineq)
    plt.xlabel('Time')
    plt.ylabel('Constraint Value')
    plt.title('Constraint Value over Time')
    
    plt.figure()
    plt.plot(g_sol)
    plt.xlabel('Time')
    plt.ylabel('Constraint Value')
    plt.title('Constraint Value over Time GT')
    
    
    # evaluate cost
    print("GT Cost: ", cost)
    P = np.array([[1,-2,1]])
    P = np.kron(P.T @ P, np.eye(3))
    f_cost = 0
    qstack = np.vstack((q_sol[:,:-2], q_sol[:,1:-1], q_sol[:,2:]))
    qhat0 = np.linspace(initial_pose.flatten(), goal_pose.flatten(), steps).T
    
    f_cost += C1*np.sum((q_sol - qhat0)**2)
    f_cost += C2*np.sum( (qstack.T @ P).T * qstack )
    print("Eval Cost: ", f_cost)
    
    f_eval = 0
    P00 = P[:3, :3]
    P01 = P[:3, 3:6]
    P02 = P[:3, 6:9]
    P10 = P[3:6, :3]
    P11 = P[3:6, 3:6]
    P12 = P[3:6, 6:9]
    P20 = P[6:9, :3]
    P21 = P[6:9, 3:6]
    P22 = P[6:9, 6:9]
    
    for i in range(2, steps):
        f_eval += q_sol[:,i-2].dot(P00.dot(q_sol[:,i-2])) + q_sol[:,i-2].dot(P01.dot(q_sol[:,i-1])) + q_sol[:,i-2].dot(P02.dot(q_sol[:,i])) + q_sol[:,i-1].dot(P10.dot(q_sol[:,i-2])) + q_sol[:,i-1].dot(P11.dot(q_sol[:,i-1])) + q_sol[:,i-1].dot(P12.dot(q_sol[:,i])) + q_sol[:,i].dot(P20.dot(q_sol[:,i-2])) + q_sol[:,i].dot(P21.dot(q_sol[:,i-1])) + q_sol[:,i].dot(P22.dot(q_sol[:,i]))
    f_eval *= C2
    
    for i in range(steps):
        f_eval += 2*C1*0.5*(q_sol[:,i] - qhat0[:,i]).dot(q_sol[:,i] - qhat0[:,i])
    plt.show()