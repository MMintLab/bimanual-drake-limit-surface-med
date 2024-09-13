from pydrake.all import (
    MathematicalProgram,
    GetProgramType,
    SnoptSolver,
    GetAvailableSolvers,
    eq
)
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
    def get_Asqrt(self):
        return np.diag([1/(self.mu_A*self.N_A), 1/(self.mu_A*self.N_A), 1/(self.mu_A*self.N_A*self.c*self.r_A)])
    def get_B(self):
        return np.diag([1/(self.mu_B*self.N_B)**2, 1/(self.mu_B*self.N_B)**2, 1/(self.mu_B*self.N_B*self.c*self.r_B)**2])
    def get_Bsqrt(self):
        return np.diag([1/(self.mu_B*self.N_B), 1/(self.mu_B*self.N_B), 1/(self.mu_B*self.N_B*self.c*self.r_B)])

def inhand_planner(obj2left_se2: np.ndarray, obj2right_se2: np.ndarray, desired_obj2left_se2: np.ndarray, desired_obj2right_se2: np.ndarray, dual_limit_surface_params: DualLimitSurfaceParams, obj_mass = 0.5, angle = 30 * np.pi/180, palm_radius = 0.04, steps = 10, kv = 0.5):
    assert dual_limit_surface_params.mu_A == dual_limit_surface_params.mu_B
    
    obj2world_se2 = np.array([0,0,0]) # +y is towards ground
    
    prog = MathematicalProgram()
    qlefts = prog.NewContinuousVariables(3, steps, "qleft")
    qrights = prog.NewContinuousVariables(3, steps, "qright")
    qobjs = prog.NewContinuousVariables(3, steps, "qobj")
    vs = prog.NewContinuousVariables(3, steps-1, "v")
    
    Qmatvec = np.array([100, 100, 100])
    for t in range(steps):
        prog.AddQuadraticCost(Qmatvec @ (desired_obj2left_se2 - qlefts[:,t])**2)
        prog.AddQuadraticCost(Qmatvec @ (desired_obj2right_se2 - qrights[:,t])**2)
        if t == 0:
            # add initial conditions
            prog.AddLinearEqualityConstraint(qlefts[:,0], obj2left_se2)
            prog.AddLinearEqualityConstraint(qrights[:,0], obj2right_se2)
            prog.AddLinearEqualityConstraint(qobjs[:,0], obj2world_se2)
            continue
        
        if t % 2 == 0:
            # right is moving, left is still
            prog.AddLinearConstraint(eq(qlefts[:,t], qlefts[:,t-1] + vs[:,t-1]))
            prog.AddConstraint(eq(qrights[:,t], Rz(vs[2,t-1]) @ qrights[:,t-1]))
        else:
            # left is moving, right is still
            prog.AddLinearConstraint(eq(qrights[:,t], qrights[:,t-1] + vs[:,t-1]))
            prog.AddConstraint(eq(qlefts[:,t], Rz(vs[2,t-1]) @ qlefts[:,t-1]))
        
        # ensure obj2left and obj2right are in circle
        # \| qleft[:2] \|^2 <= palm_radius^2
        # qleft.T @ diag(1,1,0) @ qleft - palm_radius**2 <= 0
        prog.AddQuadraticAsRotatedLorentzConeConstraint(2*np.diag([1,1,0]), np.zeros(3), -palm_radius**2, qlefts[:,t])
        
        # \| qright[:2] \|^2 <= palm_radius^2
        prog.AddQuadraticAsRotatedLorentzConeConstraint(2*np.diag([1,1,0]), np.zeros(3), -palm_radius**2, qrights[:,t])

        mg = obj_mass * 9.83
        mg_sin_theta = mg * np.sin(angle * np.pi/180)
        mg_cos_theta = mg * np.cos(angle * np.pi/180)
                
        A = dual_limit_surface_params.get_A()
        Asqrt = dual_limit_surface_params.get_Asqrt()
        Ainv = np.linalg.inv(A)
        Ainvsqrt = np.linalg.inv(Asqrt)
        B = dual_limit_surface_params.get_B()
        mu_B = dual_limit_surface_params.mu_B
        N_B = dual_limit_surface_params.N_A + mg_cos_theta
        c = (dual_limit_surface_params.N_A)**2 / (N_B)**2
        
        vy = vs[1,t-1]
        if dual_limit_surface_params.r_A == dual_limit_surface_params.r_B:
            const_sqrt_v_Ainv_v = (c - 1 + (mg_sin_theta**2)/((mu_B*N_B)**2))
            sqrt_v_Ainv_v = np.sqrt( vs[:,t-1].T @ Ainv @ vs[:,t-1] )
            c_g_v = 2 * c * mg_sin_theta * vy
            
            print("constant:", const_sqrt_v_Ainv_v)
            if const_sqrt_v_Ainv_v < 0:
                prog.AddLinearConstraint(vy >= 0)
            else:
                prog.AddConstraint( const_sqrt_v_Ainv_v * sqrt_v_Ainv_v - c_g_v >= 0 )
        else:
            sqrt_v_Ainv_v = np.sqrt( vs[:,t-1].T @ Ainv @ vs[:,t-1] )
            Ainvsqrt_v = Ainvsqrt @ vs[:,t-1]
            gf_B_gf = (mg_sin_theta**2)/((mu_B*N_B)**2)
            const_v_Ainv_B_gf = 2.0/gf_B_gf
            v_Ainv_B_gf = mg_sin_theta * vy * c
            
            cone_var = np.concatenate((np.array([(const_v_Ainv_B_gf * v_Ainv_B_gf)]), Ainvsqrt_v))
            
            prog.AddLorentzConeConstraint(cone_var)
            # prog.AddConstraint(sqrt_v_Ainv_v - const_v_Ainv_B_gf * v_Ainv_B_gf <= 0)
            
            prog.AddConstraint(kv * (vs[0,t-1] + vs[1,t-1])**2 - vs[2,t-1]**2 >= 0)
        
        # make sure qleft and qright yaw are between -pi and pi
        prog.AddBoundingBoxConstraint(-np.pi, np.pi, qlefts[2,t])
        prog.AddBoundingBoxConstraint(-np.pi, np.pi, qrights[2,t])
    
    #add terminal cost
    prog.AddQuadraticCost(1e6 * Qmatvec @ (desired_obj2left_se2 - qlefts[:,-1])**2)
    prog.AddQuadraticCost(1e6 * Qmatvec @ (desired_obj2right_se2 - qrights[:,-1])**2)
    
    print(GetProgramType(prog))
    print([solver.name() for solver in GetAvailableSolvers(GetProgramType(prog))])
    snopt_solver = SnoptSolver()
    result = snopt_solver.Solve(prog)
    if result.is_success():
        print("Success")
    else:
        print("Fail")
    
    qlefts_opt = result.GetSolution(qlefts)
    qrights_opt = result.GetSolution(qrights)
    qobjs_opt = result.GetSolution(qobjs)
    vs_opt = result.GetSolution(vs)
    return qlefts_opt, qrights_opt, vs_opt

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    palm_radius = 0.04
    obj2left_se2  = np.array([0,0,np.pi])
    obj2right_se2 = np.array([0,0,0])
    
    desired_obj2left_se2  = np.array([-0.00, -0.03, np.pi])
    desired_obj2right_se2 = np.array([-0.00, -0.03, np.pi/6])
    
    dls_params = DualLimitSurfaceParams(mu_A = 2.0, r_A = 0.04, N_A = 15.0, mu_B = 2.0, r_B = 0.03, N_B = 20.0)
    obj2left, obj2right, vs = inhand_planner(obj2left_se2, obj2right_se2, desired_obj2left_se2, desired_obj2right_se2, dls_params, steps = 5, angle = 60 * np.pi/180, palm_radius=palm_radius, kv = 20.0)
    
    print(np.round(desired_obj2left_se2 - obj2left[:,-1],4))
    print(np.round(desired_obj2right_se2 - obj2right[:,-1],4))
    
    # plot (2,1) subplots, draw xy and yaw vector
    fig, axs = plt.subplots(2,1)
    axs[0].plot(obj2left[0,:], obj2left[1,:], 'r')
    axs[0].quiver(obj2left[0,:], obj2left[1,:], np.cos(obj2left[2,:]), np.sin(obj2left[2,:]), color='b', scale=20)
    axs[0].set_title('Left Object')
    
    axs[1].plot(obj2right[0,:], obj2right[1,:], 'r')
    axs[1].quiver(obj2right[0,:], obj2right[1,:], np.cos(obj2right[2,:]), np.sin(obj2right[2,:]), color='b', scale=20)
    axs[1].set_title('Right Object')
    plt.show()