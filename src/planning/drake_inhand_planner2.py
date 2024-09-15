from pydrake.all import (
    MathematicalProgram,
    GetProgramType,
    SnoptSolver,
    GetAvailableSolvers,
    eq,
    IpoptSolver
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
    
    prog = MathematicalProgram()
    obj2lefts = prog.NewContinuousVariables(3, steps, "obj2left") #obj in left hand frame
    obj2rights = prog.NewContinuousVariables(3, steps, "obj2right") # obj in right hand frame
    left2contactframe_yaws = prog.NewContinuousVariables(steps, "obj2contactframe_yaw") # yaw of left hand frame in contact frame
    right2contactframe_yaws = prog.NewContinuousVariables(steps, "obj2contactframe_yaw") # yaw of right hand frame in contact frame
    vs = prog.NewContinuousVariables(3, steps-1, "v") #vs is in contact frame
    
    Qmat = np.diag([1e4, 1e4, 1e2])
    Rmat = np.eye(3) * 1e-2
    for t in range(steps):
        prog.AddQuadraticCost((obj2lefts[:,t] - desired_obj2left_se2).T @ Qmat @ (obj2lefts[:,t] - desired_obj2left_se2))
        prog.AddQuadraticCost((obj2rights[:,t] - desired_obj2right_se2).T @ Qmat @ (obj2rights[:,t] - desired_obj2right_se2))
        if t == 0:
            # add initial conditions
            prog.AddLinearEqualityConstraint(obj2lefts[:,0], obj2left_se2)
            prog.AddLinearEqualityConstraint(obj2rights[:,0], obj2right_se2)
            
            prog.AddLinearEqualityConstraint(left2contactframe_yaws[0], 0)
            prog.AddLinearEqualityConstraint(right2contactframe_yaws[0], 0)
            continue
        else:
            # prog.AddQuadraticCost(vs[:,t-1].T @ Rmat @ vs[:,t-1])
            pass
        
        if t % 2 == 1:
            # obj2left is moving, obj2right is still
            prog.AddConstraint(eq(obj2lefts[:,t], obj2lefts[:,t-1] + Rz(left2contactframe_yaws[t-1]).T @ vs[:,t-1]))
            prog.AddLinearConstraint(eq(obj2rights[:,t], obj2rights[:,t-1]))
            
            # left2contactframe yaw doesn't change, right2contactframe yaw changes with vs[2,t-1]
            prog.AddLinearEqualityConstraint(left2contactframe_yaws[t] - left2contactframe_yaws[t-1], 0)
            prog.AddLinearEqualityConstraint(right2contactframe_yaws[t] - (right2contactframe_yaws[t-1] + vs[2,t-1]), 0)
        else:
            # obj2right is moving, obj2left is still
            prog.AddConstraint(eq(obj2rights[:,t], obj2rights[:,t-1] + Rz(right2contactframe_yaws[t-1]).T @ vs[:,t-1]))
            prog.AddLinearConstraint(eq(obj2lefts[:,t], obj2lefts[:,t-1]))
            
            # right2contactframe yaw doesn't change, left2contactframe yaw changes with vs[2,t-1]
            prog.AddLinearEqualityConstraint(right2contactframe_yaws[t] - right2contactframe_yaws[t-1], 0)
            prog.AddLinearEqualityConstraint(left2contactframe_yaws[t] - (left2contactframe_yaws[t-1] + vs[2,t-1]), 0)
        
        # ensure obj2left and obj2right are in circle
        # \| qleft[:2] \|^2 <= palm_radius^2
        # qleft.T @ diag(1,1,0) @ qleft - palm_radius**2 <= 0
        prog.AddQuadraticAsRotatedLorentzConeConstraint(2*np.diag([1,1,0]), np.zeros(3), -palm_radius**2, obj2lefts[:,t])
        
        # \| qright[:2] \|^2 <= palm_radius^2
        prog.AddQuadraticAsRotatedLorentzConeConstraint(2*np.diag([1,1,0]), np.zeros(3), -palm_radius**2, obj2rights[:,t])
        
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
            c_g_v = -2 * c * mg_sin_theta * vy
            Ainvsqrt_v = Ainvsqrt @ vs[:,t-1]
            
            print("constant:", const_sqrt_v_Ainv_v)
            if const_sqrt_v_Ainv_v < 0:
                prog.AddLinearConstraint(vy <= 0) # go in -y direction
            else:
                # add cone constraint c*sqrt(v^T Ainv v) <= c_g_v where c > 0
                cone_var = np.concatenate((np.array([c_g_v / const_sqrt_v_Ainv_v]), Ainvsqrt_v))
                prog.AddLorentzConeConstraint(cone_var)
        else:
            Ainvsqrt_v = Ainvsqrt @ vs[:,t-1]
            gf_B_gf = (mg_sin_theta**2)/((mu_B*N_B)**2)
            const_v_Ainv_B_gf = -2.0/gf_B_gf
            v_Ainv_B_gf = mg_sin_theta * vy * c
            
            cone_var = np.concatenate((np.array([(const_v_Ainv_B_gf * v_Ainv_B_gf)]), Ainvsqrt_v))
            
            prog.AddLorentzConeConstraint(cone_var)
            
            prog.AddConstraint(kv * (vs[0,t-1] + vs[1,t-1])**2 - vs[2,t-1]**2 >= 0)
            
        prog.AddBoundingBoxConstraint(-np.pi, np.pi, obj2lefts[2,t])
        prog.AddBoundingBoxConstraint(-np.pi, np.pi, obj2rights[2,t])
        
    #add terminal cost
    # prog.AddQuadraticCost((obj2lefts[:,-1] - desired_obj2left_se2).T @ (Qmat*1e10) @ (obj2lefts[:,-1] - desired_obj2left_se2))
    # prog.AddQuadraticCost((obj2rights[:,-1] - desired_obj2right_se2).T @ (Qmat*1e10) @ (obj2rights[:,-1] - desired_obj2right_se2))
    prog.AddLinearConstraint(eq(obj2lefts[:,-1], desired_obj2left_se2))
    prog.AddLinearConstraint(eq(obj2rights[:,-1], desired_obj2right_se2))
    
    
    # initial guesses
    prog.SetInitialGuess(obj2lefts, np.ones((3,steps)) * 20.0)
    prog.SetInitialGuess(obj2rights, np.ones((3,steps)) * 20.0)
    prog.SetInitialGuess(vs, np.ones((3,steps-1)) * 0.1)
    
    print(GetProgramType(prog))
    print([solver.name() for solver in GetAvailableSolvers(GetProgramType(prog))])
    snopt_solver = IpoptSolver()
    result = snopt_solver.Solve(prog)
    if result.is_success():
        print("Success")
    else:
        print("Fail")
        print(result.GetInfeasibleConstraints(prog))

    obj2lefts_opt = result.GetSolution(obj2lefts)
    obj2rights_opt = result.GetSolution(obj2rights)
    vs_opt = result.GetSolution(vs)
    
    return obj2lefts_opt, obj2rights_opt, vs_opt
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    palm_radius = 0.032
    obj2left_se2  = np.array([0,0,0])
    obj2right_se2 = np.array([0,0,np.pi])
    
    desired_obj2left_se2 = np.array([0.0, 0.03, 0])
    desired_obj2right_se2 = np.array([0.0, -0.03, np.pi])
    
    dls_params = DualLimitSurfaceParams(mu_A = 0.75, r_A = 0.04, N_A = 20.0, mu_B = 0.75, r_B = 0.04, N_B = 20.0)
    obj2left, obj2right, vs = inhand_planner(obj2left_se2, obj2right_se2, desired_obj2left_se2, desired_obj2right_se2, dls_params, steps = 5, angle = 45, palm_radius=palm_radius, kv = 20.0)
    
    print(np.round(desired_obj2left_se2 - obj2left[:,-1],4))
    print(np.round(desired_obj2right_se2 - obj2right[:,-1],4))
    
    print(np.round(obj2left,4))
    print(np.round(obj2right,4))
    print(np.round(vs,4))
    
    
    # plot (2,1) subplots, draw xy and yaw vector
    fig, axs = plt.subplots(2,1)
    axs[0].plot(obj2left[0,:], obj2left[1,:], 'ro-')
    axs[0].quiver(obj2left[0,:], obj2left[1,:], np.cos(obj2left[2,:]), np.sin(obj2left[2,:]), color='b', scale=5)
    axs[0].set_title('Left Object')
    
    axs[1].plot(obj2right[0,:], obj2right[1,:], 'ro-')
    axs[1].quiver(obj2right[0,:], obj2right[1,:], np.cos(obj2right[2,:]), np.sin(obj2right[2,:]), color='b', scale=5)
    axs[1].set_title('Right Object')
    plt.show()