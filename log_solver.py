import gpytoolbox as gpy
import polyscope as ps
import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import inv
import polyscope.imgui as psim
import cvxpy as cp

def build_f_to_v_matrix(V, F):
    n_vertices = V.shape[0]
    n_faces = F.shape[0]

    I = F.flatten() 
    J = np.repeat(np.arange(n_faces), 3)  
    Vals = np.ones_like(I)

    A = csr_matrix((Vals, (I, J)), shape=(n_vertices, n_faces))

    face_counts = np.array(A.sum(axis=1)).flatten()
    face_counts[face_counts == 0] = 1  # avoid division by zero
    D_inv = csr_matrix((1.0 / face_counts, (np.arange(n_vertices), np.arange(n_vertices))),
                       shape=(n_vertices, n_vertices))

    F_to_V = D_inv @ A  # shape: (n_vertices, n_faces)

    return F_to_V

def norm_squared_3d(x):
    if isinstance(x, np.ndarray):
        return np.power(x, 2).reshape((3, -1)).sum(axis=0)
    return cp.power(x, 2).reshape((3, -1)).sum(axis=0)

def norm_squared(x):
    return (x**2).sum()

V, F = gpy.read_mesh('data/spot_low_resolution.obj')

L = gpy.cotangent_laplacian(V, F)
M = gpy.massmatrix(V,F)
DT = 1e-3
N_VERTICES = V.shape[0]

N_STEPS = 10

u = np.zeros(N_VERTICES)
u[[700]] = 1

method = "backward" # or "backward" or "semiimplicit" or "backward"
M_inverse = csr_matrix((1/M.data, M.indices, M.indptr))

L = M_inverse @ L

G = gpy.grad(V, F)

# averages any quantity from faces to vertices
F_to_V = build_f_to_v_matrix(V, F)

us = [u.copy()]
Gs = [norm_squared_3d(G @ u)]
VGs = [F_to_V @ Gs[0]]

if method == "semiimplicit":
    # Semi-implicit method requires solving a linear system at each step
    A = eye(N_VERTICES) + DT * L
    A_inv = inv(A)

for i in range(N_STEPS-1):
    if method=="forward":
        laplaceian = L @ u
        grad_sq = F_to_V @ norm_squared_3d(G @ u)

        u = u + DT * (laplaceian + grad_sq)

        Gs.append(norm_squared_3d(G @ u))
        VGs.append(F_to_V @ Gs[-1])
    elif method=="semiimplicit":
        u = A_inv @ (u + DT * F_to_V @ norm_squared_3d(G @ u))
        # Gs.append(norm_squared_3d(G @ u))
    elif method=="backward":
        u_t = cp.Variable(N_VERTICES)
        laplaceian = L @ u_t
        g_sq = cp.power(G @ u_t, 2).reshape((3, -1)).sum(axis=0)

        g_sq_opt = cp.Variable(g_sq.shape)
        constraints = [g_sq_opt >= g_sq]

        grad_sq_expr = F_to_V @ g_sq_opt
        expr_inner = laplaceian + grad_sq_expr

        weighted_expr = DT * expr_inner + u - u_t  # affine + affine + affine

        expression = cp.sum_squares(weighted_expr)  # weighted_expr now affine

        prob = cp.Problem(cp.Minimize(expression), constraints)
        prob.solve(verbose=True, warm_start=True)
        u = u_t.value
        
    us.append(u.copy())

ps.init()

ps_cow = ps.register_surface_mesh("cow", V, F)

ps_cow.add_scalar_quantity("heat diffusion", us[0], enabled=True)
# ps_cow.add_scalar_quantity("gradient", Gs[0], enabled=True, defined_on="faces")
# ps_cow.add_scalar_quantity("vertex gradient", VGs[0], enabled=True)

current_step = 0

def callback():
    global current_step
    
    changed, current_step = psim.SliderInt("Time step", current_step, 0, N_STEPS-1)
    
    if changed:
        # ps_cow.add_scalar_quantity("gradient", Gs[current_step], enabled=True, defined_on="faces")
        # ps_cow.add_scalar_quantity("vertex gradient", VGs[current_step], enabled=True)
        print(us[current_step])
        ps_cow.add_scalar_quantity("heat diffusion", us[current_step], enabled=True)


ps.set_user_callback(callback)

ps.show()

