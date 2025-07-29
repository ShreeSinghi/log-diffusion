import gpytoolbox as gpy
import polyscope as ps
import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import inv
import polyscope.imgui as psim

V, F = gpy.read_mesh('data/spot_low_resolution.obj')

L = gpy.cotangent_laplacian(V, F)
M = gpy.massmatrix(V,F)
DT = 1e-3
N_VERTICES = V.shape[0]

N_STEPS = 100

u = np.zeros(N_VERTICES)
u[[0, 300, 700]] = 10


M_inverse = csr_matrix((1/M.data, M.indices, M.indptr))

A = M_inverse @ L
matrix = inv(eye(N_VERTICES) + DT * A)
us = [u.copy()]
for i in range(N_STEPS-1):
    u = matrix @ u
    us.append(u.copy())


ps.init()

ps_cow = ps.register_surface_mesh("cow", V, F)

heat_quantity = ps_cow.add_scalar_quantity("heat diffusion", us[0], enabled=True, vminmax=(0, 1))

current_step = 0

def callback():
    global current_step
    
    changed, current_step = psim.SliderInt("Time step", current_step, 0, N_STEPS-1)
    
    if changed:
        ps_cow.add_scalar_quantity("heat diffusion", us[current_step], enabled=True, vminmax=(0, 1))


ps.set_user_callback(callback)

ps.show()

