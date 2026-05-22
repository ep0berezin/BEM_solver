import numpy as np
from integrate import *
from solve_Fredholm2 import *

def int_Sigma_at_x(mesh, x, nu, Phi, cnorm=False, printout=False): # Integrates arbitrary function phi over Sigma.
	norm = None 
	#Not so fancy with this separate x and Phi transfer to function. But this easily allows to calculate separately in near and far areas
	S = integral_surface_cellfunc_fdx(mesh=mesh, cfunc=nu, x=x, f=lambda y : Phi(x, y))
	if printout : print(f"Int at \n{x} = \n{S}")
	if cnorm:
		norm = np.linalg.norm(S, ord = 'fro')
		if printout : print(f"|| grad V_sigma(x) || = {np.linalg.norm(S, ord = 'fro')}")
	return S, norm

def int_Sigma_at_mesh_x(mesh_Sigma, mesh_x, nu, Phi, val_name):
	for k in range(mesh_x.ncells):
		S,_ = int_Sigma_at_x(mesh_Sigma, mesh_x.cells[k].center, nu, Phi, cnorm=False, printout=False) #calculate integral
		mesh_x.cells[k].values[val_name] = S #write value to cell
