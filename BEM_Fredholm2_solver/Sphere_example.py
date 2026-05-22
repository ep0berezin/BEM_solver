import mesh_read as msr
from solve_Fredholm2 import *
from compute_field import *
from mesh_x_create import *
import scipy.sparse as scsp
import visualize_solution as vs


class Functions:
	def __init__(self, k):
		p_bc_mag = 10. #for debug pressure Neumann BC
		self.w_bc = lambda x : np.array([x[1],-x[0],3*x[2]]) #boundary condition to filtration velocity w
		self.w_0_bc = lambda x : np.array([0.,0.,0.]) # zeroboundary condition to filtration velocity w : for tests
		self.g = lambda u, n : ((-1/k)*u.T@n) #for Neumann B.C. dp/dn_x = g = -1/k(w_bc, n)
		self.p_bc = lambda u, n : n.T@n*p_bc_mag
		#Final BC function
		self.bc_func = lambda x, n: self.g(self.w_bc(x), n)
		
		self.F = lambda x, y : (1/np.pi*4)*(1/np.linalg.norm(x-y)) #single layer potential's function : Laplace's equation fundamental solution
		self.grad_x_F = lambda x, y : -1/(np.pi*4)*(x-y)*(1/np.linalg.norm(x-y)**3)  #Gradient of Laplace's equation fundamental solution
		self.dFdn_x = lambda x, y, n : (self.grad_x_F(x,y)).T@n #dF/dn_x = grad_x F * n 
		#Final integral kernel function
		self.IntegralKernel = lambda x, y, n : self.dFdn_x(x,y,n)

class GMRES_scipy: #solver
	def __init__(self, x_0, m_Krylov, maxiter, eps=1e-10, printout=True):
		self.x_0 = x_0
		self.m = m_Krylov
		self.maxiter = maxiter
		self.eps = eps
		self.printout = printout
		self.cb = lambda res : print(f"GMRES relative residual = {res}")  if printout else None
	def create_solver(self, matvec, b):
		self.LinOp = scsp.linalg.LinearOperator((b.shape[0], b.shape[0]), matvec = matvec)
		self.b = b
	def __call__(self):
		u, _ = scsp.linalg.gmres(self.LinOp, self.b, x0=self.x_0, rtol=self.eps, restart=self.m, maxiter=self.maxiter, M=None, callback=self.cb, callback_type='pr_norm') 
		return u

def get_points_with_solution(mesh_Sigma, e1, e2, k_filter):
	R = 100.
	divs = 50
	mesh_x = planemesh_x(mesh_Sigma, e1, e2, R, divs)
	
	functions = Functions(k_filter)
	x_0 = np.zeros(mesh_Sigma.ncells)
	GMRES_solver = GMRES_scipy(x_0 = x_0, m_Krylov=15, maxiter=100, printout=True)
	Fredholm2 = Fredholm2Solver(mesh_Sigma, functions, linear_solver = GMRES_solver) 
	print("Assembling matrix and RHS")
	Fredholm2.assemble_system()
	print("Solving system")
	nu = Fredholm2.solve_system()
	print("Integrating pressure")
	#p
	int_Sigma_at_mesh_x(mesh_Sigma, mesh_x, nu, functions.F, "pressure") #get pressure values and write to mesh nodes
	print("Integrating velocity")
	#w
	int_Sigma_at_mesh_x(mesh_Sigma, mesh_x, nu, lambda x, y : -k_filter*functions.grad_x_F(x,y), "velocity") #get velocity values and write to mesh nodes
	return mesh_x
	
if __name__ == "__main__":
	mesh_Sigma = msr.triagmesh_gmsh2(f"meshes/sphere_gmsh/sphere_3.msh", printout=False)
	k_filter = 0.5
	e1 = np.array([1,0,0])
	e2 = np.array([0,1,0])
	mesh_x = get_points_with_solution(mesh_Sigma, e1, e2, k_filter)
	
	velocity = []
	pressure = []
	velocity_magnitude = []
	
	for cell in mesh_x.cells:
		velocity.append(cell.values["velocity"])
		pressure.append(cell.values["pressure"])
		velocity_magnitude.append(np.linalg.norm(cell.values["velocity"]))
	print(f"Max pressure = {np.max(pressure)} at point \n{mesh_x.cells[np.argmax(pressure)].points}")
	print(f"Max velocity magnitude = {np.max(velocity_magnitude)} at point \n{mesh_x.cells[np.argmax(velocity_magnitude)].points}")
	
	vs.plot_vector_field_2D_at_plane(mesh_x, e1, e2, "velocity")
	vs.plot_vector_field_3D(mesh_Sigma, mesh_x, "velocity")
	vs.plot_scalar_field_3D(mesh_Sigma, mesh_x, "pressure")
	vs.plot_scalar_field_2D_at_plane(mesh_x, e1, e2, "pressure")
	
