import numpy as np
import integrate as intg
import scipy.sparse as scsp

class GMRES_scipy: #solver
	def __init__(self, x_0, m_Krylov, maxiter, eps=1e-10, printout=True):
		#self.LinOp = scsp.linalg.LinearOperator((b.shape[0], b.shape[0]), matvec = matvec)
		#self.b = b
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
		u, _ = scsp.linalg.gmres(self.LinOp, self.b, x0=self.x_0, tol=self.eps, restart=self.m, maxiter=self.maxiter, M=None, callback=self.cb, callback_type='pr_norm') 
		return u

class MatrixVector:
	def __init__(self, A, c):
		self.A = A
		self.c = c #-1/2 for Fredholm II kind eq in Omega+
		self.n = A.shape[0]
		self.A_hat = self.c*np.eye(self.n)+self.A
	def mv(self, u):
		return self.A_hat@u
	def getSVD(self, r):
		self.U, self.sigma, self.V = np.linalg.svd(self.A_hat, full_matrices=False)
		self.U_t = self.U[:,:r]
		self.Sigma_t =  np.diag(self.sigma[:r])
		self.V_t = self.V[:r,:]
	def mv_SVD(self, u):
		return self.U_t@(self.Sigma_t@(self.V_t@u))

def assemble_matrix(mesh, F, eps = 1e-15, printout=False):
	N = mesh.ncells
	K = np.zeros((N,N))
	for i in range(N):
		for j in range(N):
			#Note: these variables may not be optimal, but this way is nore explicit.
			x = mesh.cells[i].center
			y = mesh.cells[j].center
			sigma_y = mesh.cells[j]
			n = mesh.cells[i].n
			res = 0.0
			if i==j: #x == y case: diagonal elements
				res = intg.singular_integral_Gauss3p2o_sigma_fdx(sigma_y, lambda y_j : F(x,y_j,n))
			elif i!=j: #x != y case: non-diagonal elements
				if np.linalg.norm(x-y) < mesh.diameter :
					res = intg.singular_integral_Gauss3p2o_sigma_fdx(sigma_y, lambda y_j : F(x,y_j,n))
				else: 
					res = intg.integral_sigma_fdx(sigma_y, lambda y_j : F(x,y_j,n))
			K[i,j] = res
			if printout : print(f"K[{i},{j}] = {res}")
	return K

def assemble_rhs(mesh, bc_func, printout=False):
	N = mesh.ncells
	gvec = np.zeros(N)
	for i in range(N):
		x = mesh.cells[i].center
		n = mesh.cells[i].n
		gvec[i] = bc_func(x,n) 
		if printout : print(f"g_{i} = {gvec[i]}")
	return gvec

class Fredholm2Solver:
	def __init__(self, mesh, functions, linear_solver=None, printout=False):
		self.mesh = mesh
		self.functions = functions
		self.linear_solver = GMRES_scipy() if linear_solver is None else linear_solver
		N = self.mesh.ncells
		self.IntegralKernel = self.functions.IntegralKernel
		self.bc_func = self.functions.bc_func
		self.printout = printout
	def assemble_system(self):
		K = assemble_matrix(self.mesh, self.IntegralKernel, printout = self.printout)
		g_tilde = assemble_rhs(self.mesh, self.bc_func, printout = self.printout)
		
		self.Matvecs = MatrixVector(K, -0.5)
		self.g_tilde = g_tilde
		
	def solve_system(self):
		mv = lambda u: self.Matvecs.mv(u)
		self.linear_solver.create_solver(matvec=mv, b=self.g_tilde)
		nu_tilde = self.linear_solver()
		return nu_tilde

###NOTES###
# * Maybe boundary conditions needs some generalization/"abstactization"
# * ???
