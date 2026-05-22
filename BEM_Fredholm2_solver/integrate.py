import numpy as np

class Transforms:
	def __init__(self, points):
		self.xlambda = lambda lda : ( lda[0]*points[0] + lda[1]*points[1] + lda[2]*points[2] )

def integral_surface_fdx(mesh, f, printout=False):
	Int_sum = 0.
	for k in range(mesh.ncells):
		Int_sum += f(mesh.cells[k].center)*mesh.cells[k].mu
		if printout : print(f"step {k} : Int_sum = {Int_sum}, f = {f(mesh.cells[k].center)}, mu = {mesh.cells[k].mu}")
	return Int_sum

def integral_surface_cellfunc_fdx(mesh, cfunc, x, f, printout=False):
	#cfunc -- function known only by cell index
	eps = mesh.diameter
	Int_sum = 0.
	for k in range(mesh.ncells):
		if np.linalg.norm(x - mesh.cells[k].center) > eps :
			res = cfunc[k]*f(mesh.cells[k].center)*mesh.cells[k].mu
		else:
			res = singular_integral_Gauss3p2o_sigma_fdx(mesh.cells[k], lambda x : cfunc[k]*f(x), printout)  #if x in a close area -- calculate using Gauss quadrature
		Int_sum += res
		if printout : print(f"step {k} : Int_sum = {Int_sum}, f = {f(mesh.cells[k].center)}, mu = {mesh.cells[k].mu}")
	return Int_sum

def integral_sigma_fdx(sigma, f, printout=False):
	Int = f(sigma.center)*sigma.mu
	if printout : print(f"step {k} : Int_sum = {Int_sum}, f = {f(sigma.center)}, mu = {sigma.mu}")
	return Int

def singular_integral_Gauss3p2o_sigma_fdx(sigma, f, printout=False):
	J = 2*sigma.mu #jacobian
	#point at parametric triangle
	q = np.array([
	[0, 0.5, 0.5],
	[0.5, 0, 0.5],
	[0.5, 0.5, 0]])
	w = 1/6 #weights are all same =1/6 for this quadrature
	Trs = Transforms(sigma.points) #transformation to physical triangle
	Int_Gauss = 0.
	for i in range(3):
		lda = q[i]
		x = Trs.xlambda(lda)
		Int_Gauss += f(x)
		if printout : print(f"step {i} : Int_sum_Gauss = {Int_sum_Gauss}, f = {f(x)}")
	return Int_Gauss*J*w
