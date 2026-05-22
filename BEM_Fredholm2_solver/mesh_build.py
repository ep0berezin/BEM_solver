import numpy as np

class mesh:
	def __init__(self, cells):
		self.cells = cells #list of mesh cells.
		self.ncells = len(cells) #cells[-1].id+1 #number of mesh cells.
		self.diameter = max(cell.h for cell in cells) #partition (i.e. mesh) diameter.
	def weightedcenter(self):
		mu_x_center = np.zeros(self.cells[0].points[:,0].shape[0])
		sum_mu = 0.
		for cell in self.cells:
			mu_x_center += cell.mu*cell.center
			sum_mu += cell.mu
		polyhedron_center = mu_x_center/sum_mu
		return polyhedron_center

class meshcell:
	def __init__(self, cellid, points, npoints, adjcells):
		#args:
		#cellid = cell unique id
		#points = ndarray( [[x1_1, x2_1, ... xnpts_1],[x1_2, .... xnpts_2], ..., [x1_n, ..., xnpts_ndim]]) so points[:,k] = k-th point coord.
		#npoints = number of points in cell
		#adjcells = list of adjacent cells (by id's)
		#vals = values associated with cell.
		self.id = cellid
		self.points = points 
		self.npoints = npoints
		self.center = np.sum(points, axis=1)/npoints
		self.adjcells = adjcells
		self.values = {}
		###
		self.partcells = [] #list of cells = partiton of this cell. Used for integrating singular integrals.

class cell0dW: #weighted point, a.k.a. point with a measure.
	def __init__(self, cellid, center, measure, normal, printout=False):
		self.id = cellid
		self.center = center
		self.mu = measure
		self.h = 0.
		self.n = normal
		self.celltype = "Weighted point"
		if printout:
			print(f"cell info: \ncell_id: {self.id}, type = {self.celltype}")
			print(f"cell vector variables: \ncenter = \n{self.center}")
			print(f"cell scalar variables: \nmu = {self.mu}")

class cell0d(meshcell): #0d cell -- i.e. point
	def __init__(self, cellid, points, adjcells, printout=False):
		super().__init__(cellid, points, 1, adjcells)
		self.celltype = "0d"
		self.h = 0.
		if printout:
			print(f"cell info: \ncell_id: {self.id}, type = {self.celltype}, adjacents: {self.adjcells}")
			print(f"cell vector variables: \npoints = \n{self.points}, \ncenter = \n{self.center}")
			#print(f"cell scalar variables: \nmu = {self.mu}, diameter = {self.h}")

class cell1d(meshcell): #1d cell such as interval
	def __init__(self, cellid, points, adjcells, printout=False):
		super().__init__(cellid, points, 2, adjcells)
		self.mu = np.linalg.norm(points[:,1] - points[:,0], ord = 2)
		self.h = self.mu #mu(sigma)
		self.celltype = "1d"
		if printout:
			print(f"cell info: \ncell_id: {self.id}, type = {self.celltype}, adjacents: {self.adjcells}")
			print(f"cell vector variables: \npoints = \n{self.points}, \ncenter = \n{self.center}")
			print(f"cell scalar variables: \nmu = {self.mu}, diameter = {self.h}")

class cell2d_3(meshcell): #2d (i.e. facet) triangular cell
	def __init__(self, cellid, points, adjcells, printout=False):
		super().__init__(cellid, points, 3, adjcells)
		edge0 = points[:,0]-points[:,1] 
		edge1 = points[:,2]-points[:,1]
		edge2 = points[:,0]-points[:,2]
		cross = np.cross(edge1,edge0)
		self.mu = 0.5*np.linalg.norm(cross, ord = 2)
		self.n = cross/np.linalg.norm(cross, ord = 2)
		self.h = np.max([np.linalg.norm(edge0, ord = 2), np.linalg.norm(edge1, ord = 2), np.linalg.norm(edge2, ord = 2)] )
		self.celltype = "2d_3"
		if printout:
			print(f"cell info: \ncell_id: {self.id}, type = {self.celltype}, adjacents: {self.adjcells}")
			print(f"cell vector variables: \npoints = \n{self.points}, \ncenter = \n{self.center}, \nnormal vector = \n{self.n}")
			print(f"cell scalar variables: \nmu = {self.mu}, diameter = {self.h}")

class cell2d_4(meshcell): #2d (i.e. facet) tetragonal cell
	def __init__(self, cellid, points, adjcells, printout=False):
		super().__init__(cellid, points, 4, adjcells)
		edge0 = points[:,0]-points[:,1]
		edge1 = points[:,1]-points[:,2]
		edge2 = points[:,2]-points[:,3]
		edge3 = points[:,3]-points[:,0]
		diag0 = points[:,0]-points[:,2]
		diag1 = points[:,1]-points[:,3]
		
		vec0 = 0.5*(points[:,3] + points[:,0]) - 0.5*(points[:,0] + points[:,1])
		vec1 = 0.5*(points[:,2] + points[:,1]) - 0.5*(points[:,0] + points[:,1])
		
		
		cross = np.cross(vec1,vec0)
		self.mu = 2.*np.linalg.norm(cross, ord = 2)
		self.n = cross/np.linalg.norm(cross, ord = 2)
		self.h = np.max(
		[np.linalg.norm(edge0, ord = 2), 
		np.linalg.norm(edge1, ord = 2), 
		np.linalg.norm(edge2, ord = 2), 
		np.linalg.norm(edge3, ord = 2),
		np.linalg.norm(diag0, ord = 2),
		np.linalg.norm(diag1, ord = 2)] )
		self.celltype = "2d_4"
		if printout:
			print(f"cell info: \ncell_id: {self.id}, type = {self.celltype}, adjacents: {self.adjcells}")
			print(f"cell vector variables: \npoints = \n{self.points}, \ncenter = \n{self.center}, \nnormal vector = \n{self.n}")
			print(f"cell scalar variables: \nmu = {self.mu}, diameter = {self.h}")
