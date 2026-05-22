import numpy as np
import mesh_build as msb

#Marchuk INM format
def triagmesh_inmf(fname, printout=False): #for triangular meshes
	print(f"Warning! This mesh doesnt'support adjacency, so adjacents set as -1")
	cells = []
	
	file = open(fname,"r")

	npoints = int(file.readline())
	points = np.zeros((3,npoints))
	for k in range(npoints):
		points_vector = np.array(file.readline().split()).astype(float)
		points[:,k] = points_vector
	ncells = int(file.readline()) 
	for k in range(ncells):
		cell_points = np.array(file.readline().split()).astype(int)
		p0 = points[:,cell_points[0]-1].reshape(-1,1)
		p1 = points[:,cell_points[1]-1].reshape(-1,1)
		p2 = points[:,cell_points[2]-1].reshape(-1,1)
		p = np.hstack((p0,p1,p2))
		cell = msb.cell2d_3(k, p, [-1], printout)
		cells.append(cell)

	file.close()
	mesh = msb.mesh(cells)
	return mesh

def tetramesh_inmf(fname, printout=False): #for tetragonal meshes
	print(f"Warning! This mesh doesnt'support adjacency, so adjacents set as -1")
	cells = []
	
	file = open(fname,"r")

	npoints = int(file.readline())
	points = np.zeros((3,npoints))
	for k in range(npoints):
		points_vector = np.array(file.readline().split()).astype(float)
		points[:,k] = points_vector
	ncells = int(file.readline()) 
	for k in range(ncells):
		cell_points = np.array(file.readline().split()).astype(int)
		p0 = points[:,cell_points[0]-1].reshape(-1,1)
		p1 = points[:,cell_points[1]-1].reshape(-1,1)
		p2 = points[:,cell_points[2]-1].reshape(-1,1)
		p3 = points[:,cell_points[3]-1].reshape(-1,1)
		p = np.hstack((p0,p1,p2,p3))
		cell = msb.cell2d_4(k, p, [-1], printout)
		cells.append(cell)

	file.close()
	mesh = msb.mesh(cells)
	return mesh

#GMSH ver.2 format

def triagmesh_gmsh2(fname, printout=False): #for triangular meshes
	print(f"Warning! This mesh doesnt'support adjacency, so adjacents set as -1")
	cells = []
	
	file = open(fname,"r")
	for line in file:
		#find $Nodes
		#print(line)
		if line.strip() == "$Nodes":
			if printout : print(f"$Nodes")
			npoints = int(file.readline())
			points = np.zeros((3,npoints)) 
			for k in range(npoints):
				points_vector = np.array(file.readline().split()).astype(float)[1:]
				points[:,k] = points_vector
			
		#find $Elements
		if line.strip() == "$Elements":
			if printout : print(f"$Elements")
			ncells = int(file.readline()) #ncells = n of ALL cells, i.e. lines and points -- because thats how gmsh works.
			face_k = -1 
			for k in range(ncells):
				line = file.readline().split()
				if line[1] == '2':
					face_k += 1
					cell_points_ids = np.array(line).astype(int)[-3:]
					p0 = points[:,cell_points_ids[0]-1].reshape(-1,1)
					p1 = points[:,cell_points_ids[1]-1].reshape(-1,1)
					p2 = points[:,cell_points_ids[2]-1].reshape(-1,1)
					p = np.hstack((p0,p1,p2))
					cell = msb.cell2d_3(face_k, p, [-1], printout)
					cells.append(cell)

	file.close()
	mesh = msb.mesh(cells)
	return mesh


