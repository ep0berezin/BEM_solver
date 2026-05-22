import mesh_build as msb
import pyvista as pv
import numpy as np

def normals_method(mesh_Sigma, points): #normals method to check if point is in the polyhedron
	#WARNING : works only for convex polyhedra!
	outside_idxs = []
	for k in range(points.shape[0]):
		for cell in mesh_Sigma.cells:
			dotprod = (cell.center - point[k]).T@cell.n
			if dotprod < 0 : outside_idxs.append(k)
	return points[outside_idx]

def filter_points_pyvista(points, mesh_Sigma, show_visualization=True):

	mesh_triangles = []
	mesh_vertices = []
	
	for k in range(mesh_Sigma.ncells):
		cell_points = mesh_Sigma.cells[k].points
		mesh_vertices.append(cell_points[:, 0])
		mesh_vertices.append(cell_points[:, 1]) 
		mesh_vertices.append(cell_points[:, 2])
		mesh_triangles.append([3*k, 3*k + 1, 3*k + 2])
	
	mesh_vertices = np.array(mesh_vertices)
	mesh_triangles = np.array(mesh_triangles)

	faces = np.insert(mesh_triangles, 0, 3, axis=1).flatten()
	mesh = pv.PolyData(mesh_vertices, faces)

	point_cloud = pv.PolyData(points)
	enclosed = point_cloud.select_enclosed_points(mesh, check_surface=False)
	outside_mask = enclosed['SelectedPoints'] == 0
	
	print(f"Total points: {len(points)}")
	print(f"Outside: {np.sum(~outside_mask)}")
	print(f"Inside: {np.sum(outside_mask)}")
	
	if show_visualization:
		plotter = pv.Plotter()
		plotter.add_mesh(mesh, color='lightblue', opacity=0.7, label='Polyhedron')
		
		outside_points = points[outside_mask]
		if len(outside_points) > 0:
			outside_cloud = pv.PolyData(outside_points)
			plotter.add_mesh(outside_cloud, color='green', point_size=8, label='Outside')
		
		inside_points = points[~outside_mask]
		if len(inside_points) > 0:
			inside_cloud = pv.PolyData(inside_points)
			plotter.add_mesh(inside_cloud, color='red', point_size=8, label='Inside')
		
		plotter.add_legend()
		plotter.show()
	
	return points[outside_mask]
    
def planemesh_x(mesh_Sigma, e1, e2, R, npts_resolution, printout=False):
	#mesh_Sigma -- surface mesh over surface Sigma.
	#e1, e2 -- directional vectors of plane.
	#R -- distance from mesh_Sigma at which mesh_x exists.
	meshcenter = mesh_Sigma.weightedcenter()
	lim1 = R*(e1/np.linalg.norm(e1))
	lim2 = R*(e2/np.linalg.norm(e2))
	e1_linspace = np.linspace(meshcenter - lim1, meshcenter + lim1, npts_resolution)
	e2_linspace = np.linspace(meshcenter - lim2, meshcenter + lim2, npts_resolution)
	i_idxs, j_idxs = np.meshgrid(np.arange(npts_resolution), np.arange(npts_resolution))
	points_from_e1 = e1_linspace[i_idxs]
	points_from_e2 = e2_linspace[j_idxs]
	plane_points = points_from_e1 + points_from_e2 - meshcenter
	points = plane_points.reshape(-1, 3)
	points = filter_points_pyvista(points, mesh_Sigma)
	cells = []
	for k in range(len(points)):
		cell = msb.cell0d(k, points[k].reshape(-1,1), [-1])
		cells.append(cell)
	mesh_x = msb.mesh(cells)
	return mesh_x

def linmesh_x(mesh_Sigma, e1, R, npts_resolution, printout=False):
	meshcenter = mesh_Sigma.weightedcenter()
	lim1 = R*(e1/np.linalg.norm(e1))
	e1_linspace = np.linspace(meshcenter - lim1, meshcenter + lim1, npts_resolution)
	points = []
	for i in range(len(e1_linspace)):
		points.append(e1_linspace[i])
	points = np.array(points)
	points = points.reshape(-1, 3)
	points = filter_points_pyvista(points, mesh_Sigma)
	cells = []
	for k in range(len(points)):
		cell = msb.cell0d(k, points[k].reshape(-1,1), [-1])
		cells.append(cell)
	mesh_x = msb.mesh(cells)
	return mesh_x
