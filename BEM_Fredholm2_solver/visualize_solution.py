import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt

def plot_vector_field_3D(mesh_Sigma, mesh_x, field_name):
	
	#draw mesh_Sigma
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
	
	#draw data from mesh_x
	
	plotter = pv.Plotter()
	
	plotter.add_mesh(mesh, color='lightblue', opacity=0.7, label='Polyhedron')
	points = []
	vectors = []
	for cell_x in mesh_x.cells:
		points.append(cell_x.points.reshape(-1))
		vectors.append(cell_x.values[field_name])
	points = np.array(points)
	vectors = np.array(vectors)

	magnitudes = np.linalg.norm(vectors, axis = 1)
	
	point_cloud = pv.PolyData(points)
	point_cloud['vectors'] = vectors
	point_cloud['magnitude'] = magnitudes
	scale = 10.0
	
	#outdated plotting
	#plotter.add_arrows(point_cloud.points, point_cloud['vectors'],
	#mag=scale,
	#cmap='plasma',
	#scalar_bar_args={'title': f"Magnitude of {field_name}",'title_font_size': 16,'label_font_size': 12,'color': 'black', 'vertical': True })
	
	
	arrows = point_cloud.glyph(orient='vectors', scale=False, factor=scale)
	plotter.add_mesh(
		arrows, 
		cmap='plasma',
		scalars='magnitude',
		show_scalar_bar=True,
		scalar_bar_args={
			'title': f"Magnitude of {field_name}",
			'title_font_size': 16,
			'label_font_size': 12,
			'color': 'black',
			'vertical': False
		}
	)
	
	plotter.add_axes(x_color='red', y_color='green',z_color='blue',xlabel=r'X_1',ylabel=r'X_2',zlabel=r'X_3',labels_off=False, line_width=4)
	#plotter.add_scalar_bar(title=f"{field_name} magnitude")
	plotter.show()

def plot_scalar_field_3D(mesh_Sigma, mesh_x, field_name):
	
	#draw mesh_Sigma
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
	
	#draw data from mesh_x
	
	plotter = pv.Plotter()
	
	plotter.add_mesh(mesh, color='lightblue', opacity=0.7, label='Polyhedron')
	points = []
	scalfield = []
	for cell_x in mesh_x.cells:
		points.append(cell_x.points.reshape(-1))
		scalfield.append(cell_x.values[field_name])
	points = np.array(points)
	scalfield = np.array(scalfield)
	point_cloud = pv.PolyData(points)
	point_cloud.point_data[field_name] = scalfield
	
	plotter.add_mesh(point_cloud, scalars=field_name, show_scalar_bar=True, point_size=5, cmap='plasma',
	scalar_bar_args={'title': f"{field_name}",'title_font_size': 16,'label_font_size': 12,'color': 'black', 'vertical': False })
	plotter.add_axes(x_color='red', y_color='green',z_color='blue',xlabel=r'X_1',ylabel=r'X_2',zlabel=r'X_3',labels_off=False, line_width=4)
	plotter.show()
	
def plot_vector_field_2D_at_plane(mesh_x, e1, e2, field_name):
	fig = plt.figure()
	ax = fig.add_subplot(111) 
	points = []
	vectors = []
	for cell_x in mesh_x.cells:
		points.append(cell_x.points.reshape(-1))
		vectors.append(cell_x.values[field_name])
	points = np.array(points)
	vectors = np.array(vectors)
	
	#Transform from x = [x1 x2 x3] to theta = [theta1 theta2] -- plane with base vectors e1, e2
	E = np.array([e1, e2]).T
	transform = lambda x : np.linalg.pinv(E)@x
	
	#project vectors onto plane
	n = np.cross(e1,e2)
	n = n/np.linalg.norm(n)
	dot_products = vectors@n
	vectors = vectors - dot_products[:, np.newaxis]*n
	#apply transform
	vectors_theta = vectors.T # transform(vectors.T)
	points_theta = transform(points.T)
	
	X, Y = points_theta[0,:], points_theta[1,:]
	U_2D, V_2D = vectors_theta[0,:], vectors_theta[1,:]
	magnitude = np.sqrt(U_2D**2 + V_2D**2)
	
	quiver = ax.quiver(X, Y, U_2D, V_2D, array=magnitude, cmap='plasma')
	fig.colorbar(quiver, ax=ax, label=fr"{field_name}")
	#ax.scatter(X,Y)
	
	ax.set_title(f"{field_name}")
	ax.set_xlabel(r'$\theta_1$')
	ax.set_ylabel(r'$\theta_2$')
	
	ax.axis('equal')
	plt.show()

	
def plot_scalar_field_2D_at_plane(mesh_x, e1, e2, field_name):
	fig = plt.figure()
	ax = fig.add_subplot(111) 
	points = []
	scalfield = []
	for cell_x in mesh_x.cells:
		points.append(cell_x.points.reshape(-1))
		scalfield.append(cell_x.values[field_name])
	points = np.array(points)
	scalfield = np.array(scalfield)
	
	#Transform from x = [x1 x2 x3] to theta = [theta1 theta2] -- plane with base vectors e1, e2
	E = np.array([e1, e2]).T
	transform = lambda x : np.linalg.pinv(E)@x
	
	#apply transform
	points_theta = transform(points.T)
	
	X, Y = points_theta[0,:], points_theta[1,:]
	
	scat = plt.scatter(X,Y, s=5., marker="+", color="black")
	cont = plt.tricontourf(X,Y,scalfield, levels=10, alpha = 0.7)
	fig.colorbar(cont, ax=ax, label=fr"{field_name}")
	
	ax.set_title(fr"{field_name}")
	ax.set_xlabel(r'$\theta_1$')
	ax.set_ylabel(r'$\theta_2$')
	
	ax.axis('equal')
	plt.show()

def plot_along_line(mesh_x, e1, R, field_name):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	points = []
	vals = []
	for cell_x in mesh_x.cells:
		points.append(cell_x.points.reshape(-1))
		vals.append(cell_x.values[field_name])
	points = np.array(points)
	vals = np.array(vals)
	plt.plot(np.linspace(0.,R, len(points)), vals)
	plt.show()

