import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import matplotlib.style as mplstyle
mplstyle.use('fast')


def draw_surface(mesh, ax, polys=False, cellnums=False, normals=False):
	normalscale = 1.
	#fig = plt.figure()
	#ax = plt.axes(projection="3d")
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')

	for cell in mesh.cells:
		ax.plot(cell.points[0,:], cell.points[1,:], cell.points[2,:], color='black', linewidth=0.1, rasterized=True)
		ax.scatter(cell.points[0,:], cell.points[1,:], cell.points[2,:], color='black', s=0.1, rasterized=True)
		ax.plot([cell.points[0,0], cell.points[0,-1]], [cell.points[1,0], cell.points[1,-1]], [cell.points[2,0], cell.points[2,-1]], color='black', linewidth=0.1, rasterized=True) #close cell contour
		
		if polys:
			verts = [cell.points.T]
			srf = Poly3DCollection(verts, alpha=.25, facecolor='green')
			plt.gca().add_collection3d(srf)
			ax.scatter(cell.center[0], cell.center[1], cell.center[2], color='red', marker = "x")
		if cellnums: ax.text(cell.center[0], cell.center[1], cell.center[2], cell.id)
		if normals : ax.quiver(cell.center[0], cell.center[1], cell.center[2], cell.n[0],  cell.n[1],  cell.n[2], length=cell.mu*normalscale, color="blue")

def draw_curve(mesh):
	#normalscale = 1.
	fig = plt.figure()
	ax = plt.axes(projection="3d")
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')

	for cell in mesh.cells:
		ax.plot(cell.points[0,:], cell.points[1,:], cell.points[2,:], color='black')
		ax.scatter(cell.points[0,:], cell.points[1,:], cell.points[2,:], color='black')
		#ax.plot([cell.points[0,0], cell.points[0,-1]], [cell.points[1,0], cell.points[1,-1]], [cell.points[2,0], cell.points[2,-1]], color='black') #close cell contour
		
		#verts = [cell.points.T]
		#srf = Poly3DCollection(verts, alpha=.25, facecolor='green')
		#plt.gca().add_collection3d(srf)
		
		ax.scatter(cell.center[0], cell.center[1], cell.center[2], color='red', marker = "x")
		ax.text(cell.center[0], cell.center[1], cell.center[2], cell.id)
		
		#ax.quiver(cell.center[0,:], cell.center[1,:], cell.center[2,:], cell.n[0,:],  cell.n[1,:],  cell.n[2,:], length=cell.mu*normalscale, color="blue")
