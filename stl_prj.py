import pyvista as pv
import numpy as np
from scipy.spatial import cKDTree


def simplify_stl(stl_filepath, reduction_factor=0.5):
    # Read STL
    mesh = pv.read(stl_filepath)

    # Using pyvista for simplification
    simplified_mesh = mesh.decimate(reduction_factor)

    return simplified_mesh

def find_dental_model_axes(mesh):   

    points = mesh.points
    # Center Detection
    center = np.mean(points, axis=0)
    # Center offset
    centered_points = points - center

    # PCA (Principal Component Analysis)
    covariance_matrix= np.matmul(np.transpose(centered_points),centered_points)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # PCA result sorting
    order = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, order]    

    # PCA resulted axes
    x_axis = eigenvectors[:, 0]
    z_axis = eigenvectors[:, 2]
    y_axis = np.cross(z_axis, x_axis)
    

    return center, x_axis, y_axis, z_axis

def plot_stl_and_axes(stl_filepath, center, x_axis, y_axis, z_axis, local_maxima,axis,plane,peak):
    # Read STL
    mesh = pv.read(stl_filepath)

    # Generate mesh
    mesh_actor = pv.PolyData(mesh.points, mesh.faces)

    # XYZ Generation
    x_axis_actor = pv.Arrow(center, x_axis * 100, scale=10)
    y_axis_actor = pv.Arrow(center, y_axis * 100, scale=10)
    z_axis_actor = pv.Arrow(center, z_axis * 100, scale=10)

    # Plot
    plotter = pv.Plotter()

    # Adding mesh and axis arrows
    plotter.add_mesh(mesh_actor, color='lightgrey', opacity=0.5)
    if axis:
        plotter.add_mesh(x_axis_actor, color='red')
        plotter.add_mesh(y_axis_actor, color='green')
        plotter.add_mesh(z_axis_actor, color='blue')

    # XY, YZ and XZ Planes
    xy_plane = pv.Plane(center=center, direction=z_axis, i_size=100, j_size=100)
    yz_plane = pv.Plane(center=center, direction=x_axis, i_size=100, j_size=100)
    xz_plane = pv.Plane(center=center, direction=y_axis, i_size=100, j_size=100)

    # adding planes and peak markers
    if plane:
        plotter.add_mesh(xy_plane, color='red', opacity=0.3)
        plotter.add_mesh(yz_plane, color='green', opacity=0.3)
        plotter.add_mesh(xz_plane, color='blue', opacity=0.3)
    if peak:
        plotter.add_mesh(pv.PolyData(local_maxima), color='yellow', point_size=10)

    plotter.show()    # Show
    plotter.view_vector(z_axis) #Viewpoint
    


def find_peaks(stl_filepath, center, z_axis, height_threshold=15.0):
    mesh = pv.read(stl_filepath)

    # Center Offset and collecting heights
    centered_points = mesh.points - center
    heights = np.dot(centered_points, z_axis)
    highest_height = np.max(heights)
    
    # Filtering with height_treshold
    valid_indices = np.where(heights >= highest_height - height_threshold)[0]
    valid_points = mesh.points[valid_indices]
    valid_heights = heights[valid_indices]

    # Neighborhood information collecting with KD tree
    tree = cKDTree(valid_points)
    local_maxima = []
    
    for i, point in enumerate(valid_points):
        # Collecting local peaks
        indices = tree.query_ball_point(point, r=2.0)  # r can be adjusted
        if all(valid_heights[i] > valid_heights[j] for j in indices if j != i):
            local_maxima.append(point)
    
    return np.array(local_maxima)


# STL Path should be added
stl_filepath = 'HenryKraft.stl'



#Axis detection
simplified_mesh = simplify_stl(stl_filepath, reduction_factor=0.5)
center, x_axis, y_axis, z_axis = find_dental_model_axes(simplified_mesh)

# Finding Peaks
local_maxima = find_peaks(stl_filepath, center, z_axis)

#Plot configuration
show_axis= False
show_planes=False
show_peaks=True

# Plotting
plot_stl_and_axes(stl_filepath, center, x_axis, y_axis, z_axis, local_maxima,show_axis,show_planes,show_peaks)

