import pyvista as pv
import numpy as np
from scipy.spatial import cKDTree
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm
from numpy.linalg import lstsq
from scipy.optimize import curve_fit
import seaborn as sns
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
import concurrent.futures
from joblib import Parallel, delayed


### First
def simplify_stl(mesh, reduction_factor=0.5):

    # Simplifikasyon işlemi için pyvista kütüphanesini kullan
    simplified_mesh = mesh.decimate(reduction_factor)


    return simplified_mesh

def find_dental_model_axes(mesh):

    # Mesh'in noktalarını al
    points = mesh.points

    # Noktaların merkezini bul
    center = np.mean(points, axis=0)
    # Merkezi noktaya göre noktaları düzelt
    centered_points = points - center

    # PCA (Principal Component Analysis) uygula
    #covariance_matrix = np.cov(centered_points, rowvar=False)
    covariance_matrix= np.matmul(np.transpose(centered_points),centered_points)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # PCA sonuçlarına göre sırala
    order = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, order]    

    # PCA sonuçlarına göre xyz eksenlerini çıkart
    x_axis = eigenvectors[:, 0]
    z_axis = eigenvectors[:, 2]

    y_axis = np.cross(z_axis, x_axis)
    if round(np.dot(y_axis, z_axis),5) <= 0:
        z_axis = -z_axis   

    return center, x_axis, y_axis, z_axis

def find_dental_axes(mesh):
    
    center, x_axis, y_axis, z_axis = find_dental_model_axes(mesh)
    for i in range(3):
        affine_matrix = np.eye(4) 
        affine_matrix[:3, 3] = -center 
        mesh=mesh.transform(affine_matrix)  
        center, x_axis, y_axis, z_axis = find_dental_model_axes(mesh)    

    for i in range(3):
        if round(np.dot(y_axis, z_axis),5) <= 0:
            z_axis=-z_axis
        affine_matrix = np.eye(4)  
        affine_matrix[:3, :3] = np.array([x_axis, y_axis, z_axis])  
        mesh=mesh.transform(affine_matrix) 
        center, x_axis, y_axis, z_axis = find_dental_model_axes(mesh)    
    center, x_axis, y_axis, z_axis = find_dental_model_axes(mesh)    

    return center, x_axis, y_axis, z_axis

## Second


def find_peaks_from_center(mesh, center, z_axis, height_threshold,radius):

    cent_loc=mesh.cell_centers()
    centered_points = cent_loc.points  - center
    heights = np.dot(centered_points, z_axis)
    highest_height = np.max(heights)
    
    valid_indices = np.where(heights >= highest_height - height_threshold)[0]
    valid_points = cent_loc.points[valid_indices]
    valid_heights = heights[valid_indices]

    tree = cKDTree(valid_points)
    local_maxima = []
    
    for i, point in tqdm(enumerate(valid_points), total=len(valid_points), desc="Searching peaks"):
        indices = tree.query_ball_point(point, r=radius)  
        if len(np.where(np.all(valid_heights[i] >= valid_heights[indices],axis=0))[0])>0:
            local_maxima.append(point)
    
    return np.array(local_maxima)

def filter_peaks_by_horizontal_vertical_variation(peaks, center, z_axis, max_horizontal_variation=0.2, max_vertical_variation=0.2):
    centered_peaks = peaks - center
    horizontal_variation = np.linalg.norm(centered_peaks[:, [0, 1]], axis=1)
    vertical_variation = np.abs(np.dot(centered_peaks, z_axis))
    ratio=horizontal_variation/vertical_variation

    filtered_peaks=peaks[abs(ratio - np.mean(ratio)) < 1.5 * np.std(ratio)]

    return np.array(filtered_peaks)
#Third

def calculate_curvature(mesh, triangle1_idx, triangle2_idx):
    # Mesh points
    points1 = mesh.points[mesh.faces[4 * triangle1_idx + 1: 4 * triangle1_idx + 4]]
    points2 = mesh.points[mesh.faces[4 * triangle2_idx + 1: 4 * triangle2_idx + 4]]

    # Mesh center and normal calculation
    center1 = np.mean(points1, axis=0)
    center2 = np.mean(points2, axis=0)
    normal1 = np.cross(points1[1] - points1[0], points1[2] - points1[0])
    normal1 /= np.linalg.norm(normal1)
    normal2 = np.cross(points2[1] - points2[0], points2[2] - points2[0])
    normal2 /= np.linalg.norm(normal2)

    # Displacement between meshes
    delta_x = center2 - center1

    # Curvatuere and sign calculations
    angle = np.arccos(np.clip(np.dot(normal1, normal2), -1.0, 1.0))
    curvature= np.cross(normal1,normal2)/np.linalg.norm(delta_x)
    signed_curvature = -np.sign(np.dot(normal1, delta_x)) * np.linalg.norm(curvature)

    if signed_curvature>0:
        signed_curvature=0
    else:
        signed_curvature=-signed_curvature

    return signed_curvature

def flood_fill_with_costs(mesh, peak_index, curvature_threshold, radius, global_visited,total_costs,color_index):
    # Initialize
    cent_loc = mesh.cell_centers()
    queue = deque([(peak_index, 0)])  
    visited = set()
    costlist=[]
    total_costs = np.full(mesh.n_cells, 0,dtype='float_') 
    added_costs = np.full(mesh.n_cells, 0,dtype='float_')
    while queue:
        current_index, current_cost = queue.popleft()
        if current_index in visited or  np.linalg.norm(cent_loc.points[current_index]-cent_loc.points[peak_index]) > radius: #or current_cost >= curvature_threshold 
            continue
        visited.add(current_index)
        global_visited.add(current_index)
        total_costs[current_index] = current_cost
        
        for neighbor_index in list(mesh.cell_neighbors(current_index, "edges")):
            if neighbor_index not in visited or neighbor_index not in global_visited:
                added_cost_temp=calculate_curvature(mesh, current_index, neighbor_index)

                added_costs[current_index]=added_cost_temp
                new_cost = current_cost+added_cost_temp
                if new_cost < curvature_threshold: 
                    queue.append((neighbor_index, new_cost))
                    costlist.append(new_cost)

                else:
                    total_costs[neighbor_index]=0


    
    #a,b=np.unique(total_costs,axis=0,return_counts=True)
    #c=range(len(a))
    #der = np.diff(a) / np.diff(c)
    #plt.plot(a,c)
    colors = np.zeros((mesh.n_cells, 3))  # Initialize all cells with zero color
    max_cost = np.max(total_costs[total_costs < np.inf]) if np.any(total_costs < np.inf) else 1

    if len(costlist)>1:
        x = np.array(list(range(len(costlist))))
        y = costlist
        def linear_func(x, a, b):
            return a * x + b

        popt, pcov = curve_fit(linear_func, x, y)

        # Modeli kullanarak tahminler yapalım
        y_pred = linear_func(x, *popt)

        # Kalanları hesaplayalım
        residuals = y - y_pred
        max_residual_index = np.argmax(residuals)
        max_residual_point = (x[max_residual_index], residuals[max_residual_index])
        smoothed_residuals = lowess(residuals, x, frac=0.2)
        max_sm_residual_index = np.argmax(smoothed_residuals[:,1])
        max_sm_residual_point = (x[max_sm_residual_index], smoothed_residuals[max_sm_residual_index,1])
        window_size = len(costlist) // 5
        cost_series = pd.Series(costlist)
        moving_average = cost_series.rolling(window=window_size).mean()    
        curvature_threshold_limit=moving_average[max_sm_residual_index]
    else:
        curvature_threshold_limit=-999
    for i, cost in enumerate(total_costs):
        if cost < curvature_threshold_limit:  #   
            normalized_cost = cost/max_cost
            colors[i] = plt.cm.viridis(normalized_cost)[:3] # Convert colormap to RGB
 #   return colors, global_visited, total_costs
    return color_index,colors

def partition_model_into_teeth(mesh, peak_points, curvature_threshold, radius):
    tooth_regions_colors = [0]* len(peak_points)    
    global_visited=set()
    total_costs = np.full(mesh.n_cells, curvature_threshold*0.999,dtype='float_')  # Start with infinite costs    
    results =Parallel(n_jobs=-1)(delayed(flood_fill_with_costs)(mesh, mesh.find_containing_cell(peak_point), curvature_threshold, radius, global_visited, total_costs,np.where(peak_points==peak_point)[0][0]) for peak_point in tqdm(peak_points, desc="Processing peak indices"))
    for index, colors in results:
        tooth_regions_colors[index] = colors
    return tooth_regions_colors


def determine_regions(tooth_regions, local_maxima ,perp_points,closure_treshold):
    print("Merging regions")
    checker=True
    while checker:    
        num_regions = len(tooth_regions)
        main_colors = []
        non_black_indices_list = []
        for colors in tooth_regions:
            unique_colors, counts = np.unique(colors.reshape(-1, 3), axis=0, return_counts=True)
            main_color = unique_colors[np.argmax(counts)]
            non_black_indices = np.where(np.any(colors != main_color, axis=1))[0]
            main_colors.append(main_color)
            non_black_indices_list.append(non_black_indices)

        indexlist = []
        for i in range(num_regions):
            if i in indexlist:
                continue
            for j in range(i + 1, num_regions):
                if j in indexlist:
                    continue
                intersection = np.intersect1d(non_black_indices_list[i], non_black_indices_list[j])
                if len(intersection) > 0:
                    merged_nonblack=np.unique(np.concatenate((non_black_indices_list[i], non_black_indices_list[j]), axis=None))
                    tooth_regions[i][merged_nonblack]=main_colors[i]

                    indexlist.append(i)
        if len(indexlist)>0:
            indexlist=np.unique(indexlist)      
            for i in enumerate(reversed(indexlist)):
                tooth_regions=np.delete(tooth_regions, i[1], 0)
                local_maxima=np.delete(local_maxima, i[1], 0)
        else:
            checker=False

    return tooth_regions, local_maxima


#Plotter

def quadratic_generator(peaks,center):
    centered_peaks=peaks#-center
    def quadratic_curve(x, a, b, c):
        return a*x**2 + b*x + c
    x = centered_peaks[:, 0]
    y = centered_peaks[:, 1]  

    params, _ = curve_fit(quadratic_curve, x, y)

    x_range = np.linspace(x.min()-center[0], x.max()-center[0], 1000)
    y_range = quadratic_curve(x_range, *params) -center[1]

    curve_points = np.column_stack((x_range, y_range, np.full_like(x_range,10)))
    curve = pv.PolyData(curve_points) 


    return curve ,curve_points

def find_perpendicular_points(curve_points, peaks):
    perpendicular_points = []
    perpendicular_list  =  []
    point_list= []
    for i,peak in enumerate(peaks):

        peak_x, peak_y = peak[0], peak[1]

        idx = np.argmin(np.linalg.norm(curve_points[:, :2] - [peak_x, peak_y], axis=1))
        nearest_point = curve_points[idx]
        point_list.append(nearest_point)
        nearest_x, nearest_y = nearest_point[0], nearest_point[1]

        perpendicular_distance = np.linalg.norm([peak_x - nearest_x, peak_y - nearest_y])

        perpendicular_points.append((idx, peak_x, peak_y, peak[2], perpendicular_distance))
        perpendicular_list.append((i,idx))
    return perpendicular_points, perpendicular_list, point_list

def plot_stl_and_axes(mesh, center, x_axis, y_axis, z_axis, local_maxima, tooth_regions, addaxis, addplane, addpeaks, grid):

    # Generate Mesh
    mesh_actor = pv.PolyData(mesh.points, mesh.faces)

    # xyz eksenlerini oluştur
    x_axis_actor = pv.Arrow(center, x_axis * 100, scale=10)
    y_axis_actor = pv.Arrow(center, y_axis * 100, scale=10)
    z_axis_actor = pv.Arrow(center, z_axis * 100, scale=10)

    plotter = pv.Plotter()

    # Mesh and Axis
    plotter.add_mesh(mesh_actor, color='lightgrey', opacity=0.5)
    if addaxis:
        plotter.add_mesh(x_axis_actor, color='red')
        plotter.add_mesh(y_axis_actor, color='green')
        plotter.add_mesh(z_axis_actor, color='blue')

    # XY, YZ ve XZ Plane Plot
    xy_plane = pv.Plane(center=center, direction=z_axis, i_size=100, j_size=100)
    yz_plane = pv.Plane(center=center, direction=x_axis, i_size=100, j_size=100)
    xz_plane = pv.Plane(center=center, direction=y_axis, i_size=100, j_size=100)

    if addplane:
        plotter.add_mesh(xy_plane, color='red', opacity=0.3)
        plotter.add_mesh(yz_plane, color='green', opacity=0.3)
        plotter.add_mesh(xz_plane, color='blue', opacity=0.3)
    if addpeaks:
        plotter.add_mesh(pv.PolyData(local_maxima), color='yellow', point_size=10)
    all_colors = np.full((mesh.n_cells,3), [plt.cm.cividis(60)[:3]])  
    unique_colors = plt.cm.get_cmap('hsv', len(tooth_regions))  
    unique_colors = np.random.rand(len(tooth_regions), 3)
    for index, colors in enumerate(tooth_regions):
        unique_rows,counts=np.unique(colors.reshape(-1,3),axis=0,return_counts=True)
        non_black_indices = np.where(np.any(colors != unique_rows[np.argmax(counts)], axis=1))[0]
        unique_color = unique_colors[index]
        all_colors[non_black_indices] = unique_color  #
    
    ## Quad
    plotter.add_mesh(grid, color='white', line_width=5, render_lines_as_tubes=True) 

    colored_mesh = mesh.copy()
    colored_mesh.cell_data['colors'] = all_colors
    plotter.add_mesh(colored_mesh, show_edges=False, scalars='colors', lighting=False, rgb=True)
    plotter.add_scalar_bar(title="Regions", n_labels=len(tooth_regions), label_font_size=22, title_font_size=24)

    plotter.show()

###MAIN###

stl_filepath = 'C:\\Users\\Tufekcioglu\\Desktop\\dev_clone\\identalfied\\12 year old male.stl'
mesh = pv.read(stl_filepath)

center, x_axis, y_axis, z_axis = find_dental_axes(mesh)
simplified_mesh = simplify_stl(mesh, reduction_factor=1-round(41000/len(mesh.points),2))


local_maxima = find_peaks_from_center(simplified_mesh, center, z_axis,height_threshold=5.5,radius=0.5)
local_maxima = filter_peaks_by_horizontal_vertical_variation(local_maxima, center, z_axis, max_horizontal_variation=0.2, max_vertical_variation=0.2)
tooth_regions=partition_model_into_teeth(simplified_mesh, local_maxima, curvature_threshold=5, radius=30)

curve, curve_points= quadratic_generator(local_maxima, center)
perpendicular_points, perp_list , poiint_list=find_perpendicular_points(curve_points, local_maxima)
perp_list.sort(key=lambda a: a[1])

tooth_regions, local_maxima =determine_regions(tooth_regions,local_maxima,perp_list,100)
curve, curve_points= quadratic_generator(local_maxima, center)
perpendicular_points, perp_list , poiint_list=find_perpendicular_points(curve_points, local_maxima)

curve = pv.PolyData(poiint_list)
addaxis= True
addplane= False
addpeaks= True


plot_stl_and_axes(simplified_mesh, center, x_axis, y_axis, z_axis, local_maxima, tooth_regions, addaxis, addplane, addpeaks, curve)


reference_points_upper = np.array([
    # Upper İncisorlar
    [-5, 0, 3], [-2.5, 0, 3], [2.5, 0, 3], [5, 0, 3],
    # Upper Kaninler
    [-7.5, 0, 2.5], [7.5, 0, 2.5],
    # Upper Premolarlar
    [-10, 0, 2], [-12.5, 0, 2], [10, 0, 2], [12.5, 0, 2],
    # Upper Molarlar
    [-15, 0, 1.5], [-17.5, 0, 1], [15, 0, 1.5], [17.5, 0, 1],
])
reference_points_lower = np.array([
    # Lower İncisor
    [-5, 0, -3], [-2.5, 0, -3], [2.5, 0, -3], [5, 0, -3],
    # Lower Kanin
    [-7.5, 0, -2.5], [7.5, 0, -2.5],
    # Lower Premolar
    [-10, 0, -2], [-12.5, 0, -2], [10, 0, -2], [12.5, 0, -2],
    # Lower Molar
    [-15, 0, -1.5], [-17.5, 0, -1], [15, 0, -1.5], [17.5, 0, -1]
])


def calculate_blob_costs(reference_points, blob_points):

    costs = np.sqrt(((blob_points[:, None, :] - reference_points[None, :, :]) ** 2).sum(axis=2))
    return costs

def plot_blob_costs_heatmap(costs):

    plt.figure(figsize=(12, 8))
    plt.imshow(costs, cmap='viridis', aspect='auto')
    plt.colorbar(label='Cost')
    plt.title('Blob Costs Heatmap')
    plt.xlabel('Tooth Types')
    plt.ylabel('Blobs')
    plt.xticks(range(costs.shape[1]), [f'Type {i+1}' for i in range(costs.shape[1])], rotation=90)
    plt.yticks(range(costs.shape[0]), [f'Blob {i+1}' for i in range(costs.shape[0])])
    plt.grid(False)  
    plt.show()


costs_upper = calculate_blob_costs(reference_points_upper, local_maxima)
plot_blob_costs_heatmap(costs_upper)
costs_lower = calculate_blob_costs(reference_points_lower, local_maxima)
plot_blob_costs_heatmap(costs_lower)