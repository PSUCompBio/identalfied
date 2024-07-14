import bpy
import math
from mathutils import Vector
from heapq import heappop, heappush
import numpy as np
import ast

controls_max = { "L1" : 2 , "L2" : 1, "L3" : 1 , "R1" : 2, "R2" : 2 , "R3" : 2 , "R4" : 2, "R5" : 2, "R6" : 2}

controls_min = { "L1" : 1 , "L2" : 1, "L5" : 0 , "R1" : 1}

thickness = 6

filename_centroids = "Outputs\\centroids.txt"
filename_min = "Outputs\\min_coords.txt"
filename_max = "Outputs\\max_coords.txt"

coordinates = [[-18.989004 , 2.49134793 , 5.02635869], [21.37638531 ,-4.69348574 , 5.22460919],[-21.41151684 , -5.26406629 , 4.99220662], [-23.70476265 ,-15.38073274 , 4.15185814], [20.12315651 ,4.21101178 , 4.48912539], [ 24.3411898 , -14.65882079  ,4.37077542], [-17.26862539 , 9.57301484 , 4.751961  ], [-5.1401908 , 20.54918814 , 3.1695357 ], [12.8205849 , 16.57466938 , 3.03112211], [16.54583193 ,10.23664677 , 4.00893114], [ 4.93446551 ,20.88632488 , 3.06768471], [-13.39225487 , 16.15103723 ,  3.62514866]]
coordinates = []
with open(filename_centroids, 'r') as file:
    for line in file:
        array = ast.literal_eval(line.strip())
        coordinates.append(array)

min_coords = [[-21.69279099 , 5.46345202 , 1.71738549], [25.81974983,-1.81901673,-0.15976323], [-25.91950226 , -2.50392636 , -0.63969668], [-29.25651296 , -14.92009322 , -3.03469221], [21.7255586 , 6.41146533 , 0.28532448], 
[29.36602211 , -13.79758485 , -0.57500548], [-19.25904719 , 10.62123235 , 0.85620173], [-5.71570333,20.38642565,-3.22707589], [12.84471003,17.58736165,-1.80159585],[18.16371028,12.03306548,0.02026811],  [4.72471809,20.41322962,-3.71867347], [-13.43736744,16.86110814,-0.66337164]]
min_coords = []
with open(filename_min, 'r') as file:
    for line in file:
        array = ast.literal_eval(line.strip())
        min_coords.append(array)


max_coords_2 = [[-17.02765528,0.40995704,6.43885358], [19.05027135,-4.44540167,7.32627312], [-19.02100817,-6.12751643,7.1202879,], 
[-21.15375074,-17.48965327,7.24505822], [20.47983233,3.51608046,6.0270977,], 
[21.737758,-15.86221822,7.32187017], [-17.16126442,11.1145649,6.52075768], [-5.46424802,21.62669182,6.35972738], [13.78541374,16.25265471,5.79531574],[17.70253372,10.3689909,6.15421502], [3.92034348,21.78133901,6.29381625], [-13.79625924,16.78437487,6.47573105]]
max_coords = []
with open(filename_max, 'r') as file:
    for line in file:
        array = ast.literal_eval(line.strip())
        max_coords.append(array)

# Calculate the distances between each pair of vertices
def calculate_distances(coords):
    num_coords = len(coords)
    distances = []
    for i in range(num_coords):
        for j in range(i + 1, num_coords):
            dist = (Vector(coords[i]) - Vector(coords[j])).length
            distances.append((dist, i, j))
    return distances

# Find the minimum spanning tree using Prim's algorithm
def find_mst(coords):
    num_coords = len(coords)
    if num_coords == 0:
        return []

    distances = calculate_distances(coords)
    graph = {i: [] for i in range(num_coords)}

    for dist, i, j in distances:
        graph[i].append((dist, j))
        graph[j].append((dist, i))

    mst_edges = []
    visited = set()
    min_heap = [(0, 0, None)]  # (cost, to_vertex, from_vertex)

    while min_heap and len(visited) < num_coords:
        cost, to_vertex, from_vertex = heappop(min_heap)

        if to_vertex in visited:
            continue

        visited.add(to_vertex)
        if from_vertex is not None:
            mst_edges.append((from_vertex, to_vertex))

        for next_cost, next_vertex in graph[to_vertex]:
            if next_vertex not in visited:
                heappush(min_heap, (next_cost, next_vertex, to_vertex))

    return mst_edges

def find_dental_model_axes(vertices):
    points = np.array([v.co for v in vertices])

    center = np.mean(points, axis=0)
    centered_points = points - center

    covariance_matrix = np.matmul(centered_points.T, centered_points)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    order = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, order]

    x_axis = eigenvectors[:, 0]
    z_axis = eigenvectors[:, 2]
    y_axis = np.cross(z_axis, x_axis)

    if round(np.dot(y_axis, z_axis), 5) <= 0:
        z_axis = -z_axis

    return center, x_axis, y_axis, z_axis

def apply_transformation(vertices, matrix):
    points = np.array([v.co for v in vertices])
    points_h = np.c_[points, np.ones((points.shape[0], 1))]
    transformed_points = points_h @ matrix.T
    transformed_points = transformed_points[:, :3]
    for i, v in enumerate(vertices):
        v.co = transformed_points[i]

def find_dental_axes(vertices):
    center, x_axis, y_axis, z_axis = find_dental_model_axes(vertices)

    for _ in range(3):
        affine_matrix = np.eye(4)
        affine_matrix[:3, 3] = -center
        apply_transformation(vertices, affine_matrix)
        center, x_axis, y_axis, z_axis = find_dental_model_axes(vertices)

    for _ in range(3):
        if round(np.dot(y_axis, z_axis), 5) <= 0:
            z_axis = -z_axis
        affine_matrix = np.eye(4)
        affine_matrix[:3, :3] = np.array([x_axis, y_axis, z_axis])
        apply_transformation(vertices, affine_matrix)
        center, x_axis, y_axis, z_axis = find_dental_model_axes(vertices)
        
    return center, x_axis, y_axis, z_axis


def make_increase_width():
    obj = bpy.context.active_object
    
    right_indices = []
    left_indices = []
    scale_size = [] 
    
    for index,vertex in enumerate(obj.data.vertices):
        if vertex.co[0] < 0:
            left_indices.append(index)
        else:
            right_indices.append(index)
            
    for index,vertex in enumerate(obj.data.vertices):
        id = 0
        if index in left_indices:
            id = "L" + str(len(left_indices)-index)
        else:
            id = "R" + str( (index + 1) % len(left_indices))
            
        min = min_coords[index]
        if controls_min.get(id):
            min[2] -= controls_min.get(id)
        max = max_coords[index]
        if controls_max.get(id):
            max[2] += controls_max.get(id)
        
        delta_z = adjust_center_point(coordinates[index],min, max)
        vertex.co = (vertex.co[0], vertex.co[1], vertex.co[2] + delta_z)
        
        scale_size.append(equalize_distances(coordinates[index],min, max))    
    
    for index,vertex in enumerate(obj.data.skin_vertices[""].data):
        vertex.radius = (thickness, scale_size[index])

            
def get_distance(point1 , point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2 + (point2[2] - point1[2]) ** 2)
    
def get_distances_between_coords(coords1, coords2):
    coords = []
    for i in range(len(coords2)):
        coords.append(get_distance(coords1[i], coords2[i]))
    return coords

def adjust_center_point(centroid_point, min_point, max_point):
    # Calculate the required vertical movement
    delta_z = (get_distance(centroid_point , max_point) - get_distance(centroid_point, min_point)) / 2
    
    return delta_z

def equalize_distances(centroid_point, min_point, max_point):
    # Calculate the required vertical movement
    distance_max = get_distance(centroid_point , max_point)
    distance_min = get_distance(centroid_point, min_point)
    scale_size = (distance_max + distance_min) / 2
    
    return scale_size

def equalize_minmax_z():
    for index,vertex in enumerate(coordinates):
        min_coords[index][0] = vertex[0]
        min_coords[index][1] = vertex[1]
        max_coords[index][0] = vertex[0]
        max_coords[index][1] = vertex[1]

filePath = "E:\\Projects\\MouthGuard\\Models\\"
modelName = "12 year old male.stl"
bpy.ops.import_mesh.stl(filepath=filePath + modelName, filter_glob="*.stl")


print("Model imported.")

teeth_obj = bpy.context.object
mesh = teeth_obj.data

# Ensure we are in object mode
bpy.ops.object.mode_set(mode='OBJECT')

teeth_obj.modifiers.new("SIMPLIFY",'DECIMATE')
teeth_obj.modifiers.get("SIMPLIFY").decimate_type = 'COLLAPSE'
teeth_obj.modifiers.get("SIMPLIFY").ratio = 0.1
bpy.ops.object.modifier_apply(modifier="SIMPLIFY", report=True)


print("Model mesh simplified.")


center, x_axis, y_axis, z_axis = find_dental_axes(mesh.vertices)
# Apply final transformation to align the mesh correctly
align_matrix = np.eye(4)
align_matrix[:3, :3] = np.array([x_axis, y_axis, z_axis])
align_matrix[:3, 3] = -center
apply_transformation(mesh.vertices, align_matrix)

# Update the mesh to apply the transformations
mesh.update()
teeth_obj.select_set(False)

print("Dental scan oriented correctly.")


equalize_minmax_z()


# Get the edges based on the minimum spanning tree
edges = find_mst(coordinates)

# Create a new mesh and object
mesh = bpy.data.meshes.new(name="CustomMesh")
obj = bpy.data.objects.new(name="CustomObject", object_data=mesh)

# Link the object to the current scene
scene = bpy.context.scene
scene.collection.objects.link(obj)

# Set the object as the active object
bpy.context.view_layer.objects.active = obj
obj.select_set(True)

# Define vertices and edges
vertices = coordinates

# Create the mesh from vertices and edges
mesh.from_pydata(vertices, edges, [])

# Update the mesh with new data
mesh.update()

# Optional: Set object mode to OBJECT
bpy.ops.object.mode_set(mode='OBJECT')


print("Generated edges from coordinates.")

bpy.ops.object.convert(target='CURVE')
bpy.ops.object.convert(target='MESH')


print("Fixed indices.")


bpy.ops.object.modifier_add(type='SKIN')

make_increase_width()


print("Added Thickness.")

bpy.ops.object.modifier_add(type='BEVEL')
bpy.context.object.modifiers["Bevel"].width = 0.5
bpy.context.object.modifiers["Bevel"].segments = 5
bpy.context.object.modifiers["Bevel"].angle_limit = 0.872665

bpy.ops.object.modifier_add(type='SUBSURF')
bpy.context.object.modifiers["Subdivision"].levels = 2



bpy.ops.object.modifier_add(type='BOOLEAN')
bpy.context.object.modifiers["Boolean"].object = teeth_obj
bpy.context.object.modifiers["Boolean"].solver = 'FAST'


outputPath = ".\\STL_Output\\output1.stl"
bpy.ops.export_mesh.stl(filepath=outputPath, check_existing=True, filter_glob='*.stl', use_selection=True)
