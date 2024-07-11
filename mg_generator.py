import bpy
import math
from mathutils import Vector
from heapq import heappop, heappush
import numpy as np

coordinates = [-18.989004 , 2.49134793 , 5.02635869] , [21.37638531 , -4.69348574 , 5.22460919] , [-21.41151684 , -5.26406629 , 4.99220662] , [-23.70476265 , -15.38073274 , 4.15185814] , [20.12315651 , 4.21101178 , 4.48912539] , [ 24.3411898 , -14.65882079 , 4.37077542] , [-17.26862539 , 9.57301484 , 4.751961  ] , [-5.1401908 , 20.54918814 , 3.1695357 ] , [12.8205849 , 16.57466938 , 3.03112211] , [ 4.93446551 , 20.88632488 , 3.06768471] , [-13.39225487 , 16.15103723 , 3.62514866]

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
    size = 5
    
    
    right_indices = []
    left_indices = []
    index_dict = { "L1" : (7,7) , "L2" : (7,7) , "R1" : (7,7)}
    
    for index,vertex in enumerate(obj.data.vertices):

        if vertex.co[0] < 0:
            left_indices.append(index)
        else:
            right_indices.append(index)
            
    
    for index,vertex in enumerate(obj.data.skin_vertices[""].data):
        left = index in left_indices
        id = 0
        if left:
            id = "L" + str(len(left_indices)-index)
        else:
            id = "R" + str( (index + 1) % len(left_indices))
        if index_dict.get(id):
            vertex.radius = index_dict.get(id)
        else:
            vertex.radius = (7,7)

filePath = ".\\Models\\"
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
