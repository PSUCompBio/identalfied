import pyvista as pv

outputPath = ".\\STL_Output\\output1.stl"

def plot_mesh(stl_filepath):
    mesh = pv.read(stl_filepath)
    mesh_actor = pv.PolyData(mesh.points, mesh.faces)
    plotter = pv.Plotter()
    plotter.add_mesh(mesh_actor, color='green', opacity=1)

    plotter.show()


plot_mesh(outputPath)
