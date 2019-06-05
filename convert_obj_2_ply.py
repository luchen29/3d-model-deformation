import pymesh

file_path = "/Users/luchenliu/Desktop/Intern/1_projection/glasses.obj"
mesh = pymesh.load_mesh(file_path)
pymesh.save_mesh_raw(file_path.replace(".obj",".ply"), mesh.vertices, mesh.faces, mesh.voxels)