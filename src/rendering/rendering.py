import numpy as np
import pyrender
import trimesh

def create_side_camera():
    camera = pyrender.PerspectiveCamera(yfov=1)

    camera_pose = np.array([
        [0, 0, -1, -7],
        [0, 1, 0, 0.8],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ])

    return camera, camera_pose


def create_light():
    return pyrender.DirectionalLight(color=np.ones(3), intensity=6.0)


def render_mesh(vertices, faces, camera, camera_pose, light, renderer, bg_color=[0, 0, 0, 1]):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    scene = pyrender.Scene(bg_color=bg_color)
    scene.add(pyrender.Mesh.from_trimesh(mesh))
    scene.add(camera, pose=camera_pose)
    scene.add(light, pose=camera_pose)

    color, depth = renderer.render(scene)
    return color