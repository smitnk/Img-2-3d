"""Load a GLB/OBJ as one welded triangle mesh (vertices, faces, name); meshes over max_faces are decimated (test meshes only)."""
import os

import numpy as np
import trimesh


def load_mesh(path, max_faces=60000):
    m = trimesh.load(path, force="mesh")
    m.merge_vertices()
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    if len(m.faces) > max_faces:
        try:
            m = m.simplify_quadric_decimation(face_count=max_faces)
        except Exception as e:  # pragma: no cover
            print("decimation failed", repr(e), flush=True)
    name = os.path.splitext(os.path.basename(path))[0]
    print(f"MESH {name}: {len(m.vertices)} vertices, {len(m.faces)} faces, watertight={m.is_watertight}", flush=True)
    return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64), name
