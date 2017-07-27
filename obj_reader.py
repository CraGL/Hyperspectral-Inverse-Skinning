"""
Loads an Wavefront OBJ file into a OBJ recordclass.

Written by Yotam Gingold; Zachary Ferguson
"""

from __future__ import print_function, division
from recordclass import recordclass

import numpy

UV    = recordclass('UV', ['u', 'v'])
XYZ   = recordclass('XYZ', ['x', 'y', 'z'])

# 'extra' is for extra lines
OBJ = recordclass('OBJ', ['v', 'vt', 'vn', 'vc', 'f', 'extra', 'filename'])
FaceVertex = recordclass('FaceVertex', ['v', 'vt', 'vn'])


def load_obj(filename):
    """ Load a Wavefront OBJ file with the given filename.  """

    print("Loading:", filename)

    with open(filename) as lines:

        v = []
        vt = []
        vn = []
        vc = []
        f = []

        extra_lines = []

        for line in lines:
            sline = line.strip().split()
            if len(sline) == 0:
                continue
            if sline[0] == 'v':
                v.append(XYZ(*list(map(float, sline[1:4]))))
                if(len(sline) > 4):
                    vc.append(list(map(float, sline[4:])))
            elif sline[0] == 'vt':
                # Could also use UVW coordinates
                vt.append(UV(*list(map(float, sline[1:]))[:2]))
            elif sline[0] == 'vn':
                vn.append(XYZ(*list(map(float, sline[1:]))))
            elif sline[0] == 'f':
                # Pad bundle with two extra '//' and then take the first three
                # values in between. This ensures that we always get enough
                # data for a FaceVertex.
                face = tuple(
                    [FaceVertex(
                        *[(int(val) - 1 if len(val) > 0 else None)
                            for val in (bundle + '//').split('/')[:3]]
                    ) for bundle in sline[1:]]
                )
                f.append(face)
            else:
                extra_lines.append(line[:-1])

        result = OBJ(v = v, vt = vt, vn = vn, vc = vc, f = f,
            extra = extra_lines, filename = filename)
        return result


def quads_to_triangles(mesh):
    """
    Convert Quad faces to Triangular ones.
    Inputs:
        mesh - an OBJ object loaded from load_obj()
    Outputs:
        Modifies the mesh.f and returns mesh.
    """
    newFaces = []
    for i, face in enumerate(mesh.f):
        if(len(face) != 3):
            assert len(face) == 4
            newFaces.append((face[0], face[1], face[2]))
            newFaces.append((face[2], face[3], face[0]))
        else:
            newFaces.append(face)
    mesh.f = newFaces
    return mesh


def convert_to_counterclockwise_UVs(mesh):
    """
    Converts any clockwise UV triangles to counter-clockwise order.
    !!! WARNING: This may break find_seam_fast.find_seam() !!!
    Inputs:
        mesh - an OBJ object loaded from load_obj(); must be all
            triangles (use quads_to_triangles())
    Output:
        Modifies the mesh.f and returns mesh.
    """
    for i, face in enumerate(mesh.f):
        assert len(face) == 3
        uvs = [mesh.vt[fv.vt] for fv in face]
        # Create matrix as specified (http://goo.gl/BDPYIT)
        mat = numpy.array([[1, uv.u, uv.v] for uv in uvs])
        if(numpy.linalg.det(mat) < 0): # If order is clockwise
            mesh.f[i] = (face[1], face[0], face[2]) # Swap order
    return mesh
