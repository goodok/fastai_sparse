# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

import numpy as np
import pandas as pd
import re
import json
import os
from os.path import join
from string import Template
from trimesh import util
try:
    from trimesh.resources import get_resource
except:
    from trimesh.resources import get as get_resource
import trimesh


# Read/write meshes
# from pyntcloud

def read_materials(filename):
    mats = []
    mtl = None  # current material
    for line in open(filename, "r"):
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue
        if values[0] == 'newmtl':
            if mtl is not None:
                mats.append(mtl)
            mtl = {}
            mtl['name'] = values[1]
        elif values[0] == 'Kd':
            mtl['Kd'] = " ".join(values[1:])

    if mtl is not None:
        mats.append(mtl)

    return pd.DataFrame(mats)


def read_obj(filename, verbose=0):
    """ Reads and obj file and return the elements as pandas Dataframes.

    Parameters
    ----------
    filename: str
        Path to the obj file.

    Returns
    -------
    Each obj element found as pandas Dataframe.

    Notes
    -----
    https://ru.wikipedia.org/wiki/Obj

    """
    v = []
    vn = []
    f = []
    f_mat = []

    materials = None
    mat = None

    with open(filename) as obj:
        for line in obj:
            values = line.split()
            if line.startswith('v '):
                v.append(line.strip()[1:].split())

            elif line.startswith('vn'):
                vn.append(line.strip()[2:].split())

            elif line.startswith('f'):
                f.append(line.strip()[2:])
                f_mat.append(mat)

            elif line.startswith('mtllib '):
                fn_materials = line.strip()[7:]
                fn_materials = join(os.path.dirname(filename), fn_materials)
                materials = read_materials(fn_materials)
            elif values[0] in ('usemtl', 'usemat'):
                mat = values[1]

    points = pd.DataFrame(v, dtype='f4', columns=['x', 'y', 'z'])

    if len(vn) > 0:
        vn = pd.DataFrame(vn, dtype='f4', columns=['nx', 'ny', 'nz'])

        # print("points.shape:", points.shape)
        # print("vn.shape:", vn.shape)
        # points = points.join(vn)

    if len(f) > 0 and "//" in f[0]:
        mesh_columns = ['v1', 'vn1', 'v2', 'vn2', 'v3', 'vn3']
    elif len(vn) > 0:
        mesh_columns = ['v1', 'vt1', 'vn1', 'v2',
                        'vt2', 'vn2', 'v3', 'vt3', 'vn3']
    else:
        if len(re.split(r'\D+', f[0])) == 3:
            mesh_columns = ['v1', 'v2', 'v3']
        else:
            mesh_columns = ['v1', 'vt1', 'v2', 'vt2', 'v3', 'vt3']

    f = [re.split(r'\D+', x) for x in f]

    if verbose:
        print(f[0])
        print("mesh_columns:", mesh_columns)

        print("f[0]:", f[0])
        print("len(f_mat)", len(f_mat))

        for fi in f:
            if len(fi) != len(mesh_columns):
                print(fi)
                break

        print("np.array(f).shape:", np.array(f).shape)

    mesh = pd.DataFrame(f, dtype='i4', columns=mesh_columns)
    # start index at 0
    mesh -= 1

    mesh = mesh.assign(mtl=f_mat)

    data = {'points': points, 'mesh': mesh, 'materials': materials}

    if len(vn) > 0:
        data['vn'] = vn

    return data


def load_off(fn):
    file = open(fn, 'r')
    first_line = file.readline().rstrip()
    use_colors = (first_line == 'COFF')
    colors = []

    # FIX OFFx  y  z
    off, xzy = first_line.split('OFF')
    if xzy.strip() != '':
        line = xzy
    else:
        # handle blank and comment lines after the first line
        line = file.readline()

    while line.isspace() or line[0] == '#':
        line = file.readline()

    vcount, fcount, ecount = [int(x) for x in line.split()]
    verts = []
    facets = []
    edges = []
    i = 0
    while i < vcount:
        line = file.readline()
        if line.isspace():
            continue  # skip empty lines
        try:
            bits = [float(x) for x in line.split()]
            px = bits[0]
            py = bits[1]
            pz = bits[2]
            if use_colors:
                colors.append(
                    [float(bits[3]) / 255, float(bits[4]) / 255, float(bits[5]) / 255])

        except ValueError:
            i = i + 1
            continue
        verts.append((px, py, pz))
        i = i + 1

    i = 0
    while i < fcount:
        line = file.readline()
        if line.isspace():
            continue  # skip empty lines
        try:
            splitted = line.split()
            ids = list(map(int, splitted))
            if len(ids) > 3:
                facets.append(tuple(ids[1:]))
            elif len(ids) == 3:
                edges.append(tuple(ids[1:]))
        except ValueError:
            i = i + 1
            continue
        i = i + 1

    return np.array(verts), np.array(edges), np.array(facets), colors


def read_off(fn):
    verts, edges, facets, colors = load_off(fn)

    points = pd.DataFrame(verts, dtype='f4', columns=['x', 'y', 'z'])
    mesh = pd.DataFrame(facets, dtype='i4', columns=['v1', 'v2', 'v3'])

    data = {'points': points, 'mesh': mesh}

    return data


def get_faces_colors(mesh):

    materials = mesh['materials']
    faces_mtl = mesh['mesh']['mtl']

    mat_name_to_color = {}
    for i, mat_row in materials.iterrows():
        color = mat_row.Kd.split(' ')
        color = [float(c) for c in color]
        mat_name_to_color[mat_row['name']] = color

    colors = [mat_name_to_color[mat_name] for mat_name in faces_mtl]
    return np.array(colors)


# from trimesh

def load_trimesh_from_obj(fn, subtract_labels=10):
    mesh_wavefront = read_obj(fn)
    mesh = trimesh.Trimesh(
        vertices=mesh_wavefront['points'][['x', 'y', 'z']].values,
        faces=mesh_wavefront['mesh'][['v1', 'v2', 'v3']].values,
        face_colors=get_faces_colors(mesh_wavefront),
        process=False,
    )

    # TODO: load labels
    return mesh


def export_ply(mesh,
               faces_labels=None,
               vertex_labels=None,
               label_field_name='label',
               label_type='ushort',
               encoding='binary',
               vertex_normal=None):
    """
    Export a mesh to the PLY format including labels.

    Parameters
    ----------
    mesh : Trimesh object
    encoding : ['ascii'|'binary_little_endian']
    vertex_normal : include vertex normals

    Returns
    -------
    export : bytes of result


    Notes
    -----
    Based on `trimesh.exchange.ply.export_ply`
    https://github.com/mikedh/trimesh/blob/master/trimesh/exchange/ply.py

    """
    # evaluate input args
    # allow a shortcut for binary
    if encoding == 'binary':
        encoding = 'binary_little_endian'
    elif encoding not in ['binary_little_endian', 'ascii']:
        raise ValueError('encoding must be binary or ascii')
    # if vertex normals aren't specifically asked for
    # only export them if they are stored in cache
    if vertex_normal is None:
        vertex_normal = 'vertex_normal' in mesh._cache

    is_multilabels = isinstance(label_field_name, list) or isinstance(label_field_name, tuple)

    # custom numpy dtypes for exporting
    dtype_face = [('count', '<u1'),
                  ('index', '<i4', (3))]
    dtype_vertex = [('vertex', '<f4', (3))]
    # will be appended to main dtype if needed
    dtype_vertex_normal = ('normals', '<f4', (3))
    dtype_color = ('rgba', '<u1', (4))

    if label_type == 'char':
        dtype_label = ('label', '<i1')
        dtype_label_numpy = np.int8  # TODO: check it
    elif label_type == 'ushort':
        if is_multilabels:
            dtype_label = []
            for i, lfn in enumerate(label_field_name):
                dtype_label.append(('label_' + str(i), '<u2'))
            dtype_label_numpy = np.uint16
        else:
            dtype_label = ('label', '<u2')
            dtype_label_numpy = np.uint16

    if faces_labels is not None:
        assert faces_labels.dtype == dtype_label_numpy, faces_labels.dtype
    if vertex_labels is not None:
        assert vertex_labels.dtype == dtype_label_numpy, vertex_labels.dtype

    # get template strings in dict
    templates = json.loads(get_resource('ply.template'))
    # append labels in template
    if is_multilabels:
        for i, lfn in enumerate(label_field_name):
            templates['label_' + str(i)] = 'property {} {}\n'.format(
                label_type, lfn)
    else:
        templates['label'] = 'property {} {}\n'.format(
            label_type, label_field_name)

    # start collecting elements into a string for the header
    header = templates['intro']
    header += templates['vertex']

    # if we're exporting vertex normals add them
    # to the header and dtype
    if vertex_normal:
        header += templates['vertex_normal']
        dtype_vertex.append(dtype_vertex_normal)

    # if mesh has a vertex coloradd it to the header
    if mesh.visual.kind == 'vertex' and encoding != 'ascii':
        header += templates['color']
        dtype_vertex.append(dtype_color)
    if vertex_labels is not None and encoding != 'ascii':
        if is_multilabels:
            for i, lfn in enumerate(label_field_name):
                header += templates['label_' + str(i)]
                dtype_vertex.append(dtype_label[i])
        else:
            header += templates['label']
            dtype_vertex.append(dtype_label)

    # create and populate the custom dtype for vertices
    vertex = np.zeros(len(mesh.vertices),
                      dtype=dtype_vertex)
    vertex['vertex'] = mesh.vertices
    if vertex_normal:
        vertex['normals'] = mesh.vertex_normals
    if mesh.visual.kind == 'vertex':
        vertex['rgba'] = mesh.visual.vertex_colors
    if vertex_labels is not None and encoding != 'ascii':
        if is_multilabels:
            for i, lfn in enumerate(label_field_name):
                vertex['label_' + str(i)] = vertex_labels[:, i]
        else:
            vertex['label'] = vertex_labels

    header += templates['face']
    if mesh.visual.kind == 'face' and encoding != 'ascii':
        header += templates['color']
        dtype_face.append(dtype_color)

    if faces_labels is not None and encoding != 'ascii':
        header += templates['label']
        dtype_face.append(dtype_label)

    # put mesh face data into custom dtype to export
    faces = np.zeros(len(mesh.faces), dtype=dtype_face)
    faces['count'] = 3
    faces['index'] = mesh.faces
    if mesh.visual.kind == 'face' and encoding != 'ascii':
        faces['rgba'] = mesh.visual.face_colors
    if faces_labels is not None and encoding != 'ascii':
        faces['label'] = faces_labels

    header += templates['outro']

    header_params = {'vertex_count': len(mesh.vertices),
                     'face_count': len(mesh.faces),
                     'encoding': encoding}

    export = Template(header).substitute(header_params).encode('utf-8')

    if encoding == 'binary_little_endian':
        export += vertex.tostring()
        export += faces.tostring()
    elif encoding == 'ascii':
        raise NotImplementedError()
        # ply format is: (face count, v0, v1, v2)
        fstack = np.column_stack((np.ones(len(mesh.faces),
                                          dtype=np.int64) * 3,
                                  mesh.faces))

        # if we're exporting vertex normals they get stacked
        if vertex_normal:
            vstack = np.column_stack((mesh.vertices,
                                      mesh.vertex_normals))
        else:
            vstack = mesh.vertices

        # add the string formatted vertices and faces
        _s = util.array_to_string(vstack, col_delim=' ', row_delim='\n')
        _s += '\n'
        _s += util.array_to_string(fstack, col_delim=' ', row_delim='\n')
        export += _s.encode('utf-8')
    else:
        raise ValueError('encoding must be ascii or binary!')

    return export
