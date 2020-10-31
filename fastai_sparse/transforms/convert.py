# -*- coding: utf-8 -*-

import numpy as np

from .main import Transform, transfer_keys
from ..data_items import PointsItem, MeshItem, SparseItem

from .. import utils

__all__ = ['TfmConvertItem', 'to_points_cloud', 'to_sparse_voxels', 'merge_features']


class TfmConvertItem(Transform):
    order = 0
    pass


def _to_points_cloud(x: MeshItem, method='centers', normals=True):
    assert method in ['centers', 'vertices']
    if method == 'centers':
        d = _to_points_cloud_by_centers(x, normals=normals)
    elif method == 'vertices':
        d = _to_points_cloud_by_vertices(x, normals=normals)

    example_id = x.data.metadata['example_id']
    d['id'] = example_id
    state = np.random.get_state()
    d['random_seed'] = f'{state[1][0]}_{state[2]}'
    return PointsItem(d)


def _to_points_cloud_by_centers(x, normals=False):
    """
    Extract points as center of faces.
    """
    # TODO: option alpha channel
    # TODO: labels can be optional
    # TODO: colors can be optional
    assert x.is_colors_from_vertices
    assert not x.is_labels_from_vertices

    mesh = x.data

    points = np.array(mesh.vertices, dtype=np.float32)
    colors = np.array(x.colors, dtype=np.float32)[:, :3]  # without alpha channel
    assert len(points) == len(colors)

    faces = np.array(mesh.faces)
    labels = x.labels
    if labels is not None:
        assert len(faces) == len(labels)

    faces_xyz = points[faces]
    faces_rgb = colors[faces]

    # calculate centers
    points = np.mean(faces_xyz, axis=1)
    colors = np.mean(faces_rgb, axis=1)

    d = {'points': points, 'colors': colors, 'labels': labels}

    if normals:
        d['normals'] = np.array(mesh.face_normals, dtype=np.float32)

    return d


def _to_points_cloud_by_vertices(x, normals=False):
    # TODO: labels can be optional
    # TODO: colors can be optional

    mesh = x.data

    points = np.array(mesh.vertices, dtype=np.float32)
    colors = x.colors
    labels = x.labels

    assert len(points) == len(colors)
    is_multilabels = isinstance(labels, (list, tuple))
    if is_multilabels:
        for l in labels:
            assert len(points) == len(l)
    else:
        assert len(points) == len(labels)

    d = {'points': points, 'colors': colors, 'labels': labels}

    if normals:
        d['normals'] = np.array(mesh.vertex_normals, dtype=np.float32)
    return d


to_points_cloud = TfmConvertItem(_to_points_cloud)


def _to_sparse_voxels(x: PointsItem, add_local_pos=False):
    d = x.data.copy()

    points = d['points']

    # TODO: is floor better then simply astype(np.int64) ? For x > 0 there is no differences
    # Some spreadsheet programs calculate the “floor-towards-zero”, in other words floor(-2.5) == -2. NumPy instead uses
    # the definition of floor where floor(-2.5) == -3.
    # >>> a = np.array([-1.7, -1.5, -0.2, 0.2, 0.5, 0.7, 1.3, 1.5, 1.7, 2.0, 2.5, 2.9])
    # >>> b = np.floor(a)
    # >>> c = a.astype(np.int64)
    # >>> pd.DataFrame([a, b, c])
    #    	-1.7 	-1.5 	-0.2 	0.2 	0.5 	0.7 	1.3 	1.5 	1.7 	2.0 	2.5 	2.9
    #    	-2.0 	-2.0 	-1.0 	0.0 	0.0 	0.0 	1.0 	1.0 	1.0 	2.0 	2.0 	2.0
    #    	-1.0 	-1.0 	0.0 	0.0 	0.0 	0.0 	1.0 	1.0 	1.0 	2.0 	2.0 	2.0

    coords = np.floor(points).astype(np.int64)

    # TODO: filter result, coords.min() must be >=0, warn

    labels = d['labels']
    is_multilabels = isinstance(labels, (list, tuple))
    if is_multilabels:
        labels_new = []
        for l in labels:
            labels_new.append(_convert_labels_dtype(l))
    else:
        labels = _convert_labels_dtype(labels)
    
    if add_local_pos is True:
        dxdydz = np.array(points - coords - 0.5, dtype=np.float32)
        d["features"] = np.hstack((d["features"], dxdydz))

    res = {'coords': coords,
           'features': d['features'],
           'labels': labels,
           }

    transfer_keys(d, res)

    return SparseItem(res)


def _convert_labels_dtype(x):
    if np.issubdtype(x.dtype, np.integer):
        return x.astype(np.int64)
    else:
        return x.astype(np.float32)


to_sparse_voxels = TfmConvertItem(_to_sparse_voxels)


def _merge_features(x: PointsItem, ones=True, normals=False, colors=False):
    # TODO: inplace

    append_ones = ones
    append_normals = normals
    append_colors = colors

    d = x.data.copy()

    points = d['points']
    normals = d.get('normals', None)
    colors = d.get('colors', None)
    n_points = points.shape[0]

    # create features
    features = []
    if append_ones:
        features.append(np.ones((n_points, 1)).astype(np.float32))

    if append_normals:
        if normals is not None:
            features.append(normals)
        else:
            utils.warn_always('merge_features: append_normals is True, but there is no normals')

    if append_colors:
        if colors is not None:
            features.append(colors)
        else:
            utils.warn_always('merge_features: append_colors is True, but there is no colors')

    features = np.hstack(features)

    res = {'points': points, 'features': features, 'labels': d['labels']}

    # TODO: global/parameter ?
    transfer_keys(d, res)

    return PointsItem(res)


merge_features = TfmConvertItem(_merge_features)
