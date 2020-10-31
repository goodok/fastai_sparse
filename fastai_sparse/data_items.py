# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import trimesh
import lzma
from pathlib import Path
from os.path import splitext
import warnings
from abc import abstractmethod

from . import visualize
from .utils import log, warn_always
from .core import is_listy, Collection


class ItemBase():
    def __init__(self, data, *args, **kwargs):
        self.data = data
        self._affine_mat = None
        self.verbose = 0
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return f'{self.__class__.__name__} {str(self)}'

    def apply_tfms(self, tfms: Collection, do_resolve: bool = True, verbose=0, refresh_always=False, **kwargs):
        "Apply data augmentation with `tfms` to this `ItemBase`."

        verbose_bak = self.verbose

        # disturb random state flow
        # if do_resolve:
        #    _resolve_tfms(tfms)

        # if flow of affine transforms is ended, then refresh (apply)
        is_affine = [getattr(tfm.tfm, '_wrap', None) == 'affine' for tfm in tfms]
        is_do_refresh = np.diff(is_affine, append=0) == -1

        # x = self.clone()
        x = self
        for tfm, do_refresh in zip(tfms, is_do_refresh):

            if do_resolve:
                tfm.resolve()

            x.verbose = verbose

            x = tfm(x)
            if refresh_always or do_refresh:
                x.refresh()

        self.verbose = verbose_bak
        return x

    @property
    def affine_mat(self):
        "Get the affine matrix that will be applied by `refresh`."
        if self._affine_mat is None:
            # Transformation matrix in homogeneous coordinates for 3D is 4x4.
            self._affine_mat = np.eye(4).astype(np.float32)
            self._mat_list = []
        return self._affine_mat

    @affine_mat.setter
    def affine_mat(self, v) -> None:
        self._affine_mat = v

    def affine(self, func, *args, **kwargs):
        "Equivalent to `self.affine_mat = self.affine_mat @ func()`."
        m = func(*args, **kwargs)
        assert m.shape == (4, 4)
        if self.verbose:
            print("* affine:", func.__name__)
            print("affine_mat: was:")
            print(repr(self.affine_mat))
            print("m:")
            print(repr(m))

        # fixed order
        # self.affine_mat = self.affine_mat @ m
        self.affine_mat = m @ self.affine_mat
        self._mat_list += [m]

        if self.verbose:
            print("affine_mat: became:")
            print(repr(self.affine_mat))
        return self

    def refresh(self):
        "Apply affine (and others) transformations that have been sent to and store in the `ItemBase`."
        if self._affine_mat is not None:
            if self.verbose:
                print('refresh:', self._affine_mat)
            self.aplly_affine(self._affine_mat)
            self.last_affine_mat = self._affine_mat
            self._affine_mat = None
        return self

    @abstractmethod
    def aplly_affine(self, affine_mat):
        "Apply affine (and others) transformations that have been sent to and store in the `ItemBase`."
        # https://en.wikipedia.org/wiki/Transformation_matrix#Affine_transformations
        # https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/geometry/geo-tran.html
        # http://qaru.site/questions/144684/difference-between-numpy-dot-and-python-35-matrix-multiplication

    @abstractmethod
    def show(self, **kwargs):
        pass


class MeshItem(ItemBase):

    def __init__(self, *args, **kwargs):
        self._labels = None
        self._colors = None
        self.is_labels_from_vertices = True
        self.is_colors_from_vertices = True
        super().__init__(*args, **kwargs)

    def copy(self):
        d = self.data.copy()
        o = MeshItem(d)
        for k in ['is_labels_from_vertices', 'is_colors_from_vertices', '_labels', '_colors']:
            setattr(o, k, getattr(self, k))
        return o

    def __str__(self):
        # return str(self.obj)
        d = self.data
        fn = d.metadata['file_name']
        num_v = d.vertices.shape[0]
        # num_f = d.faces.shape[0]
        return f"({fn}, vertices:{num_v})"

    @property
    def colors(self):
        return self._colors

    @colors.setter
    def colors(self, v):
        self._colors = v

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, v):
        self._labels = v

    @property
    def vertices(self):
        return self.data.vertices

    @vertices.setter
    def vertices(self, v):
        mesh = self.data
        mesh.vertices = v
        mesh._cache.clear()
        mesh._cache.id_set()
        mesh.face_normals = None
        mesh.vertex_normals = None

    def describe(self):
        d = self.data
        cn = self.__class__.__name__
        fn = d.metadata['file_name']
        print(f"{cn} ({fn})")
        log('vertices:', d.vertices)
        log('faces:', d.faces)
        log('colors:', self.colors)
        log('labels:', self.labels)

        if self.colors is not None:
            if self.is_colors_from_vertices:
                s = "vertices"
            else:
                s = "faces"
            print(f"Colors from {s}")

        if self.labels is not None:
            if self.is_labels_from_vertices:
                s = "vertices"
            else:
                s = "faces"
            print(f"Labels from {s}")

    @classmethod
    def from_file(cls,
                  fn,
                  example_id=None,
                  extract_id=lambda fn: splitext(Path(fn).name)[0],
                  label_field='label', colors_from_vertices=True, labels_from_vertices=True,
                  pop_metadata=True, need_lzma=False,

                  **kwargs):
        assert Path(fn).exists()
        
        try:
            if need_lzma is True:
                mesh = trimesh.load_mesh(lzma.open(str(fn)), file_type='ply', process=False)
            else:
                mesh = trimesh.load_mesh(str(fn), file_type='ply', process=False)
        except ValueError as ve:
            raise ValueError(str(ve), f'file name: "{fn}"; need_lzma = {need_lzma}')
        
        o = cls(mesh)
        o.parse_additional_data(label_field=label_field,
                                colors_from_vertices=colors_from_vertices,
                                labels_from_vertices=labels_from_vertices,
                                pop_metadata=pop_metadata,
                                **kwargs)

        if example_id is None:
            example_id = extract_id(fn)
        o.data.metadata['example_id'] = example_id
        o.data.metadata['fname'] = fn
        return o

    def get_id(self):
        return self.data.metadata['example_id']

    def parse_additional_data(self, label_field='label', colors_from_vertices=True, labels_from_vertices=True, pop_metadata=False):

        # TODO:
        # --- normals
        """

        --- MeshItem.parse_additional_data

        - rgb[a] from vertices or from faces
        - mark it `is_colors_from_vertices'

        - lables from vertices of from faces
        - mark it `is_lables_from_vertices'

        faces = faces_features_dict['vertex_indices']['f1']  # or vertex_index

        """
        mesh = self.data
        assert mesh.metadata.get(
            'processed', False) is False, "Mesh must be loaded with , process=False"

        if pop_metadata:
            ply_raw = mesh.metadata.pop('ply_raw')
        else:
            ply_raw = mesh.metadata.get('ply_raw')

        vertex_data = ply_raw['vertex']['data']
        face_data = ply_raw['face']['data']

        # vertices = np.column_stack([vertex_data[i] for i in ['x', 'y', 'z']])

        self.is_colors_from_vertices = colors_from_vertices
        if colors_from_vertices:
            d = vertex_data
        else:
            d = face_data
        color_keys = self.color_keys(d)
        colors = None
        if len(color_keys):
            colors = np.column_stack([d[i] for i in color_keys])

        if colors is None:
            if len(self.color_keys(vertex_data)) or len(self.color_keys(face_data)):
                if colors_from_vertices:
                    warn_always(
                        'Try read colors from vertices but there are colors in faces only. Set `colors_from_vertices` = False to load them')
                else:
                    warn_always(
                        'Try read colors from faces but there are colors in vertices only. Set `colors_from_vertices` = True to load them')

        self.is_labels_from_vertices = labels_from_vertices
        if labels_from_vertices:
            d = vertex_data
        else:
            d = face_data
        fields = d.dtype.fields

        labels = None
        if is_listy(label_field):
            labels = [d[lf] for lf in label_field]
            # labels = np.stack(labels).T
        else:
            if label_field in fields:
                labels = d[label_field]

        self.labels = labels
        self.colors = colors
        # TODO: compare with self.data.vertices
        # self.vertices = vertices

    def color_keys(self, raw_data):
        fields = raw_data.dtype.fields
        return [i for i in ['red', 'green', 'blue', 'alpha'] if i in fields]

    def show(self, method=None, labels=None, point_size_value=1., with_normals=False, **kwargs):
        """
        Show mesh.

        Notes
        -----

        If visualize.options.interactive == False

        'For rendering run command in terminal:\n\n    chromium-browser --remote-debugging-port=9222\n'
        """
        # TODO: colorise faces: by vertices colors or by face colors (if exists in trimesh)
        if method is None:
            method = visualize.options.mesh_method
        assert method in ['ipyvolume', 'trimesh']

        d = self.data
        if method == 'ipyvolume':
            # TODO: colors, normals
            v = d.vertices
            v = np.array(v, dtype=np.float64)

            if labels is None:
                labels = self.labels

            vertex_labels = None
            face_labels = None
            if self.is_labels_from_vertices:
                vertex_labels = labels
            else:
                face_labels = labels

            vertex_colors = None
            face_colors = None
            colors = self.colors
            if self.is_colors_from_vertices:
                vertex_colors = colors
            else:
                face_colors = colors

            if self.is_colors_from_vertices:
                vertex_colors = self.colors

            vertex_normals = None
            if with_normals:
                # TODO: case of `face_normals`
                vertex_normals = d.vertex_normals

            f = visualize.show_mesh(v, d.faces,
                                    vertex_colors=vertex_colors, vertex_labels=vertex_labels,
                                    face_labels=face_labels, face_colors=face_colors,
                                    point_size_value=point_size_value,
                                    vertex_normals=vertex_normals,
                                    **kwargs)
            return f
        else:
            return d.show()

    def aplly_affine(self, affine_mat):
        "Apply affine (and others) transformations that have been sent to and store in the `ItemBase`."
        # https://en.wikipedia.org/wiki/Transformation_matrix#Affine_transformations
        # https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/geometry/geo-tran.html
        # http://qaru.site/questions/144684/difference-between-numpy-dot-and-python-35-matrix-multiplication

        self.data.apply_transform(affine_mat)


class PointsItem(ItemBase):
    def __str__(self):
        # return str(self.obj)
        _id = self.data['id']
        _size = self.data['points'].shape

        return f"('{_id}', n: {_size[0]})"

    def copy(self):
        d = self.data.copy()
        o = PointsItem(d)
        return o

    def describe(self):
        d = self.data
        cn = self.__class__.__name__
        _id = d['id']
        print(f"{cn} ({_id})")
        log('points', d['points'])
        for k in ['labels', 'colors', 'normals', 'features']:
            v = d.get(k, None)
            if v is not None:
                log(k, v)

    @property
    def colors(self):
        return self.data.get('colors', None)

    @colors.setter
    def colors(self, v):
        self.data['colors'] = v

    @property
    def labels(self):
        return self.data.get('labels', None)

    @labels.setter
    def labels(self, v):
        self.data['labels'] = v

    def show(self, labels=None, colors=None, with_normals=False, point_size_value=1., normals_size_value=1., **kwargs):
        """Show"""
        d = self.data
        if labels is None:
            labels = d.get('labels', None)
        points = d['points']
        points = np.array(points, dtype=np.float64)

        normals = None
        if with_normals:
            normals = d.get('normals', None)

        colors = d.get('colors', colors)
        return visualize.scatter(points,
                                 labels=labels, colors=colors, normals=normals,
                                 point_size_value=point_size_value,
                                 vector_size_value=normals_size_value,
                                 **kwargs)

    def aplly_affine(self, affine_mat):
        "Apply affine (and others) transformations that have been sent to and store in the `ItemBase`."
        # https://en.wikipedia.org/wiki/Transformation_matrix#Affine_transformations
        # https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/geometry/geo-tran.html
        # http://qaru.site/questions/144684/difference-between-numpy-dot-and-python-35-matrix-multiplication

        # 3x3 for rotation, reflection, scale, shear
        m = affine_mat[:3, :3]
        # column 3x1 for transpose  (shifting)
        v = affine_mat[:3, 3]

        d = self.data

        points = d['points']
        normals = d.get('normals', None)

        points = np.matmul(points, m.T)   # = (m @ points.T).T
        points += v

        # incorrect, correct only for rotation
        if normals is not None:
            warnings.warn(
                'Item has normals, but normals affine transformation is not full implemented (only rotation, flip and transpose)')
            normals = np.dot(normals, m)

        d['points'] = points

        if normals is not None:
            d['normals'] = normals

        # TODO:
        # in common case, the normals are not transforms similar like points and vectors
        # normals is valid for rotation and flippings, but not for (not simmetric) scaling
        # https://paroj.github.io/gltut/Illumination/Tut09%20Normal%20Transformation.html
        # https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/geometry/transforming-normals
        # In fact, the solution to transforming normals, is not to multiply them by the same matrix used for transforming points and vectors,
        # but to multiply them by the transpose of the inverse of that matrix


class SparseItem(ItemBase):
    def __str__(self):
        # return str(self.obj)
        return f"('{self.data['id']}')"

    def describe(self):
        d = self.data

        print('id:', d['id'])
        coords = self.data['coords']
        log('coords', coords)
        log('features', d['features'])
        log('x', coords[:, 0])
        log('y', coords[:, 1])
        log('z', coords[:, 2])
        if 'labels' in d:
            log('labels', d['labels'])

        n_voxels = self.num_voxels()
        n_points = len(coords)
        # print('points:', n_points)
        print('voxels:', n_voxels)
        print('points / voxels:', n_points / n_voxels)

    @property
    def labels(self):
        return self.data.get('labels', None)

    @labels.setter
    def labels(self, v):
        self.data['labels'] = v

    def num_voxels(self):
        coords = self.data['coords']
        df = pd.DataFrame(coords, columns=['x', 'y', 'z'])
        n_voxels = len(df.groupby(['x', 'y', 'z']).count())
        return n_voxels

    def show(self, labels=None, point_size_value=1., **kwargs):
        # The same as PointsItem.show but points = d['coords']
        d = self.data
        if labels is None:
            labels = d['labels']

        points = d['coords']
        points = np.array(points, dtype=np.float64)

        return visualize.scatter(points, labels=labels, point_size_value=point_size_value, **kwargs)

    def apply_tfms(self, tfms: Collection, **kwargs):
        "Subclass this method if you want to apply data augmentation with `tfms` to this `SparseItem`."
        if tfms:
            raise NotImplementedError(f" Transformation for {self.__class__.__name__} is not implemented.")
        return self


def extract_data(b: Collection):
    "Recursively map lists of items in `b ` to their wrapped data."
    if is_listy(b):
        return [extract_data(o) for o in b]
    return b.data if isinstance(b, ItemBase) else b
