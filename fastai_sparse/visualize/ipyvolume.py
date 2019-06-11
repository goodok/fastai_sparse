# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division


from matplotlib import cm
from matplotlib.colors import Normalize
import numpy as np
from pathlib import Path
from dataclasses import dataclass

import ipyvolume as ipv
from ipywidgets import FloatSlider, VBox, jslink, Label, RadioButtons
from IPython.display import FileLink, display, Image

from . import utils
# from ..utils import warn_always

__all__ = ['options', 'scatter', 'show_mesh']


@dataclass
class OptionsOfShow:
    mesh_method: str = 'ipyvolume'  # or `trimesh` for mesh
    # embed in IPython notebook as image, for publication
    # needed: chromium-browser --remote-debugging-port=9222
    interactive: bool = True
    save_images: bool = False       # work only when interactive==False
    save_htmls: bool = False
    fig_number: int = 1
    filename_pattern_image: str = 'fig_{fig_number}'
    filename_pattern_html: str = None
    verbose: int = 0                # display names of saved file


options = OptionsOfShow()


def display_widgets(widget_list, filename=None, ):
    assert options.mesh_method == 'ipyvolume'

    if not options.interactive:

        # For rendering run command in terminal:
        #    chromium-browser --remote-debugging-port=9222
        # try:
        # use headless chrome to save figure

        d = ipv.pylab._screenshot_data(headless=True, format='png')
        # except Exception as e:
        #     utils.warn_always('For rendering run command in terminal:\n\n    chromium-browser --remote-debugging-port=9222\n')
        #     raise(e)
        img = Image(d)
        #   img = Image(value=d, format='png')

        if options.save_images:
            if filename is None:
                pattern = str(options.filename_pattern_image)
                filename = pattern.format(
                    fig_number=options.fig_number) + '.png'
            filename = Path(filename)
            if not filename.parent.exists():
                filename.parent.mkdir(parents=True)

            with open(filename, "wb") as f:
                f.write(d)
                options.fig_number += 1
                if options.verbose:
                    display(FileLink(
                        filename, result_html_prefix="Saved to file: ", result_html_suffix=''))

        # TODO: save html
        if options.save_htmls:
            pattern = str(options.filename_pattern_image)
            filename = pattern.format(fig_number=options.fig_number) + '.html'
            filename = Path(filename)
            if not filename.parent.exists():
                filename.parent.mkdir(parents=True)

            title = filename.name[:-4]
            ipv.save(filename, title=title)
            if options.verbose:
                display(FileLink(
                    filename, result_html_prefix="Saved to file: ", result_html_suffix=''))

        return img

    if options.interactive:
        # TODO: save_html

        # ipv.save('test.html')
        return VBox(widget_list)


def encode_labels_unique(labels):
    """
    Encode labels to be unique and in the range(0, len(unique()).

    To apply color map.
    """
    u = np.unique(labels)
    res = labels.copy()
    for i, label in enumerate(u):
        res[np.where(labels == label)] = i
    return res


def set_axes_lims(points, axeslim='auto', aspect_ratio_preserve=True):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    if axeslim == 'auto':
        if aspect_ratio_preserve:
            bounds = np.array([(v.min(), v.max()) for v in [x, y, z]]).T
            widths = bounds[1] - bounds[0]
            max_width = widths.max()

            pads = (max_width - widths) / 2
            pads = np.vstack([-pads, pads])
            axeslim = (bounds + pads).T

            ipv.xlim(axeslim[0][0], axeslim[0][1])
            ipv.ylim(axeslim[1][0], axeslim[1][1])
            ipv.zlim(axeslim[2][0], axeslim[2][1])

        else:
            ipv.xlim(x.min(), x.max())
            ipv.ylim(y.min(), y.max())
            ipv.zlim(z.min(), z.max())
    else:
        ipv.xlim(-axeslim, axeslim)
        ipv.ylim(-axeslim, axeslim)
        ipv.zlim(-axeslim, axeslim)


def calc_colors_by_labels(labels, cmap=cm.RdBu, reorder_colors=True):
    """
    Calculate rgb colors by labels and color map.
    """
    if labels is not None:
        # calculate colors by labels and color map
        if np.issubdtype(labels.dtype, np.integer):
            u, color_indices, color_indices_reverse = np.unique(
                labels, return_index=True, return_inverse=True)
            colors_seg = cmap(np.linspace(0, 1, len(u)))
            if reorder_colors:
                colors_seg = reorder_contrast(colors_seg)
            # assert len(np.unique(labels)) == labels.max() + 1  # expect labels = 0, 1, 2... n
            colors_by_labels = np.array([colors_seg[ind][:3]
                                         for ind in color_indices_reverse])
        else:
            norm = Normalize(vmin=labels.min(), vmax=labels.max())
            colors_by_labels = cmap(norm(labels))[:, :3]
    else:
        colors_by_labels = None
    return colors_by_labels


def calc_colors_dict_by_labels(labels, cmap=cm.RdBu, reorder_colors=True):
    if labels is not None:
        # calculate colors by labels and color map
        u, color_indices, color_indices_reverse = np.unique(
            labels, return_index=True, return_inverse=True)
        colors_seg = cmap(np.linspace(0, 1, len(u)))
        if reorder_colors:
            colors_seg = reorder_contrast(colors_seg)
        d = dict(zip(u, colors_seg[:, :3]))
    else:
        d = None
    return d


def get_color_switch_widget(colors_by_labels, colors_rgb, scatter_widget):
    w = None
    options = []
    if colors_by_labels is not None:
        options = ['labels']
        is_multilabels = isinstance(colors_by_labels, (list, tuple))
        if is_multilabels:
            options = ['labels_' + str(i) for i in range(len(colors_by_labels))]
    if colors_rgb is not None:
        options += ['rgb']
    if len(options) > 1:
        value = options[-1]
        w = RadioButtons(options=options,
                         description='colors', value=value)

        def on_switch_colors(e):
            value = e['new']
            if value.startswith('labels'):
                if is_multilabels:
                    ind = int(value.split('_')[1])
                    current_colors = colors_by_labels[ind]
                else:
                    current_colors = colors_by_labels
            else:
                current_colors = colors_rgb

            with scatter_widget.hold_trait_notifications():
                scatter_widget.color = current_colors

        w.observe(on_switch_colors, 'value')
    return w


def draw_error_points(points, labels, labels_gt):
    if labels_gt is not None:
        # accuracy
        assert len(labels_gt) == len(labels)
        # accuracy = (labels_gt == labels).sum() / len(labels)
        # print("Accuracy: {}".format(accuracy) )

        # draw errors
        points_error = points[(labels_gt != labels)]
        x = points_error[:, 0]
        y = points_error[:, 1]
        z = points_error[:, 2]

        sc2 = ipv.scatter(x, y, z, size=2, marker="sphere", color='red')

        point_size2 = FloatSlider(
            min=0, max=2, step=0.02, description='Error point size')
        jslink((sc2, 'size'), (point_size2, 'value'))
    else:
        return None, None
    return sc2, point_size2


def scatter(points, labels=None, labels_gt=None,
            colors=None, cmap=cm.RdBu, reorder_colors=True,
            normals=None,
            width=800, height=600,
            axeslim='auto', aspect_ratio_preserve=True,
            point_size_value=0.5, vector_size_value=0.5, title=None):

    if normals is not None:
        assert len(points) == len(normals), "Length incorrect. These are not normals of points, may be."

    points = points.astype(np.float32)

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # draw
    ipv.figure(width=width, height=height)

    set_axes_lims(points, axeslim=axeslim,
                  aspect_ratio_preserve=aspect_ratio_preserve)

    is_multilabels = isinstance(labels, (list, tuple))
    if is_multilabels:
        colors_by_labels = []
        for vl in labels:
            colors_by_labels.append(calc_colors_by_labels(
                vl, cmap=cmap, reorder_colors=reorder_colors))
    else:
        colors_by_labels = calc_colors_by_labels(
            labels, cmap=cmap, reorder_colors=reorder_colors)
    colors_rgb = colors

    if colors_by_labels is None and colors_rgb is None:
        current_colors = 'red'
    elif colors_rgb is not None:
        current_colors = colors_rgb
    elif is_multilabels:
        current_colors = colors_by_labels[0]
    else:
        current_colors = colors_by_labels

    sc = ipv.scatter(x, y, z, size=point_size_value,
                     marker="sphere", color=current_colors)

    point_size = FloatSlider(min=0, max=2, step=0.02, description='Point size')
    jslink((sc, 'size'), (point_size, 'value'))

    w_switch_colors = get_color_switch_widget(colors_by_labels, colors_rgb, sc)

    sc_errors, point_size_errors = draw_error_points(points, labels, labels_gt)

    container = ipv.gcc()
    fig = container.children[0]

    widget_list = [fig, point_size]

    if point_size_errors is not None:
        widget_list.append(point_size_errors)

    if w_switch_colors is not None:
        widget_list.append(w_switch_colors)

    # vertex normals
    if normals is not None:
        u = normals[:, 0]
        v = normals[:, 1]
        w = normals[:, 2]

        quiver = ipv.quiver(
            x, y, z, u, v, w, size=vector_size_value, marker="arrow", color='green')

        vector_size = FloatSlider(
            min=0, max=5, step=0.1, description='Nomals size')
        jslink((quiver, 'size'), (vector_size, 'value'))
        widget_list.append(vector_size)

    if title is not None:
        widget_list = [Label(title)] + widget_list

    return display_widgets(widget_list)


def show_mesh(verts, triangles,
              face_colors=None, face_labels=None, face_cmap=cm.RdBu, face_reorder_colors=True,
              vertex_colors=None, vertex_labels=None, vertex_cmap=cm.RdBu, vertex_reorder_colors=True,
              vertex_normals=None,
              point_size_value=0.5,
              vector_size_value=0.5,
              width=800, height=600, axeslim='auto', aspect_ratio_preserve=True,
              verbose=0):
    """
    vertex_normals - normals of vertices.
    """

    if vertex_normals is not None:
        assert len(verts) == len(vertex_normals), "Length incorrect. These are not normals of points, may be."

    x = verts[:, 0]
    y = verts[:, 1]
    z = verts[:, 2]

    ipv.figure(width=width, height=height)

    set_axes_lims(verts, axeslim=axeslim,
                  aspect_ratio_preserve=aspect_ratio_preserve)

    # faces
    if face_labels is not None:
        face_color_dict = calc_colors_dict_by_labels(
            face_labels, cmap=face_cmap, reorder_colors=face_reorder_colors)
        for label in face_color_dict.keys():
            triangles_set = triangles[face_labels == label]
            color = face_color_dict[label]
            _ = ipv.plot_trisurf(
                x, y, z, triangles=triangles_set, color=color)
    else:
        if face_colors is None:
            face_colors = '#f0f0f0'
        _ = ipv.plot_trisurf(
            x, y, z, triangles=triangles, color=face_colors)

    # vertices
    is_multilabels = isinstance(vertex_labels, (list, tuple))
    if is_multilabels:
        vertex_colors_by_labels = []
        for vl in vertex_labels:
            vertex_colors_by_labels.append(calc_colors_by_labels(
                vl, cmap=vertex_cmap, reorder_colors=vertex_reorder_colors))
    else:
        vertex_colors_by_labels = calc_colors_by_labels(
            vertex_labels, cmap=vertex_cmap, reorder_colors=vertex_reorder_colors)
    vertex_colors_rgb = vertex_colors

    if vertex_colors_by_labels is None and vertex_colors_rgb is None:
        vertex_current_colors = 'red'
    elif vertex_colors_rgb is not None:
        vertex_current_colors = vertex_colors_rgb
    elif is_multilabels:
        vertex_current_colors = vertex_colors_by_labels[0]
    else:
        vertex_current_colors = vertex_colors_by_labels

    sc = ipv.scatter(x, y, z, size=point_size_value,
                     marker='sphere', color=vertex_current_colors)

    point_size = FloatSlider(min=0, max=2, step=0.1, description='Vertex size')
    jslink((sc, 'size'), (point_size, 'value'))

    w_switch_vertex_colors = get_color_switch_widget(
        vertex_colors_by_labels, vertex_colors_rgb, sc)

    widget_list = [ipv.gcc(), point_size]

    # vertex normals
    if vertex_normals is not None:
        u = vertex_normals[:, 0]
        v = vertex_normals[:, 1]
        w = vertex_normals[:, 2]

        quiver = ipv.quiver(
            x, y, z, u, v, w, size=vector_size_value, marker="arrow", color='green')

        vector_size = FloatSlider(
            min=0, max=5, step=0.1, description='Nomals size')
        jslink((quiver, 'size'), (vector_size, 'value'))
        widget_list.append(vector_size)

    if w_switch_vertex_colors is not None:
        widget_list.append(w_switch_vertex_colors)

    return display_widgets(widget_list)


def scatter_normals(points, normals, labels=None, labels_gt=None,
                    with_normals_errors_only=False,
                    width=800, height=600, cmap=cm.RdBu, cmap_normals=cm.cool, aspect_ratio_preserve=True, axeslim='auto',
                    reorder_colors=True,
                    point_size_value=0.2, vector_size_value=1.0,
                    title=None,
                    ):

    assert len(points) == len(
        normals), "Length incorrect. These are not normals of points, may be."

    points = points.astype(np.float32)
    normals = normals.astype(np.float32)

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    u = normals[:, 0]
    v = normals[:, 1]
    w = normals[:, 2]

    if labels is None:
        labels = np.ones(shape=len(points), dtype=np.int)

    # draw
    ipv.figure(width=width, height=height)

    if axeslim == 'auto':
        if aspect_ratio_preserve:
            bounds = np.array([(v.min(), v.max()) for v in [x, y, z]]).T
            widths = bounds[1] - bounds[0]
            max_width = widths.max()

            pads = (max_width - widths) / 2
            pads = np.vstack([-pads, pads])
            axeslim = (bounds + pads).T

            ipv.xlim(axeslim[0][0], axeslim[0][1])
            ipv.ylim(axeslim[1][0], axeslim[1][1])
            ipv.zlim(axeslim[2][0], axeslim[2][1])

        else:

            ipv.xlim(x.min(), x.max())
            ipv.ylim(y.min(), y.max())
            ipv.zlim(z.min(), z.max())
    else:
        ipv.xlim(-axeslim, axeslim)
        ipv.ylim(-axeslim, axeslim)
        ipv.zlim(-axeslim, axeslim)

    # calc point colors
    colors_seg = cmap(np.linspace(0, 1, labels.max() + 1))
    if reorder_colors:
        colors_seg = reorder_contrast(colors_seg)
    colors = np.array([colors_seg[label - 1][:3] for label in labels])

    sc = ipv.scatter(x, y, z, size=point_size_value,
                     marker="sphere", color=colors)

    colors_seg = cmap_normals(np.linspace(0, 1, labels.max() + 1))
    if reorder_colors:
        colors_seg = reorder_contrast(colors_seg)
    colors_normals = np.array([colors_seg[label - 1][:3] for label in labels])

    if labels_gt is None:
        quiver = ipv.quiver(x, y, z, u, v, w, size=vector_size_value,
                            marker="arrow", color=colors_normals)
    else:
        if not with_normals_errors_only:
            quiver = ipv.quiver(x, y, z, u, v, w, size=vector_size_value,
                                marker="arrow", color=colors_normals)

        # accuracy
        assert len(labels_gt) == len(labels)
        accuracy = (labels_gt == labels).sum() / len(labels)
        print("Accuracy: {}".format(accuracy))

        # draw errors
        points_error = points[(labels_gt != labels)]
        x = points_error[:, 0]
        y = points_error[:, 1]
        z = points_error[:, 2]

        # normals with errors
        normals_error = normals[(labels_gt != labels)]
        u = normals_error[:, 0]
        v = normals_error[:, 1]
        w = normals_error[:, 2]

        sc2 = ipv.scatter(x, y, z, size=point_size_value,
                          marker="sphere", color='red')

        point_size2 = FloatSlider(
            min=0, max=2, step=0.02, description='Error point size')
        jslink((sc2, 'size'), (point_size2, 'value'))

        if with_normals_errors_only:

            quiver = ipv.quiver(x, y, z, u, v, w, size=vector_size_value,
                                marker="arrow", color=colors_normals)

    point_size = FloatSlider(min=0, max=2, step=0.1, description='Point size')
    vector_size = FloatSlider(min=0, max=5, step=0.1,
                              description='Vector size')

    jslink((sc, 'size'), (point_size, 'value'))
    jslink((quiver, 'size'), (vector_size, 'value'))

    if labels_gt is not None:
        widget_list = [ipv.gcc(), point_size, point_size2, vector_size]
    else:
        widget_list = [ipv.gcc(), point_size, vector_size]

    if title is not None:
        widget_list = [Label(title)] + widget_list

    return display_widgets(widget_list)


def reorder_contrast(colors):
    i = np.arange(len(colors))
    half = len(i) // 2
    half1 = i[:half]
    half2 = i[half:]
    if len(i) % 2:
        # append value
        half1 = np.hstack([half1, -np.ones(1, dtype=np.int)])
    assert len(half1) == len(half2)
    i = np.vstack([half1, half2]).reshape(-1, order='F')
    i = i[i >= 0]
    return colors[i]


def plot_line(p1=(0, 0, 0), p2=(1, 1, 1), color='red'):
    p1 = np.array(p1)
    p2 = np.array(p2)
    points = np.stack([p1, p2])
    print(points.shape)

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # draw lines from the first vertex to second
    lines = [[0, 1]]

    ipv.plot_trisurf(x, y, z, lines=lines, color=color)


def plot_box(corner1=(0.1, 0.1, 0.1), corner2=(0.5, 0.5, 0.5), color='red'):
    c1 = np.array(corner1)
    c2 = np.array(corner2)
    c = np.stack([c1, c2])

    points = np.array([
        [c[0, 0], c[0, 1], c[0, 2]],
        [c[1, 0], c[0, 1], c[0, 2]],
        [c[1, 0], c[1, 1], c[0, 2]],
        [c[0, 0], c[1, 1], c[0, 2]],
        [c[0, 0], c[0, 1], c[1, 2]],
        [c[1, 0], c[0, 1], c[1, 2]],
        [c[1, 0], c[1, 1], c[1, 2]],
        [c[0, 0], c[1, 1], c[1, 2]],
    ])

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    lines = [[0, 1], [1, 2], [2, 3], [3, 0],
             [4, 5], [5, 6], [6, 7], [7, 4],
             [0, 4], [1, 5], [2, 6], [3, 7],
             ]
    ipv.plot_trisurf(x, y, z, lines=lines, color=color)


def show_OFF(fn):
    verts, edges, facets, colors = utils.load_off(fn)

    return show_mesh(verts, facets)


def float2color(a):
    vmin = a.min()
    vmax = a.max()
    w = vmax - vmin

    res = 255 * (a - vmin) / w
    return res.astype(np.int)
