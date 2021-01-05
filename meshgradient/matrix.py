from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import progressbar

from .utils import get_cycle, get_area_from_points, get_triangles

def build_CON_matrix(mesh):
    """Build connectivity matrix to compute gradient on boundaries.
        
    A[i,j] = area[j] / total_area[i]
        area[j] is the area of the cell j
        total_area is the sum of all cell areas where node i is a vertex of a cell.
    shape = (#vertex, #cells)
    
    Arguments:
        mesh: a meshio object

    Returns:
        A albumentation functions to pass our images to.
    Raises:
    """
    points = mesh.points
    triangles = get_triangles(mesh)

    tf_indices = []
    tf_values = []
    tf_shape = (len(points), len(triangles))

    #for indx_point in progressbar.progressbar(range(len(points))):
    for indx_point in range(len(points)):
        indx_triangles = np.argwhere(triangles == indx_point)[:, 0]
        cell_triangles = triangles[indx_triangles]
        n_triangle = len(indx_triangles)
        total_area = 0
        areas = []
        for i in range(n_triangle):
            triangle_points = cell_triangles[i]
            area = get_area_from_points(mesh, triangle_points)
            total_area += area
            areas.append(area)
        for i, indx_triangle in enumerate(indx_triangles):
            tf_indices.append([indx_point, indx_triangle])
            tf_values.append(areas[i] / total_area)

    Sp_tf_CON_matrix = tf.sparse.SparseTensor(
        tf_indices, tf.cast(tf_values, dtype=tf.float32), tf_shape
    )

    return Sp_tf_CON_matrix


def build_PCE_matrix(mesh):
    """Build Per Cell Average matrix to compute gradient on cells.
        
    shape = (3 * #cells, #points)
    
    Arguments:
        mesh: a meshio object

    Returns:
        A sparse tensor to compute per cell gradient
    Raises:
    """
    triangles = get_triangles(mesh)

    tf_indices = []
    tf_values = []
    tf_shape = (3 * len(triangles), len(mesh.points))

    rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])

    #for i in progressbar.progressbar(range(len(triangles))):
    for i in range(len(triangles)):
        dot_products = []
        curr_triangle = triangles[i]
        area = get_area_from_points(mesh, curr_triangle) * 2

        for j in range(len(curr_triangle)):
            prev = curr_triangle[j]
            curr = curr_triangle[(j + 1) % len(curr_triangle)]
            next = curr_triangle[(j + 2) % len(curr_triangle)]

            u = mesh.points[next] - mesh.points[curr]
            v = mesh.points[curr] - mesh.points[prev]

            if np.cross(u, -v)[2] > 0:
                prev, next = next, prev
                u = mesh.points[next] - mesh.points[curr]
                v = mesh.points[curr] - mesh.points[prev]

            u_90 = np.matmul(rot, u)
            v_90 = np.matmul(rot, v)
            u_90_n = u_90 / np.linalg.norm(u_90)
            v_90_n = v_90 / np.linalg.norm(v_90)

            vert_contr = u_90_n * np.linalg.norm(u) + v_90_n * np.linalg.norm(v)
            vert_contr = vert_contr / area

            for k in range(3):
                tf_indices.append([i * 3 + k, curr])
                tf_values.append(vert_contr[k])

    Sp_tf_PCE_matrix = tf.sparse.SparseTensor(
        tf_indices, tf.cast(tf_values, dtype=tf.float32), tf_shape
    )

    return Sp_tf_PCE_matrix


def build_AGS_matrix(mesh):
    """Build Average Gradient Star matrix to compute gradient on cells.
        
    shape = (3 * #vertex, #vertex)
    
    Arguments:
        mesh: a meshio object

    Returns:
        A sparse tensor to compute per cell gradient
    Raises:
    """
    tf_indices = []
    tf_values = []

    n_nodes = len(mesh.points)
    tf_shape = (3 * n_nodes, n_nodes)

    rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
    #for i in progressbar.progressbar(range(n_nodes)):
    for i in range(n_nodes):

        indx_node = i
        triangles, flag_b = get_cycle(mesh, indx_node)
        if len(triangles) > 0:
            prev_triangle = triangles[0]

            area = 0.0

            vert_contr = []

            for i in range(1, len(triangles) + (1 - int(flag_b))):
                curr_triangle = triangles[i % len(triangles)]
                vid = curr_triangle[1]

                prev = prev_triangle[0]
                curr = prev_triangle[2]
                next = curr_triangle[2]

                if i == 0 and flag_b:
                    area += get_area_from_points(mesh, (prev, vid, curr))

                area += get_area_from_points(mesh, (curr, vid, next))

                c_prev = np.matmul(rot, (mesh.points[curr] - mesh.points[prev]))
                c_next = np.matmul(rot, (mesh.points[next] - mesh.points[curr]))

                vert_contr.append((curr, 0.5 * (c_prev + c_next)))

                if flag_b:
                    if i == 0:
                        vert_contr.append((vid, 0.5 * (c_prev)))
                    if i == len(triangles) - 1:
                        vert_contr.append((vid, 0.5 * (c_next)))

                prev_triangle = curr_triangle

            for col, value in vert_contr:
                for i in range(3):
                    tf_indices.append([indx_node * 3 + i, col])
                    tf_values.append(value[i] / area)

    Sp_tf_AGS_matrix = tf.sparse.SparseTensor(
        tf_indices, tf.cast(tf_values, dtype=tf.float32), tf_shape
    )

    return Sp_tf_AGS_matrix