from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Dict, Tuple, Optional, List, Any

import psutil
import ray

import numpy as np
import tensorflow as tf
import progressbar
import meshio

from .utils import get_cycle, get_area_from_points, get_triangles

num_cpus = psutil.cpu_count(logical=False)
ray.init(num_cpus=num_cpus)

@ray.remote 
class PointProcessCON(object):
    def __init__(self,mesh,triangles):
        self.mesh = mesh
        self.triangles = triangles 
        self.tf_indices = []
        self.tf_values = []
    
    def process(self,indx_point):
        indx_triangles: np.ndarray = np.argwhere(self.triangles == indx_point)[:, 0]
        cell_triangles: np.ndarray = self.triangles[indx_triangles]

        areas: List = [get_area_from_points(self.mesh, cell) for cell in cell_triangles]
        total_area: int = sum(areas)

        for i, indx_triangle in enumerate(indx_triangles):
            self.tf_indices.append([indx_point, indx_triangle])
            self.tf_values.append(areas[i] / total_area)
            
    def get_tensor_info(self):
        return [self.tf_indices, self.tf_values]

def build_CON_matrix_multiprocess(mesh: meshio.Mesh) -> tf.sparse.SparseTensor:
    points: np.ndarray = mesh.points
    triangles: np.ndarray = get_triangles(mesh)

    tf_indices: List
    tf_values: List
    tf_shape: Tuple[int]
    tf_indices, tf_values, tf_shape = [], [], (len(points), len(triangles))
    # for indx_point in progressbar.progressbar(range(len(points))):
    indx_point: int
    i: int
                  
    streaming_actors = [PointProcessCON.remote(mesh,triangles) for _ in range(num_cpus)]
    for indx_point in range(len(points)):
        streaming_actors[indx_point % num_cpus].process.remote(indx_point)
        
    results = ray.get([actor.get_tensor_info.remote() for actor in streaming_actors])
    results = np.array(results)
    tf_indices = results[:,0][0]
    tf_values = results[:,1][0]
    Sp_tf_CON_matrix: tf.sparse.SparseTensor = tf.sparse.SparseTensor(
        tf_indices, tf.cast(tf_values, dtype=tf.float32), tf_shape
    )

    return Sp_tf_CON_matrix

@ray.remote 
class TriangleProcessPCE(object):
    def __init__(self,mesh):
        self.mesh = mesh
        self.rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
        self.tf_indices = []
        self.tf_values = []
    
    def process(self,indx,curr_triangle):
        area = get_area_from_points(self.mesh, curr_triangle) * 2
        for j, prev in enumerate(curr_triangle):
            curr = curr_triangle[(j + 1) % len(curr_triangle)]
            next = curr_triangle[(j + 2) % len(curr_triangle)]

            u: np.ndarray = self.mesh.points[next] - self.mesh.points[curr]
            v: np.ndarray = self.mesh.points[curr] - self.mesh.points[prev]

            if np.cross(u, -v)[2] > 0:
                prev, next = next, prev
                u = self.mesh.points[next] - self.mesh.points[curr]
                v = self.mesh.points[curr] - self.mesh.points[prev]

            u_90, v_90 = np.matmul(self.rot, u), np.matmul(self.rot, v)
            u_90 /= np.linalg.norm(u_90)
            v_90 /= np.linalg.norm(v_90)

            vert_contr: np.ndarray = (
                u_90 * np.linalg.norm(u) + v_90 * np.linalg.norm(v)
            ) / area
            for k in range(3):
                self.tf_indices.append([indx * 3 + k, curr])
                self.tf_values.append(vert_contr[k])
            
    def get_tensor_info(self):
        return [self.tf_indices, self.tf_values]

def build_PCE_matrix_multiprocess(mesh: meshio.Mesh) -> tf.sparse.SparseTensor:
    """Build Per Cell Average matrix to compute gradient on cells.

    shape = (3 * #cells, #points)

    Arguments:
        mesh: a meshio object
    Returns:
        A sparse tensor to compute per cell gradient
    Raises:
    """
    triangles: np.ndarray = get_triangles(mesh)
    tf_indices: List
    tf_values: List
    tf_shape: Tuple[int]
    tf_indices, tf_values, tf_shape = [], [], (3 * len(triangles), len(mesh.points))

    i: int
    curr_triangle: np.ndarray
    
    streaming_actors = [TriangleProcessPCE.remote(mesh) for _ in range(num_cpus)]
    for i, curr_triangle in enumerate(triangles):
        streaming_actors[i % num_cpus].process.remote(i, curr_triangle)
        
    results = ray.get([actor.get_tensor_info.remote() for actor in streaming_actors])
    results = np.array(results)
    tf_indices = results[:,0][0]
    tf_values = results[:,1][0]
    
    Sp_tf_PCE_matrix: tf.sparse.SparseTensor = tf.sparse.SparseTensor(
        tf_indices, tf.cast(tf_values, dtype=tf.float32), tf_shape
    )

    return Sp_tf_PCE_matrix

@ray.remote 
class NodeProcessAGS(object):
    def __init__(self,mesh):
        self.mesh = mesh
        self.rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
        self.tf_indices = []
        self.tf_values = []
    
    def process(self,indx_node, node):
        triangles: List[Tuple[int]]
        flag_b: bool 
        triangles, flag_b = get_cycle(self.mesh, indx_node)
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

                if i == 0 and flag_b: area += get_area_from_points(self.mesh, (prev, vid, curr))

                area += get_area_from_points(self.mesh, (curr, vid, next))

                c_prev = np.matmul(self.rot, (self.mesh.points[curr] - self.mesh.points[prev]))
                c_next = np.matmul(self.rot, (self.mesh.points[next] - self.mesh.points[curr]))

                vert_contr.append((curr, 0.5 * (c_prev + c_next)))

                if flag_b:
                    if i == 0:
                        vert_contr.append((vid, 0.5 * (c_prev)))
                    if i == len(triangles) - 1:
                        vert_contr.append((vid, 0.5 * (c_next)))

                prev_triangle = curr_triangle

            for col, value in vert_contr:
                for i in range(3):
                    self.tf_indices.append([indx_node * 3 + i, col])
                    self.tf_values.append(value[i] / area)
            
    def get_tensor_info(self):
        return [self.tf_indices, self.tf_values]

def build_AGS_matrix_multiprocess(mesh: meshio.Mesh) -> tf.sparse.SparseTensor:
    """Build Average Gradient Star matrix to compute gradient on cells.

    shape = (3 * #vertex, #vertex)

    Arguments:
        mesh: a meshio object

    Returns:
        A sparse tensor to compute per cell gradient
    Raises:
    """

    tf_indices: List
    tf_values: List
    tf_shape: Tuple[int]
    tf_indices, tf_values, tf_shape = [], [], (3 * len(mesh.points), len(mesh.points))
    indx_node: int
    node: np.ndarray
    
    streaming_actors = [NodeProcessAGS.remote(mesh) for _ in range(num_cpus)]
    for indx_node, node in enumerate(mesh.points):
        streaming_actors[indx_node % num_cpus].process.remote(indx_node, node)
        
    results = ray.get([actor.get_tensor_info.remote() for actor in streaming_actors])
    results = np.array(results)
    tf_indices = results[:,0][0]
    tf_values = results[:,1][0]

    Sp_tf_AGS_matrix: tf.sparse.SparseTensor = tf.sparse.SparseTensor(
        tf_indices, tf.cast(tf_values, dtype=tf.float32), tf_shape
    )

    return Sp_tf_AGS_matrix
