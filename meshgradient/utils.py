from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Dict, Tuple, Optional, List, Any

import numpy as np
import meshio


def get_area_from_points(
    mesh: meshio.Mesh, points: List[int], min_value: float = 1e-10
) -> float:
    """Computes the area of a triangle.
    Arguments:
        mesh: A mesh object
        points: a list of 3 indx, the three points that builds a triangle
        min_value: minimum value to clip the area of the triangle by.
    Returns:
        a float
    Raises:
    """
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    a, b, c = mesh.points[points[0]], mesh.points[points[1]], mesh.points[points[2]]
    return max(0.5 * np.linalg.norm((np.cross(a - c, b - c))), min_value)


def get_triangles(mesh: meshio.Mesh) -> np.ndarray:
    """Get the list of triangles in the mesh.
    Arguments:
        mesh: A mesh object
    Returns:
        triangles: a 2D numpy array with the list of triangles (by nodes index)
    Raises:
    """
    triangles: np.ndarray = None
    cell: Tuple[str, np.ndarray]
    for cell in mesh.cells:
        if cell[0] == "triangle":
            triangles = cell[1]
    return triangles


def get_cycle(mesh: meshio.Mesh, indx_node: int) -> Tuple[List[Tuple[int]], bool]:
    """Construct the ordered cycle of the one star ring neighbors of a node.
    Arguments:
        mesh: A mesh object
        indx_node: The indx of the node we want to use as center of our ring.
    Returns:
        list_of_triangles: list of triangle for which indx_node is a vertex
        flag_b: A boolean, True if the node is on a boundary
    Raises:
    """
    triangles: np.ndarray = get_triangles(mesh)
    indx_triangle: np.ndarray = triangles[np.argwhere(triangles == indx_node)][:, 0]

    if len(indx_triangle) == 0:
        return [], False

    unique_indx: np.ndarray
    counts_indx: np.ndarray 
    unique_indx, counts_indx = np.unique(indx_triangle.flatten(), return_counts=True)
    list_of_triangles: List[Tuple[int]] = []

    cnt: int
    prev: int
    curr: int
    next: int
    flag_b: bool
    curr_triangle: np.ndarray
    next_triangle: np.ndarray
    mask: np.ndarray

    if np.any(counts_indx == 1):
        indx_start: int = np.argwhere(counts_indx == 1)[0, 0]
        prev, curr = unique_indx[indx_start], indx_node
        cnt = 1

        mask = np.any(indx_triangle == prev, axis=1) * np.any(
            indx_triangle == curr, axis=1
        )

        curr_triangle = indx_triangle[mask, :][0]
        next = curr_triangle[(curr_triangle != prev) * (curr_triangle != curr)][0]
        flag_b = True
    else:
        curr_triangle = indx_triangle[0]
        cnt = 1
        curr = indx_node
        prev, next = tuple(curr_triangle[curr_triangle != curr])

        a: np.ndarray = mesh.points[prev] - mesh.points[curr]
        b: np.ndarray = mesh.points[next] - mesh.points[curr]
        if np.cross(a, b)[2] > 0:
            prev, next = next, prev

        flag_b = False

    list_of_triangles.append((prev, curr, next))

    while cnt < len(indx_triangle):
        mask = np.any(indx_triangle == next, axis=1) * np.all(
            indx_triangle != prev, axis=1
        )
        next_triangle = indx_triangle[mask, :][0]

        prev = next
        next = next_triangle[(next_triangle != prev) * (next_triangle != curr)][0]
        cnt += 1

        list_of_triangles.append((prev, curr, next))
    return list_of_triangles, flag_b
