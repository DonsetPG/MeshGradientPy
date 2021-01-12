from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Dict, Tuple, Optional, List, Any, Union

import numpy as np
import tensorflow as tf


def compute_div_from_grad(
    grad: Union[tf.Tensor, np.ndarray]
) -> Union[tf.Tensor, np.ndarray]:
    """Compute divergence of a 2D vector field from the gradient of such vector field.
    Arguments:
        grad: gradient vector field
    Returns:
        divergence scalar field
    Raises:
    """
    return grad[:, 0] + grad[:, 4]


def compute_sca_grad_from_grad(
    field: Union[tf.Tensor, np.ndarray], grad: Union[tf.Tensor, np.ndarray]
) -> tf.Tensor:
    """Compute scalar gradient operator for NS equations in 2D.
    Arguments:
        field: vector field (velocity)
        grad: gradient field
    Returns:
        results of the operator
    Raises:
    """
    output_x: tf.Tensor = tf.math.multiply(field[:, 0], grad[:, 0]) + tf.math.multiply(
        field[:, 1], grad[:, 1]
    )
    output_x = tf.reshape(output_x, (len(output_x), 1))
    output_y: tf.Tensor = tf.math.multiply(field[:, 0], grad[:, 3]) + tf.math.multiply(
        field[:, 1], grad[:, 4]
    )
    output_y = tf.reshape(output_y, (len(output_y), 1))
    output: tf.Tensor = tf.concat([output_x, output_y], axis=1)
    return output


def compute_gradient_per_points(
    gradient_matrices: Tuple[tf.sparse.SparseTensor],
    F: Union[tf.Tensor, np.ndarray],
    b1: Optional[Union[tf.Tensor, np.ndarray]] = None,
    b2: Optional[Union[tf.Tensor, np.ndarray]] = None,
    b3: Optional[Union[tf.Tensor, np.ndarray]] = None,
    b4: Optional[Union[tf.Tensor, np.ndarray]] = None,
) -> tf.Tensor:
    """Compute gradient on vertex of a scalar fields.

    For most cells, gradient is computed accordingly to AGS methods and for
    the boundaries we use another average of cells gradient

    Arguments:
        gradient_matrices: matrices used to compute gradient (from matrix.py)
        F: scalar field on which gradient is computed
        b1: boundary flag
        b2: boundary flag
        b3: boundary flag
        b4: boundary flag
    Returns:
        The gradient field.
    Raises:
    """
    Sp_AGS_tf: tf.sparse.SparseTensor
    Sp_PCE_tf: tf.sparse.SparseTensor
    Sp_CON_tf: tf.sparse.SparseTensor
    (Sp_AGS_tf, Sp_PCE_tf, Sp_CON_tf) = gradient_matrices

    tf_F: tf.Tensor = tf.cast(tf.expand_dims(F, axis=-1), tf.float32)
    # Compute gradient on points
    gp_F: tf.Tensor = tf.sparse.sparse_dense_matmul(Sp_AGS_tf, tf_F)

    # Compute gradient on boundaries
    gc_F: tf.Tensor = tf.sparse.sparse_dense_matmul(Sp_PCE_tf, tf_F)
    gc_F = tf.reshape(gc_F, (Sp_CON_tf.shape[1], 3))
    gb_F: tf.Tensor = tf.sparse.sparse_dense_matmul(Sp_CON_tf, gc_F)
    gb_F = tf.reshape(gb_F, (gb_F.shape[0] * 3, 1))

    g_F: tf.Tensor
    if b1 is not None:
        mask: tf.Tensor = 1 - tf.clip_by_value(
            (b1 + b2 + b3 + b4), clip_value_min=0.0, clip_value_max=1.0
        )
        mask = tf.repeat(mask, 3)
        mask = tf.reshape(mask, (len(mask), 1))
        g_F = tf.math.multiply(gp_F, mask) + tf.math.multiply(gb_F, 1 - mask)
    else:
        g_F = gp_F

    g_F = tf.reshape(g_F, (len(F), 3))
    return g_F


def compute_laplacian_scalar_field(
    gradient_matrices: Tuple[tf.sparse.SparseTensor],
    F: Union[tf.Tensor, np.ndarray],
    b1: Optional[Union[tf.Tensor, np.ndarray]] = None,
    b2: Optional[Union[tf.Tensor, np.ndarray]] = None,
    b3: Optional[Union[tf.Tensor, np.ndarray]] = None,
    b4: Optional[Union[tf.Tensor, np.ndarray]] = None,
) -> tf.Tensor:
    """Compute laplacian on vertex of a scalar fields.

    Arguments:
        G: matrices to compute gradient
        F: scalar field on which gradient is computed
        b1: boundary flag
        b2: boundary flag
        b3: boundary flag
        b4: boundary flag
    Returns:
        The laplacian field.
    Raises:
    """
    grad_SF: tf.Tensor = compute_gradient_per_points(
        gradient_matrices, F, b1, b2, b3, b4
    )

    grad_grad_SF_x: tf.Tensor = compute_gradient_per_points(
        gradient_matrices, grad_SF[:, 0], b1, b2, b3, b4
    )
    grad_grad_SF_y: tf.Tensor = compute_gradient_per_points(
        gradient_matrices, grad_SF[:, 1], b1, b2, b3, b4
    )
    grad_grad_SF: tf.Tensor = tf.concat([grad_grad_SF_x, grad_grad_SF_y], axis=1)

    laplacian_SF: tf.Tensor = compute_div_from_grad(grad_grad_SF)
    laplacian_SF = tf.reshape(laplacian_SF, (len(laplacian_SF), 1))
    return laplacian_SF


def compute_laplacian_vector_field(
    gradient_matrices: Tuple[tf.sparse.SparseTensor],
    F: Union[tf.Tensor, np.ndarray],
    b1: Optional[Union[tf.Tensor, np.ndarray]] = None,
    b2: Optional[Union[tf.Tensor, np.ndarray]] = None,
    b3: Optional[Union[tf.Tensor, np.ndarray]] = None,
    b4: Optional[Union[tf.Tensor, np.ndarray]] = None,
) -> tf.Tensor:
    """Compute laplacian on vertex of a vector fields.

    Arguments:
        gradient_matrices: matrices to compute gradient
        F: scalar field on which gradient is computed
        b1: boundary flag
        b2: boundary flag
        b3: boundary flag
        b4: boundary flag
    Returns:
        The laplacian field.
    Raises:
    """
    laplacian_SF_x: tf.Tensor = compute_laplacian_scalar_field(
        gradient_matrices, F[:, 0], b1, b2, b3, b4
    )
    laplacian_SF_y: tf.Tensor = compute_laplacian_scalar_field(
        gradient_matrices, F[:, 1], b1, b2, b3, b4
    )
    laplacian_SF: tf.Tensor = tf.concat([laplacian_SF_x, laplacian_SF_y], axis=1)
    return laplacian_SF
