from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

def compute_div_from_grad(grad):
    """Compute divergence of a 2D vector field.
    Arguments:
        grad: gradient vector field
    Returns:
        divergence scalar field
    Raises:
    """
    return grad[:, 0] + grad[:, 4]


def compute_sca_grad_from_grad(field, grad):
    """Compute scalar gradient operator for NS equations in 2D.
    Arguments:
        field: vector field (velocity)
        grad: gradient field
    Returns:
        results of the operator
    Raises:
    """
    output_x = tf.math.multiply(field[:, 0], grad[:, 0]) + tf.math.multiply(
        field[:, 1], grad[:, 1]
    )
    output_x = tf.reshape(output_x, (len(output_x), 1))
    output_y = tf.math.multiply(field[:, 0], grad[:, 3]) + tf.math.multiply(
        field[:, 1], grad[:, 4]
    )
    output_y = tf.reshape(output_y, (len(output_y), 1))
    output = tf.concat([output_x, output_y], axis=1)
    return output


def compute_gradient_per_points(
    gradient_matrices, F, b1=None, b2=None, b3=None, b4=None
):
    """Compute gradient on vertex of a scalar fields.

    For most cells, gradient is computed accordingly to AGS methods and for 
    the boundaries we use another average of cells gradient

    Arguments:
        gradient_matrices: matrices to compute gradient
        F: scalar field on which gradient is computed
        b1: boundary flag
        b2: boundary flag
        b3: boundary flag
        b4: boundary flag
        
    Returns:
        The gradient field.
    Raises:
    """
    (Sp_AGS_tf, Sp_PCE_tf, Sp_CON_tf) = gradient_matrices

    tf_F = tf.cast(tf.expand_dims(F, axis=-1), tf.float32)
    # Compute gradient on points
    gp_F = tf.sparse.sparse_dense_matmul(Sp_AGS_tf, tf_F)

    # Compute gradient on boundaries
    gc_F = tf.sparse.sparse_dense_matmul(Sp_PCE_tf, tf_F)
    gc_F = tf.reshape(gc_F, (Sp_CON_tf.shape[1], 3))
    gb_F = tf.sparse.sparse_dense_matmul(Sp_CON_tf, gc_F)
    gb_F = tf.reshape(gb_F, (gb_F.shape[0] * 3, 1))

    if b1 is not None:
        mask = 1 - tf.clip_by_value(
            (b1 + b2 + b3 + b4), clip_value_min=0.0, clip_value_max=1.0
        )
        mask = tf.repeat(mask, 3)
        mask = tf.reshape(mask, (len(mask), 1))
        g_F = tf.math.multiply(gp_F, mask) + tf.math.multiply(gb_F, 1 - mask)
    else:
        g_F = gp_F

    g_F = tf.reshape(g_F, (len(F), 3))
    return g_F


def compute_laplacian_scalar_field(G, SF, b1, b2, b3, b4):
    """Compute laplacian on vertex of a scalar fields.

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
    grad_SF = compute_gradient_per_points(G, SF, b1, b2, b3, b4)

    grad_grad_SF_x = compute_gradient_per_points(G, grad_SF[:, 0], b1, b2, b3, b4)
    grad_grad_SF_y = compute_gradient_per_points(G, grad_SF[:, 1], b1, b2, b3, b4)
    grad_grad_SF = tf.concat([grad_grad_SF_x, grad_grad_SF_y], axis=1)

    laplacian_SF = compute_div_from_grad(grad_grad_SF)
    laplacian_SF = tf.reshape(laplacian_SF, (len(laplacian_SF), 1))
    return laplacian_SF


def compute_laplacian_vector_field(G, VF, b1, b2, b3, b4):
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
    laplacian_SF_x = compute_laplacian_scalar_field(G, VF[:, 0], b1, b2, b3, b4)
    laplacian_SF_y = compute_laplacian_scalar_field(G, VF[:, 1], b1, b2, b3, b4)
    laplacian_SF = tf.concat([laplacian_SF_x, laplacian_SF_y], axis=1)
    return laplacian_SF