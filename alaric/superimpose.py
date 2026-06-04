"""Functions related to Kabsch superimposition

Author: Sjoerd de Vries

This file is part of the Nefertiti project.
"""

import numpy as np
from numpy import sqrt
from numpy.linalg import det, svd


def superimpose(coor1, coor2):
    """Returns the rotation matrix and RMSD between coor1 and coor2

    Do coor1.dot(rotmat) to perform the superposition
    """
    assert coor1.ndim == 2 and coor1.shape[-1] == 3, coor1.shape
    assert coor2.ndim == 2 and coor2.shape[-1] == 3, coor2.shape
    assert coor2.shape[0] == coor1.shape[0]
    natoms = coor1.shape[0]

    c1 = coor1 - coor1.mean(axis=0)
    c2 = coor2 - coor2.mean(axis=0)

    residual1 = (c1 * c1).sum()
    residual2 = (c2 * c2).sum()
    covar = c1.T.dot(c2)

    v, s, wt = svd(covar)
    if det(v) * det(wt) < 0:
        s[-1] *= -1
        v[:, -1] *= -1
    rotmat = v.dot(wt)
    ss = (residual1 + residual2) - 2 * s.sum()
    if ss < 0:
        ss = 0
    rmsd = sqrt(ss / natoms)
    return rotmat, rmsd


def superimpose_array(coor1_array, coor2):
    """Returns the rotation matrix and RMSD between coor1 and coor2
    where coor1 is every element of coor1_array

    Do np.einsum("ijk,ikl->ijl", coor1_array, rotmat) to perform the superpositions
    """
    assert coor1_array.ndim == 3 and coor1_array.shape[-1] == 3, coor1_array.shape
    assert coor2.ndim == 2 and coor2.shape[-1] == 3, coor2.shape
    assert coor2.shape[0] == coor1_array.shape[1]
    natoms = coor2.shape[0]

    c1_array = coor1_array - coor1_array.mean(axis=1)[:, None, :]
    c2 = coor2 - coor2.mean(axis=0)

    residual1 = np.einsum("ijk,ijk->i", c1_array, c1_array)
    residual2 = (c2 * c2).sum()
    covar = np.einsum("ijk,jl->ikl", c1_array, c2)

    v, s, wt = svd(covar)
    reflect = det(v) * det(wt)
    s[:, -1] *= reflect
    v[:, :, -1] *= reflect[:, None]
    rotmat = np.einsum("...ij,...jk->...ik", v, wt)
    ss = (residual1 + residual2) - 2 * s.sum(axis=1)
    ss = np.maximum(ss, 0)
    rmsd = sqrt(ss / natoms)
    return rotmat, rmsd


def superimpose_array_from_covar(covar, residuals, natoms, return_sd: bool):
    """Returns the rotation matrix and RMSD between coor1 and coor2
    where coor1 is every element of coor1_array

    Instead of the coordinates themselves, provide the covariance
     and residuals.

    Do np.einsum("ijk,ikl->ijl", coor1_array, rotmat) to perform the superpositions

    If return_sd, calculate the SD instead of (rotmat, rmsd)
    """
    v, s, wt = svd(covar)
    reflect = det(v) * det(wt)
    s[:, -1] *= reflect
    sd = residuals - 2 * s.sum(axis=1)
    sd = np.maximum(sd, 0)
    if not return_sd:
        v[:, :, -1] *= reflect[:, None]
        rotmat = np.einsum("...ij,...jk->...ik", v, wt)
        rmsd = sqrt(sd / natoms)
        return rotmat, rmsd
    else:
        return sd
