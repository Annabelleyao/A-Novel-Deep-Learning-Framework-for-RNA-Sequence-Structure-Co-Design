import numpy as np
import math
import torch
import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch.nn as nn
import torch.nn.functional as F
import gvp

def calculate_dihedral(p0, p1, p2, p3):
        """
        Calculate the dihedral angle between four points p0 - p1 - p2 - p3.
        The angle is returned in degrees.
        """
        b0 = p1 - p0
        b1 = p2 - p1
        b2 = p3 - p2
    
        # Normalize b1
        b1 /= np.linalg.norm(b1)
    
        # Orthogonal vectors
        v = b0 - np.dot(b0, b1) * b1
        w = b2 - np.dot(b2, b1) * b1
    
        # Angle between v and w
        x = np.dot(v, w)
        y = np.dot(np.cross(b1, v), w)
        angle = np.degrees(np.arctan2(y, x))
        return angle
    
def torsion( i, allcord):  # i>=2, i is position
    torsion_angles = dict()
    try:
        # Alpha (α): O3'(i-1) - P(i) - O5'(i) - C5'(i)
        torsion_angles['alpha'] = calculate_dihedral(
            allcord[i-1]['O3\''], allcord[i]['P'], allcord[i]['O5\''], allcord[i]['C5\'']
        )
    except:
        torsion_angles['alpha'] = 0
    # Beta (β): P(i) - O5'(i) - C5'(i) - C4'(i)
    torsion_angles['beta'] = calculate_dihedral(
        allcord[i]['P'], allcord[i]['O5\''], allcord[i]['C5\''], allcord[i]['C4\'']
    )
    # Gamma (γ): O5'(i) - C5'(i) - C4'(i) - C3'(i)
    torsion_angles['gamma'] = calculate_dihedral(
        allcord[i]['O5\''], allcord[i]['C5\''], allcord[i]['C4\''], allcord[i]['C3\'']
    )
    # Delta (δ): C5'(i) - C4'(i) - C3'(i) - O3'(i)
    torsion_angles['delta'] = calculate_dihedral(
        allcord[i]['C5\''], allcord[i]['C4\''], allcord[i]['C3\''], allcord[i]['O3\'']
    )
    try:
        # Epsilon (ε): C4'(i) - C3'(i) - O3'(i) - P(i+1)
        torsion_angles['epsilon'] = calculate_dihedral(
            allcord[i]['C4\''], allcord[i]['C3\''], allcord[i]['O3\''], allcord[i+1]['P']
        )
    except:
        torsion_angles['epsilon'] = 0
    try:
        # Zeta (ζ): C3'(i) - O3'(i) - P(i+1) - O5'(i+1)
        torsion_angles['zeta'] = calculate_dihedral(
            allcord[i]['C3\''], allcord[i]['O3\''], allcord[i +
                                                            1]['P'], allcord[i+1]['O5\'']
        )
    except:
        torsion_angles['zeta'] = 0
    torsion_angles['chi_pyrimidine'] = calculate_dihedral(
        allcord[i]['O4\''], allcord[i]['C1\''], allcord[i]['N1'], allcord[i]['C2']
    )
    try:
        # Example for purines (e.g., Adenine, Guanine)
        torsion_angles['chi_purine'] = calculate_dihedral(
            allcord[i]['O4\''], allcord[i]['C1\''], allcord[i]['N9'], allcord[i]['C4']
        )
    except:
        torsion_angles['chi_purine'] = 0
    return torsion_angles
    
def calculate_vector(i, allcord):
      try:
          vector = allcord[i]["C4\'"]-allcord[i+1]["C4\'"]
          magnitude = math.sqrt(
              vector[0]*vector[0] + vector[1]*vector[1] + vector[2]*vector[2])
          if magnitude == 0:
              return np.zeros(3)
          return vector / magnitude
      except:
          return np.zeros(3)
      
def nan_to_num(ts, val=0.0):
    """
    Replaces nans in tensor with a fixed value.    
    """
    val = torch.tensor(val, dtype=ts.dtype, device=ts.device)
    return torch.where(~torch.isfinite(ts), val, ts)
def norm(tensor, dim, eps=1e-8, keepdim=False):
    """
    Returns L2 norm along a dimension.
    """
    return torch.sqrt(
            torch.sum(torch.square(tensor), dim=dim, keepdim=keepdim) + eps)
def normalize(tensor, dim=-1):
    """
    Normalizes a tensor along a dimension after removing nans.
    """
    return nan_to_num(
        torch.div(tensor, norm(tensor, dim=dim, keepdim=True))
    )

def get_rotation_frames(coords):
    """
    Returns a local rotation frame defined by N, CA, C positions.

    Args:
        coords: coordinates, tensor of shape (batch_size x length x 3 x 3)
        where the third dimension is in order of N, CA, C

    Returns:
        Local relative rotation frames in shape (batch_size x length x 3 x 3)
    """
    v1 = coords[:, :, 2] - coords[:, :, 1]
    v2 = coords[:, :, 0] - coords[:, :, 1]
    e1 = normalize(v1, dim=-1)
    u2 = v2 - e1 * torch.sum(e1 * v2, dim=-1, keepdim=True)
    e2 = normalize(u2, dim=-1)
    e3 = torch.cross(e1, e2, dim=-1)
    R = torch.stack([e1, e2, e3], dim=-2)
    return R

def rotate(v, R):
    """
    Rotates a vector by a rotation matrix.
    
    Args:
        v: 3D vector, tensor of shape (length x batch_size x channels x 3)
        R: rotation matrix, tensor of shape (length x batch_size x 3 x 3)

    Returns:
        Rotated version of v by rotation matrix R.
    """
    R = R.unsqueeze(-3)
    v = v.unsqueeze(-1)
    return torch.sum(v * R, dim=-2)