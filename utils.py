import numpy as np
import json
from datetime import datetime


class Params():
    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        return self.__dict__


def cartesian_coords_to_dihedral_angles(atomic_coords):
  """Convert Cartisian coordiantes of atoms on the backbone of protein to succesive dihedral angles.
  Args:
    - atomic_coords: array with shape [seq_length * 3, 3]
  Returns:
    - angles: array of radian angles in [-pi, pi] with shape [seq_length * 3, 3]
  """

  zero_tensor = 0.
  dihedral_list = [zero_tensor, zero_tensor]
  dihedral_list.extend(compute_dihedral_angles(atomic_coords))
  dihedral_list.append(zero_tensor)
  angles = np.array(dihedral_list).reshape(-1, 3)

  return angles


def compute_dihedral_angles(atomic_coords):
  ba = atomic_coords[1:] - atomic_coords[:-1]
  ba /= np.expand_dims(np.linalg.norm(ba, axis=1), 1)
  ba_neg = -1 * ba

  n1_vec = np.cross(ba[:-2], ba_neg[1:-1], axis=1)
  n2_vec = np.cross(ba_neg[1:-1], ba[2:], axis=1)
  n1_vec /= np.expand_dims(np.linalg.norm(n1_vec, axis=1), 1)
  n2_vec /= np.expand_dims(np.linalg.norm(n2_vec, axis=1), 1)
  m1_vec = np.cross(n1_vec, ba_neg[1:-1], axis=1)

  x = np.sum(n1_vec * n2_vec, axis=1)
  y = np.sum(m1_vec * n2_vec, axis=1)

  angles = np.arctan2(y, x)
  # NOTE: there are 'nan' values due to masked coords, should reset them to zero
  angles[np.isnan(angles)] = 0.

  return angles


def time_string():
  return datetime.now().strftime('%m-%d %H:%M:%S')
