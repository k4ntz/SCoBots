from colormath import color_diff_matrix
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
import scobi.utils.color_dicts as cdicts
import numpy as np

# TODO: RF: is this copypaste from colormath?
def _get_lab_color1_vector(color):
    """
    Converts an LabColor into a NumPy vector.

    :param LabColor color:
    :rtype: numpy.ndarray
    """
    if not color.__class__.__name__ == 'LabColor':
        raise ValueError(
            "Delta E functions can only be used with two LabColor objects.")
    return np.array([color.lab_l, color.lab_a, color.lab_b])

def _get_lab_color2_matrix(color):
    """
    Converts an LabColor into a NumPy matrix.

    :param LabColor color:
    :rtype: numpy.ndarray
    """
    if not color.__class__.__name__ == 'LabColor':
        raise ValueError(
            "Delta E functions can only be used with two LabColor objects.")
    return np.array([(color.lab_l, color.lab_a, color.lab_b)])

def delta_e_cmc(color1, color2_matrix, pl=2, pc=1):
    """
    Calculates the Delta E (CMC) of two colors.

    CMC values
      Acceptability: pl=2, pc=1
      Perceptability: pl=1, pc=1
    """
    color1_vector = _get_lab_color1_vector(color1)
    # color2_matrix = _get_lab_color2_matrix(color2)
    delta_e = color_diff_matrix.delta_e_cmc(
        color1_vector, color2_matrix, pl=pl, pc=pc)[0]
    return delta_e.item()

colors_matrices = {}
for color_name, rgb in cdicts.CSS3_NAMES_TO_RGB.items():
    color2_rgb = sRGBColor(*rgb)
    color2_lab = convert_color(color2_rgb, LabColor)
    color2_lab_matrix = _get_lab_color2_matrix(color2_lab)
    colors_matrices[color_name] = color2_lab_matrix

def _colordist(rgb1, color2_lab_matrix):
    rgb1 = (np.array(rgb1)/255).tolist()
    color1_lab = convert_color(sRGBColor(*rgb1), LabColor)
    delta_e = delta_e_cmc(color1_lab, color2_lab_matrix)
    return delta_e

# returns str and int of closest color to input rgb
def get_closest_color(rgb):
    comp_list = [(c[0], _colordist(rgb, c[1])) for c in colors_matrices.items()] 
    comp_list2 = sorted(comp_list, key=lambda el: el[1])
    color_name = comp_list2[0][0]
    color_int = cdicts.COLOR_TO_INT[color_name]
    return color_name, color_int