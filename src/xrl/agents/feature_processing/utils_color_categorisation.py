from colormath import color_diff_matrix
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
import numpy as np


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


COLOR_TO_CATEGORY = {
 'alice_blue': 'blue', 'aliceblue': 'white', 'alizarin_crimson': 'red',
 'antique_white': 'white', 'antiquewhite': 'white', 'aqua': 'blue',
 'aquamarine': 'cyan', 'aquamarine_medium': 'cyan',
 'aureoline_yellow': 'yellow', 'azure': 'white',  'banana': 'yellow',
 'beige': 'brown', 'bisque': 'white', 'black': 'black',
 'blanched_almond': 'white', 'blanchedalmond': 'brown',
 'blue': 'blue', 'blue_light': 'blue', 'blue_medium': 'blue',
 'blue_violet': 'magenta', 'blueviolet': 'purple', 'brick': 'red',
 'brown': 'brown', 'brown_madder': 'brown', 'brown_ochre': 'brown',
 'burlywood': 'brown', 'burnt_sienna': 'brown', 'burnt_umber': 'brown',
 'cadet': 'blue', 'cadetblue': 'blue', 'cadmium_lemon': 'yellow',
 'cadmium_orange': 'orange', 'cadmium_red_deep': 'red',
 'cadmium_red_light': 'orange', 'cadmium_yellow': 'yellow',
 'cadmium_yellow_light': 'yellow', 'carrot': 'orange',
 'cerulean': 'blue', 'chartreuse': 'green', 'chocolate': 'brown',
 'chrome_oxide_green': 'green', 'cinnabar_green': 'green', 'cobalt': 'blue',
 'cobalt_green': 'green', 'cobalt_violet_deep': 'magenta', 'cold_grey': 'grey',
 'coral': 'red', 'coral_light': 'red', 'cornflower': 'blue',
 'cornflowerblue': 'blue', 'cornsilk': 'white', 'crimson': 'red',
 'cyan': 'cyan', 'cyan_white': 'cyan', 'dark_orange': 'orange',
 'darkblue': 'blue', 'darkcyan': 'green', 'darkgoldenrod': 'brown',
 'darkgray': 'gray', 'darkgreen': 'green', 'darkkhaki': 'yellow',
 'darkmagenta': 'purple', 'darkolivegreen': 'green', 'darkorange': 'orange',
 'darkorchid': 'purple', 'darkred': 'red',  'darksalmon': 'red',
 'darkseagreen': 'green', 'darkslateblue': 'purple', 'darkslategray': 'gray',
 'darkturquoise': 'blue', 'darkviolet': 'purple',
 'deep_ochre': 'brown', 'deep_pink': 'red', 'deeppink': 'pink',
 'deepskyblue': 'blue', 'dim_grey': 'grey', 'dimgray': 'gray',
 'dodger_blue': 'blue', 'dodgerblue': 'blue', 'eggshell': 'white',
 'emerald_green': 'green', 'english_red': 'red', 'firebrick': 'red',
 'flesh': 'brown', 'flesh_ochre': 'brown', 'floral_white': 'white',
 'floralwhite': 'white', 'forest_green': 'green', 'forestgreen': 'green',
 'fuchsia': 'purple', 'gainsboro': 'white', 'geranium_lake': 'red',
 'ghost_white': 'white', 'ghostwhite': 'white',
 'gold': 'yellow', 'gold_ochre': 'brown', 'goldenrod': 'yellow',
 'goldenrod_dark': 'yellow', 'goldenrod_light': 'yellow',
 'goldenrod_pale': 'yellow', 'gray': 'gray', 'green': 'green',
 'green_dark': 'green', 'green_pale': 'green', 'green_yellow': 'green',
 'greenish_umber': 'brown', 'greenyellow': 'green', 'grey': 'grey',
 'honeydew': 'white', 'hot_pink': 'red', 'hotpink': 'pink', 'indian_red': 'red',
 'indianred': 'red', 'indigo': 'blue', 'ivory': 'white',
 'ivory_black': 'black', 'khaki': 'brown', 'khaki_dark': 'brown',
 'lamp_black': 'black', 'lavender': 'white', 'lavender_blush': 'white',
 'lavenderblush': 'white', 'lawn_green': 'green', 'lawngreen': 'green',
 'lemon_chiffon': 'white', 'lemonchiffon': 'yellow', 'light_beige': 'brown',
 'light_goldenrod': 'yellow', 'light_grey': 'grey', 'light_salmon': 'red',
 'lightblue': 'blue', 'lightcoral': 'red', 'lightcyan': 'blue',
 'lightgoldenrodyellow': 'yellow', 'lightgreen': 'green', 'lightgray': 'gray',
 'lightgrey': 'gray', 'light_gray': 'gray',
 'lightpink': 'pink', 'lightsalmon': 'orange', 'lightseagreen': 'green',
 'lightskyblue': 'blue', 'lightslategray': 'gray', 'lightsteelblue': 'blue',
 'lightyellow': 'yellow', 'lime': 'green', 'lime_green': 'green',
 'limegreen': 'green', 'linen': 'white', 'madder_lake_deep': 'red',
 'magenta': 'magenta', 'manganese_blue': 'blue', 'maroon': 'red',
 'mars_orange': 'orange', 'mars_yellow': 'orange', 'mediumaquamarine': 'green',
 'mediumblue': 'blue', 'mediumorchid': 'purple', 'mediumpurple': 'purple',
 'mediumseagreen': 'green', 'mediumslateblue': 'purple',
 'mediumspringgreen': 'green', 'mediumturquoise': 'blue',
 'mediumvioletred': 'pink', 'melon': 'yellow', 'midnight_blue': 'blue',
 'midnightblue': 'blue', 'mint': 'green', 'mint_cream': 'white',
 'mintcream': 'white', 'misty_rose': 'white',  'mistyrose': 'white',
 'moccasin': 'white', 'naples_yellow_deep': 'yellow', 'navajo_white': 'white',
 'navajowhite': 'brown', 'navy': 'blue', 'old_lace': 'white',
 'oldlace': 'white', 'olive': 'green', 'olive_drab': 'green',
 'olive_green_dark': 'green', 'olivedrab': 'green', 'orange': 'orange',
 'orange_red': 'orange', 'orangered': 'orange', 'orchid': 'magenta',
 'orchid_dark': 'magenta', 'orchid_medium': 'magenta',
 'palegoldenrod': 'yellow', 'palegreen': 'green', 'paleturquoise': 'blue',
 'palevioletred': 'pink', 'papaya_whip': 'white', 'papayawhip': 'yellow',
 'peach_puff': 'white', 'peachpuff': 'yellow', 'peacock': 'blue',
 'permanent_green': 'green', 'permanent_red_violet': 'magenta', 'peru': 'brown',
 'pink': 'red', 'pink_light': 'red', 'plum': 'magenta', 'powder_blue': 'blue',
 'powderblue': 'blue', 'purple': 'magenta', 'purple_medium': 'magenta',
 'raspberry': 'red', 'raw_sienna': 'brown', 'raw_umber': 'brown', 'red': 'red',
 'rose_madder': 'red', 'rosy_brown': 'brown', 'rosybrown': 'brown',
 'royal_blue': 'blue', 'royalblue': 'blue', 'saddle_brown': 'brown',
 'saddlebrown': 'brown', 'salmon': 'red', 'sandy_brown': 'brown',
 'sandybrown': 'brown', 'sap_green': 'green', 'sea_green': 'green',
 'sea_green_dark': 'green', 'sea_green_light': 'green',
 'sea_green_medium': 'green', 'seagreen': 'green', 'seashell': 'white',
 'sepia': 'brown', 'sienna': 'brown', 'silver': 'gray', 'sky_blue': 'blue',
 'sky_blue_deep': 'blue', 'sky_blue_light': 'blue', 'skyblue': 'blue',
 'slate_blue': 'blue', 'slate_blue_dark': 'blue', 'slate_blue_light': 'blue',
 'slate_blue_medium': 'blue', 'slate_grey': 'grey', 'slate_grey_dark': 'grey',
 'slate_grey_light': 'grey', 'slateblue': 'purple', 'slategray': 'gray',
 'snow': 'white', 'spring_green': 'green', 'spring_green_medium': 'green',
 'springgreen': 'green', 'steel_blue': 'blue', 'steel_blue_light': 'blue',
 'steelblue': 'blue', 'tan': 'brown', 'teal': 'green', 'terre_verte': 'green',
 'thistle': 'white', 'titanium_white': 'white', 'tomato': 'red',
 'turquoise': 'cyan', 'turquoise_blue': 'blue', 'turquoise_dark': 'cyan',
 'turquoise_medium': 'cyan', 'turquoise_pale': 'cyan', 'ultramarine': 'blue',
 'ultramarine_violet': 'magenta', 'van_dyke_brown': 'brown',
 'venetian_red': 'red', 'violet': 'magenta', 'violet_dark': 'magenta',
 'violet_red': 'magenta', 'violet_red_medium': 'magenta',
 'violet_red_pale': 'magenta', 'viridian_light': 'green', 'warm_grey': 'grey',
 'wheat': 'white', 'white': 'white', 'white_smoke': 'white',
 'whitesmoke': 'white', 'yellow': 'yellow', 'yellow_green': 'green',
 'yellow_light': 'yellow', 'yellow_ochre': 'orange', 'yellowgreen': 'green',
 'zinc_white': 'white'}

CSS3_NAMES_TO_RGB = {
 'aliceblue': [0.9411764705882353, 0.9725490196078431, 1.0],
 'antiquewhite': [0.9803921568627451, 0.9215686274509803, 0.8431372549019608],
 'aquamarine': [0.4980392156862745, 1.0, 0.8313725490196079],
 'azure': [0.9411764705882353, 1.0, 1.0],
 'beige': [0.9607843137254902, 0.9607843137254902, 0.8627450980392157],
 'bisque': [1.0, 0.8941176470588236, 0.7686274509803922],
 'black': [0.0, 0.0, 0.0],
 'blanchedalmond': [1.0, 0.9215686274509803, 0.803921568627451],
 'blue': [0.0, 0.0, 1.0],
 'blueviolet': [0.5411764705882353, 0.16862745098039217, 0.8862745098039215],
 'brown': [0.6470588235294118, 0.16470588235294117, 0.16470588235294117],
 'burlywood': [0.8705882352941177, 0.7215686274509804, 0.5294117647058824],
 'cadetblue': [0.37254901960784315, 0.6196078431372549, 0.6274509803921569],
 'chartreuse': [0.4980392156862745, 1.0, 0.0],
 'chocolate': [0.8235294117647058, 0.4117647058823529, 0.11764705882352941],
 'coral': [1.0, 0.4980392156862745, 0.3137254901960784],
 'cornflowerblue': [0.39215686274509803,
                    0.5843137254901961,
                    0.9294117647058824],
 'cornsilk': [1.0, 0.9725490196078431, 0.8627450980392157],
 'crimson': [0.8627450980392157, 0.0784313725490196, 0.23529411764705882],
 'cyan': [0.0, 1.0, 1.0],
 'darkblue': [0.0, 0.0, 0.5450980392156862],
 'darkcyan': [0.0, 0.5450980392156862, 0.5450980392156862],
 'darkgoldenrod': [0.7215686274509804,
                   0.5254901960784314,
                   0.043137254901960784],
 'darkgray': [0.6627450980392157, 0.6627450980392157, 0.6627450980392157],
 'darkgreen': [0.0, 0.39215686274509803, 0.0],
 'darkkhaki': [0.7411764705882353, 0.7176470588235294, 0.4196078431372549],
 'darkmagenta': [0.5450980392156862, 0.0, 0.5450980392156862],
 'darkolivegreen': [0.3333333333333333, 0.4196078431372549, 0.1843137254901961],
 'darkorange': [1.0, 0.5490196078431373, 0.0],
 'darkorchid': [0.6, 0.19607843137254902, 0.8],
 'darkred': [0.5450980392156862, 0.0, 0.0],
 'darksalmon': [0.9137254901960784, 0.5882352941176471, 0.47843137254901963],
 'darkseagreen': [0.5607843137254902, 0.7372549019607844, 0.5607843137254902],
 'darkslateblue': [0.2823529411764706, 0.23921568627450981, 0.5450980392156862],
 'darkslategray': [0.1843137254901961,
                   0.30980392156862746,
                   0.30980392156862746],
 'darkturquoise': [0.0, 0.807843137254902, 0.8196078431372549],
 'darkviolet': [0.5803921568627451, 0.0, 0.8274509803921568],
 'deeppink': [1.0, 0.0784313725490196, 0.5764705882352941],
 'deepskyblue': [0.0, 0.7490196078431373, 1.0],
 'dimgray': [0.4117647058823529, 0.4117647058823529, 0.4117647058823529],
 'dodgerblue': [0.11764705882352941, 0.5647058823529412, 1.0],
 'firebrick': [0.6980392156862745, 0.13333333333333333, 0.13333333333333333],
 'floralwhite': [1.0, 0.9803921568627451, 0.9411764705882353],
 'forestgreen': [0.13333333333333333, 0.5450980392156862, 0.13333333333333333],
 'gainsboro': [0.8627450980392157, 0.8627450980392157, 0.8627450980392157],
 'ghostwhite': [0.9725490196078431, 0.9725490196078431, 1.0],
 'gold': [1.0, 0.8431372549019608, 0.0],
 'goldenrod': [0.8549019607843137, 0.6470588235294118, 0.12549019607843137],
 'gray': [0.5019607843137255, 0.5019607843137255, 0.5019607843137255],
 'green': [0.0, 0.5019607843137255, 0.0],
 'greenyellow': [0.6784313725490196, 1.0, 0.1843137254901961],
 'honeydew': [0.9411764705882353, 1.0, 0.9411764705882353],
 'hotpink': [1.0, 0.4117647058823529, 0.7058823529411765],
 'indianred': [0.803921568627451, 0.3607843137254902, 0.3607843137254902],
 'indigo': [0.29411764705882354, 0.0, 0.5098039215686274],
 'ivory': [1.0, 1.0, 0.9411764705882353],
 'khaki': [0.9411764705882353, 0.9019607843137255, 0.5490196078431373],
 'lavender': [0.9019607843137255, 0.9019607843137255, 0.9803921568627451],
 'lavenderblush': [1.0, 0.9411764705882353, 0.9607843137254902],
 'lawngreen': [0.48627450980392156, 0.9882352941176471, 0.0],
 'lemonchiffon': [1.0, 0.9803921568627451, 0.803921568627451],
 'lightblue': [0.6784313725490196, 0.8470588235294118, 0.9019607843137255],
 'lightcoral': [0.9411764705882353, 0.5019607843137255, 0.5019607843137255],
 'lightcyan': [0.8784313725490196, 1.0, 1.0],
 'lightgoldenrodyellow': [0.9803921568627451,
                          0.9803921568627451,
                          0.8235294117647058],
 'lightgray': [0.8274509803921568, 0.8274509803921568, 0.8274509803921568],
 'lightgreen': [0.5647058823529412, 0.9333333333333333, 0.5647058823529412],
 'lightpink': [1.0, 0.7137254901960784, 0.7568627450980392],
 'lightsalmon': [1.0, 0.6274509803921569, 0.47843137254901963],
 'lightseagreen': [0.12549019607843137, 0.6980392156862745, 0.6666666666666666],
 'lightskyblue': [0.5294117647058824, 0.807843137254902, 0.9803921568627451],
 'lightslategray': [0.4666666666666667, 0.5333333333333333, 0.6],
 'lightsteelblue': [0.6901960784313725, 0.7686274509803922, 0.8705882352941177],
 'lightyellow': [1.0, 1.0, 0.8784313725490196],
 'lime': [0.0, 1.0, 0.0],
 'limegreen': [0.19607843137254902, 0.803921568627451, 0.19607843137254902],
 'linen': [0.9803921568627451, 0.9411764705882353, 0.9019607843137255],
 'magenta': [1.0, 0.0, 1.0],
 'maroon': [0.5019607843137255, 0.0, 0.0],
 'mediumaquamarine': [0.4, 0.803921568627451, 0.6666666666666666],
 'mediumblue': [0.0, 0.0, 0.803921568627451],
 'mediumorchid': [0.7294117647058823, 0.3333333333333333, 0.8274509803921568],
 'mediumpurple': [0.5764705882352941, 0.4392156862745098, 0.8588235294117647],
 'mediumseagreen': [0.23529411764705882,
                    0.7019607843137254,
                    0.44313725490196076],
 'mediumslateblue': [0.4823529411764706,
                     0.40784313725490196,
                     0.9333333333333333],
 'mediumspringgreen': [0.0, 0.9803921568627451, 0.6039215686274509],
 'mediumturquoise': [0.2823529411764706, 0.8196078431372549, 0.8],
 'mediumvioletred': [0.7803921568627451,
                     0.08235294117647059,
                     0.5215686274509804],
 'midnightblue': [0.09803921568627451, 0.09803921568627451, 0.4392156862745098],
 'mintcream': [0.9607843137254902, 1.0, 0.9803921568627451],
 'mistyrose': [1.0, 0.8941176470588236, 0.8823529411764706],
 'moccasin': [1.0, 0.8941176470588236, 0.7098039215686275],
 'navajowhite': [1.0, 0.8705882352941177, 0.6784313725490196],
 'navy': [0.0, 0.0, 0.5019607843137255],
 'oldlace': [0.9921568627450981, 0.9607843137254902, 0.9019607843137255],
 'olive': [0.5019607843137255, 0.5019607843137255, 0.0],
 'olivedrab': [0.4196078431372549, 0.5568627450980392, 0.13725490196078433],
 'orange': [1.0, 0.6470588235294118, 0.0],
 'orangered': [1.0, 0.27058823529411763, 0.0],
 'orchid': [0.8549019607843137, 0.4392156862745098, 0.8392156862745098],
 'palegoldenrod': [0.9333333333333333, 0.9098039215686274, 0.6666666666666666],
 'palegreen': [0.596078431372549, 0.984313725490196, 0.596078431372549],
 'paleturquoise': [0.6862745098039216, 0.9333333333333333, 0.9333333333333333],
 'palevioletred': [0.8588235294117647, 0.4392156862745098, 0.5764705882352941],
 'papayawhip': [1.0, 0.9372549019607843, 0.8352941176470589],
 'peachpuff': [1.0, 0.8549019607843137, 0.7254901960784313],
 'peru': [0.803921568627451, 0.5215686274509804, 0.24705882352941178],
 'pink': [1.0, 0.7529411764705882, 0.796078431372549],
 'plum': [0.8666666666666667, 0.6274509803921569, 0.8666666666666667],
 'powderblue': [0.6901960784313725, 0.8784313725490196, 0.9019607843137255],
 'purple': [0.5019607843137255, 0.0, 0.5019607843137255],
 'red': [1.0, 0.0, 0.0],
 'rosybrown': [0.7372549019607844, 0.5607843137254902, 0.5607843137254902],
 'royalblue': [0.2549019607843137, 0.4117647058823529, 0.8823529411764706],
 'saddlebrown': [0.5450980392156862, 0.27058823529411763, 0.07450980392156863],
 'salmon': [0.9803921568627451, 0.5019607843137255, 0.4470588235294118],
 'sandybrown': [0.9568627450980393, 0.6431372549019608, 0.3764705882352941],
 'seagreen': [0.1803921568627451, 0.5450980392156862, 0.3411764705882353],
 'seashell': [1.0, 0.9607843137254902, 0.9333333333333333],
 'sienna': [0.6274509803921569, 0.3215686274509804, 0.17647058823529413],
 'silver': [0.7529411764705882, 0.7529411764705882, 0.7529411764705882],
 'skyblue': [0.5294117647058824, 0.807843137254902, 0.9215686274509803],
 'slateblue': [0.41568627450980394, 0.35294117647058826, 0.803921568627451],
 'slategray': [0.4392156862745098, 0.5019607843137255, 0.5647058823529412],
 'snow': [1.0, 0.9803921568627451, 0.9803921568627451],
 'springgreen': [0.0, 1.0, 0.4980392156862745],
 'steelblue': [0.27450980392156865, 0.5098039215686274, 0.7058823529411765],
 'tan': [0.8235294117647058, 0.7058823529411765, 0.5490196078431373],
 'teal': [0.0, 0.5019607843137255, 0.5019607843137255],
 'thistle': [0.8470588235294118, 0.7490196078431373, 0.8470588235294118],
 'tomato': [1.0, 0.38823529411764707, 0.2784313725490196],
 'turquoise': [0.25098039215686274, 0.8784313725490196, 0.8156862745098039],
 'violet': [0.9333333333333333, 0.5098039215686274, 0.9333333333333333],
 'wheat': [0.9607843137254902, 0.8705882352941177, 0.7019607843137254],
 'white': [1.0, 1.0, 1.0],
 'whitesmoke': [0.9607843137254902, 0.9607843137254902, 0.9607843137254902],
 'yellow': [1.0, 1.0, 0.0],
 'yellowgreen': [0.6039215686274509, 0.803921568627451, 0.19607843137254902]
 }

colors_matrices = {}
for color_name, rgb in CSS3_NAMES_TO_RGB.items():
    color2_rgb = sRGBColor(*rgb)
    color2_lab = convert_color(color2_rgb, LabColor)
    color2_lab_matrix = _get_lab_color2_matrix(color2_lab)
    colors_matrices[color_name] = color2_lab_matrix

def _colordist(rgb1, color2_lab_matrix):
    rgb1 = (np.array(rgb1)/255).tolist()
    color1_lab = convert_color(sRGBColor(*rgb1), LabColor)
    delta_e = delta_e_cmc(color1_lab, color2_lab_matrix)
    return delta_e
