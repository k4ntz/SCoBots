

# TODO: Currently hardcoded bc fuck it
# extracts rgb values from coordinates
def extract_rgb_value(img, coords, gametype):
    # TODO: Implement correctly
    conv_coords = convert_coords(coords)
    rgb = img[conv_coords[0]][conv_coords[1]]
    return rgb


def convert_coords(coords):
    # TODO: implement 
    return coords
