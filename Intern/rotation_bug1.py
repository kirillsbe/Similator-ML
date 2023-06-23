import utils
from skimage import data


def test_rotated_image():
    cat = data.chelsea()
    cat_rotated = utils.rotated_image(cat)
    assert cat_rotated.shape == cat.shape
