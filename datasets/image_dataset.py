"""Create dataset for images."""
import logging
import os
import numpy as np
import cv2

logger = logging.getLogger(__name__)


__THIS_DIR = os.path.dirname(__file__)


def load_image(filename, dtype=np.float32, scaled=True, gray=False, dsize=None):
    """Load image from filename.

    Parameters
    ----------
    filename: string
        File path for the image.
    dtype:
        Data type for the loaded image.
    scaled: boolean
        If scaled is True, then the image is scaled to the range [0., 1.],
        otherwise it remains in [0, 255].  Note that this option is valid only
        for floating data type.
    gray: boolean
        If True, then the image is converted to single channel grayscale image.
    dsize: tuple or None
        If not None, then the image is rescaled to this shape.

    Returns
    -------
    img: ndarray [H, W, C]
        Image array with color channel at the last dimension.  If it is a color
        image, then the color channel is organized in RGB order.  If it is a
        grayscale image, the last dimension is always 1.
    """
    filename = os.path.join(__THIS_DIR, '..', filename)
    assert os.path.exists(filename), \
        'Path does not exist: {}'.format(filename)
    image_name = os.path.split(filename)[-1]
    gray_indicator = 'gray' in image_name.lower() or \
        'grey' in image_name.lower()
    if gray_indicator != gray:
        logger.debug(
            'Specifying gray filename as %s while requiring gray as %s',
            gray_indicator, gray
        )
    if gray:
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if dsize is not None:
        image = cv2.resize(image, dsize=dsize)
    if gray:
        image = np.expand_dims(image, axis=-1)
    image = image.astype(dtype)
    if scaled and not np.issubdtype(image.dtype, np.integer):
        image /= 255.
    return image


__image_list = {
    'lena': 'images/ece533/lena.ppm',
    'barbara': 'images/sporco_get_images/standard/barbara.bmp',
    'boat.gray': 'images/misc/boat.512.tiff',
    'house': 'images/misc/4.1.05.tiff',
    'peppers': 'images/misc/4.2.07.tiff',
    'cameraman.gray': 'images/ece533/cameraman.tif',
    'man.gray': 'images/sporco_get_images/standard/man.grey.tiff',
    'mandrill': 'images/sporco_get_images/standard/mandrill.tiff',
    'monarch': 'images/sporco_get_images/standard/monarch.png',
    'fruit': ['images/FCSC/Images/fruit_100_100/{}.jpg'.format(i+1)
              for i in range(10)],
    'city': ['images/FCSC/Images/city_100_100/{}.jpg'.format(i+1)
             for i in range(10)],
    'singles': [  # used as test images for fruit and city
        'images/FCSC/Images/singles/test1/test1.jpg',
        'images/FCSC/Images/singles/test2/test2.jpg',
        'images/FCSC/Images/singles/test3/test3.jpg',
        'images/FCSC/Images/singles/test4/test.jpg',
    ]
}


def get_image_items():
    """Gets all available images."""
    return __image_list


def create_image_blob(name, dtype=np.float32, scaled=True, gray=False,
                      dsize=None):
    """Create a 4-D image blob from a valid image name.

    Parameters
    ----------
    name: string
        The name of image or image list.
    dtype, scaled, gray, dsize
        Refer to `image_dataset.load_image` for more details.

    Returns
    -------
    blob: ndarray, [H, W, C, N]
        Loaded 4-D image blob.  The dimension is organized as [height, width,
        channels, batch], which is for the ease of internal use of SPORCO.
    """
    img_list = __image_list[name]
    if not isinstance(img_list, list):
        img_list = [img_list]
    imgs = [load_image(img, dtype, scaled, gray, dsize) for img in img_list]
    return np.stack(imgs, axis=-1)
