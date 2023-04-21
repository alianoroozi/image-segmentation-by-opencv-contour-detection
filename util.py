import cv2
import numpy as np
from PIL import Image
from scipy.spatial import distance as dist


def crop_level_with_rectangle(img, rectangle):
    """ Crop and rotate/level given image using given rectangle
    Args:
    img : Image to crop and rotate/level
    rectangle : Rectangle used to crop and rotate/level the image

    Returns:
        an ndarray, containing cropped and rotated/levelled image
    """
    # the order of the box points: bottom left, top left, top right, bottom right
    box = cv2.boxPoints(rectangle)
    box = np.int0(box)
    box = order_points(box)

    (bl, tl, tr, br) = box

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst_pts = np.array([
        [0, maxHeight - 1],
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1]], dtype="float32")

    src_pts = box.astype("float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    img_warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    # return the warped image
    return img_warped


def order_points(box_pts):
    """ Sort box points in the order bottom-left, top-left, top-right, and bottom-right
    """
    # sort the points based on their x-coordinates
    xSorted = box_pts[np.argsort(box_pts[:, 0]), :]
    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    return np.array([bl, tl, tr, br])


def image_transpose_exif(im):
    """
        Apply Image.transpose to ensure 0th row of pixels is at the visual
        top of the image, and 0th column is the visual left-hand side.
        Return the original image if unable to determine the orientation.

        As per CIPA DC-008-2012, the orientation field contains an integer,
        1 through 8. Other values are reserved.
    """

    exif_orientation_tag = 0x0112
    exif_transpose_sequences = [  # Val  0th row  0th col
        [],  # 0    (reserved)
        [],  # 1   top      left
        [Image.Transpose.FLIP_LEFT_RIGHT],  # 2   top      right
        [Image.Transpose.ROTATE_180],  # 3   bottom   right
        [Image.Transpose.FLIP_TOP_BOTTOM],  # 4   bottom   left
        [Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.ROTATE_90],  # 5   left     top
        [Image.Transpose.ROTATE_270],  # 6   right    top
        [Image.Transpose.FLIP_TOP_BOTTOM, Image.Transpose.ROTATE_90],  # 7   right    bottom
        [Image.Transpose.ROTATE_90],  # 8   left     bottom
    ]

    try:
        seq = exif_transpose_sequences[im.getexif()[exif_orientation_tag]]
    except Exception as e:
        # print(e)
        return im
    else:
        return functools.reduce(type(im).transpose, seq, im)
