import numpy as np
import cv2


def image_t(im, scale=1.0, rot=45, trans=(50, -50)):
    # the image dimensions
    (h, w) = im.shape[:2]

    # the rotation angle from degrees to radians, negative for clockwise rotation
    rot_rad = -np.deg2rad(rot)

    # the center of the image
    center = (w / 2, h / 2)

    # three points on the image before transformation
    src_points = np.float32([
        center,
        (center[0] + w / 2, center[1] - h / 2),
        (center[0] - w / 2, center[1] - h / 2)
    ])

    # the corresponding points after transformation
    dst_points = np.float32([
        # center point remains the same after rotation and scaling
        (center[0] + trans[0], center[1] + trans[1]),
        # right point after rotation, scaling and translation
        (int(center[0] + (w / 2) * scale * np.cos(rot_rad) + (h / 2) * scale * np.sin(rot_rad) + trans[0]),
         int(center[1] + (w / 2) * scale * np.sin(rot_rad) - (h / 2) * scale * np.cos(rot_rad) + trans[1])),
        # left point after rotation, scaling and translation
        (int(center[0] - (w / 2) * scale * np.cos(rot_rad) + (h / 2) * scale * np.sin(rot_rad) + trans[0]),
         int(center[1] - (w / 2) * scale * np.sin(rot_rad) - (h / 2) * scale * np.cos(rot_rad) + trans[1]))
    ])

    # the affine transformation matrix
    M = cv2.getAffineTransform(src_points, dst_points)

    # the transformation matrix to the image
    result = cv2.warpAffine(im, M, (w, h))

    return result


'''
# alternatively, the following simply does the same job

def image_t(im, scale=1.0, rot=45, trans=(50, -50)):
    # the image dimensions
    (h, w) = im.shape[:2]

    # the center of the image
    center = (w // 2, h // 2)

    # the affine transformation matrix with rotation and scaling
    M = cv2.getRotationMatrix2D(center, rot, scale)

    # the translation
    M[0, 2] += trans[0]
    M[1, 2] += trans[1]

    # the transformation matrix to the image
    result = cv2.warpAffine(im, M, (w, h))

    return result
'''

if __name__ == '__main__':
    im = cv2.imread('./misc/pearl.jpeg')

    scale = 0.5
    rot = 45
    trans = (50, -50)
    result = image_t(im, scale, rot, trans)
    cv2.imwrite('./results/affine_result.png', result)
