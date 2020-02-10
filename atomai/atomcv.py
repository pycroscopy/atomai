# TODO add 'get contours' class

"""
Module with computer vision tools for analysis of neural network output
"""

import numpy as np
import cv2
from scipy import ndimage


class locate:
    """
    Transforms pixel data from NN output into coordinate data

    Args:
        decoded_imgs: 4D numpy array
            the output of a neural network (softmax/sigmoid layer)
        threshold: float
            value at which the neural network output is thresholded
        dist_edge: int
            distance within image boundaries not to consider
        dim_order: str
            'channel_last' or 'channel_first' (Default: 'channel last')
    """
    def __init__(self,
                 nn_output,
                 threshold=0.5,
                 dist_edge=5,
                 dim_order='channel_last'):

        if nn_output.shape[-1] == 1: # Add background class for 1-channel data
            nn_output_b = 1 - nn_output
            nn_output = np.concatenate(
                (nn_output[:, :, :, None], nn_output_b[:, :, :, None]), axis=3)
        if dim_order == 'channel_first':  # make channel dim the last dim
            nn_output = np.transpose(nn_output, (0, 2, 3, 1))
        elif dim_order == 'channel_last':
            pass
        else:
            raise NotImplementedError(
                'For dim_order, use "channel_first"',
                'or "channel_last" (e.g. tensorflow)')
        self.nn_output = nn_output
        self.threshold = threshold
        self.dist_edge = dist_edge

    def get_all_coordinates(self):
        '''Extract all atomic coordinates in image
        via CoM method & store data as a dictionary
        (key: frame number)'''

        def find_com(image_data):
            '''Find atoms via center of mass methods'''
            labels, nlabels = ndimage.label(image_data)
            coordinates = np.array(
                ndimage.center_of_mass(
                    image_data, labels, np.arange(nlabels) + 1))
            coordinates = coordinates.reshape(coordinates.shape[0], 2)
            return coordinates

        d_coord = {}
        for i, decoded_img in enumerate(self.nn_output):
            coordinates = np.empty((0, 2))
            category = np.empty((0, 1))
            # we assume that class backgrpund is always the last one
            for ch in range(decoded_img.shape[2]-1):
                decoded_img_c = cv_thresh(
                    decoded_img[:, :, ch], self.threshold)
                coord = find_com(decoded_img_c)
                coord_ch = self.rem_edge_coord(coord)
                category_ch = np.zeros((coord_ch.shape[0], 1)) + ch
                coordinates = np.append(coordinates, coord_ch, axis=0)
                category = np.append(category, category_ch, axis=0)
            d_coord[i] = np.concatenate((coordinates, category), axis=1)
        return d_coord

    def rem_edge_coord(self, coordinates):
        '''Remove coordinates at the image edges'''

        def coord_edges(coordinates, w, h):
            return [coordinates[0] > w - self.dist_edge,
                    coordinates[0] < self.dist_edge,
                    coordinates[1] > h - self.dist_edge,
                    coordinates[1] < self.dist_edge]

        w, h = self.nn_output.shape[1:3]
        coord_to_rem = [
                        idx for idx, c in enumerate(coordinates)
                        if any(coord_edges(c, w, h))
                        ]
        coord_to_rem = np.array(coord_to_rem, dtype=int)
        coordinates = np.delete(coordinates, coord_to_rem, axis=0)
        return coordinates


def cv_thresh(imgdata,
              threshold):
    """
    Wrapper for opencv binary threshold method.
    Returns thresholded image.
    """
    _, thresh = cv2.threshold(
                    imgdata, 
                    threshold, 1, 
                    cv2.THRESH_BINARY)
    return thresh

