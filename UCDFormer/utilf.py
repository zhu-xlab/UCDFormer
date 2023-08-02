from ctypes import *
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from skimage.color import rgb2lab, lab2rgb
from skimage.segmentation import mark_boundaries
from random import random as r
from random import uniform

# Show image in notebook using matplotlib (instead of opencv's show() function)
def show_inplace(image):
    plt.axis('off')
    plt.imshow(image)
    plt.show()
    
# Cluster superpixels in the color space to obtain the final clustering
def cluster_superpixels(superpixels, n_regions=2, m=2):
    sums, sizes = get_sums_and_sizes(superpixels)
    sp_h, sp_w, _ = superpixels.shape
    clustered = np.zeros((sp_h, sp_w))
    markers = np.unique(superpixels.reshape((sp_h * sp_w, -1)), axis=0, return_inverse=True)[1]
    markers = markers.reshape((sp_h , sp_w))
    
    # Run Superpixel-based Fuzzy C-means multiple times and choose best result (with respect to within cluster distances)
    error = 1.7e300
    res = sizes.copy()
    for i in range(100):
        tmp_result, tmp_error = fcm_superpixels(superpixels, sums, sizes, c=n_regions, m=m)
        if tmp_error < error:
            error = tmp_error
            res = tmp_result

    # Color each region in the final mask using the mean color of that region
    for y in range(superpixels.shape[0]):
        for x in range(superpixels.shape[1]):
            clustered[y][x] = res[markers[y][x]]
    return clustered

# Use mean color of pixels to color each region in the final clustering
def color_regions(superpixels, original):
    result = original.copy()
    rows, cols = superpixels.shape
    for val in np.unique(superpixels):
        result[superpixels == val] = np.mean(original[superpixels == val], axis=0)
    return result

# Interface for the C implementation of Fuzzy SLIC
def fslic(img, initial_num_clusters=200, compactness=10.0, max_iterations=10, p=0.0, q=2.0):
    h, w, depth = img.shape
    shared_file = CDLL('/data/xuqs/Change_detection/Change_detection_1222/Baselines/PytorchStyleFormer-main/PytorchStyleFormer-main/shared.so')
    shared_file.fslic.restype = POINTER(c_double * (h * w * depth))
    l = py_object(img.astype(c_double).flatten().tolist())
    v = shared_file.fslic(l, w, h, depth, c_int(initial_num_clusters), c_double(compactness), max_iterations, c_double(p), c_double(q))
    result_1d = [x for x in v.contents]
    result = np.zeros((h, w, depth))
    for y in range(h):
        for x in range(w):
            for z in range(depth):
                result[y][x][z] = result_1d[(((y) * (w) * depth) + ((x) * depth) + z)]
    return result

# Returns if pixel coordinates are in range
def VALID(x,y,w,h):
    return 0 <= x < w and 0 <= y <h

# Returns sum of colors and number of pixels in each superpixel (necessary for Superpixel-based Fuzzy C-means)
def get_sums_and_sizes(img):
    rows, cols, depth = img.shape
    reshaped = img.reshape((rows * cols, depth))
    unique, markers = np.unique(reshaped, axis=0, return_inverse=True)
    num_regions = len(unique)
    sums = np.zeros((num_regions, depth))
    sizes = np.zeros(num_regions)    
    for i in range(num_regions):
        mask = markers == i
        sums[i] = np.sum(reshaped[mask], axis=0)
        sizes[i] = np.count_nonzero(mask)
    return sums, sizes

# Superpixel-based Fuzzy C-means
def fcm_superpixels(superpixels, region_sums, region_sizes, c=2, m=2, eta=0.0001, max_iterations=50):
    channels = superpixels.shape[-1]
    num_regions = len(region_sizes)
    exponent = -2/(m-1)
    U = np.random.rand(c, num_regions).astype(float)
    for i in range(num_regions):
        U[:, i] /= np.sum(U[:, i])
    cluster_centers = np.zeros((c, channels)).astype(float)
    prev = U.copy()
    
    # loop till convergence or max_iterations has been reached
    for i in range(max_iterations):
        prev = U.copy()
        # Recalculate cluster centers
        for j in range(c):
            numerator = denominator = 0.0
            for k in range(num_regions):
                u_klm = U[j][k] ** m
                numerator += u_klm * region_sums[k]
                denominator += u_klm * region_sizes[k]
            cluster_centers[j] = numerator / denominator

        # Recalculate membership matrix
        for j in range(c):
            center = cluster_centers[j]
            for k in range(num_regions):
                region_avg = region_sums[k] / region_sizes[k]
                diff = region_avg - center
                numerator = np.sqrt(diff.dot(diff)) ** exponent
                denominator = 0.0
                for l in cluster_centers:
                    diff = region_avg - l
                    denominator += np.sqrt(diff.dot(diff)) ** exponent
                U[j][k] = numerator/denominator
        if(np.max(U-prev) <= eta):
            break
            
    # Assign each superpixel to the cluster which has the highest membership
    assignments = np.argmax(U, axis=0)
    
    # Calculate within cluster distances (error).
    # (Needed to compare multiple runs of the algorithm (which is sensitive to initial conditions))
    error = 0.0
    for i in range(num_regions):
        avg = region_sums[i] / region_sizes[i]
        diff = avg - cluster_centers[assignments[i]]
        error += diff.dot(diff)
    return assignments, error