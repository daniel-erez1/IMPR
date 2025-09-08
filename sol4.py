import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import map_coordinates, gaussian_filter
from scipy.ndimage import convolve
import cv2
from scipy.ndimage import map_coordinates, gaussian_filter
from scipy.ndimage.measurements import label, center_of_mass
from scipy.ndimage.morphology import generate_binary_structure
import sol4_utils
import os
from tqdm import tqdm
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import label, center_of_mass
import shutil
from imageio import imwrite
def harris_corner_detector(im):
  """
  Detects harris corners.
  Make sure the returned coordinates are x major!!!
  :param im: A 2D array representing an image.
  :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.  
  """
  # Step 1: Compute derivatives using [1, 0, -1] filters
  dx_filter = np.array([[1, 0, -1]], dtype=np.float32)  # Horizontal derivative filter
  dy_filter = dx_filter.T  # Vertical derivative filter

  Ix = convolve(im.astype(np.float32), dx_filter)
  Iy = convolve(im.astype(np.float32), dy_filter)

  # Step 2: Compute products of derivatives
  Ix2 = Ix * Ix
  Iy2 = Iy * Iy
  IxIy = Ix * Iy

  # Step 3: Blur the products using cv2.GaussianBlur with kernel_size=3
  kernel_size = 3

  Ix2 = cv2.GaussianBlur(Ix2, (kernel_size, kernel_size), 0)
  Iy2 = cv2.GaussianBlur(Iy2, (kernel_size, kernel_size), 0)
  IxIy = cv2.GaussianBlur(IxIy, (kernel_size, kernel_size), 0)

  # Step 4: Compute Harris response
  # R = det(M) - k * trace(M)^2
  # where M = [[Ix2, IxIy], [IxIy, Iy2]]
  k = 0.04
  det_M = Ix2 * Iy2 - IxIy * IxIy
  trace_M = Ix2 + Iy2
  R = det_M - k * (trace_M ** 2)

  # Step 5: Find local maxima using the existing non_maximum_suppression
  corner_mask = non_maximum_suppression(R)

  # Step 6: Extract corner coordinates
  # IMPORTANT: np.where returns (rows, cols) but we need (x, y) = (cols, rows)
  corners_rows, corners_cols = np.where(corner_mask)
  corners = np.stack([corners_cols, corners_rows], axis=1)  # [x, y] format

  return corners
  

def sample_descriptor(im, pos, desc_rad):
  """
  Samples descriptors at the given corners.
  :param im: A 2D array representing an image.
  :param pos: An array with shape (N,2), where pos[i,:] are the [x,y] coordinates of the ith corner point.   
  :param desc_rad: "Radius" of descriptors to compute.
  :return: A 3D array with shape (N,K,K) containing the ith descriptor at desc[i,:,:].
  """
  K = 1 + 2 * desc_rad  # Descriptor size (7x7 for desc_rad=3)
  N = pos.shape[0]  # Number of feature points
  descriptors = np.zeros((N, K, K))

  # Create relative coordinates for the patch
  # We want a KxK grid centered at (0,0)
  patch_range = np.arange(-desc_rad, desc_rad + 1)
  patch_y, patch_x = np.meshgrid(patch_range, patch_range, indexing='ij')

  for i in range(N):
      # Get the center point coordinates
      # pos is in (x, y) format
      center_x, center_y = pos[i, 0], pos[i, 1]

      # Calculate absolute coordinates for sampling
      # These may be sub-pixel coordinates
      sample_x = patch_x.flatten() + center_x
      sample_y = patch_y.flatten() + center_y

      # Stack coordinates for map_coordinates
      # map_coordinates expects (row, col) format, so we need (y, x)
      coords = np.vstack([sample_y, sample_x])

      # Sample using bilinear interpolation
      sampled_values = map_coordinates(im, coords, order=1, prefilter=False)

      # Reshape to KxK
      descriptor = sampled_values.reshape(K, K)

      # Normalize the descriptor
      mean_val = np.mean(descriptor)
      descriptor_centered = descriptor - mean_val
      norm = np.linalg.norm(descriptor_centered)

      if norm > 0:
          descriptor = descriptor_centered / norm
      else:
          # If norm is zero, return zero descriptor
          descriptor = np.zeros((K, K))

      descriptors[i] = descriptor

  return descriptors
  

def find_features(pyr):
  """
  Detects and extracts feature points from a pyramid.
  :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
  :return: A list containing:
              1) An array with shape (N,2) of [x,y] feature location per row found in the image. 
                 These coordinates are provided at the pyramid level pyr[0].
              2) A feature descriptor array with shape (N,K,K)
  """
  # Detect features at the original resolution (pyr[0])
  # Use spread_out_corners for better spatial distribution
  feature_points = spread_out_corners(pyr[0], m=7, n=7, radius=16)

  # Convert feature coordinates from level 0 to level 2
  # According to equation (1) in the PDF: p_lj = 2^(li-lj) * p_li
  # From level 0 to level 2: p_l2 = 2^(0-2) * p_l0 = p_l0 / 4
  feature_points_l2 = feature_points / 4.0

  # Sample descriptors at level 2 of the pyramid
  # Using desc_rad=3 to get 7x7 descriptors
  descriptors = sample_descriptor(pyr[2], feature_points_l2, desc_rad=3)

  # Return features at original resolution (pyr[0]) and their descriptors
  return [feature_points, descriptors]


def match_features(desc1, desc2, min_score):
  """
  Return indices of matching descriptors.
  :param desc1: A feature descriptor array with shape (N1,K,K).
  :param desc2: A feature descriptor array with shape (N2,K,K).
  :param min_score: Minimal match score.
  :return: A list containing:
              1) An array with shape (M,) and dtype int of matching indices in desc1.
              2) An array with shape (M,) and dtype int of matching indices in desc2.
  """
  N1 = desc1.shape[0]
  N2 = desc2.shape[0]

  # Flatten descriptors for dot product computation
  # Shape: (N1, K*K) and (N2, K*K)
  desc1_flat = desc1.reshape(N1, -1)
  desc2_flat = desc2.reshape(N2, -1)

  # Compute all pairwise dot products (match scores)
  # S[i,j] = dot product between desc1[i] and desc2[j]
  # Shape: (N1, N2)
  S = desc1_flat @ desc2_flat.T

  # Lists to store matching indices
  matches_ind1 = []
  matches_ind2 = []

  # For each descriptor in desc1, check if it has a valid match in desc2
  for j in range(N1):
      # Get scores for descriptor j from image 1 with all descriptors in image 2
      scores_j = S[j, :]

      # Find the best and second best matches in desc2
      if N2 >= 2:
          # Get indices of top 2 scores
          top2_indices = np.argpartition(scores_j, -2)[-2:]
          top2_indices = top2_indices[np.argsort(scores_j[top2_indices])[::-1]]
          best_k = top2_indices[0]
          second_best_score = scores_j[top2_indices[1]]
      elif N2 == 1:
          best_k = 0
          second_best_score = -np.inf
      else:
          continue

      # Score between desc1[j] and desc2[best_k]
      score_jk = scores_j[best_k]

      # Check condition 1: S_jk >= second_best_score in row j
      # (This is automatically true since best_k is the argmax)

      # Check condition 2: S_jk >= second_best_score in column best_k
      scores_k = S[:, best_k]  # All scores for desc2[best_k]

      if N1 >= 2:
          # Get indices of top 2 scores in column
          top2_col_indices = np.argpartition(scores_k, -2)[-2:]
          top2_col_indices = top2_col_indices[np.argsort(scores_k[top2_col_indices])[::-1]]

          # Check if j is in the top 2 matches for descriptor best_k
          if j not in top2_col_indices:
              continue

          # Get the second best score in the column
          if top2_col_indices[0] == j:
              second_best_col_score = scores_k[top2_col_indices[1]]
          else:
              second_best_col_score = scores_k[top2_col_indices[0]]
      else:
          second_best_col_score = -np.inf

      # Check condition 3: Score must be greater than min_score
      if score_jk > min_score:
          # All conditions satisfied - add this match
          matches_ind1.append(j)
          matches_ind2.append(best_k)

  # Convert to numpy arrays
  matches_ind1 = np.array(matches_ind1, dtype=int)
  matches_ind2 = np.array(matches_ind2, dtype=int)

  return [matches_ind1, matches_ind2]


def apply_homography(pos1, H12):
  """
  Apply homography to inhomogenous points.
  :param pos1: An array with shape (N,2) of [x,y] point coordinates.
  :param H12: A 3x3 homography matrix.
  :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12.
  """
  N = pos1.shape[0]

  # Convert to homogeneous coordinates
  # pos1 is (N, 2) with [x, y] coordinates
  # Add a column of ones to make it (N, 3) with [x, y, 1]
  ones = np.ones((N, 1))
  pos1_homogeneous = np.hstack([pos1, ones])  # Shape: (N, 3)

  # Apply homography: p2_homo = H12 @ p1_homo^T
  # Transpose to get (3, N), apply H, then transpose back
  pos2_homogeneous = (H12 @ pos1_homogeneous.T).T  # Shape: (N, 3)

  # Convert back to inhomogeneous coordinates
  # Divide by the third coordinate (z) to normalize
  # Handle the case where z might be 0 (point at infinity)
  z_coords = pos2_homogeneous[:, 2]

  # Avoid division by zero
  valid_mask = np.abs(z_coords) > 1e-10
  pos2 = np.zeros_like(pos1)

  pos2[valid_mask, 0] = pos2_homogeneous[valid_mask, 0] / z_coords[valid_mask]  # x
  pos2[valid_mask, 1] = pos2_homogeneous[valid_mask, 1] / z_coords[valid_mask]  # y

  # Points with z=0 are at infinity, keep them as zeros or mark as invalid
  # The RANSAC process will treat these as outliers

  return pos2


def ransac_homography(points1, points2, num_iter, inlier_tol, translation_only=False):
  """
  Computes homography between two sets of points using RANSAC.
  :param pos1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
  :param pos2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
  :param num_iter: Number of RANSAC iterations to perform.
  :param inlier_tol: inlier tolerance threshold.
  :param translation_only: see estimate rigid transform
  :return: A list containing:
              1) A 3x3 normalized homography matrix.
              2) An Array with shape (S,) where S is the number of inliers,
                  containing the indices in pos1/pos2 of the maximal set of inlier matches found.
  """
  N = points1.shape[0]

  # Determine number of points needed for estimation
  if translation_only:
      n_points = 1  # Only need 1 point pair for translation
  else:
      n_points = 2  # Need 2 point pairs for rigid transform (rotation + translation)

  if N < n_points:
      # Not enough points
      return [np.eye(3), np.array([])]

  best_inliers = []
  best_homography = np.eye(3)

  for iteration in range(num_iter):
      # Step 1: Randomly sample point pairs
      sample_indices = np.random.choice(N, n_points, replace=False)
      sample_points1 = points1[sample_indices]
      sample_points2 = points2[sample_indices]

      # Step 2: Compute homography from samples
      H12 = estimate_rigid_transform(sample_points1, sample_points2, translation_only)

      # Step 3: Transform all points and compute distances
      transformed_points = apply_homography(points1, H12)

      # Compute squared Euclidean distances
      distances_squared = np.sum((transformed_points - points2) ** 2, axis=1)

      # Step 4: Find inliers
      inlier_mask = distances_squared < inlier_tol ** 2
      inlier_indices = np.where(inlier_mask)[0]

      # Step 5: Keep track of best set
      if len(inlier_indices) > len(best_inliers):
          best_inliers = inlier_indices
          best_homography = H12

  # Step 6: Recompute homography using all inliers
  if len(best_inliers) >= n_points:
      inlier_points1 = points1[best_inliers]
      inlier_points2 = points2[best_inliers]
      best_homography = estimate_rigid_transform(inlier_points1, inlier_points2, translation_only)

  return [best_homography, best_inliers]


def display_matches(im1, im2, points1, points2, inliers):
  """
  Dispalay matching points.
  :param im1: A grayscale image.
  :param im2: A grayscale image.
  :parma pos1: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
  :param pos2: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
  :param inliers: An array with shape (S,) of inlier matches.
  """
  #Create horizontally concatenated image
  h1, w1 = im1.shape
  h2, w2 = im2.shape
  h = max(h1, h2)
  combined_image = np.zeros((h, w1 + w2))
  combined_image[:h1, :w1] = im1
  combined_image[:h2, w1:] = im2

  # Display the combined image
  plt.figure(figsize=(15, 8))
  plt.imshow(combined_image, cmap='gray')

  # Plot all matched points as red dots
  plt.scatter(points1[:, 0], points1[:, 1], c='r', s=10, marker='o')
  plt.scatter(points2[:, 0] + w1, points2[:, 1], c='r', s=10, marker='o')

  # Draw lines between matches
  # Outliers in blue, inliers in yellow
  for i in range(len(points1)):
      x_coords = [points1[i, 0], points2[i, 0] + w1]
      y_coords = [points1[i, 1], points2[i, 1]]

      if i in inliers:
          # Inlier - yellow line
          plt.plot(x_coords, y_coords, 'y-', linewidth=0.5, alpha=0.7)
      else:
          # Outlier - blue line
          plt.plot(x_coords, y_coords, 'b-', linewidth=0.3, alpha=0.5)

  plt.title(f'Feature Matches - {len(inliers)} inliers (yellow) / {len(points1) - len(inliers)} outliers (blue)')
  plt.axis('off')
  plt.show()

def accumulate_homographies(H_succesive, m):
  """
  Convert a list of succesive homographies to a 
  list of homographies to a common reference frame.
  :param H_successive: A list of M-1 3x3 homography 
    matrices where H_successive[i] is a homography which transforms points
    from coordinate system i to coordinate system i+1.
  :param m: Index of the coordinate system towards which we would like to 
    accumulate the given homographies.
  :return: A list of M 3x3 homography matrices, 
    where H2m[i] transforms points from coordinate system i to coordinate system m
  """ 
  pass


def compute_bounding_box(homography, w, h):
  """
  computes bounding box of warped image under homography, without actually warping the image
  :param homography: homography
  :param w: width of the image
  :param h: height of the image
  :return: 2x2 array, where the first row is [x,y] of the top left corner,
   and the second row is the [x,y] of the bottom right corner
  """
  pass


def warp_channel(image, homography):
  """
  Warps a 2D image with a given homography.
  :param image: a 2D image.
  :param homography: homograhpy.
  :return: A 2d warped image.
  """
  pass


def warp_image(image, homography):
  """
  Warps an RGB image with a given homography.
  :param image: an RGB image.
  :param homography: homograhpy.
  :return: A warped image.
  """
  return np.dstack([warp_channel(image[...,channel], homography) for channel in range(3)])


def filter_homographies_with_translation(homographies, minimum_right_translation):
  """
  Filters rigid transformations encoded as homographies by the amount of translation from left to right.
  :param homographies: homograhpies to filter.
  :param minimum_right_translation: amount of translation below which the transformation is discarded.
  :return: filtered homographies..
  """
  translation_over_thresh = [0]
  last = homographies[0][0,-1]
  for i in range(1, len(homographies)):
    if homographies[i][0,-1] - last > minimum_right_translation:
      translation_over_thresh.append(i)
      last = homographies[i][0,-1]
  return np.array(translation_over_thresh).astype(int)


def estimate_rigid_transform(points1, points2, translation_only=False):
  """
  Computes rigid transforming points1 towards points2, using least squares method.
  points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
  :param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
  :param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
  :param translation_only: whether to compute translation only. False (default) to compute rotation as well.
  :return: A 3x3 array with the computed homography.
  """
  centroid1 = points1.mean(axis=0)
  centroid2 = points2.mean(axis=0)

  if translation_only:
    rotation = np.eye(2)
    translation = centroid2 - centroid1

  else:
    centered_points1 = points1 - centroid1
    centered_points2 = points2 - centroid2

    sigma = centered_points2.T @ centered_points1
    U, _, Vt = np.linalg.svd(sigma)

    rotation = U @ Vt
    translation = -rotation @ centroid1 + centroid2

  H = np.eye(3)
  H[:2,:2] = rotation
  H[:2, 2] = translation
  return H


def non_maximum_suppression(image):
  """
  Finds local maximas of an image.
  :param image: A 2D array representing an image.
  :return: A boolean array with the same shape as the input image, where True indicates local maximum.
  """
  # Find local maximas.
  neighborhood = generate_binary_structure(2,2)
  local_max = maximum_filter(image, footprint=neighborhood)==image
  local_max[image<(image.max()*0.1)] = False

  # Erode areas to single points.
  lbs, num = label(local_max)
  centers = center_of_mass(local_max, lbs, np.arange(num)+1)
  centers = np.stack(centers).round().astype(int)
  ret = np.zeros_like(image, dtype=np.bool)
  ret[centers[:,0], centers[:,1]] = True

  return ret


def spread_out_corners(im, m, n, radius):
  """
  Splits the image im to m by n rectangles and uses harris_corner_detector on each.
  :param im: A 2D array representing an image.
  :param m: Vertical number of rectangles.
  :param n: Horizontal number of rectangles.
  :param radius: Minimal distance of corner points from the boundary of the image.
  :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
  """
  corners = [np.empty((0,2), dtype=int)]
  x_bound = np.linspace(0, im.shape[1], n+1, dtype=int)
  y_bound = np.linspace(0, im.shape[0], m+1, dtype=int)
  for i in range(n):
    for j in range(m):
      # Use Harris detector on every sub image.
      sub_im = im[y_bound[j]:y_bound[j+1], x_bound[i]:x_bound[i+1]]
      sub_corners = harris_corner_detector(sub_im)
      sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis,:]
      corners.append(sub_corners)
  corners = np.vstack(corners)
  legit = ((corners[:,0]>radius) & (corners[:,0]<im.shape[1]-radius) &
           (corners[:,1]>radius) & (corners[:,1]<im.shape[0]-radius))
  ret = corners[legit,:]
  return ret


class PanoramicVideoGenerator:
  """
  Generates panorama from a set of images.
  """

  def __init__(self, data_dir, file_prefix, num_images, bonus=False):
    """
    The naming convention for a sequence of images is file_prefixN.jpg,
    where N is a running number 001, 002, 003...
    :param data_dir: path to input images.
    :param file_prefix: see above.
    :param num_images: number of images to produce the panoramas with.
    """
    self.bonus = bonus
    self.file_prefix = file_prefix
    self.files = [os.path.join(data_dir, '%s%03d.jpg' % (file_prefix, i + 1)) for i in range(num_images)]
    self.files = list(filter(os.path.exists, self.files))
    self.panoramas = None
    self.homographies = None
    print('found %d images' % len(self.files))

  def align_images(self, translation_only=False):
    """
    compute homographies between all images to a common coordinate system
    :param translation_only: see estimte_rigid_transform
    """
    # Extract feature point locations and descriptors.
    points_and_descriptors = []
    for file in self.files:
      image = sol4_utils.read_image(file, 1)
      self.h, self.w = image.shape
      pyramid, _ = sol4_utils.build_gaussian_pyramid(image, 3, 7)
      points_and_descriptors.append(find_features(pyramid))

    # Compute homographies between successive pairs of images.
    Hs = []
    for i in range(len(points_and_descriptors) - 1):
      points1, points2 = points_and_descriptors[i][0], points_and_descriptors[i + 1][0]
      desc1, desc2 = points_and_descriptors[i][1], points_and_descriptors[i + 1][1]

      # Find matching feature points.
      ind1, ind2 = match_features(desc1, desc2, .7)
      points1, points2 = points1[ind1, :], points2[ind2, :]

      # Compute homography using RANSAC.
      H12, inliers = ransac_homography(points1, points2, 100, 6, translation_only)

      # Uncomment for debugging: display inliers and outliers among matching points.
      # In the submitted code this function should be commented out!
      # display_matches(self.images[i], self.images[i+1], points1 , points2, inliers)

      Hs.append(H12)

    # Compute composite homographies from the central coordinate system.
    accumulated_homographies = accumulate_homographies(Hs, (len(Hs) - 1) // 2)
    self.homographies = np.stack(accumulated_homographies)
    self.frames_for_panoramas = filter_homographies_with_translation(self.homographies, minimum_right_translation=5)
    self.homographies = self.homographies[self.frames_for_panoramas]

    def generate_panoramic_images(self, number_of_panoramas):
      """
      combine slices from input images to panoramas.
      :param number_of_panoramas: how many different slices to take from each input image
      """
      if self.bonus:
        self.generate_panoramic_images_bonus(number_of_panoramas)
      else:
        self.generate_panoramic_images_normal(number_of_panoramas)

  def generate_panoramic_images_normal(self, number_of_panoramas):
    """
    combine slices from input images to panoramas.
    :param number_of_panoramas: how many different slices to take from each input image
    """
    assert self.homographies is not None

    # compute bounding boxes of all warped input images in the coordinate system of the middle image (as given by the homographies)
    self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
    for i in range(self.frames_for_panoramas.size):
      self.bounding_boxes[i] = compute_bounding_box(self.homographies[i], self.w, self.h)

    # change our reference coordinate system to the panoramas
    # all panoramas share the same coordinate system
    global_offset = np.min(self.bounding_boxes, axis=(0, 1))
    self.bounding_boxes -= global_offset

    slice_centers = np.linspace(0, self.w, number_of_panoramas + 2, endpoint=True, dtype=int)[1:-1]
    warped_slice_centers = np.zeros((number_of_panoramas, self.frames_for_panoramas.size))
    # every slice is a different panorama, it indicates the slices of the input images from which the panorama
    # will be concatenated
    for i in range(slice_centers.size):
      slice_center_2d = np.array([slice_centers[i], self.h // 2])[None, :]
      # homography warps the slice center to the coordinate system of the middle image
      warped_centers = [apply_homography(slice_center_2d, h) for h in self.homographies]
      # we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate system
      warped_slice_centers[i] = np.array(warped_centers)[:, :, 0].squeeze() - global_offset[0]

    panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(int) + 1

    # boundary between input images in the panorama
    x_strip_boundary = ((warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2)
    x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                  x_strip_boundary,
                                  np.ones((number_of_panoramas, 1)) * panorama_size[0]])
    x_strip_boundary = x_strip_boundary.round().astype(int)

    self.panoramas = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0], 3), dtype=np.float64)
    for i, frame_index in enumerate(self.frames_for_panoramas):
      # warp every input image once, and populate all panoramas
      image = sol4_utils.read_image(self.files[frame_index], 2)
      warped_image = warp_image(image, self.homographies[i])
      x_offset, y_offset = self.bounding_boxes[i][0].astype(int)
      y_bottom = y_offset + warped_image.shape[0]

      for panorama_index in range(number_of_panoramas):
        # take strip of warped image and paste to current panorama
        boundaries = x_strip_boundary[panorama_index, i:i + 2]
        image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[1] - x_offset]
        x_end = boundaries[0] + image_strip.shape[1]
        self.panoramas[panorama_index, y_offset:y_bottom, boundaries[0]:x_end] = image_strip

    # crop out areas not recorded from enough angles
    # assert will fail if there is overlap in field of view between the left most image and the right most image
    crop_left = int(self.bounding_boxes[0][1, 0])
    crop_right = int(self.bounding_boxes[-1][0, 0])
    assert crop_left < crop_right, 'for testing your code with a few images do not crop.'
    print(crop_left, crop_right)
    self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]

  def generate_panoramic_images_bonus(self, number_of_panoramas):
    """
    The bonus
    :param number_of_panoramas: how many different slices to take from each input image
    """
    pass

  def save_panoramas_to_video(self):
    assert self.panoramas is not None
    out_folder = 'tmp_folder_for_panoramic_frames/%s' % self.file_prefix
    try:
      shutil.rmtree(out_folder)
    except:
      print('could not remove folder')
      pass
    os.makedirs(out_folder)
    # save individual panorama images to 'tmp_folder_for_panoramic_frames'
    for i, panorama in enumerate(self.panoramas):
      imwrite('%s/panorama%02d.png' % (out_folder, i + 1), panorama)
    if os.path.exists('%s.mp4' % self.file_prefix):
      os.remove('%s.mp4' % self.file_prefix)
    # write output video to current folder
    os.system('ffmpeg -framerate 3 -i %s/panorama%%02d.png %s.mp4' %
              (out_folder, self.file_prefix))


  def show_panorama(self, panorama_index, figsize=(20, 20)):
    assert self.panoramas is not None
    plt.figure(figsize=figsize)
    plt.imshow(self.panoramas[panorama_index].clip(0, 1))
    plt.show()
