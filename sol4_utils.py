import cv2

def read_image(image_path, representation):
    """
    Reads an image from a file according to the correct
    1 - Grayscale
    2 - RGB

    Args:
        image_path (str): The path to the image file.
        representation (int): The representation to convert the image to.
    Returns:
        image (numpy.ndarray): The image in the specified representation.
    """
    image = cv2.imread(image_path)
    if representation == 1:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # elif representation == 2:
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    Builds a Gaussian pyramid from an image.
    
    Args:
        im (numpy.ndarray): Input image.
        max_levels (int): The maximal number of levels in the resulting pyramid.
        filter_size (int): Size of the Gaussian filter (odd number).
        
    Returns:
        tuple: A tuple containing:
            1) A list of images, where the first element is the original image
               and each subsequent element is a blurred and downsampled version
               of the previous image.
            2) A list of filter sizes used for each level.
    """
    # Initialize pyramid with original image
    pyramid = [im]
    filter_vec = [filter_size]
    
    # Create pyramid levels
    for i in range(max_levels - 1):
        # Blur the current image
        blurred = cv2.GaussianBlur(pyramid[-1], (filter_size, filter_size), 0)
        
        # Downsample by factor of 2
        downsampled = blurred[::2, ::2]
        
        # Add to pyramid
        pyramid.append(downsampled)
        filter_vec.append(filter_size)
    
    return pyramid, filter_vec

