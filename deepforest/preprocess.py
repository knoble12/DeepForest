# Deepforest Preprocessing model
"""The preprocessing module is used to reshape data into format suitable for
training or prediction.

For example cutting large tiles into smaller images.
"""
import os
import numpy as np
import pandas as pd
import slidingwindow
from PIL import Image
import torch
import warnings
import rasterio
import geopandas as gpd
from deepforest.utilities import read_file
from shapely import geometry

def preprocess_image(image):
    """Preprocess a single RGB numpy array as a prediction from channels last, to channels first"""
    image = torch.tensor(image).permute(2, 0, 1)
    image = image / 255

    return image


def image_name_from_path(image_path):
    """Convert path to image name for use in indexing."""
    image_name = os.path.basename(image_path)
    image_name = os.path.splitext(image_name)[0]

    return image_name


def compute_windows(numpy_image, patch_size, patch_overlap):
    """Create a sliding window object from a raster tile.

    Args:
        numpy_image (array): Raster object as numpy array to cut into crops

    Returns:
        windows (list): a sliding windows object
    """

    if patch_overlap > 1:
        raise ValueError("Patch overlap {} must be between 0 - 1".format(patch_overlap))

    # Generate overlapping sliding windows
    windows = slidingwindow.generate(numpy_image,
                                     slidingwindow.DimOrder.HeightWidthChannel,
                                     patch_size, patch_overlap)

    return (windows)

def select_annotations(annotations, window):
    """Select annotations that overlap with selected image crop.

    Args:
        annotations: a geopandas dataframe of annotations with a geometry column
        windows: A sliding window object (see compute_windows)
    Returns:
        selected_annotations: a pandas dataframe of annotations
    """
    window_xmin, window_ymin, w, h = window.getRect()
    
    # Create a shapely box from the window
    window_box = geometry.box(window_xmin, window_ymin, window_xmin + w, window_ymin + h)
    selected_annotations = annotations[annotations.intersects(window_box)]
    selected_annotations.geometry = selected_annotations.geometry.translate(xoff=-window_xmin, yoff=-window_ymin)

    # cut off any annotations over the border
    original_area = selected_annotations.geometry.area
    clipped_annotations = gpd.clip(selected_annotations, window_box)

    if clipped_annotations.empty:
        return clipped_annotations
    
    # For points, keep all annotations.
    if selected_annotations.iloc[0].geometry.type == "Point":
        return selected_annotations
    else:
        # Only keep clipped boxes if they are more than 50% of the original size.
        clipped_area = clipped_annotations.geometry.area
        clipped_annotations = clipped_annotations[(clipped_area/original_area) > 0.5]
        
    return clipped_annotations

def save_crop(base_dir, image_name, index, crop):
    """Save window crop as image file to be read by PIL.

    Filename should match the image_name + window index
    """
    # create dir if needed
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    im = Image.fromarray(crop)
    image_basename = os.path.splitext(image_name)[0]
    filename = "{}/{}_{}.png".format(base_dir, image_basename, index)
    im.save(filename)

    return filename


def split_raster(annotations_file,
                 path_to_raster=None,
                 numpy_image=None,
                 base_dir=None,
                 patch_size=400,
                 patch_overlap=0.05,
                 allow_empty=False,
                 image_name=None,
                 save_dir="."):
    """Divide a large tile into smaller arrays. Each crop will be saved to file.

    Args:
        numpy_image: a numpy object to be used as a raster, usually opened from rasterio.open.read(), in order (height, width, channels)
        path_to_raster: (str): Path to a tile that can be read by rasterio on disk
        annotations_file (str or pd.DataFrame): A pandas dataframe or path to annotations csv file. In the format -> image_path, xmin, ymin, xmax, ymax, label
        save_dir (str): Directory to save images
        base_dir (str): Directory to save images
        patch_size (int): Maximum dimensions of square window
        patch_overlap (float): Percent of overlap among windows 0->1
        allow_empty: If True, include images with no annotations
            to be included in the dataset
        image_name (str): If numpy_image arg is used, what name to give the raster?

    Returns:
        A pandas dataframe with annotations file for training. 
        A copy of this file is written to save_dir as a side effect.
    """
    # Set deprecation warning for base_dir and set to save_dir
    if base_dir:
        warnings.warn(
            "base_dir argument will be deprecated in 2.0. The naming is confusing, the rest of the API uses 'save_dir' to refer to location of images. Please use 'save_dir' argument.",
            DeprecationWarning)
        save_dir = base_dir

    # Load raster as image
    if (numpy_image is None) & (path_to_raster is None):
        raise IOError("supply a raster either as a path_to_raster or if ready "
                      "from existing in memory numpy object, as numpy_image=")

    if path_to_raster:
        numpy_image = rasterio.open(path_to_raster).read()
        numpy_image = np.moveaxis(numpy_image, 0, 2)
    else:
        if image_name is None:
            raise (IOError("If passing an numpy_image, please also specify a image_name"
                           " to match the column in the annotation.csv file"))

    # Confirm that raster is H x W x C, if not, convert, assuming image is wider/taller than channels
    if numpy_image.shape[0] < numpy_image.shape[-1]:
        warnings.warn(
            "Input rasterio had shape {}, assuming channels first. Converting to channels last"
            .format(numpy_image.shape), UserWarning)
        numpy_image = np.moveaxis(numpy_image, 0, 2)

    # Check that its 3 band
    bands = numpy_image.shape[2]
    if not bands == 3:
        warnings.warn(
            "Input rasterio had non-3 band shape of {}, ignoring "
            "alpha channel".format(numpy_image.shape), UserWarning)
        try:
            numpy_image = numpy_image[:, :, :3].astype("uint8")
        except:
            raise IOError("Input file {} has {} bands. "
                          "DeepForest only accepts 3 band RGB rasters in the order "
                          "(height, width, channels). "
                          "Selecting the first three bands failed, "
                          "please reshape manually.If the image was cropped and "
                          "saved as a .jpg, please ensure that no alpha channel "
                          "was used.".format(path_to_raster, bands))

    # Check that patch size is greater than image size
    height = numpy_image.shape[0]
    width = numpy_image.shape[1]
    if any(np.array([height, width]) < patch_size):
        raise ValueError("Patch size of {} is larger than the image dimensions {}".format(
            patch_size, [height, width]))

    # Compute sliding window index
    windows = compute_windows(numpy_image, patch_size, patch_overlap)

    # Get image name for indexing
    if image_name is None:
        image_name = os.path.basename(path_to_raster)

    # Load annotations file and coerce dtype
    if type(annotations_file) == str:
        annotations = read_file(annotations_file)
    elif type(annotations_file) == gpd.GeoDataFrame:
        annotations = annotations_file
    else:
        raise TypeError(
            "annotations file must either by a path or a gpd.Dataframe, found {}".format(
                type(annotations_file)))

    # open annotations file
    image_annotations = annotations[annotations.image_path == image_name]
    image_basename = os.path.splitext(image_name)[0]

    # Sanity checks
    if image_annotations.empty:
        raise ValueError(
            "No image names match between the file:{} and the image_path: {}. "
            "Reminder that image paths should be the relative "
            "path (e.g. 'image_name.tif'), not the full path "
            "(e.g. path/to/dir/image_name.tif)".format(annotations_file, image_name))

    annotations_files = []
    for index, window in enumerate(windows):
        # Crop image
        crop = numpy_image[windows[index].indices()]

        # skip if empty crop
        if crop.size == 0:
            continue

        # Find annotations, image_name is the basename of the path
        crop_annotations = select_annotations(image_annotations, window = windows[index])
        
        if crop_annotations.empty:
            if allow_empty:
                crop_annotations.loc[0, "image_path"] = "{}_{}.png".format(image_basename, index)
            else:
                raise ValueError(
                    "Input file has no overlapping annotations and allow_empty is {}".format(
                        allow_empty))
        else:
            crop_annotations["image_path"] = "{}_{}.png".format(image_basename, index)

        annotations_files.append(crop_annotations)
        save_crop(save_dir, image_name, index, crop)
    
    if len(annotations_files) == 0:
        raise ValueError(
            "Input file has no overlapping annotations and allow_empty is {}".format(
                allow_empty))

    annotations_files = pd.concat(annotations_files)

    # Checkpoint csv files, useful for parallelization
    # Use filename of the raster path to save the annotations
    image_basename = os.path.splitext(image_name)[0]
    file_path = image_basename + ".csv"
    file_path = os.path.join(save_dir, file_path)
    annotations_files.to_csv(file_path, index=False, header=True)

    return annotations_files
