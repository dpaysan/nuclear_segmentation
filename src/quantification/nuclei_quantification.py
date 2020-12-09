from skimage import measure
import numpy as np
import pandas as pd


def get_basic_properties(nuclei_mask, intensity_image):
    # 3D features
    volume = np.sum(nuclei_mask)
    iso_verts, iso_faces, _, _ = measure.marching_cubes(nuclei_mask)
    surface_area = measure.mesh_surface_area(iso_verts, iso_faces)
    sa_vo = surface_area / volume

    # 2D features
    props = measure.regionprops(nuclei_mask.astype(np.uint8).max(axis=0), intensity_image.max(axis=0))[0]
    area = props.area
    eccentricity = props.eccentricity
    aspect_ratio = max(props.intensity_image.shape) / min(props.intensity_image.shape)
    convexity = area / props.convex_area
    major_axis_length = props.major_axis_length
    minor_axis_length = props.minor_axis_length

    basic_properties = {'volume': volume, 'surface_area': surface_area, 'sa/vol': sa_vo, 'area': area,
                        'eccentricity': eccentricity, 'aspect_ratio': aspect_ratio, 'convexity': convexity,
                        'minor_axis_length': minor_axis_length, 'major_axis_length': major_axis_length}

    return basic_properties


def get_hc_ec_properties(hc_mask, ec_mask):
    hc_volume = np.sum(hc_mask)
    ec_volume = np.sum(ec_mask)
    nuclei_volume = hc_volume + ec_volume
    hc_ec_ratio = hc_volume / ec_volume
    ec_ratio = ec_volume / nuclei_volume
    hc_ratio = hc_volume / nuclei_volume
    hc_ec_properties = {'ec_ratio': ec_ratio, 'hc_ratio': hc_ratio, 'hc_ec_ratio': hc_ec_ratio}
    return hc_ec_properties
