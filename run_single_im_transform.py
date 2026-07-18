"""Run and test generated single-image or pbr transformation modules for all generated transformations in folder.

Generated transformation folders are expected to contain a ``generate.py`` file
with a ``transform(input_map, numsteps=...)`` function. This runner loads that
module, applies it to either RGB images or packed PBR texture maps, and writes
the resulting transformation sequence to disk for inspection.
"""

import random

import numpy as np
import os
import cv2
import re
import json
from pathlib import Path


def transform(import_path, input_map,numsteps=-1):
    """Import a generated single-image module and run its transform function on image or map."""
    import os, sys

    # Remove cached module so the new path is actually used
    if 'generate' in sys.modules:
        del sys.modules['generate']

    # Remove any previously inserted paths to avoid accumulation
    sys.path = [p for p in sys.path if p != os.path.abspath(import_path)] # set folder path for import

    sys.path.insert(0, os.path.abspath(import_path))
    import generate
    if numsteps>0:
       list_trans = generate.transform(input_map,numsteps=numsteps)
    else:
        list_trans = generate.transform(input_map)
    return list_trans


# Common aliases used to identify PBR texture maps before standardizing names.
NORMAL_MAP_NAMES = [
    "normal",
    "normalmap",
    "normal_map",
    "normal-map",
    "norm",
    "normals",
    "Normal",
    "NormalMap",
    "Normal_Map",
    "NORMAL",
    "NORM"]

HEIGHT_MAP_NAMES = [
    "height",
    "heightmap",
    "height_map",
    "height-map",
    "displ",
    "displacement",
    "displacementmap",
    "displacement_map",
    "dispmap",
    "depth",
    "depthmap",
    "depth_map",
    "elevation",
    "bump",
    "bumpmap",
    "bump_map",
    "Height",
    "HeightMap",
    "HEIGHT",
    "Displacement",
    "Bump",
    "Depth"]

AO_MAP_NAMES = [
    "ao",
    "ambientocclusion",
    "ambient_occlusion",
    "ambient-occlusion",
    "occlusion",
    "occlusionmap",
    "occlusion_map",
    "ao_map",
    "aomap",
    "occ",
    "AO",
    "AmbientOcclusion",
    "Ambient_Occlusion",
    "Occlusion",
    "OCC",
    # Common suffixes
    "_ao",
    "_aomap",
    "_ao_map",
    "_ambientocclusion",
    "_ambient_occlusion",
    "_occlusion",
    "_occlusionmap",
    "_occlusion_map"]


BLENDER_PRINCIPLED_DEFAULTS = {
    "BaseColor":        (204, 204, 204),   # 0.8,0.8,0.8
    "Roughness":        128,               # 0.5
    "Metallic":         0,                 # 0.0
    "Transmission":     0,                 # 0.0
    "Emission":         (0, 0, 0),         # black
    "Normal":           (128, 128, 255),   # flat normal (implicit)
    "AmbientOcclusion": 255,               # no AO (not a Principled input)
    "Specular":         128,               # 0.5 (older Blender) / IOR Level 0.5 in newer Blender
 }
def remove_reductant_map(dc):
    """Remove maps that can be regenerated from a height/displacement map."""
    height=False
    for ky in dc:
        for nm in HEIGHT_MAP_NAMES:
            if nm.lower() in ky.lower():
                height = True
    #-------------------Remove reductran map-----------------------------
    new_maps={}
    for ky in dc:
        add=True
        if height:
            for nm in AO_MAP_NAMES+NORMAL_MAP_NAMES:
                if nm.lower() in ky.lower():
                    add = False
        if add:
            new_maps[ky]=dc[ky]
    return new_maps

def load_pbr_dir(indir):
    """Load image files from a PBR directory into a name->image dictionary."""
    maps = {}
    for fl in os.listdir(indir):
        if  os.path.splitext(fl)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif"}:
            maps[os.path.splitext(os.path.basename(fl))[0]] = cv2.imread(os.path.join(indir,fl))
    return maps


###################Standarized PBR dir maps names#####################################################################################################################
def standardize_pbr_texture_dict(texture_dict):
    """
    Rename keys (paths/filenames) in a texture dictionary to standard PBR names.

    Args:
        texture_dict (dict):
            {
                "/path/to/wood_albedo.jpg": image,
                "/path/to/wood_roughness.exr": image,
                ...
            }

    Returns:
        dict:
            {
                "BaseColor": image,
                "Roughness": image,
                ...
            }

    Removes ``Normal`` if ``Height`` exists, because a normal map can be
    regenerated from height in downstream tooling.
    """

    texture_patterns = {
        "BaseColor": [
            "basecolor", "base_color", "albedo", "diffuse",
            "color", "colour", "col", "diff"
        ],

        "Roughness": [
            "roughness", "rough", "rgh"
        ],

        "Metallic": [
            "metallic", "metalness", "metal", "mtl"
        ],

        "Height": [
            "height", "displacement", "disp", "depth", "bump"
        ],

        "Transmission": [
            "transmission", "transparency", "opacity", "alpha"
        ],

        "Emission": [
            "emission", "emissive", "emit", "glow"
        ],

        "Normal": [
            "normal", "normalmap", "norm", "nrm", "nor"
        ],

        "AmbientOcclusion": [
            "ambientocclusion", "ambient_occlusion",
            "ao", "occlusion"
        ],

        "Specular": [
            "specular", "spec", "reflection", "refl"
        ]
    }

    out = {}
    has_height = False

    for filepath, image in texture_dict.items():
        filename = os.path.basename(filepath).lower()
        matched = False

        for canonical_name, aliases in texture_patterns.items():
            for alias in aliases:
                if re.search(
                    rf'(^|[_\-\s]){re.escape(alias)}([_\-\s.]|$)',
                    filename
                ):
                    out[canonical_name] = image
                    matched = True

                    if canonical_name == "Height":
                        has_height = True

                    break

            if matched:
                break

        # Keep unknown maps unchanged
        if not matched:
            out[os.path.basename(filepath)] = image

    # Remove normal map if height exists
    if has_height:
        out.pop("Normal", None)

    # Keep only known PBR channels after canonicalization.
    ky_to_del =[]
    for ky in out.keys():
        if ky not in list(texture_patterns.keys()):
           ky_to_del.append(ky)
    for ky in ky_to_del:
              out.pop(ky, None)
    return out

##################################################################################################
def dict_to_multichannel(images: dict, channel_mapping: dict = None) -> tuple[np.ndarray, dict]:
    """
    Stack a dictionary of images into a single multi-channel image.

    Args:
        images:          {name: image} where image is HxW or HxWxC (1-3 channels)
        channel_mapping: optional {name: [channel_indices]} from a previous call.
                         If provided, keys are placed at the specified channels.

    Returns:
        stacked:         HxW x total_channels image
        mapping:         {name: [channel_indices]} e.g. {"BaseColor": [0,1,2], "Roughness": [3]}
    """
    if channel_mapping is not None:
        # --- Use the provided mapping (e.g. to match a previously saved layout) ---

        # Make sure every key in the mapping actually exists in the images dict
        for key in channel_mapping:
            if key not in images:
                raise KeyError(f"channel_mapping key '{key}' not found in images dict")

        # Total channels = highest channel index across all maps + 1
        total_channels = max(idx for indices in channel_mapping.values() for idx in indices) + 1

        h, w = next(iter(images.values())).shape[:2]
        stacked = np.zeros((h, w, total_channels), dtype=np.uint8)

        for name, indices in channel_mapping.items():
            img = images[name]
            img_channels = 1 if img.ndim == 2 else img.shape[2]

            # Sanity check: the image's channel count must match what the mapping expects
            if img_channels != len(indices):
                raise ValueError(
                    f"'{name}' has {img_channels} channels but mapping expects {len(indices)}"
                )

            if img.ndim == 2:
                # Grayscale: write the single plane directly
                stacked[:, :, indices[0]] = img
            else:
                # Multi-channel: write each plane to its designated slot
                for i, ch_idx in enumerate(indices):
                    stacked[:, :, ch_idx] = img[:, :, i]

        return stacked, channel_mapping

    # --- Build the mapping automatically (sequential, no gaps) ---
    mapping = {}
    cursor = 0  # Tracks the next free channel slot
    for name, img in images.items():
        n_channels = 1 if img.ndim == 2 else img.shape[2]
        # Assign the next N consecutive slots to this map
        mapping[name] = list(range(cursor, cursor + n_channels))
        cursor += n_channels

    # Allocate the output array now that we know the total channel count
    h, w = next(iter(images.values())).shape[:2]
    stacked = np.zeros((h, w, cursor), dtype=np.uint8)

    # Write each image into its assigned channels
    for name, indices in mapping.items():
        img = images[name]
        if img.ndim == 2:
            stacked[:, :, indices[0]] = img
        else:
            for i, ch_idx in enumerate(indices):
                stacked[:, :, ch_idx] = img[:, :, i]

    return stacked, mapping

#################################################################################################
def multichannel_to_dict(stacked: np.ndarray, channel_mapping: dict) -> dict:
    """
    Split a multi-channel image back into a dictionary of images.

    Args:
        stacked:         HxW x total_channels image
        channel_mapping: {name: [channel_indices]} as returned by dict_to_multichannel

    Returns:
        images: {name: image} where image is HxW (grayscale) or HxWxC (multi-channel)
    """
    images = {}
    for name, indices in channel_mapping.items():
        if len(indices) == 1:
            # Single-channel map: return as HxW (no extra dimension)
            images[name] = stacked[:, :, indices[0]]
        else:
            # Multi-channel map: stack the planes back into HxWxC
            images[name] = np.stack([stacked[:, :, i] for i in indices], axis=2)
    return images
###########################################################################################################################
def run_transformation_on_pbrs(transformation_dir,pbr1,outdir,numsteps=-1):
    """Run one generated single-image transform on a PBR material folder."""
    if not os.path.exists(outdir): os.makedirs(outdir)
    pbr1 = load_pbr_dir(pbr1)
    pbr1 = remove_reductant_map(pbr1)
    pbr1 = standardize_pbr_texture_dict(pbr1)



    stacked1, mapping = dict_to_multichannel(pbr1)
    list_trans = transform(transformation_dir, stacked1,numsteps)

    for ky, pbr_map in pbr1.items():
        path = os.path.join(outdir, ky)
        if not os.path.exists(path): os.makedirs(path)
        cv2.imwrite(os.path.join(path, "start.jpg"), pbr1[ky])


    for i in range(len(list_trans)):
        pbrt = multichannel_to_dict(list_trans[i], mapping)

        for ky in pbr1.keys():
            path = os.path.join(outdir, ky)
            cv2.imwrite(os.path.join(path, str(i) + ".jpg"), pbrt[ky])
####################################################################################################################################
def run_transformation_image(rgb1,outdir,transformation_dir,num_steps=-1):
    """Run one generated single-image transform on an RGB image file."""
    if not os.path.exists(outdir): os.makedirs(outdir)
    im1 = cv2.imread(rgb1)
    if im1 is None:
        raise FileNotFoundError(f"Could not read image: {rgb1}")
    list_trans = transform(transformation_dir, im1,num_steps)
    cv2.imwrite(os.path.join(outdir, "start.jpg"), im1)
    for i in range(len(list_trans)):
        cv2.imwrite(os.path.join(outdir, str(i) + ".jpg"), list_trans[i])
#############################################################################################################################################
def run_test(transformation_dir,outdir):
    """Smoke-test a generated transform on fixed sample PBR and RGB inputs."""
    outdir_pbr = os.path.join(outdir, "PBR")
    pbr1=r"pbrs/pbr1"
    run_transformation_on_pbrs(transformation_dir,pbr1,outdir_pbr)
    ####################################################################################3
    outdir_RGB = os.path.join(outdir, "RGB")
    if not os.path.exists(outdir_RGB): os.makedirs(outdir_RGB)
    rgb1 = r"images/im1.jpg"
    run_transformation_image(rgb1,outdir_RGB,transformation_dir)
##################################################################################################################################
###########################################################################################################3
def fix_trans_dir(transdir,trans_file="generate.py"):
    """Patch a common generated-code typo: ``num_steps`` vs ``numsteps``."""
    with open(transdir+"//"+trans_file,"r") as fl:
        txt = fl.read()
    if ("num_steps" in txt) and (not "numsteps" in txt):
            txt = txt.replace("num_steps","numsteps")
            with open(transdir + "//" + trans_file, "w") as fl:
                fl.write(txt)
####################################################################################333
def get_PBRS_list_emergent_texture(pbr_main_dir):
    """Return PBR directories from the Emergent Texture nested folder layout."""
    pbr_dirs = []
    for topic_subdir in os.listdir(pbr_main_dir):
        topic_dir = pbr_main_dir + "//" + topic_subdir + "//"
        if not os.path.isdir(topic_dir): continue
        for sdir in os.listdir(topic_dir):
            all_pbr_dir = topic_dir + "//" + sdir + "//"
            for spbrdir in os.listdir(all_pbr_dir):
                pbr_dir = all_pbr_dir + "//" + spbrdir
                if os.path.isdir(pbr_dir):
                    pbr_dirs.append(pbr_dir)
    return  pbr_dirs
#################################################################################################
def get_PBRS_list_vastexture(pbr_main_dir):
    """Return PBR directories from a flat Vastexture-style folder layout."""
    pbr_dirs = []
    for pbr_sdir in os.listdir(pbr_main_dir):
        pbr_dir = pbr_main_dir + "//" + pbr_sdir + "//"
        if os.path.isdir(pbr_dir):
                    pbr_dirs.append(pbr_dir)
    return  pbr_dirs
###########################################################################################################################
def run_all_transformation_PBR(trans_main_dir,pbr_dirs,outmaindir,max_steps=100):
    """Run every generated single-image transform on random PBR materials."""
    if not os.path.exists(outmaindir): os.makedirs(outmaindir)
    if not pbr_dirs:
        raise ValueError("Need at least one PBR directory to run transformations.")

    for topic_subdir in os.listdir(trans_main_dir):
        topic_dir = trans_main_dir + "//"  + topic_subdir + "//"
        if not(os.path.isdir(topic_dir)): continue
        for sdir in os.listdir(topic_dir):
            trans_dir = topic_dir + "//" + sdir +"//"
            if not os.path.isdir(trans_dir): continue
            fix_trans_dir(trans_dir)
            numsteps = -1
            trans_file = trans_dir + "//manual_setting.json"
            if os.path.isfile(trans_file):
                data = json.load(open(trans_file, "r"))
                if 'recommended number steps' in data:
                    numsteps = data['recommended number steps']
            pbr_path = random.choice(pbr_dirs)
            pbr_name = Path(pbr_path).resolve().name
            if numsteps>max_steps: numsteps = max_steps
            outdir = outmaindir + "//" + topic_subdir + "_PBR_" + pbr_name + "_Trans_" + sdir
            print(outdir)
           #### if os.path.exists(trans_dir + "//FDFDFDFDFDFDFDFD.txt"): continue

            if numsteps>0:
                try:
                     run_transformation_on_pbrs(trans_dir, pbr_path, outdir, numsteps)

                except Exception as error:
                    print("PBR transformation failed:", trans_dir, error)
                    continue
            else:
                try:
                   print(trans_dir)

                   run_transformation_on_pbrs(trans_dir, pbr_path, outdir)
                   print(outdir)

                except Exception as e:
                   print("PBR transformation failed:", trans_dir, e)
                   continue

#
# path = "/home/user/project/data/images/"
# last_subdir = Path(path).resolve().name
###########################################################################################################################
def run_all_transformation_images(trans_main_dir,image_dir,outmaindir,max_steps=100):
    """Run every generated single-image transform on random RGB images."""
    if not os.path.exists(outmaindir): os.makedirs(outmaindir)

    list_images = []
    for im in os.listdir(image_dir):
        if os.path.isfile(image_dir + "//" + im):
            list_images.append(image_dir + "//" + im)
    if not list_images:
        raise ValueError("Need at least one image to run transformations.")

    for topic_subdir in os.listdir(trans_main_dir):
        topic_dir = trans_main_dir + "//"  + topic_subdir + "//"
        if not(os.path.isdir(topic_dir)): continue
        for sdir in os.listdir(topic_dir):
            trans_dir = topic_dir + "//" + sdir +"//"
            if not os.path.isdir(trans_dir): continue
            fix_trans_dir(trans_dir)
            numsteps = -1
            trans_file = trans_dir + "//manual_setting.json"
            if os.path.isfile(trans_file):
                data = json.load(open(trans_file, "r"))
                if 'recommended number steps' in data:
                    numsteps = data['recommended number steps']
            img_path = random.choice(list_images)

            if numsteps>max_steps: numsteps = max_steps
            topic_out_dir = outmaindir + "//" + topic_subdir
            if not os.path.exists(topic_out_dir): os.makedirs(topic_out_dir)
            outdir = outmaindir + "//" + topic_subdir + "//" + sdir
            print(outdir)


            if numsteps>0:
                try:
                   run_transformation_image(rgb1=img_path, outdir=outdir, transformation_dir=trans_dir, num_steps=numsteps)

                except Exception as error:
                    print("RGB transformation failed:", trans_dir, error)
                    continue
            else:
                try:
                   print(trans_dir)

                   run_transformation_image(rgb1=img_path, outdir=outdir, transformation_dir=trans_dir)
                   print(outdir)

                except Exception as e:
                   print("RGB transformation failed:", trans_dir, e)
                   continue



###########################################################################################################################
if __name__  == "__main__":
    # -----------Run single image transformation-----------------------------------------------
    transformation_dir = "samples_single_image_transformation/Buoyancy_Driven_Convection__Rayleigh_B_nard_Instability_"
    rgb1 = "samples/sample_images/im1.jpg"
    outdir = "output_image_transformation/"
    run_transformation_image(rgb1, outdir, transformation_dir,num_steps=20)
    # --------------Run PBR  transformation----------------------------------------------------
    pbr_path1 = "samples/sample_pbrs/pbr1"
    trans_dir = "samples_single_image_transformation/Buoyancy_Driven_Convection__Rayleigh_B_nard_Instability_"
    outdir = "output_pbr_transformation/"
    run_transformation_on_pbrs(trans_dir, pbr_path1, outdir, numsteps=20)


