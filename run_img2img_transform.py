"""Run and test generated image-to-image transformation modules (or pbr to pbr trasformation, could work with any set of maps).

Generated transformation folders are expected to contain a ``generate.py`` file
with a ``transform(start_map, end_map, numsteps=...)`` function. This runner
loads that module, applies it to either RGB images or packed PBR texture maps,
and writes the resulting transformation sequence to disk for inspection.
"""

import numpy as np
import os
import cv2
import re

import json
from pathlib import Path
import random


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

def transform(import_path, start_map, end_map,numsteps=-1):
    """Import a generated transformation module and run its transform function."""
    import os, sys

    # Remove cached module so the new path is actually used
    if 'generate' in sys.modules:
        del sys.modules['generate']

    # Remove any previously inserted paths to avoid accumulation
    sys.path = [p for p in sys.path if p != os.path.abspath(import_path)]

    sys.path.insert(0, os.path.abspath(import_path))
    import generate
    if numsteps >0:
           list_trans = generate.transform(start_map, end_map,numsteps=numsteps)
    else:
        list_trans = generate.transform(start_map, end_map)
    return list_trans
###############################################################################

def load_pbr_dir(indir):
    """Load image files from a PBR directory into a name->image dictionary."""
    maps = {}
    if not os.path.isdir(indir):
         print(indir + " is not a directory.")
         exit(0)
    for fl in os.listdir(indir):
        if  os.path.splitext(fl)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif"}:
            maps[os.path.splitext(os.path.basename(fl))[0]] = cv2.imread(os.path.join(indir,fl))
    return maps
###################Normalize pbr#####################################################################################################################


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
    ky_to_del = []
    for ky in out.keys():
        if ky not in list(texture_patterns.keys()):
            ky_to_del.append(ky)
    for ky in ky_to_del:
        out.pop(ky, None)
    return out


def sync_pbrs(pbr1, pbr2, h0=None,w0=None):
    """Make two PBR dictionaries contain the same channels and image size."""
    pbr1 = remove_reductant_map(pbr1)
    pbr1 = standardize_pbr_texture_dict(pbr1)
    pbr2 = remove_reductant_map(pbr2)
    pbr2 = standardize_pbr_texture_dict(pbr2)
    all_keys = set(set(pbr1.keys()) | set(pbr2.keys()))
    if h0 is None:
            h0 = pbr1[list(pbr1.keys())[0]].shape[0]
            w0 = pbr1[list(pbr1.keys())[0]].shape[1]



    for ky in all_keys:
        try:
            if not ky in pbr1:
                 if ky in BLENDER_PRINCIPLED_DEFAULTS:
                     pbr1[ky]=np.zeros_like(pbr2[ky]) + BLENDER_PRINCIPLED_DEFAULTS[ky]
                 else:
                     pbr1[ky] = np.zeros_like(pbr2[ky])
            if not ky in pbr2:
                if ky in BLENDER_PRINCIPLED_DEFAULTS:
                     pbr2[ky] = np.zeros_like(pbr1[ky]) + BLENDER_PRINCIPLED_DEFAULTS[ky]
                else:
                    pbr2[ky] = np.zeros_like(pbr1[ky])
            if h0!=pbr1[ky].shape[0] or w0!=pbr1[ky].shape[1]:
                    pbr1[ky]=cv2.resize(pbr1[ky], (w0,h0))
            if h0!=pbr2[ky].shape[0] or w0!=pbr2[ky].shape[1]:
                    pbr2[ky]=cv2.resize(pbr2[ky], (w0,h0))
        except Exception as error:
           print("Skipping incompatible PBR channel", ky, "because:", error)
           if ky in pbr1: del pbr1[ky]
           if ky in pbr2: del pbr2[ky]

    return pbr1, pbr2
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
def run_transformation_on_pbrs(transformation_dir,pbr1,pbr2,outdir,numsteps=-1):
    """Run one generated image-to-image transform on two PBR material folders."""
    if not os.path.exists(outdir): os.makedirs(outdir)
    pbr1 = load_pbr_dir(pbr1)
    pbr1 = remove_reductant_map(pbr1)
    pbr1 = standardize_pbr_texture_dict(pbr1)

    pbr2 = load_pbr_dir(pbr2)
    pbr2 = remove_reductant_map(pbr2)
    pbr2 = standardize_pbr_texture_dict(pbr2)

    pbr1, pbr2 = sync_pbrs(pbr1, pbr2)
    stacked1, mapping = dict_to_multichannel(pbr1)
    stacked2, mapping = dict_to_multichannel(pbr2, mapping)
    list_trans = transform(transformation_dir, stacked1, stacked2,numsteps)

    for ky, pbr_map in pbr1.items():
        path = os.path.join(outdir, ky)
        if not os.path.exists(path): os.makedirs(path)
        cv2.imwrite(os.path.join(path, "start.jpg"), pbr1[ky])
        cv2.imwrite(os.path.join(path, "end.jpg"), pbr2[ky])

    for i in range(len(list_trans)):
        pbrt = multichannel_to_dict(list_trans[i], mapping)

        for ky in pbr1.keys():
            path = os.path.join(outdir, ky)
            cv2.imwrite(os.path.join(path, str(i) + ".jpg"), pbrt[ky])
####################################################################################################################################
def run_transformation_image(rgb1,rgb2,outdir,transformation_dir,numsteps=-1):
    """Run one generated image-to-image transform on two RGB image files transforming one to the other."""
    if not os.path.exists(outdir): os.makedirs(outdir)
    im1 = cv2.imread(rgb1)
    im2 = cv2.imread(rgb2)
    if im1 is None:
        raise FileNotFoundError(f"Could not read start image: {rgb1}")
    if im2 is None:
        raise FileNotFoundError(f"Could not read end image: {rgb2}")
    im2 = cv2.resize(im2, [im1.shape[1], im1.shape[0]])
    list_trans = transform(transformation_dir, im1, im2,numsteps)

    cv2.imwrite(os.path.join(outdir, "start.jpg"), im1)
    cv2.imwrite(os.path.join(outdir, "end.jpg"), im2)

    for i in range(len(list_trans)):
        cv2.imwrite(os.path.join(outdir, str(i) + ".jpg"), list_trans[i])
#############################################################################################################################################
def run_test(transformation_dir,outdir):
    """Smoke-test a generated transform on fixed sample PBR and RGB inputs."""
    outdir_pbr = os.path.join(outdir, "PBR")
    pbr1=r"pbrs/pbr1"
    pbr2=r"pbrs/pbr2"
    run_transformation_on_pbrs(transformation_dir,pbr1,pbr2,outdir_pbr)
    ####################################################################################3
    outdir_RGB = os.path.join(outdir, "RGB")
    if not os.path.exists(outdir_RGB): os.makedirs(outdir_RGB)
    rgb1 = r"images/im1.jpg"
    rgb2 = r"images/im2.jpg"
    if not os.path.exists(rgb1) or not os.path.exists(rgb2):
        print("MISSING ",rgb1,"or ",rgb2)

    run_transformation_image(rgb1,rgb2,outdir_RGB,transformation_dir)
###########################################################################################################3
def fix_trans_dir(transdir,trans_file="generate.py"):
    """Patch a common generated-code typo: ``num_steps`` vs ``numsteps``."""
    with open(transdir+"//"+trans_file,"r") as fl:
        txt = fl.read()
    if ("num_steps" in txt) and (not "numsteps" in txt):
            txt = txt.replace("num_steps","numsteps")
            with open(transdir + "//" + trans_file, "w") as fl:
                fl.write(txt)
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

def get_PBRS_list_vastexture(pbr_main_dir):
    """Return PBR directories from a flat Vastexture-style folder layout."""
    pbr_dirs = []
    for pbr_sdir in os.listdir(pbr_main_dir):
        pbr_dir = pbr_main_dir + "//" + pbr_sdir + "//"
        if os.path.isdir(pbr_dir):
                    pbr_dirs.append(pbr_dir)
    return  pbr_dirs

###########################################################################################################################
def run_all_transformation_PBRS(trans_main_dir,pbr_dirs,outmaindir,max_steps=100):
    """Run every generated image-to-image transform on random PBR pairs."""

    if not os.path.exists(outmaindir): os.makedirs(outmaindir)
    if len(pbr_dirs) < 2:
        raise ValueError("Need at least two PBR directories to run image-to-image transformations.")

    for topic_subdir in os.listdir(trans_main_dir):
        topic_dir = trans_main_dir + "//"  + topic_subdir + "//"
        if not(os.path.isdir(topic_dir)): continue
        for sdir in os.listdir(topic_dir):
            trans_dir = topic_dir + "//" + sdir +"//"
            if not os.path.isdir(trans_dir): continue
            fix_trans_dir(trans_dir)
            numsteps = -1
            trans_file = trans_dir + "//manual_setting.json"
            if os.path.exists(trans_file):
                data = json.load(open(trans_file, "r"))
                if 'recommended number steps' in data:
                    numsteps = data['recommended number steps']
            while (True):
                pbr_path1 = random.choice(pbr_dirs)
                pbr_name1 = Path(pbr_path1).resolve().name

                pbr_path2 = random.choice(pbr_dirs)
                pbr_name2 = Path(pbr_path2).resolve().name
                print(trans_dir,"\n",pbr_path1,"\n",pbr_path2)
                if pbr_path1 == pbr_path2: continue
                break

            if numsteps>max_steps: numsteps = max_steps
            outdir = outmaindir + "//" + topic_subdir + "_PBR1_" + pbr_name1+ "_PBR2_" + pbr_name2 + "_Trans_" + sdir
            print(outdir)
            if os.path.exists(outdir): continue
          #  if os.path.exists(trans_dir + "//FDFDFDFDFDFDFDFD.txt"): continue
            if numsteps>0:
                try:
                     run_transformation_on_pbrs(trans_dir, pbr_path1,pbr_path2, outdir, numsteps)
                except Exception as error:
                    print("PBR transformation failed:", trans_dir, error)
                    continue
            else:
                try:
                   print(trans_dir)

                   run_transformation_on_pbrs(trans_dir, pbr_path1,pbr_path2, outdir)

                except Exception as e:
                   print("PBR transformation failed:", trans_dir, e)
                   continue
###########################################################################################################################
def run_all_transformation_images(trans_main_dir,image_dir,outmaindir,max_steps=100):
    """Run every generated image-to-image transform on random RGB image pairs."""

    if not os.path.exists(outmaindir): os.makedirs(outmaindir)

    list_images = []
    for im in os.listdir(image_dir):
        if os.path.isfile(image_dir + "//" + im):
            list_images.append(image_dir + "//" + im)
    if len(list_images) < 2:
        raise ValueError("Need at least two images to run image-to-image transformations.")


#--------------------------------------------------------------------------------
    for topic_subdir in os.listdir(trans_main_dir):
        topic_dir = trans_main_dir + "//"  + topic_subdir + "//"
        if not(os.path.isdir(topic_dir)): continue
        for sdir in os.listdir(topic_dir):
            trans_dir = topic_dir + "//" + sdir +"//"
            if not os.path.isdir(trans_dir): continue
            fix_trans_dir(trans_dir)
            trans_file =  trans_dir + "//manual_setting.json"

            numsteps = -1
            if os.path.exists(trans_file):
                data = json.load(open(trans_file, "r"))
                if 'recommended number steps' in data:
                    numsteps = data['recommended number steps']
            while (True):
                image_path1 = random.choice(list_images)
                image_path2 = random.choice(list_images)
                if image_path1!=image_path2: break


            if numsteps>max_steps: numsteps = max_steps
            topic_out_dir = outmaindir + "//" + topic_subdir
            if not os.path.exists(topic_out_dir): os.makedirs(topic_out_dir)
            outdir = outmaindir + "//" + topic_subdir + "//" + sdir
            print(outdir)
            if os.path.exists(outdir): continue
          #  if os.path.exists(trans_dir + "//FDFDFDFDFDFDFDFD.txt"): continue
            if numsteps>0:
                try:
                    run_transformation_image(rgb1=image_path1, rgb2=image_path2, outdir=outdir, transformation_dir=trans_dir, numsteps=numsteps)

                except Exception as error:
                    print("RGB transformation failed:", trans_dir, error)
                    continue
            else:
                try:
                   print(trans_dir)

                   run_transformation_image(rgb1=image_path1, rgb2=image_path2, outdir=outdir, transformation_dir=trans_dir)

                except Exception as e:
                   print("RGB transformation failed:", trans_dir, e)
                   continue



###########################################################################################################################
if __name__  == "__main__":
    #-----------Run single image to image transformation-----------------------------------------------
    transformation_dir = "samples_img2img_tranformation/Aeolian_Landform_Sculpting__Wind_Deflation_and_Loess_Mantling"
    rgb1 = "samples/sample_images/im1.jpg"
    rgb2 = "samples/sample_images/im2.jpg"
    outdir = "output_image2image_transformation/"
    run_transformation_image(rgb1, rgb2, outdir, transformation_dir,numsteps=12)
    #--------------Run PBR to PBR transformation----------------------------------------------------
    pbr_path1 =  "samples/sample_pbrs/pbr1"
    pbr_path2 =  "samples/sample_pbrs/pbr2"
    trans_dir = "samples_img2img_tranformation/Biofilm_Quorum_Sensing_Colonization_and_Dispersal_Morph"
    outdir = "output_pbr2pbr_transformation/"
    run_transformation_on_pbrs(trans_dir, pbr_path1, pbr_path2, outdir,numsteps=12)
   #


