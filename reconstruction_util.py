from pathlib import Path
from HiLo.hloc import extract_features, match_features, reconstruction, pairs_from_retrieval
from HiLo.hloc.utils import viz_3d
import os
import calculate_zoom as calz
import numpy as np
from PIL import Image

def struct_from_motion(pathstring, num_matched):
    work_dir = os.path.dirname(os.path.abspath(__file__))
    pathstring = os.path.join(work_dir, pathstring)

    pathstring += "/"
    imagesstring = pathstring + 'Images'
    images = Path(imagesstring)
    pathoutputsstring = pathstring + 'output/'
    outputs = Path(pathoutputsstring)
    sfm_pairs = outputs / 'pairs-netvlad.txt'
    pathsfm_dirstring = pathoutputsstring + 'sparse'
    sfm_dir = outputs / 'sparse'

    txt_path = pathstring + "output/sparse/"  # for calculate_zoom.py

    retrieval_conf = extract_features.confs['netvlad']
    feature_conf = extract_features.confs['superpoint_max']
    matcher_conf = match_features.confs['superglue']

    retrieval_path = extract_features.main(retrieval_conf, images, outputs)
    pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=num_matched)  # 5 xujx update 40

    feature_path = extract_features.main(feature_conf, images, outputs)

    match_path = match_features.main(matcher_conf, sfm_pairs, feature_conf['output'], outputs)

    reconstruction.main(sfm_dir, images, sfm_pairs, feature_path, match_path)

    converter = "colmap model_converter" + " --input_path " + pathsfm_dirstring + " --output_path " + pathsfm_dirstring + " --output_type TXT"
    os.system(converter)

    viz_3d.init_figure()

    calz.read_cameras_txt(txt_path + "cameras.txt")

    # openMVS
    converter = "cd /usr/local/bin/OpenMVS"
    converter = converter + "\n"

    os.system(converter)
def multi_view_stero(pathstring):
    work_dir = os.path.dirname(os.path.abspath(__file__))
    pathstring = os.path.join(work_dir, pathstring)

    pathstring += "/"
    imagesstring = pathstring + 'target'
    pathoutputsstring = pathstring + 'output/'


    # openMVS
    converter = "cd /usr/local/bin/OpenMVS"
    converter = converter + "\n"

    converter = converter + "./InterfaceCOLMAP -i "
    converter = converter + pathoutputsstring
    converter = converter + " -o "
    converter = converter + pathoutputsstring + "scene.mvs"
    converter = converter + " --image-folder "
    converter = converter + imagesstring
    converter = converter + "\n"

    converter = converter + "./DensifyPointCloud -w "
    converter = converter + pathoutputsstring
    converter = converter + " -i scene.mvs -o dense_scene.mvs --remove-dmaps 1"
    converter = converter + "\n"

    os.system(converter)


def extract_from_numpy(work_dir, target_label):
    mask_dir = os.path.join(work_dir, 'masks')
    image_dir = os.path.join(work_dir, 'Images')
    output_dir = os.path.join(work_dir, 'target')

    img_format = os.listdir(image_dir)[0].split(".")[-1]

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    masks = os.listdir(mask_dir)

    for mask_filename in masks:
        mask_path = os.path.join(mask_dir, mask_filename)
        image_path = os.path.join(image_dir, mask_filename.replace('npy', img_format))
        mask = np.load(mask_path)
        image = Image.open(image_path).convert('RGBA')  # 转换为RGBA模式
        result_image = Image.new('RGBA', (mask.shape[1], mask.shape[0]), (0, 0, 0, 0))
        result_pixels = result_image.load()
        image_pixels = image.load()

        for x in range(mask.shape[1]):
            for y in range(mask.shape[0]):
                if mask[y][x] == target_label:
                    result_pixels[x, y] = image_pixels[x, y]

        output_path = os.path.join(output_dir, mask_filename.replace('npy', img_format))
        result_image.save(output_path, "PNG")
