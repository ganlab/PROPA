import cv2
import os
import numpy as np
from tapnet import tapir_model
import functools
from tapnet.utils import model_utils
import haiku as hk
import jax
import tree
import mediapy as media
from tapnet.utils import transforms
from num_samples import get_num_samples
import math
from tqdm import tqdm
from paint_util import mask_painter


def uniform_sampling(mask, distance):
    """
    Perform uniform sampling within the '1' regions of a mask.

    Parameters:
    - mask: A 2D numpy array with 0s and 1s
    - distance: The approximate distance between the sampled points

    Returns:
    - points: A list of (x, y) tuples representing the sampled points
    """
    points = []
    if distance == 0:
        distance = 1
    for i in range(0, mask.shape[0], distance):
        for j in range(0, mask.shape[1], distance):
            if mask[i, j] == 1:
                points.append((i, j))
    points = np.asarray(points)
    points = points[:, [1, 0]]
    return points


def create_video(images, image_folder, video_name):
    video_path = os.path.join(image_folder, video_name)

    height, width, layers = images[0].shape

    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    for image in images:
        video.write(image)

    cv2.destroyAllWindows()
    video.release()


def inference(frames, query_points, params, state, model_apply):
    """Inference on one video.

  Args:
    frames: [num_frames, height, width, 3], [0, 255], np.uint8
    query_points: [num_points, 3], [0, num_frames/height/width], [t, y, x]

  Returns:
    tracks: [num_points, 3], [-1, 1], [t, y, x]
    visibles: [num_points, num_frames], bool
  """
    # Preprocess video to match model inputs format
    frames = model_utils.preprocess_frames(frames)
    num_frames, height, width = frames.shape[0:3]
    query_points = query_points.astype(np.float32)
    frames, query_points = frames[None], query_points[None]  # Add batch dimension

    # Model inference
    rng = jax.random.PRNGKey(42)
    outputs, _ = model_apply(params, state, rng, frames, query_points)
    outputs = tree.map_structure(lambda x: np.array(x[0]), outputs)
    tracks, occlusions, expected_dist = outputs['tracks'], outputs['occlusion'], outputs['expected_dist']

    # Binarize occlusions
    visibles = model_utils.postprocess_occlusions(occlusions, expected_dist)
    return tracks, visibles


def build_model(frames, query_points, model_type='tapir'):
    """Compute point tracks and occlusions given frames and query points."""
    if model_type == 'tapir':
        model = tapir_model.TAPIR(bilinear_interp_with_depthwise_conv=False, pyramid_level=0)
    elif model_type == 'bootstapir':
        model = tapir_model.TAPIR(
            bilinear_interp_with_depthwise_conv=False,
            pyramid_level=1,
            extra_convs=True,
            softmax_temperature=10.0,
        )
    outputs = model(
        video=frames,
        is_training=False,
        query_points=query_points,
        query_chunk_size=64,
    )
    return outputs


def convert_mask_to_img(image, mask, label):
    mask_color = label + 2
    mask_alpha = 0.7
    contour_color = 2
    contour_width = 3

    painted_image = mask_painter(image, mask.astype('uint8'), mask_color, mask_alpha, contour_color, contour_width)

    return painted_image


def get_points_and_labels(mask, factor):
    max_label = np.amax(mask)

    all_points = None
    all_labels = None

    for i in range(1, max_label + 1):
        mask_array = (mask == i)
        if np.amax(mask_array) == 0:
            continue

        if i == 0:
            distance = 30
        else:
            _, points_count = np.unique(mask_array, return_counts=True)
            distance = round(math.sqrt(points_count[1]) / factor)

        mask_points = uniform_sampling(mask_array, distance)
        mask_labels = np.full(mask_points.shape[0], i)

        if all_points is None and all_labels is None:
            all_points = mask_points
            all_labels = mask_labels
        else:
            all_points = np.vstack((all_points, mask_points))
            all_labels = np.hstack((all_labels, mask_labels))
    if all_points is None:
        all_points = np.array([[1, 1]])
        all_labels = np.array([0])
    return all_points, all_labels


def convert_query_points(all_points):
    query_points = np.zeros((all_points.shape[0], 3))
    query_points[:, 1:3] = all_points
    query_points = query_points[:, [0, 2, 1]]
    return query_points


def save_image(mask, result_path, i):
    save_path = os.path.join(result_path, f'frame{i}.jpg')

    save_arr = np.dstack((mask * 20 % 255, mask * 105 % 255, mask * 208 % 255))
    cv2.imwrite(save_path, save_arr)


def track_from_mask(mask: np.ndarray, video_path, factor, predictor):
    all_points, all_labels = get_points_and_labels(mask, factor)

    query_points = convert_query_points(all_points)

    checkpoint_path = 'tapnet/checkpoints/bootstapir_checkpoint_v2.npy'

    ckpt_state = np.load(checkpoint_path, allow_pickle=True).item()
    params, state = ckpt_state['params'], ckpt_state['state']

    build_model_fn = functools.partial(build_model, model_type='bootstapir')
    model = hk.transform_with_state(build_model_fn)
    model_apply = jax.jit(model.apply)

    video = media.read_video(video_path)
    height, width = video.shape[1:3]

    resize_height = 256
    resize_width = 256
    frames = media.resize_video(video, (resize_height, resize_width))
    query_points = transforms.convert_grid_coordinates(
        query_points, (1, height, width), (1, resize_height, resize_width), coordinate_format='tyx')

    tracks, visibles = inference(frames, query_points, params, state, model_apply)
    tracks = transforms.convert_grid_coordinates(tracks, (resize_width, resize_height), (width, height))

    video_name = os.path.basename(video_path)
    video_name = os.path.splitext(video_name)[0]

    frames = []

    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break

    # 路径保存结果
    if not os.path.exists('tracking'):
        os.mkdir('tracking')
    result_path = os.path.join('tracking', video_name)
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    result_images = []
    for i, image in enumerate(tqdm(frames)):
        points = tracks[:, i, :]
        visibilities = np.asarray(visibles[:, i])

        points = points[visibilities]
        labels = all_labels[visibilities]

        if points.shape[0] == 0:
            continue

        max_label = np.amax(labels)
        painted_image = image.copy()
        evaluate_image = np.zeros(image.shape[:2])
        for label in range(1, max_label + 1):
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
            cur_index = labels == label

            positive_points = points[cur_index]
            positive_labels = np.ones((positive_points.shape[0]))
            negative_points = points[~cur_index]
            negative_labels = np.zeros((negative_points.shape[0]))

            cur_points = np.vstack((positive_points, negative_points))
            cur_labels = np.hstack((positive_labels, negative_labels))

            if positive_points.shape[0] > 0:
                new_point = None
                new_label = None
                while True:
                    try:
                        new_point, new_label, is_full = get_num_samples(image, cur_points.astype(np.int16), cur_labels,
                                                                        new_point,
                                                                        new_label, mask)
                    except:
                        break

                    if is_full:
                        break
                    else:
                        predictor.set_image(image)

                        masks, _, _ = predictor.predict(
                            point_coords=new_point,
                            point_labels=new_label,
                            multimask_output=False
                        )

                    mask = masks[0]

            painted_image = convert_mask_to_img(painted_image, mask, label)
            evaluate_image[mask] = label

        result_images.append(painted_image)
        save_image(evaluate_image, result_path, i)
    create_video(result_images, result_path, f'{video_name}.mp4')

