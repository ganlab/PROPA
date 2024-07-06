import numpy as np
from scipy import spatial
from scipy.spatial import ConvexHull
from alpha_shape import generate_mask_from_points, db_eval_iou


def remove_boundary_points(points):
    try:
        points = np.array(points)

        # convex hull
        hull = ConvexHull(points)

        # bool array for boundary points
        is_boundary_point = np.zeros(len(points), dtype=bool)

        is_boundary_point[hull.vertices] = True

        # remove boundary points
        inner_points = points[~is_boundary_point]

        if len(inner_points) == 0:
            inner_points = points
    except:
        inner_points = points

    return inner_points


def points_to_image(points, image_shape):
    image = np.zeros(image_shape)

    for point in points:
        image[point[1], point[0]] = 1

    return image


def bool_to_int(mask):
    return np.where(mask, 0, 1)


def find_farthest_to_centroid(points, mask):
    # whether mask is empty
    if not mask.max():
        if len(points) < 3:
            farthest_point = points[0]
            tmp = farthest_point[0]
            farthest_point[0] = farthest_point[1]
            farthest_point[1] = tmp
        else:
            hull = ConvexHull(points)

            # center of convex hull
            centroid = np.mean(points[hull.vertices, :], axis=0)

            tree = spatial.KDTree(points)

            dist, idx = tree.query([centroid], k=1)
            farthest_point = points[idx[0]]
            tmp = farthest_point[0]
            farthest_point[0] = farthest_point[1]
            farthest_point[1] = tmp
        return farthest_point, 999
    else:
        mask_points = np.argwhere(mask == True)

        tree = spatial.KDTree(mask_points)

        max_distance = 0
        farthest_point = None

        for point in points:
            distance, _ = tree.query(point, k=1)

            if distance > max_distance:
                max_distance = distance
                farthest_point = point

        tmp = farthest_point[0]
        farthest_point[0] = farthest_point[1]
        farthest_point[1] = tmp
        return farthest_point, max_distance


def filter_points(input_point, input_label):

    if input_point is None or input_label is None:
        undo_points = []
        prompt_points = []
    else:
        undo_points = [input_point[i] for i in range(len(input_label)) if input_label[i] == 0]
        prompt_points = [input_point[i] for i in range(len(input_label)) if input_label[i] == 1]

    return undo_points, prompt_points


def get_num_samples(image, all_point, all_label, part_point, part_label, mask):
    resolution = tuple(mask.shape)

    all_undo_points, all_prompt_points = filter_points(all_point, all_label)
    if len(all_undo_points) == 0:
        all_undo_points.append(np.array([1, 1]))
        all_undo_points.append(np.array([1, 1]))
        all_undo_points.append(np.array([1, 1]))

    try:
        point_mask = generate_mask_from_points(np.asarray(all_prompt_points, dtype='int64'), resolution)
        iou = db_eval_iou(point_mask, mask)
        if iou > 0.8:
            return None, None, True
    except Exception as e:
        print(e)

    nearest_num = 1
    prompt_tree = spatial.KDTree(all_prompt_points)
    undo_tree = spatial.KDTree(all_undo_points)
    prompt_dist, prompt_nearest_idx = prompt_tree.query(all_undo_points, k=nearest_num)
    undo_dist, undo_nearest_idx = undo_tree.query(all_prompt_points, k=nearest_num)

    prompt_nearest_idx = prompt_nearest_idx.ravel()
    undo_nearest_idx = undo_nearest_idx.ravel()

    unique_prompt_idx = np.unique(prompt_nearest_idx)
    unique_undo_idx = np.unique(undo_nearest_idx)

    if len(all_undo_points) > nearest_num:
        all_undo_points = np.delete(all_undo_points, unique_undo_idx, axis=0)
    if len(all_prompt_points) > nearest_num:
        all_prompt_points = np.delete(all_prompt_points, unique_prompt_idx, axis=0)

    all_prompt_points = remove_boundary_points(all_prompt_points)
    all_undo_points = remove_boundary_points(all_undo_points)

    all_point_image = points_to_image(all_prompt_points, mask.shape)
    mask_int = bool_to_int(mask)

    not_in_mask_points = all_point_image * mask_int
    not_in_mask_points = np.argwhere(not_in_mask_points > 0)

    if len(not_in_mask_points) != 0:
        point, dist = find_farthest_to_centroid(not_in_mask_points, mask)

        if part_point is None:
            part_point = np.array([point])
            part_label = np.array([1])
        else:
            part_point = np.append(part_point, [point], axis=0)
            part_label = np.append(part_label, 1)

    else:
        dist = 0

    part_undo_points, part_prompt_points = filter_points(part_point, part_label)

    if len(part_prompt_points) > 10 or dist < 10:

        undo_points_image = points_to_image(all_undo_points, mask.shape)

        in_mask_undos = undo_points_image * mask

        if np.any(in_mask_undos) and len(part_undo_points) < 5:
            in_mask_pts = np.argwhere(mask)
            in_mask_undos = np.argwhere(in_mask_undos == 1)
            tree = spatial.KDTree(in_mask_undos)
            centroid = np.mean(in_mask_pts, axis=0)

            dist, idx = tree.query([centroid], k=1)

            nearest = in_mask_undos[idx[0]]

            tmp = nearest[0]
            nearest[0] = nearest[1]
            nearest[1] = tmp

            part_point = np.append(part_point, [nearest], axis=0)
            part_label = np.append(part_label, 0)
        else:
            return None, None, True

    return part_point, part_label, False
