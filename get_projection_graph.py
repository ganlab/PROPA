from read_para_before_db import get_matches_and_points_from_h5, get_pair_connections
import numpy as np


def get_one_time_projection(template_arr, db_path, template_frame, image_resolution):
    project_segment = []
    for resolution in image_resolution:
        temp = np.ones((resolution[0], resolution[1])) * -1
        project_segment.append(temp)
    image_points, points_matches, image_info = get_matches_and_points_from_h5(db_path)

    for key, value in points_matches.items():
        # id
        img_id1, img_id2 = key[0], key[1]

        if img_id1 in template_arr:
            # array, row: index of point match in two images
            points_related = points_matches.get(key)

            for i in range(points_related.shape[0]):
                # index of two points
                points_pair_index = points_related[i]
                # xy of template
                template_xy = image_points.get(img_id1)[points_pair_index[0]]
                # xy to map
                projection_xy = image_points.get(img_id2)[points_pair_index[1]]
                # index in template_array
                template_arr_index = template_arr.index(img_id1)
                # label in template
                label = template_frame[template_arr_index][int(template_xy[1])][int(template_xy[0])]

                project_segment[img_id2 - 1][int(projection_xy[1])][int(projection_xy[0])] = label
        elif img_id2 in template_arr:
            points_related = points_matches.get(key)

            for i in range(points_related.shape[0]):

                points_pair_index = points_related[i]

                template_xy = image_points.get(img_id2)[points_pair_index[1]]

                projection_xy = image_points.get(img_id1)[points_pair_index[0]]

                template_arr_index = template_arr.index(img_id2)

                label = template_frame[template_arr_index][int(template_xy[1])][int(template_xy[0])]

                project_segment[img_id1 - 1][int(projection_xy[1])][int(projection_xy[0])] = label
        else:
            continue

    return project_segment