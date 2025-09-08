from read_para_before_db import get_matches_and_points_from_h5, get_pair_connections
import numpy as np


def get_projection_graph(template_arr, out_path, template_frame, image_resolution):
    project_segment = []
    for resolution in image_resolution:
        temp = np.ones((resolution[0], resolution[1])) * -1
        project_segment.append(temp)

    image_points, points_matches, image_info_sparse = get_matches_and_points_from_h5(out_path)
    return_list = get_pair_connections(points_matches)

    for i in range(len(return_list)):
        map = return_list[i]

        contains_template = False
        template_label = -1

        for point_pair in map:
            # index
            possible_image_index = int(point_pair.split('-')[0])
            possible_point_index = int(point_pair.split('-')[1])
            if possible_image_index in template_arr:
                contains_template = True
                template_xy = image_points.get(possible_image_index)[possible_point_index]
                template_arr_index = template_arr.index(possible_image_index)
                template_label = template_frame[template_arr_index][int(template_xy[1])][int(template_xy[0])]
                break
        if contains_template:
            for point_pair in map:
                image_index_sparse = int(point_pair.split('-')[0])
                point_index = int(point_pair.split('-')[1])
                point_xy = image_points.get(image_index_sparse)[point_index]
                if image_index_sparse in template_arr:
                    new_template_arr_index = template_arr.index(image_index_sparse)
                    new_label = template_frame[new_template_arr_index][int(point_xy[1])][int(point_xy[0])]
                    template_label = new_label

                project_segment[image_index_sparse - 1][int(point_xy[1])][int(point_xy[0])] = template_label
    return project_segment


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