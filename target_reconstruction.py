import numpy as np
import os
import cv2
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
from get_projection_graph import get_one_time_projection
from num_samples import get_num_samples
from read_para_before_db import get_image_info
from tqdm import tqdm


def correct_mask(mask_to_correct, positive_prompt):
    mask_uint8 = mask_to_correct.astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(mask_uint8)

    for region in range(num_labels):
        region_mask = (labels == region)

        wrong_flag = False
        for i in range(positive_prompt.shape[0]):
            point_result = region_mask[positive_prompt[i][1], positive_prompt[i][0]]
            if point_result:
                wrong_flag = True
                break

        if not wrong_flag:
            mask_to_correct[region_mask] = False

    return mask_to_correct


def prompt_for_reconstruction(work_path):
    image_pre = os.path.join(work_path, 'Images')

    image_format = os.listdir(image_pre)[0].split(".")[-1]

    output_path = os.path.join(work_path, 'output')

    image_info = get_image_info(output_path)

    image_name_to_index = {}
    for key, value in image_info.items():
        image_name_to_index[value] = int(key)

    image_resolution = []

    for key, value in image_info.items():
        value = os.path.join(image_pre, value)
        img = Image.open(value)
        width, height = img.size
        image_resolution.append([height, width])

    template_folder = os.path.join(work_path, 'masks')
    template_names = os.listdir(template_folder)

    template_frame = []
    template_arr = []

    for template_name in template_names:
        cur_frame = np.load(os.path.join(template_folder, template_name))
        template_frame.append(cur_frame)
        cur_index = image_name_to_index.get(template_name.replace('npy', image_format))
        template_arr.append(cur_index)

    sam_checkpoint = "checkpoints/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    image_num = len(image_info)

    # init Predictor
    predictor = SamPredictor(sam)

    mask_2d = []
    for i in range(image_num):
        temp = np.zeros((image_resolution[i][0], image_resolution[i][1]))
        mask_2d.append(temp)

    for i in range(len(template_arr)):
        mask_2d[template_arr[i] - 1] = template_frame[i]

    empty_list = [i for i in range(1, image_num + 1)]
    for template_single_index in template_arr:
        empty_list.remove(template_single_index)

    project_segment = get_one_time_projection(
        template_arr, output_path, template_frame, image_resolution)

    for image_index in tqdm(empty_list):
        image_name = os.path.split(image_info.get(image_index))[-1]

        image = cv2.imread(os.path.join(image_pre, image_info.get(image_index)))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        project_cur = project_segment[image_index - 1]
        project_cur.astype(int)

        result_image = np.zeros((image.shape[0], image.shape[1]))

        max_label = np.amax(project_cur)

        for i in range(1, int(max_label) + 1):
            input_point = []
            input_label = []

            indices_positive = np.where(project_cur == i)
            if indices_positive[0].shape[0] == 0:
                continue

            row_indices_positive = indices_positive[0]
            col_indices_positive = indices_positive[1]
            for m in range(row_indices_positive.shape[0]):
                prompt_point = [col_indices_positive[m], row_indices_positive[m]]
                input_point.append(prompt_point)
                input_label.append(1)

            indices_negative = np.where(np.logical_and(np.logical_and(project_cur != i, project_cur != -1),
                                                       0 != project_cur))
            row_indices_negative = indices_negative[0]
            col_indices_negative = indices_negative[1]
            for n in range(row_indices_negative.shape[0]):
                prompt_point = [col_indices_negative[n], row_indices_negative[n]]
                input_point.append(prompt_point)
                input_label.append(0)

            input_point = np.asarray(input_point)
            input_label = np.asarray(input_label)

            mask = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
            new_point = None
            new_label = None
            prompt_index = 0

            while True:
                try:
                    new_point, new_label, is_full = get_num_samples(image, input_point, input_label, new_point, new_label,
                                                                mask)
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

                    mask = correct_mask(mask, new_point)

                    prompt_index += 1

            result_image[mask] = i

        image_number = image_name.split('.')[0]

        np.save(os.path.join(template_folder, f'{image_number}.npy'), result_image)

        mask_2d[image_index - 1] = result_image
