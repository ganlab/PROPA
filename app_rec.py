import os
import zipfile, tarfile, gzip
import gradio as gr
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import json
from paint_util import mask_painter, point_painter
import cv2
from reconstruction_util import struct_from_motion, multi_view_stero
import shutil
from target_reconstruction import prompt_for_reconstruction
from reconstruction_util import extract_from_numpy
import open3d as o3d


def get_point_cloud_capture(reconstruction_folder):
    pcd_path = os.path.join(reconstruction_folder, 'output', 'dense_scene.ply')
    out_path = pcd_path.replace('dense_scene.ply', 'result.png')
    pcd = o3d.io.read_point_cloud(pcd_path)

    visualizer = o3d.visualization.Visualizer()

    visualizer.create_window(
        visible=False
    )

    visualizer.add_geometry(pcd)

    visualizer.poll_events()
    visualizer.update_renderer()

    visualizer.capture_screen_image(out_path, do_render=True)

    image = cv2.imread(out_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image



def show_mask(image_state, interactive_state, mask_dropdown):
    mask_dropdown.sort()
    select_frame = image_state["origin_images"]

    for i in range(len(mask_dropdown)):
        mask_number = int(mask_dropdown[i].split("_")[1]) - 1
        mask = interactive_state["multi_mask"]["masks"][mask_number]
        select_frame = mask_painter(select_frame, mask.astype('uint8'), mask_color=mask_number + 2)

    operation_log = [("", ""), ("Select {} for tracking or inpainting".format(mask_dropdown), "Normal")]
    return select_frame, operation_log


def get_sam_result(image, points, labels):
    mask_color = 3
    mask_alpha = 0.7
    point_color_ne = 8
    point_color_ps = 50
    point_alpha = 0.9
    point_radius = 15
    contour_color = 2
    contour_width = 5

    masks, scores, logits = predictor.predict(
        point_coords=points,
        point_labels=labels,
        multimask_output=False
    )
    mask = masks[0]

    painted_image = mask_painter(image, mask.astype('uint8'), mask_color, mask_alpha, contour_color, contour_width)
    painted_image = point_painter(painted_image, np.squeeze(points[np.argwhere(labels > 0)], axis=1), point_color_ne,
                                  point_alpha, point_radius, contour_color, contour_width)
    painted_image = point_painter(painted_image, np.squeeze(points[np.argwhere(labels < 1)], axis=1), point_color_ps,
                                  point_alpha, point_radius, contour_color, contour_width)
    # painted_image = Image.fromarray(painted_image)

    return mask, painted_image


def get_prompt(click_state, click_input):
    inputs = json.loads(click_input)
    points = click_state[0]
    labels = click_state[1]
    for input in inputs:
        points.append(input[:2])
        labels.append(input[2])
    click_state[0] = points
    click_state[1] = labels
    prompt = {
        "prompt_type": ["click"],
        "input_point": click_state[0],
        "input_label": click_state[1],
        "multimask_output": "True",
    }
    return prompt


def sam_refine(image_state, point_prompt, click_state, interactive_state, evt: gr.SelectData):
    if point_prompt == "Positive":
        coordinate = "[[{},{},1]]".format(evt.index[0], evt.index[1])
        interactive_state["positive_click_times"] += 1
    else:
        coordinate = "[[{},{},0]]".format(evt.index[0], evt.index[1])
        interactive_state["negative_click_times"] += 1

    predictor.set_image(image_state["origin_images"])

    prompt = get_prompt(click_state=click_state, click_input=coordinate)

    mask, painted_image = get_sam_result(
        image=image_state["origin_images"],
        points=np.array(prompt["input_point"]),
        labels=np.array(prompt["input_label"])
    )

    image_state["masks"] = mask
    image_state["painted_images"] = painted_image

    operation_log = [("", ""), (
        "Use SAM for segment. You can try add positive and negative points by clicking. Or press Clear clicks button to refresh the image. Press Add mask button when you are satisfied with the segment",
        "Normal")]

    return painted_image, image_state, interactive_state, operation_log


def create_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)


def init_prompt_propagation(file, interactive_state):
    create_folder('images')
    file_name = os.path.basename(file.name)
    image_path = os.path.join('images', os.path.splitext(file_name)[0])
    create_folder(image_path)

    format = file_name.split('.')[-1]

    if format == 'zip':
        with zipfile.ZipFile(file.name, 'r') as zip_ref:
            zip_ref.extractall(image_path)
    elif format == 'tar':
        tar = tarfile.open(file.name)
        names = tar.getnames()
        for name in names:
            tar.extract(name, image_path)

        filename = file.name
        tf = tarfile.open(filename)
        tf.extractall(image_path)


    else:
        operation_log = [("Unsupported file format, please choose from 'zip', 'tar.gz'", "Error"), ("", "")]
        return None, None, None, None, None, None, None, None, None, operation_log

    images = os.listdir(image_path)
    images.sort()

    cur_image_name = [("", ""), ("", ""), (f"{images[0]}", "Image")]

    for i in range(len(images)):
        images[i] = os.path.join(image_path, images[i])

    first_image = cv2.imread(images[0])
    first_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2RGB)

    image_state = {
        "origin_images": first_image,
        "painted_images": first_image.copy(),
        "masks": np.zeros((first_image.shape[0], first_image.shape[1]), np.uint8)
    }

    image_info = f'resolution: {first_image.shape[0]} × {first_image.shape[1]}'

    interactive_state["negative_click_times"] = 0
    interactive_state["positive_click_times"] = 0
    interactive_state["multi_mask"]["mask_names"] = []
    interactive_state["multi_mask"]["masks"] = []

    click_state = [[], []]

    operation_log = [("", ""), ("Successfully initialize PROPA, try click the image for adding masks.", "normal")]

    possible_nm = int(len(images) / 3)

    return first_image, images, gr.update(maximum=len(images) - 1), image_state, image_info, interactive_state, \
        click_state, gr.update(maximum=len(images) - 1, value=possible_nm), cur_image_name, operation_log


def add_multi_mask(image_state, interactive_state, mask_dropdown):
    try:
        mask = image_state["masks"]
        interactive_state["multi_mask"]["masks"].append(mask)
        interactive_state["multi_mask"]["mask_names"].append(
            "mask_{:03d}".format(len(interactive_state["multi_mask"]["masks"])))
        mask_dropdown.append("mask_{:03d}".format(len(interactive_state["multi_mask"]["masks"])))
        frame_result, run_status = show_mask(image_state, interactive_state, mask_dropdown)

        operation_log = [("", ""), ("Added a new mask successfully.", "Normal")]
    except:
        operation_log = [("Please click the image to generate mask.", "Error"), ("", "")]
    return interactive_state, gr.update(choices=interactive_state["multi_mask"]["mask_names"],
                                        value=mask_dropdown), frame_result, [[], []], operation_log


def remove_multi_mask(interactive_state):
    interactive_state["multi_mask"]["mask_names"] = []
    interactive_state["multi_mask"]["masks"] = []

    operation_log = [("", ""), ("Remove all mask, please add new masks", "Normal")]
    return interactive_state, gr.update(choices=[], value=[]), operation_log


def clear_click(image_state):
    click_state = [[], []]
    template_frame = image_state["origin_images"]
    operation_log = [("", ""), ("Clear points history and refresh the image.", "Normal")]
    return template_frame, click_state, operation_log


def save_mask(mask_dropdown, interactive_state, image_number, image_series):
    full_name = image_series[image_number]
    category_name = full_name.split('/')[-2]
    save_name = os.path.basename(full_name)
    save_name = os.path.splitext(save_name)[0]
    create_folder('mask')
    create_folder(os.path.join('mask', category_name))

    if len(mask_dropdown) == 0:
        mask_dropdown = ["mask_001"]
    mask_dropdown.sort()
    template_mask = interactive_state["multi_mask"]["masks"][int(mask_dropdown[0].split("_")[1]) - 1] * (
        int(mask_dropdown[0].split("_")[1]))
    for i in range(1, len(mask_dropdown)):
        mask_number = int(mask_dropdown[i].split("_")[1]) - 1
        template_mask = np.clip(
            template_mask + interactive_state["multi_mask"]["masks"][mask_number] * (mask_number + 1), 0,
            mask_number + 1)
    np.save(f'mask/{category_name}/{save_name}.npy', template_mask)
    operation_log = [("", ""), (f"Mask saved in this path: 'mask/{category_name}/{save_name}.npy'.", "Normal")]

    all_masks_list = os.listdir(os.path.join('mask', category_name))
    all_masks = " ".join(all_masks_list)

    return all_masks, operation_log


def change_image(image_series, image_slider, interactive_state):
    cur_image = image_series[int(image_slider)]
    image_name = os.path.basename(cur_image)
    cur_image = cv2.imread(cur_image)
    cur_image = cv2.cvtColor(cur_image, cv2.COLOR_BGR2RGB)

    image_state = {
        "origin_images": cur_image,
        "painted_images": cur_image.copy(),
        "masks": np.zeros((cur_image.shape[0], cur_image.shape[1]), np.uint8)
    }

    image_info = f'resolution: {cur_image.shape[0]} × {cur_image.shape[1]}'

    interactive_state["negative_click_times"] = 0
    interactive_state["positive_click_times"] = 0
    interactive_state["multi_mask"]["mask_names"] = []
    interactive_state["multi_mask"]["masks"] = []

    click_state = [[], []]

    operation_log = [("", ""), (f"Change image to {image_name}, try click the image for adding masks.", "Normal")]

    cur_image_name = [("", ""), ("", ""), (f"{image_name}", "Image")]

    return interactive_state, cur_image, image_state, image_info, click_state, gr.update(choices=[], value=[]), cur_image_name, operation_log


def start_reconstruction(number_matches, image_series):
    full_name = image_series[0]
    category_name = full_name.split('/')[-2]

    create_folder('reconstruction')
    reconstruction_folder = os.path.join('reconstruction', category_name)
    if os.path.exists(reconstruction_folder):
        shutil.rmtree(reconstruction_folder)
    create_folder(reconstruction_folder)

    # copy images and rename
    shutil.copytree(os.path.join('images', category_name), os.path.join(reconstruction_folder, 'Images'))

    shutil.copytree(os.path.join('mask', category_name), os.path.join(reconstruction_folder, 'masks'))

    struct_from_motion(reconstruction_folder, number_matches)

    prompt_for_reconstruction(reconstruction_folder)

    extract_from_numpy(reconstruction_folder, 1)

    multi_view_stero(reconstruction_folder)

    return get_point_cloud_capture(reconstruction_folder)



title = """<p><h1 align="center">PROPA</h1></p>"""
# 设置SAM参数
sam_checkpoint = "checkpoints/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

# 初始化SAM
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

with gr.Blocks() as iface:
    click_state = gr.State([[], []])
    image_series = gr.State([])

    image_state = gr.State({
        "origin_images": None,
        "painted_images": None,
        "masks": None

    })

    interactive_state = gr.State({
        "negative_click_times": 0,
        "positive_click_times": 0,
        "multi_mask": {
            "mask_names": [],
            "masks": []
        }
    })

    gr.Markdown(title)
    with gr.Row():
        with gr.Column():
            with gr.Row():
                # Upload compressed file
                compressed_file = gr.File(label="Upload Compressed File (.zip/.tar/.gz)")
            with gr.Row():
                image_input = gr.Image(interactive=False, visible=True, label="Input Image", height=360)
            with gr.Row():
                highlighted_text = gr.HighlightedText(label="Current image name",
                                                      value=[("Text", "Error"),
                                                             ("to be", "Label 2"),
                                                             ("highlighted", "Label 3")],
                                                      visible=True)
            with gr.Row():
                image_slider = gr.Slider(minimum=0, maximum=100, step=1, value=0, label="Image Number", visible=True)
                # image_number = gr.Number(minimum=0, maximum=100, step=1, value=0, label="Image Number", visible=True)
            with gr.Row():
                point_prompt = gr.Radio(
                    choices=["Positive", "Negative"],
                    value="Positive",
                    label="Point prompt",
                    interactive=True,
                    visible=True)
                remove_mask_button = gr.Button(value="Remove mask", interactive=True, visible=True)
                clear_button_click = gr.Button(value="Clear clicks", interactive=True,
                                               visible=True)
                add_mask_button = gr.Button(value="Add mask", interactive=True, visible=True)

                mask_dropdown = gr.Dropdown(multiselect=True, value=[], label="Mask selection",
                                            info=".",
                                            visible=True)

            with gr.Row():
                generate_template_button = gr.Button(value="Generate mask", interactive=True, visible=True)

        with gr.Column():
            image_info = gr.Textbox(label="Image Info")

            run_status = gr.HighlightedText(label="Log",
                                            value=[("Text", "Error"), ("to be", "Label 2"), ("highlighted", "Label 3")],
                                            visible=True)
            current_mask = gr.Textbox(interactive=False, label="All masks", visible=True)
            number_matches = gr.Slider(label="Number Matches", value=5, minimum=0, maximum=100, step=1)
            start_reconstruction_button = gr.Button(value="Start Reconstruction", interactive=True, visible=True)
            result_image = gr.Image(interactive=False, visible=True, label="Result Point Cloud", height=360)

    compressed_file.upload(
        fn=init_prompt_propagation,
        inputs=[compressed_file, interactive_state],
        outputs=[image_input, image_series, image_slider, image_state, image_info, interactive_state, click_state, number_matches, highlighted_text, run_status]
    )

    # second step use sam to segment
    image_input.select(
        fn=sam_refine,
        inputs=[image_state, point_prompt, click_state, interactive_state],
        outputs=[image_input, image_state, interactive_state, run_status]
    )

    add_mask_button.click(
        fn=add_multi_mask,
        inputs=[image_state, interactive_state, mask_dropdown],
        outputs=[interactive_state, mask_dropdown, image_input, click_state, run_status]
    )

    remove_mask_button.click(
        fn=remove_multi_mask,
        inputs=[interactive_state],
        outputs=[interactive_state, mask_dropdown, run_status]
    )

    clear_button_click.click(
        fn=clear_click,
        inputs=[image_state],
        outputs=[image_input, click_state, run_status],
    )

    generate_template_button.click(
        fn=save_mask,
        inputs=[mask_dropdown, interactive_state, image_slider, image_series],
        outputs=[current_mask, run_status]
    )

    image_slider.release(
        fn=change_image,
        inputs=[image_series, image_slider, interactive_state],
        outputs=[interactive_state, image_input, image_state, image_info, click_state, mask_dropdown,
                 highlighted_text, run_status]
    )

    start_reconstruction_button.click(
        fn=start_reconstruction,
        inputs=[number_matches, image_series],
        outputs=[result_image]
    )

iface.launch(debug=True, server_port=8000, server_name="127.0.0.1")
