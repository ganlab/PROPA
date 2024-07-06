import gradio as gr
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import json
from paint_util import mask_painter, point_painter
import cv2
from tracking_util import track_from_mask
import os


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


def remove_multi_mask(interactive_state, mask_dropdown):
    interactive_state["multi_mask"]["mask_names"] = []
    interactive_state["multi_mask"]["masks"] = []

    operation_log = [("", ""), ("Remove all mask, please add new masks", "Normal")]
    return interactive_state, gr.update(choices=[], value=[]), operation_log


def clear_click(image_state):
    click_state = [[], []]
    template_frame = image_state["origin_images"]
    operation_log = [("", ""), ("Clear points history and refresh the image.", "Normal")]
    return template_frame, click_state, operation_log


def start_tracking(video_info, mask_dropdown, interactive_state, factor):
    video_path = video_info["video_path"]
    video_name = os.path.basename(video_path)
    video_name = os.path.splitext(video_name)[0]

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

    track_from_mask(template_mask, video_path, factor, predictor)

    result_path = os.path.join('tracking', video_name, f'{video_name}.mp4')

    operation_log = [("", ""), (f"Finish tracking'.", "Normal")]

    return result_path, operation_log


def init_prompt_propagation(video_input, interactive_state):
    interactive_state["negative_click_times"] = 0
    interactive_state["positive_click_times"] = 0
    interactive_state["multi_mask"]["mask_names"] = []
    interactive_state["multi_mask"]["masks"] = []

    click_state = [[], []]
    video_path = video_input
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if success:
        image_state = {
            "origin_images": frame,
            "painted_images": frame.copy(),
            "masks": np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
        }

        video_info = {
            "video_path": video_path
        }

        image_info = f'resolution: {frame.shape[0]} × {frame.shape[1]}'
        operation_log = [("", ""),
                         ("Upload video already. Try click the image for adding masks.", "Normal")]

        return click_state, interactive_state, video_info, frame, image_state, image_info, operation_log
    else:
        return None, None, None, None, None, None, None


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
    video_info = gr.State({
        "video_path": None
    })

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
            video_input = gr.Video()
            image_input = gr.Image(interactive=False, visible=True, label="Input Image", height=360)
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

            factor_slider = gr.Slider(label="Uniform factor", maximum=20, minimum=1, value=8, visible=True, step=1)
            start_tracking_button = gr.Button(value="Start tracking", interactive=True, visible=True)

        with gr.Column():
            image_info = gr.Textbox(label="Image Info")

            run_status = gr.HighlightedText(
                value=[("Text", "Error"), ("to be", "Label 2"), ("highlighted", "Label 3")], visible=True)

            result_video = gr.Video(interactive=False)

    video_input.upload(
        fn=init_prompt_propagation,
        inputs=[video_input, interactive_state],
        outputs=[click_state, interactive_state, video_info, image_input, image_state, image_info, run_status]
    )

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
        inputs=[interactive_state, mask_dropdown],
        outputs=[interactive_state, mask_dropdown, run_status]
    )

    clear_button_click.click(
        fn=clear_click,
        inputs=[image_state],
        outputs=[image_input, click_state, run_status],
    )

    start_tracking_button.click(
        fn=start_tracking,
        inputs=[video_info, mask_dropdown, interactive_state, factor_slider],
        outputs=[result_video, run_status]
    )


iface.launch(debug=True, server_port=8080, server_name="127.0.0.1")