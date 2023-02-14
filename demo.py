import math

import torch

MODEL_NAME = "separable_half"
DEVICE_NAME = 'cuda'
device = torch.device(DEVICE_NAME)


def load_poser(model: str, device: torch.device):
    print("Using the %s model." % model)
    if model == "standard_float":
        from tha3.poser.modes.standard_float import create_poser
        return create_poser(device)
    elif model == "standard_half":
        from tha3.poser.modes.standard_half import create_poser
        return create_poser(device)
    elif model == "separable_float":
        from tha3.poser.modes.separable_float import create_poser
        return create_poser(device)
    elif model == "separable_half":
        from tha3.poser.modes.separable_half import create_poser
        return create_poser(device)
    else:
        raise RuntimeError("Invalid model: '%s'" % model)


poser = load_poser(MODEL_NAME, DEVICE_NAME)
poser.get_modules();


import PIL.Image
import io
from io import StringIO, BytesIO
import IPython.display
import numpy
import ipywidgets
import time
import threading
import torch
from tha3.util import resize_PIL_image, extract_PIL_image_from_filelike, \
    extract_pytorch_image_from_PIL_image, convert_output_image_from_torch_to_numpy
import matplotlib.pyplot as plt
from matplotlib.image import imread

FRAME_RATE = 30.0

last_output_image = None
torch_input_image = None

import cv2



from tha3.poser.modes.pose_parameters import get_pose_parameters
pose_parameters = get_pose_parameters()
pose_size = poser.get_num_parameters()
last_pose = torch.zeros(1, pose_size, dtype=poser.get_dtype()).to(device)


iris_small_left_index = pose_parameters.get_parameter_index("iris_small_left")
iris_small_right_index = pose_parameters.get_parameter_index("iris_small_right")
iris_rotation_x_index = pose_parameters.get_parameter_index("iris_rotation_x")
iris_rotation_y_index = pose_parameters.get_parameter_index("iris_rotation_y")
head_x_index = pose_parameters.get_parameter_index("head_x")
head_y_index = pose_parameters.get_parameter_index("head_y")
neck_z_index = pose_parameters.get_parameter_index("neck_z")
body_y_index = pose_parameters.get_parameter_index("body_y")
body_z_index = pose_parameters.get_parameter_index("body_z")
breathing_index = pose_parameters.get_parameter_index("breathing")


class Status:
    def __init__(self):
        self.eyebrow_status = "troubled"  # options = ["troubled", "angry", "lowered", "raised", "happy", "serious"]
        self.eyebrow_left = 0.0  # 0.0 ~ 1.0
        self.eyebrow_right = 0.0  # 0.0 ~ 1.0
        self.eye_status = "wink"  # options=["wink", "happy_wink", "surprised", "relaxed", "unimpressed", "raised_lower_eyelid"]
        self.eye_left = 0.0  # 0.0 ~ 1.0
        self.eye_right = 0.0  # 0.0 ~ 1.0
        self.mouth_status = "aaa"  # options=["aaa", "iii", "uuu", "eee", "ooo", "delta", "lowered_corner", "raised_corner", "smirk"]
        self.mouth_left = 0.0  # 0.0 ~ 1.0
        self.mouth_right = 0.0  # 0.0 ~ 1.0
        self.iris_small_left = 0.0  # 0.0 ~ 1.0
        self.iris_small_right = 0.0  # 0.0 ~ 1.0
        self.iris_rotation_x = 0.0  # -1.0 ~ 1.0
        self.iris_rotation_y = 0.0  # -1.0 ~ 1.0
        self.head_x = 0.0  # -1.0 ~ 1.0
        self.head_y = 0.0  # -1.0 ~ 1.0
        self.neck_z = 0.0  # -1.0 ~ 1.0
        self.body_y = 0.0  # -1.0 ~ 1.0
        self.body_z = 0.0  # -1.0 ~ 1.0
        self.breathing = 0.0  # 0.0 ~ 1.0





def get_pose():
    global status
    pose = torch.zeros(1, pose_size, dtype=poser.get_dtype())

    eyebrow_name = f"eyebrow_{status.eyebrow_status}"
    eyebrow_left_index = pose_parameters.get_parameter_index(f"{eyebrow_name}_left")
    eyebrow_right_index = pose_parameters.get_parameter_index(f"{eyebrow_name}_right")
    pose[0, eyebrow_left_index] = status.eyebrow_left
    pose[0, eyebrow_right_index] = status.eyebrow_right

    eye_name = f"eye_{status.eye_status}"
    eye_left_index = pose_parameters.get_parameter_index(f"{eye_name}_left")
    eye_right_index = pose_parameters.get_parameter_index(f"{eye_name}_right")
    pose[0, eye_left_index] = status.eye_left
    pose[0, eye_right_index] = status.eye_right

    mouth_name = f"mouth_{status.mouth_status}"
    if mouth_name == "mouth_lowered_corner" or mouth_name == "mouth_raised_corner":
        mouth_left_index = pose_parameters.get_parameter_index(f"{mouth_name}_left")
        mouth_right_index = pose_parameters.get_parameter_index(f"{mouth_name}_right")
        pose[0, mouth_left_index] = status.mouth_left
        pose[0, mouth_right_index] = status.mouth_right
    else:
        mouth_index = pose_parameters.get_parameter_index(mouth_name)
        pose[0, mouth_index] = status.mouth_left

    pose[0, iris_small_left_index] = status.iris_small_left
    pose[0, iris_small_right_index] = status.iris_small_right
    pose[0, iris_rotation_x_index] = status.iris_rotation_x
    pose[0, iris_rotation_y_index] = status.iris_rotation_y
    pose[0, head_x_index] = status.head_x
    pose[0, head_y_index] = status.head_y
    pose[0, neck_z_index] = status.neck_z
    pose[0, body_y_index] = status.body_y
    pose[0, body_z_index] = status.body_z
    pose[0, breathing_index] = status.breathing

    return pose.to(device)

import sys
from matplotlib import animation


def update():
    global last_pose
    global last_output_image
    global output_image

    if torch_input_image is None:
        return None
    if last_output_image is None:
        last_output_image = torch_input_image
        return torch_input_image
    needs_update = False

    pose = get_pose()
    # print(pose)
    if (pose - last_pose).abs().max().item() > 0:
        needs_update = True

    if not needs_update:
        return last_output_image

    output_image = poser.pose(torch_input_image, pose)[0]

    last_pose = pose
    last_output_image = output_image
    return output_image


def upload_image(image):
    global torch_input_image
    content = image
    if content is not None:
        pil_image = resize_PIL_image(extract_PIL_image_from_filelike(content), size=(512,512))
        w, h = pil_image.size
        if pil_image.mode != 'RGBA':
            torch_input_image = None
            print("Image must have an alpha channel!!!", file=sys.stderr)
        else:
            torch_input_image = extract_pytorch_image_from_PIL_image(pil_image).to(device)
            if poser.get_dtype() == torch.half:
                torch_input_image = torch_input_image.half()
        update()

def show_pytorch_image(pytorch_image):
    output_image = pytorch_image.detach().cpu()
    numpy_image = numpy.uint8(numpy.rint(convert_output_image_from_torch_to_numpy(output_image) * 255.0))
    pil_image = PIL.Image.fromarray(numpy_image, mode='RGBA')
    plt.imshow(pil_image)
    plt.show()


if __name__ == '__main__':
    status = Status()
    upload_image("data/images/crypko_00.png")
    frame_cnt = 0
    while True:
        pytorch_image = update()
        output_image = pytorch_image.detach().cpu()
        numpy_image = numpy.uint8(numpy.rint(convert_output_image_from_torch_to_numpy(output_image) * 255.0))
        numpy_image_rgba = numpy_image[:, :, [2, 1, 0, 3]]
        cv2.imshow("image", numpy_image_rgba)
        cv2.waitKey(1)
        frame_cnt += 1
        if frame_cnt % 100 == 0:
            print(frame_cnt)

        #status.breathing = 0.5 + 0.5 * math.sin(frame_cnt / 100.0)
        status.iris_rotation_y = 0.5 + 0.5 * math.sin(frame_cnt / 10.0)
        # print(status.iris_rotation_y)
