import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
import multiprocessing as mp

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import cv2
import numpy as np
import re

from torch.cuda.amp import autocast

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from models import add_avism_config
from predictor import VisualizationDemo


# constants
WINDOW_NAME = "avis video demo"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_avism_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="acvis demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/acvis_R50_COCO.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'"
        "this will be treated as frames of a video",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--save-frames",
        default=True,
        help="Save frame level image outputs.",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False

def extract_number(filename):
    return int(re.search(r'(\d+).jpg$', filename).group(1))


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if args.output:
        os.makedirs(args.output, exist_ok=True)

    input_dir = "datasets/test/JPEGImages/"
    output_dir = "results/acvis_R50_COCO"
    for video_name in os.listdir(input_dir):
        print(video_name)
        vid_frames = []
        for path in sorted(os.listdir(os.path.join(input_dir, video_name)), key=extract_number):
            img = read_image(os.path.join(input_dir, video_name, path), format="BGR")
            vid_frames.append(img)

        # audio_pth = os.path.join("datasets/test/FEATAudios", video_name + ".npy")
        audio_pth = os.path.join("datasets/test/MELAudios", video_name + ".npy")

        audio_feats = np.load(audio_pth)

        start_time = time.time()
        with autocast():
            predictions, visualized_output = demo.run_on_video(vid_frames, audio_feats)

        os.makedirs(os.path.join(output_dir, video_name), exist_ok=True)

        for path, _vis_output in zip(sorted(os.listdir(os.path.join(input_dir, video_name)), key=extract_number), visualized_output):
            out_filename = os.path.join(output_dir, video_name, path)
            _vis_output.save(out_filename)
