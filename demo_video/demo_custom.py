import os
import subprocess
import numpy as np


os.environ["CUDA_VISIBLE_DEVICES"] = "1"  

import argparse
import multiprocessing as mp
import tempfile
import time
import cv2
import re

from torch.cuda.amp import autocast
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from mask2former import add_maskformer2_config
from models import add_avism_config
from predictor import VisualizationDemo


WINDOW_NAME = "avis video demo"

def setup_cfg(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_avism_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="avis demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/acvis_R50_COCO.yaml",
        metavar="FILE",
        help="path to detectron2 config file",
    )
    parser.add_argument(
        "--input-dir",
        default="datasets/test/JPEGImages",
        help="root folder containing subfolders of frames (one folder per video), or directly a folder of frames for a single video",
    )
    parser.add_argument(
        "--audio-dir",
        default="datasets/test/FEATAudios",
        help="folder containing .npy audio features (one .npy per video)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/acvis_R50_COCO",
        help="where to save visualized output frames",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

def extract_number(filename):
    m = re.search(r'(\d+)\.jpg$', filename)
    return int(m.group(1)) if m else -1

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()

    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info(f"Arguments: {args}")

    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)

    
    os.makedirs(args.output_dir, exist_ok=True)

    
    all_entries = sorted(os.listdir(args.input_dir))
    subdirs = [d for d in all_entries if os.path.isdir(os.path.join(args.input_dir, d))]
    if subdirs:
        video_folders = [(d, os.path.join(args.input_dir, d)) for d in subdirs]
    else:
        npy_files = [f for f in sorted(os.listdir(args.audio_dir)) if f.endswith('.npy')]
        if len(npy_files) == 1:
            video_name = os.path.splitext(npy_files[0])[0]
        else:
            video_name = os.path.basename(os.path.normpath(args.input_dir))
        video_folders = [(video_name, args.input_dir)]

    for video_name, video_folder in video_folders:
        print(f"Processing video: {video_name}")

        
        frame_files = sorted(
            [f for f in os.listdir(video_folder) if f.lower().endswith('.jpg')],
            key=extract_number
        )
        if not frame_files:
            logger.warning(f"No .jpg frames found in {video_folder}, skipping.")
            continue

        vid_frames = [
            read_image(os.path.join(video_folder, f), format="BGR")
            for f in frame_files
        ]

        
        audio_pth = os.path.join(args.audio_dir, video_name + ".npy")
        if not os.path.exists(audio_pth):
            logger.warning(f"Audio feature not found for {video_name}, skipping.")
            continue
        audio_feats = np.load(audio_pth)

        
        start = time.time()
        with autocast():
            predictions, visualized_outputs = demo.run_on_video(vid_frames, audio_feats)
        duration = time.time() - start
        print(f"Inference done in {duration:.1f}s")

        
        save_folder = os.path.join(args.output_dir, video_name)
        os.makedirs(save_folder, exist_ok=True)
        for fname, vis in zip(frame_files, visualized_outputs):
            out_path = os.path.join(save_folder, fname)
            vis.save(out_path)

        print(f"Saved visualizations to {save_folder}\n")
