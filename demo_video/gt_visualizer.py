





import os, argparse, cv2, numpy as np
from detectron2.data.detection_utils import read_image
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from models.data.aviseval.datasets.avis import AVIS
from visualizer import TrackVisualizer
from detectron2.structures import Instances    
from pycocotools import mask as mask_utils     
import torch
import pandas as pd

def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--gt_folder",   default="datasets")          
    p.add_argument("--gt_file",     default="test.json")
    p.add_argument("--img_root",    default="datasets/test/JPEGImages")
    p.add_argument("--out_root",    default="results/gt_vis")
    p.add_argument("--std_root",    default="datasets/test/num_ins_CSVs")
    return p

def load_avis_dataset(gt_folder, gt_file):
    cfg = {
        "GT_FOLDER": gt_folder,
        "GT_File":   gt_file,
        "TRACKERS_FOLDER":   "./dummy",   
        "TRACKERS_TO_EVAL":  [],          
    }
    avis = AVIS(cfg)
    return avis


def register_metadata(avis, dataset_name="avis_test"):
    cats = sorted(avis.gt_data["categories"], key=lambda c: c["id"])
    thing_classes = [c["name"] for c in cats]
    id_to_contig   = {c["id"]: i for i, c in enumerate(cats)}

    meta = MetadataCatalog.get(dataset_name)
    meta.set(
        thing_classes=thing_classes,
        thing_dataset_id_to_contiguous_id=id_to_contig
    )
    return meta, id_to_contig



def visualize_gt(avis, img_root, out_root, meta):
    os.makedirs(out_root, exist_ok=True)
    
    anns_by_vid = {}
    for ann in avis.gt_data["annotations"]:
        anns_by_vid.setdefault(ann["video_id"], []).append(ann)
    
    for vid in avis.gt_data["videos"]:
        vid_id      = vid["id"]
        video_name  = vid["file_names"][0].split("/")[0]        
        width, height = vid["width"], vid["height"]
        
        out_dir = os.path.join(out_root, video_name)
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)
        
        for f_idx, fname in enumerate(vid["file_names"]):
            record = {
                "file_name": os.path.join(img_root, fname),
                "height": height,
                "width":  width,
                "annotations": []
            }
            
            for trk in anns_by_vid.get(vid_id, []):
                seg = trk["segmentations"][f_idx]
                if not seg:      
                    continue
                record["annotations"].append({
                    "segmentation": seg,               
                    "category_id":  trk["category_id"],
                    "iscrowd":      trk.get("iscrowd", 0)
                })
           
            if len(record["annotations"]) == 0:
                continue
            
            img = read_image(record["file_name"], format="BGR")
            viz = Visualizer(img[:, :, ::-1], meta, instance_mode=ColorMode.IMAGE)
            vis = viz.draw_dataset_dict(record).get_image()[:, :, ::-1]
            cv2.imwrite(os.path.join(out_dir, os.path.basename(fname)), vis)

            print(f"[{video_name}] frame {f_idx+1}/{len(vid['file_names'])} saved")

    print(f"\n Done -> {out_root}")


def visualize_gt_with_trackvis(avis, img_root, out_root, meta, id2contig, std_root):
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(std_root, exist_ok=True)
    anns_by_vid = {}
    for ann in avis.gt_data["annotations"]:
        anns_by_vid.setdefault(ann["video_id"], []).append(ann)

    for vid in avis.gt_data["videos"]:
        vid_id      = vid["id"]
        video_name  = vid["file_names"][0].split("/")[0]
        H, W        = vid["height"], vid["width"]

        track_ids = sorted({ann["id"] for ann in anns_by_vid[vid_id]})
        id_map = {g_id: l_id for l_id, g_id in enumerate(track_ids)}

        out_dir = os.path.join(out_root, video_name)
        os.makedirs(out_dir, exist_ok=True)
        instance_cnt_list = []
        for f_idx, fname in enumerate(vid["file_names"]):
            frame_path = os.path.join(img_root, fname)
            img_bgr    = read_image(frame_path, format="BGR")
            masks, classes, obj_ids = [], [], []  
            for trk in anns_by_vid.get(vid_id, []):
                seg = trk["segmentations"][f_idx]

                if not seg:          
                    continue
                
                if isinstance(seg, list):
                    rle = mask_utils.merge(mask_utils.frPyObjects(seg, H, W))
                else:
                    rle = seg
                mask = mask_utils.decode(rle).astype(bool)  
                masks.append(torch.from_numpy(mask))
                classes.append(trk["category_id"])
                obj_ids.append(id_map[trk["id"]])

            instance_cnt = len(masks)
            instance_cnt_list.append(instance_cnt)
            if instance_cnt == 0:      
                cv2.imwrite(os.path.join(out_dir, os.path.basename(fname)), img_bgr)      
                print(f"[{video_name}] frame {f_idx+1}/{len(vid['file_names'])} saved (no instances)")
                continue
            
            ins = Instances((H, W))
            ins.pred_masks = torch.stack(masks, dim=0)
            contig_classes = [id2contig[cid] for cid in classes]
            ins.pred_classes = torch.tensor(contig_classes, dtype=torch.int64)
            ins.ID           = torch.tensor(obj_ids, dtype=torch.int64)
            
            
            vis = TrackVisualizer(img_bgr[:, :, ::-1], meta).draw_with_id(ins)
            vis_img = vis.get_image()[:, :, ::-1]             

            cv2.imwrite(os.path.join(out_dir, os.path.basename(fname)), vis_img)
            print(f"[{video_name}] frame {f_idx+1}/{len(vid['file_names'])} saved")
        
        csv_path = os.path.join(std_root, f"{video_name}.csv")
        ins_df = pd.DataFrame({'instance_cnt': instance_cnt_list})
        ins_df.to_csv(csv_path, index=False)
        print(f"instance counts saved → {csv_path}")
    

    print(f"\n GT visualization saved to  {out_root}")





if __name__ == "__main__":
    args = get_parser().parse_args()

    avis = load_avis_dataset(args.gt_folder, args.gt_file)   
    meta, id2contig = register_metadata(avis)

    visualize_gt_with_trackvis(avis, args.img_root, args.out_root, meta, id2contig, args.std_root)
