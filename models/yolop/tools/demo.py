import argparse
import os, sys
import shutil
import time
from pathlib import Path
import imageio
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import scipy.special
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm

# --- 1. NEW IMPORT: Ultralytics for YOLOv8n ---
# We use this to detect Humans, Cattle, and Dogs because the base YOLOP model
# is only good at detecting Cars.
from ultralytics import YOLO

# --- 2. LOAD AUXILIARY MODEL ---
# We load this ONCE at the start (Global Scope) so we don't reload it every frame.
# 'yolov8n.pt' is the "Nano" version: smallest, fastest, but less accurate than 'Large'.
print("Loading Object Detector (YOLOv8n)...")
object_model = YOLO('yolov8n.pt') 

# Setup Local Imports from the 'lib' folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from lib.config import cfg
from lib.config import update_config
from lib.utils.utils import create_logger, select_device, time_synchronized
from lib.models import get_net
from lib.dataset import LoadImages, LoadStreams
from lib.core.general import non_max_suppression, scale_coords
from lib.utils import plot_one_box, show_seg_result
from lib.core.function import AverageMeter
from lib.core.postprocess import morphological_process, connect_lane

# --- IMPORT OUR CUSTOM SAFETY ENGINE ---
# This is the "Brain" of the project that handles the Green Carpet and Logic.
from modules.safety import SafetyManager 

# Image Normalization Constants (Standard ImageNet mean/std)
# This scales pixel values so the AI can understand them better.
normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

def detect(cfg, opt):
    # Setup Logging and Device (CPU vs GPU)
    logger, _, _ = create_logger(cfg, cfg.LOG_DIR, 'demo')
    device = select_device(logger, opt.device)
    
    # --- OUTPUT SETUP ---
    # We create a unique timestamped folder for every run.
    # This prevents us from overwriting old experiments.
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(opt.save_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True) 
    opt.save_dir = save_dir 

    # --- MAIN MODEL LOADING (YOLOP) ---
    # YOLOP is a "Multi-Task" Network: It does 3 things at once:
    # 1. Object Detection (Cars)
    # 2. Drivable Area Segmentation (Where is the road?)
    # 3. Lane Line Segmentation (Where are the white lines?)
    half = device.type != 'cpu' 
    model = get_net(cfg)
    weights = opt.weights[0] if isinstance(opt.weights, list) else opt.weights
    checkpoint = torch.load(weights, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    if half:
        model.half() # Use FP16 (Half Precision) to speed up inference on GPU

    # --- DATASET LOADING ---
    if opt.source.isnumeric():
        cudnn.benchmark = True
        dataset = LoadStreams(opt.source, img_size=opt.img_size)
        bs = len(dataset)
    else:
        dataset = LoadImages(opt.source, img_size=opt.img_size)
        bs = 1

    names = model.module.names if hasattr(model, 'module') else model.names
    
    # --- INITIALIZE SAFETY MANAGER ---
    # We set fps=30.0 because standard dashcam footage is 30fps.
    # This value is crucial for the Time-To-Collision (TTC) math.
    safety_engine = SafetyManager(fps=30.0)

    # Performance Timers
    t0 = time.time()
    vid_path, vid_writer = None, None
    inf_time = AverageMeter()
    nms_time = AverageMeter()
    
    # Run a dummy pass to "warm up" the GPU
    img = torch.zeros((1, 3, opt.img_size, opt.img_size), device=device)
    _ = model(img.half() if half else img) if device.type != 'cpu' else None
    model.eval()

    # ================================================================
    #                      MAIN VIDEO LOOP
    # ================================================================
    for i, (path, img, img_det, vid_cap, shapes) in tqdm(enumerate(dataset), total=len(dataset)):
        
        # Preprocess Image (Resize & Normalize)
        img = transform(img).to(device)
        img = img.half() if half else img.float()
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
            
        # --- 1. YOLOP INFERENCE ---
        t1 = time_synchronized()
        
        # det_out: Bounding Boxes (Cars)
        # da_seg_out: Drivable Area Mask (Green)
        # ll_seg_out: Lane Line Mask (Red)
        det_out, da_seg_out, ll_seg_out = model(img)
        
        t2 = time_synchronized()
        inf_out, _ = det_out
        inf_time.update(t2-t1, img.size(0))

        # --- 2. NON-MAX SUPPRESSION (NMS) ---
        # The AI suggests 1000s of boxes. NMS removes duplicates.
        # It keeps only the "best" box for each object.
        t3 = time_synchronized()
        det_pred = non_max_suppression(inf_out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, classes=None, agnostic=False)
        t4 = time_synchronized()
        nms_time.update(t4-t3, img.size(0))
        
        det = det_pred[0] # Get detections for this frame

        # Setup Save Path
        save_path = str(opt.save_dir + '/' + Path(path).name) if dataset.mode != 'stream' else str(opt.save_dir + '/' + "web.mp4")

        # Get Image Dimensions for Scaling
        _, _, height, width = img.shape
        h, w, _ = img_det.shape
        pad_w, pad_h = shapes[1][1]
        pad_w, pad_h = int(pad_w), int(pad_h)
        ratio = shapes[1][0][1]

        # --- 3. PROCESS SEGMENTATION MASKS ---
        # Resize the output mask back to the original image size
        
        # A. Drivable Area (Road)
        da_predict = da_seg_out[:, :, pad_h:(height-pad_h), pad_w:(width-pad_w)]
        da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=int(1/ratio), mode='bilinear')
        _, da_seg_mask = torch.max(da_seg_mask, 1)
        da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()

        # B. Lane Lines
        ll_predict = ll_seg_out[:, :, pad_h:(height-pad_h), pad_w:(width-pad_w)]
        ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=int(1/ratio), mode='bilinear')
        _, ll_seg_mask = torch.max(ll_seg_mask, 1)
        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()

        # --- OPTIMIZATION: ROI BLINDER ---
        # The top of the image usually contains sky, trees, and buildings.
        # These look like "Road Lines" to the AI, causing glitches.
        # We forcibly BLACK OUT the top 45% of the mask to ignore them.
        roi_limit = int(h * 0.45) 
        da_seg_mask[0:roi_limit, :] = 0
        ll_seg_mask[0:roi_limit, :] = 0

        # --- 4. SAFETY ENGINE: GENERATE CARPET ---
        # This function runs the Kalman Filter & Cluster Analysis
        # to create a stable, non-flickering green lane.
        safety_engine.generate_dynamic_carpet(da_seg_mask, ll_seg_mask, h, w)

        final_status_text = "PATH CLEAR"
        final_color = (0, 255, 0)

        # =============================================================
        #      NEW FEATURE: CATTLE & HUMAN DETECTION (Time Sliced)
        # =============================================================
        
        # Initialize the 'memory' list if it doesn't exist
        if 'detected_obstacles' not in locals():
            detected_obstacles = []

        # We run the heavy YOLOv8n model only once every 5 frames.
        # This keeps the FPS high while still detecting slow-moving animals.
        if i % 5 == 0:
            # We use the raw original image (img_det)
            # Classes: 0=Person, 15=Cat, 16=Dog, 17=Horse, 18=Sheep, 19=Cow
            results_yolo = object_model(img_det, classes=[0, 15, 16, 17, 18, 19], conf=0.45, verbose=False)
            
            detected_obstacles = [] # Reset list
            for res in results_yolo:
                boxes = res.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    cls_id = int(box.cls[0])
                    label = object_model.names[cls_id]
                    detected_obstacles.append((x1, y1, x2, y2, label))

        # --- DRAW OBSTACLES (Every Frame) ---
        # Even if we didn't run the model this frame, we draw the old boxes (Persistence)
        for (x1, y1, x2, y2, label) in detected_obstacles:
            # Color Logic: Red for Animals, Blue for Humans
            color = (0, 0, 255) if label in ['cow', 'horse', 'sheep', 'dog'] else (255, 0, 0)
            
            cv2.rectangle(img_det, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_det, label.upper(), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # HAZARD CHECK: Is the Cow inside the Green Carpet?
            if safety_engine.tunnel_poly is not None:
                cx_feet = (x1 + x2) // 2
                # Check pointPolygonTest on feet location (y2)
                dist = cv2.pointPolygonTest(safety_engine.tunnel_poly, (cx_feet, y2), True)
                if dist > 0: # Inside Lane!
                     cv2.putText(img_det, "OBSTACLE!", (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)


        # --- 5. SAFETY ENGINE: HAZARD CHECK (CARS) ---
        # Now we check the standard YOLOP detections (Vehicles)
        if len(det):
            # Scale boxes from 640x640 back to original resolution (e.g. 1920x1080)
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_det.shape).round()

            # Prepare detection list for the Safety Manager
            current_detections = []
            for *xyxy, conf, cls in reversed(det):
                current_detections.append({
                    'id': -1, # ID will be assigned by CentroidTracker inside safety.py
                    'box': [int(x) for x in xyxy],
                    'cls': int(cls)
                })

            # Run Math Logic (TTC, Growth Rate, Lane Position)
            results, global_col, global_txt = safety_engine.check_hazard_status(current_detections, w, h)
            
            final_status_text = global_txt
            final_color = global_col

            # Draw Car Boxes with INDIVIDUAL Safety Colors
            for j, detection in enumerate(current_detections):
                box = detection['box']
                cls_idx = detection['cls']
                
                # Get the calculated status (Safe/Warn/Crash)
                obj_result = results[j]
                obj_color = obj_result['color']
                obj_status = obj_result['status']
                
                label = f"{names[cls_idx]} | {obj_status}"
                
                # Plot box
                plot_one_box(box, img_det, label=label, color=obj_color, line_thickness=2)
        
        # --- 6. DRAW FINAL UI ---
        # Overlays the Green Carpet and the Dashboard Status Bar
        img_det = safety_engine.draw_ui(img_det, final_status_text, final_color)

        # Save or Show Frame
        if dataset.mode == 'images':
            cv2.imwrite(save_path, img_det)

        elif dataset.mode == 'video':
            if vid_path != save_path:
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()

                fourcc = 'mp4v'
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            vid_writer.write(img_det)
        
        else:
            cv2.imshow('image', img_det)
            cv2.waitKey(1)

    print('Results saved to %s' % Path(opt.save_dir))
    print('Done. (%.3fs)' % (time.time() - t0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/End-to-end.pth', help='model.pth path(s)')
    parser.add_argument('--source', type=str, default='inference/videos', help='source')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    with torch.no_grad():
        detect(cfg, opt)