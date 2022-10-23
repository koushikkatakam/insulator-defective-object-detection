import numpy as np
import os
import systemcheck
import cv2
import torch

from models.common import DetectMultiBackend
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import Annotator, colors
from utils.torch_utils import select_device


def get_model(weights,  # model.pt path(s)
        imgsz=640,  # inference size (pixels)
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):

    
    ###################### Load Model for Detection  #########################################  
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit = model.stride, model.names, model.pt, model.jit
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    print("Names:", names)

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup

    ###################### Model ready for Detection  #########################################  
    model_yolo = (device,model,stride, names, pt, jit, imgsz, half )
    print("Model ready for Detection")
    return model_yolo


def detect(model_yolo, 
        source, 
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        visualize = False,
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        conf_thres=0.30,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=100,  # maximum detections per image
        ):

    device,model,stride, names, pt, jit, imgsz, half = model_yolo
     # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)

    for path, im, raw, vid_cap, s in dataset:
    
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        pred = model(im)
       
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, max_det=max_det)

        # Process predictions
        
        for i, det in enumerate(pred):  # per image
            
            save_path = os.getcwd()+"/"+source  # im.jpg
            annotator = Annotator(raw, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to raw size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], raw.shape).round()
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))
                  
            annotated_image = annotator.result()
        
    return annotated_image
            
                

if __name__ == '__main__':

    ls = os.listdir("testframes")
    ls.sort()

    model_yolo = get_model("bestv5l.pt")
    video = list()

  # ls = ("testframes\0348.png")
   

    for img in ls[:]:
        if ".jpg" in img or ".jpeg" in img or ".png" in img: 
            print("Processing:", img)
            image = detect(model_yolo, source = f"testframes/{img}")
            video.append(image)

            cv2.imshow("YOLO Output", image)
            cv2.waitKey(2000)
    
    print("Streaming Output Video Frames")
    for frame in video:
        cv2.imshow("YOLO Output", frame)
        cv2.waitKey(20000)
        