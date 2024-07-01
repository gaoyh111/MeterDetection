import onnxruntime
import numpy as np
import cv2
import cv2.dnn
import math
import os
import torch
import torchvision
current_dir = os.path.dirname(os.path.abspath(__file__))
from ultralytics import YOLO
import torch


class PointRecognizer:
                            
    def __init__(self):
        self.sess = onnxruntime.InferenceSession('/usr/local/ev_sdk/model/2point.onnx',None)
        #self.sess = onnxruntime.InferenceSession('/project/train/src_repo/models/2point.onnx',None)


    def resize_norm_img(self, img):
    
        h, w = img.shape[:2]
        img = img[:, :, ::-1]
        ch,cw=img.shape[:2]
        img = cv2.resize(img, (640,640), interpolation=cv2.INTER_LINEAR)
        img_mean=127.5
        img_scale=1/127.5
        img = (img - img_mean) * img_scale
        img = np.asarray(img, dtype=np.float32)
        img = np.expand_dims(img,0)
        input = img.transpose(0,3,1,2)


        return input

    def predict_text(self, im):
        centerx=-1
        centery=-1
        pointx=-1
        pointy=-1
        max_score_box=[]


        [height, width, _] = im.shape
        scale_h = height / 640
        scale_w = width/640


        img = self.resize_norm_img(im)
        ch=640
        cw=640
        # print(h,ch,w,cw)
        input_name = self.sess.get_inputs()[0].name
        output = self.sess.run([], {input_name: img})
        x=output[0]
        outputs = np.array([cv2.transpose(x[0])])
        rows = outputs.shape[1]
        boxes = []
        scores = []
        class_ids = []
        kpts=[]
        for i in range(rows):
            classes_scores = outputs[0][i][4]
            kpt=outputs[0][i][5:]
            
            if classes_scores >= 0.25:
                box = [
                    outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                    outputs[0][i][2]+outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][3]+outputs[0][i][1] - (0.5 * outputs[0][i][3])]
                boxes.append(box)
                kpts.append(kpt)
                scores.append(classes_scores)
        # print("大于置信度的检测框数量",len(boxes))
        if len(boxes)>0:
            result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)
            # print("nms筛选后的数量",len(result_boxes))
            max_score_box=boxes[result_boxes[0]]
            # print(max_score_box)
            # print(scores[result_boxes[0]])
            p1 = kpts[result_boxes[0]][:2]  # 第一个点的坐标
            p2 = kpts[result_boxes[0]][3:5]  # 第二个点的坐标
            centerx=float(p1[0]*scale_w)
            centery=float(p1[1]*scale_h)
            pointx=float(p2[0]*scale_w)
            pointy=float(p2[1]*scale_h)
            print("x1,y1,x2,y2",centerx,centery,pointx,pointy)
            return centerx,centery,pointx,pointy,max_score_box

        return centerx,centery,pointx,pointy, max_score_box
if __name__ == '__main__':
    """Test python api,注意修改图片路径
    """
    img = cv2.imread("/project/train/src_repo/edgeai-yolov5/images/ZDSappearance20230404_V2_sample_office_1_76.jpg")
    
    predictor = PointRecognizer()
    # result = process_image(predictor, img)
    predictor.predict_text(img)
    # print(result)
