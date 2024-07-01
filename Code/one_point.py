from ultralytics import YOLO
import torch

import cv2

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class OnePoint:
                                    
    def __init__(self):

        self.model = YOLO('/usr/local/ev_sdk/model/1point.pt')
        #self.model = YOLO('/project/train/src_repo/models/1point.pt')

        
        self.model.to(device)




    def get_dim(lst):
        if isinstance(lst, list):
                return [len(lst)] + get_dim(lst[0])
        else:
            return []
    #     return input

    def predict_text(self, im):
        results = self.model(im)
        num_bbox = len(results[0].boxes.cls)
        print('预测出 {} 个框'.format(num_bbox))
        print("每个框的置信度",results[0].boxes.conf)
        results[0].boxes.xyxy# 每个框的：左上角XY坐标、右下角XY坐标
        # 转成整数的 numpy array
        bboxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype('uint32')
        # print(bboxes_xyxy)
        # 每个框，每个关键点的 XY坐标 置信度
        # results[0].keypoints.shape
        bboxes_keypoints = results[0].keypoints.cpu().numpy().astype('uint32')
        # print(self.get_dim(bboxes_keypoints))
        # print(bboxes_keypoints)
        graduations=[]
        gra_points=[]
        for idx in range(num_bbox): # 遍历每个框
            
            # 获取该框坐标
            bbox_xyxy = bboxes_xyxy[idx] 
            
            # 获取框的预测类别（对于关键点检测，只有一个类别）
            # bbox_label = results[0].names[0]
            graduations.append(bbox_xyxy)
            
            bbox_keypoint = bboxes_keypoints[idx] # 该框所有关键点坐标和置信度
            gra_points.append(bbox_keypoint)

            # print("bbox_keypoint",bbox_keypoint)
            # print(gra_points[0][0][0],gra_points[0][0][1])
        return graduations,gra_points
        # centerx=-1
        # centery=-1
        # pointx=-1
        # pointy=-1
        # max_score_box=[]


        # [height, width, _] = im.shape
        # scale_h = height / 640
        # scale_w = width/640


        # img = self.resize_norm_img(im)
        # ch=640
        # cw=640
        # # print(h,ch,w,cw)
        # input_name = self.sess.get_inputs()[0].name
        # output = self.sess.run([], {input_name: img})
        # x=output[0]
        # outputs = np.array([cv2.transpose(x[0])])
        # rows = outputs.shape[1]
        # boxes = []
        # scores = []
        # class_ids = []
        # kpts=[]
        # for i in range(rows):
        #     classes_scores = outputs[0][i][4]
        #     kpt=outputs[0][i][5:]
            
        #     if classes_scores >= 0.25:
        #         box = [
        #             outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
        #             outputs[0][i][2]+outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][3]+outputs[0][i][1] - (0.5 * outputs[0][i][3])]
        #         boxes.append(box)
        #         kpts.append(kpt)
        #         scores.append(classes_scores)
        # print("大于置信度的检测框数量",len(boxes))
        # if len(boxes)>0:
        #     result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)
        #     print("nms筛选后的数量",len(result_boxes))
        #     max_score_box=boxes[result_boxes[0]]
        #     print(max_score_box)
        #     print(scores[result_boxes[0]])
        #     p1 = kpts[result_boxes[0]][:2]  # 第一个点的坐标
        #     p2 = kpts[result_boxes[0]][3:5]  # 第二个点的坐标
        #     centerx=int(p1[0]*scale_w)
        #     centery=int(p1[1]*scale_h)
        #     pointx=int(p2[0]*scale_w)
        #     pointy=int(p2[1]*scale_h)
        #     return centerx,centery,pointx,pointy,max_score_box

        # return centerx,centery,pointx,pointy, max_score_box
if __name__ == '__main__':
    """Test python api,注意修改图片路径
    """
    img = cv2.imread("/project/train/src_repo/edgeai-yolov5/images/ZDSappearance20230404_V2_sample_office_1_76.jpg")
    
    predictor = OnePoint()
    # result = process_image(predictor, img)
    graduations,gra_points=predictor.predict_text(img)
    for i in range(len(graduations)):
        
        print(graduations[i][0],graduations[i][1],graduations[i][2],graduations[i][3])
        print(gra_points[i][0][0],gra_points[i][0][1])
        # print(result)
