import json
import logging
import os
import onnxruntime
import cv2
import numpy as np
import torch
import torchvision
import tensorrt as trt
from collections import OrderedDict, namedtuple
from pathlib import Path
import time
import math
import copy 
from point_det import PointRecognizer
from one_point import OnePoint
from mmocr.apis import TextDetInferencer,TextRecInferencer
from get_out_point_cv import get_out_point_opencv
os.environ["KMP_DUPLICATE_LIB_OK"] = 'TRUE'
from collections import Counter   
logger = trt.Logger(trt.Logger.INFO)
LOGGER = logging.getLogger("yolo")


class TextRec:
    def __init__(self):
                
        self.model = TextRecInferencer(model='/project/train/src_repo/mmocr/configs/textrecog/satrn/my_satrn_shallow_5e_st_mj.py',
                # weights='/project/train/src_repo/mmocr/checkpoints/satrn_shallow_5e_st_mj.pth',device='cuda:0')
                weights='/usr/local/ev_sdk/model/my_satrn_shallow_5e_st_mj/best_recog_word_acc_epoch_10.pth',device='cuda:0')

    def recog_text(self,img):
        pred=self.model(img)
        return pred
    
    def text_postprocess(self,text):
        text=text.replace(",", ".")
        text=text.replace("o", "0")
        text=text.replace("O", "0")
        if text.startswith('-'):
            if text != '-0.1':
                text=text[1:len(text)]
        if text.startswith('.'):
            text=text[1:len(text)]
        return text

    def is_number(self,s):    
    
        try:    # 如果能运⾏ float(s) 语句，返回 True（字符串 s 是浮点数）        
            float(s)
            if '.' not in str(s):
                int(s)

            return True    
        except ValueError:  # ValueError 为 Python 的⼀种标准异常，表⽰"传⼊⽆效的参数"        
            pass  # 如果引发了 ValueError 这种异常，不做任何事情（pass：不做任何事情，⼀般⽤做占位语句）    
        try:        
            import unicodedata  # 处理 ASCII 码的包        
            unicodedata.numeric(s)  # 把⼀个表⽰数字的字符串转换为浮点数返回的函数        
            return True    
        except (TypeError, ValueError):        
            pass    
            return False   





    def re_rec(self,polygon,img,k,pre_num):
        crop=PerspectiveTransform(polygon,img)                
        pred1=self.recog_text(crop)
        pred1_text=pred1['predictions'][0]['text']
        pred1_text=self.text_postprocess(pred1_text)
        if self.is_number(pred1_text):
            if float(pred1_text)-pre_num<=k and float(pred1_text)-pre_num>=0:
                            
                if '.' in str(pred1_text):
                    pred1_text=float(pred1_text)     
                else:
                    pred1_text=int(pred1_text)  
                return pred1_text
        # -90
        rot1 = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        pred1=self.recog_text(rot1)
        pred1_text=pred1['predictions'][0]['text']
        pred1_text=self.text_postprocess(pred1_text)
        if self.is_number(pred1_text):
            if float(pred1_text)-pre_num<=k and float(pred1_text)-pre_num>=0:
                            
                if '.' in str(pred1_text):
                    pred1_text=float(pred1_text)     
                else:
                    pred1_text=int(pred1_text)  
                return pred1_text
        # -180
        rot2 = cv2.rotate(rot1, cv2.ROTATE_90_CLOCKWISE)
        pred1=self.recog_text(rot2)
        pred1_text=pred1['predictions'][0]['text']
        pred1_text=self.text_postprocess(pred1_text)
        if self.is_number(pred1_text):
            if float(pred1_text)-pre_num<=k and float(pred1_text)-pre_num>=0:
                                    
                if '.' in str(pred1_text):
                    pred1_text=float(pred1_text)     
                else:
                    pred1_text=int(pred1_text)  
                return pred1_text
        # -270
        rot3 = cv2.rotate(rot2, cv2.ROTATE_90_CLOCKWISE)
        pred1=self.recog_text(rot3)
        pred1_text=pred1['predictions'][0]['text']
        pred1_text=self.text_postprocess(pred1_text)
        if self.is_number(pred1_text):
            if float(pred1_text)-pre_num<=k and float(pred1_text)-pre_num>=0:
                                
                if '.' in str(pred1_text):
                    pred1_text=float(pred1_text)     
                else:
                    pred1_text=int(pred1_text)  
                return pred1_text
        return -1

            
    def recog4direc(self,img):
        digit=[]
        
        # 0
        pred1=self.recog_text(img)
        pred1_text=pred1['predictions'][0]['text']
        pred1_text=self.text_postprocess(pred1_text)
        
        pred1['predictions'][0].update(text=pred1_text)

        print(pred1_text)
        if self.is_number(pred1_text):
            digit.append(pred1)
            scores=pred1['predictions'][0]['scores']
            print(f'{pred1_text}:{scores}')
        
        # -90
        rot1 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        pred2=self.recog_text(rot1)
        pred2_text=pred2['predictions'][0]['text']
        pred2_text=self.text_postprocess(pred2_text)
        pred2['predictions'][0].update(text=pred2_text)

        print(pred2_text)
        if self.is_number(pred2_text):
            digit.append(pred2)
            scores=pred2['predictions'][0]['scores']
            print(f'{pred2_text}:{scores}')
        
        # -180
        rot2 = cv2.rotate(rot1, cv2.ROTATE_90_CLOCKWISE)
        pred3=self.recog_text(rot2)
        pred3_text=pred3['predictions'][0]['text']
        pred3_text=self.text_postprocess(pred3_text)
        pred3['predictions'][0].update(text=pred3_text)

        print(pred3_text)
        if self.is_number(pred3_text):
            digit.append(pred3)
            scores=pred3['predictions'][0]['scores']
            print(f'{pred3_text}:{scores}')
        
        # -270
        rot3 = cv2.rotate(rot2, cv2.ROTATE_90_CLOCKWISE)
        pred4=self.recog_text(rot3)
        pred4_text=pred4['predictions'][0]['text']
        pred4_text=self.text_postprocess(pred4_text)
        pred4['predictions'][0].update(text=pred4_text)

        print(pred4_text)
        if self.is_number(pred4_text):
            digit.append(pred4)
            scores=pred4['predictions'][0]['scores']
            print(f'{pred4_text}:{scores}')
        max_score=0.0
        if len(digit)==0:
            return f'NaN',max_score
    
        pred_final_text=[]
        for i in range(len(digit)):
            if digit[i]['predictions'][0]['scores'] > max_score:
                max_score=digit[i]['predictions'][0]['scores']
                pred_final_text=digit[i]['predictions'][0]['text']
        if pred_final_text==[]:
            return f'NaN',0
        return pred_final_text,max_score

def correct_nan(nums):
    is_float=False
    if nums[0]=='6':
        nums[0]=0
    for i in range(len(nums)):
        if '.' in str(nums[i]):
            is_float=True
        if nums[i]=='9':
            nums[i]=6
        if nums[i]=='09':
            nums[i]=60
    if is_float:
        for i in range(len(nums)):
            if nums[i]=='NaN':
                nums[i]=-1
            else:
                nums[i]=float(nums[i])
    else:
        for i in range(len(nums)):
            if nums[i]=='NaN':
                nums[i]=-1
            else:
                if len(str(nums[i]))>1 and int(nums[i])==0:
                    nums[i]=-1
                else:
                    nums[i]=int(nums[i])
    return nums

# def correct_int_nums(nums):#改最后一个点
#     is_int=False
#     for i in range(len(nums)):
#         if isinstance(nums[i],int):
#             is_int=True
#             break
    
#     if(is_int):
#         nums = [round(x) for x in nums]

#         # 处理转换后的数列，确保其仍然保持递增
#         for i in range(1, len(nums)):
#             if nums[i] <= nums[i-1]:
#                     nums[i] = nums[i-1] + 1
 
#     return nums

def correct_float_nums(nums):#3
    """遍历整个数字列表，检查其中是否包含浮点数。

    如果有浮点数，将列表中的所有浮点数保留三位小数。
    如果某个浮点数的小数部分全部为0，则将其转换为整数0。

    返回处理后的数字列表。

    需要注意的是，这段代码可能会修改输入的数字列表。
    如果在程序的后续部分中需要使用原始数字列表，则应在调用该函数之前创建一个副本。
    """
    is_float=False
    for i in range(len(nums)):
        if '.' in str(nums[i]):
            is_float=True
    if is_float:
        for i in range(len(nums)):
            if str(nums[i])=='0.0':
                 nums[i]=int(0)   
            nums[i]=round(nums[i],3)
    return nums
# 过点集（刻度序号，刻度读数）最多的直线
def getLine( pointlist):
                                    
    # write code here
    max_line = 0
    max_slope = None
    max_intercept = None
    slope_dict = {}
    intercept_dict ={}
    for i in range(len(pointlist)-1):
        for j in range(i+1, len(pointlist)):
            k = 1.0 * (pointlist[j][1] - pointlist[i][1]) / (pointlist[j][0] - pointlist[i][0])
            b=pointlist[i][1] -k*pointlist[i][0]
            if k in slope_dict:
                slope_dict[k] += 1
            else:
                slope_dict[k] = 1
                intercept_dict[k]=b
    for item in slope_dict:
        if slope_dict[item] > max_line:
            max_line = slope_dict[item]
            max_slope = item
            max_intercept = intercept_dict[item]
    return max_slope, max_intercept


def find_pointer(resorted_polygons,resorted_angle):
    pointer_angle=0.0
    pointer_index=0
    for i in range(len(resorted_polygons)):
        if(np.array_equal(resorted_polygons[i], [[0, 0], [1, 0], [1, 1], [0, 1]])):
            pointer_angle=resorted_angle[i]
            pointer_index=i
    return pointer_index,pointer_angle

def is_occlusion(resorted_angle,resorted_polygons):
    
    diff_angle1=[]
    diff_angle2=[]
    # 0:无遮挡，1：遮挡无框，2：遮挡有框
    occlusion=0
    pointer_index,pointer_angle=find_pointer(resorted_polygons,resorted_angle)

    for i in range(len(resorted_angle)-1):
        diff=resorted_angle[i+1]-resorted_angle[i]
        if diff<0:
                diff+=360
        diff_angle1.append(diff)
    max_index = diff_angle1.index(max(diff_angle1))
    if max_index!=pointer_index and max_index!=pointer_index+1:
        diff_angle1[max_index]=np.mean(diff_angle1)
    max_1=max(diff_angle1)
    min_1=min(diff_angle1)
    
    # remove pointer
    for i in range(len(resorted_angle)-1):
        
        if(i==pointer_index-1 and pointer_index!= len(resorted_angle)-1):
                                
            diff=resorted_angle[i+2]-resorted_angle[i]
            if(diff<0):
                diff+=360
            diff_angle2.append(diff)
        elif(i==pointer_index-1 and pointer_index== len(resorted_angle)-1):
                                                        
            continue
        elif(i==pointer_index or i==pointer_index+1 ):
            continue
        else:
            diff=resorted_angle[i+1]-resorted_angle[i]
            if(diff<0):
                diff+=360
            diff_angle2.append(diff)
    max_index = diff_angle2.index(max(diff_angle2))
    if max_index!=pointer_index and max_index!=pointer_index-1:
        diff_angle2[max_index]=np.mean(diff_angle2)
    max_2=max(diff_angle2)
    min_2=min(diff_angle2)

    if(max_1!=max_2 and max_2>1.6*max_1):
            occlusion=1
    if (min_1!=min_2 and min_1<5):
        occlusion=2

    return occlusion

def get_pointer_num(nums_list,resorted_angle,pointer_index):
                                                                                                
    if pointer_index==0:
        diff_angle1=resorted_angle[1]-resorted_angle[0]
        diff_angle2=resorted_angle[2]-resorted_angle[1]
        if diff_angle1<0:
            diff_angle1+=360
        if diff_angle2<0:
            diff_angle2+=360
        nums_list[0]=round(nums_list[1]-(diff_angle1/diff_angle2)*(nums_list[2]-nums_list[1]),3)
    elif pointer_index==1 and len(resorted_angle)>=4:
        diff_angle1=resorted_angle[2]-resorted_angle[1]
        diff_angle2=resorted_angle[3]-resorted_angle[2]    
        if diff_angle1<0:
            diff_angle1+=360
        if diff_angle2<0:
            diff_angle2+=360
        nums_list[1]=round(nums_list[2]-(diff_angle1/diff_angle2)*(nums_list[3]-nums_list[2]),3)
    elif pointer_index==len(resorted_angle)-1:
       nums_list[pointer_index]=nums_list[pointer_index-1]
    else:
        diff_angle1=resorted_angle[pointer_index]-resorted_angle[pointer_index-1]
        diff_angle2=resorted_angle[pointer_index+1]-resorted_angle[pointer_index]
        if diff_angle1<0:
            diff_angle1+=360
        if diff_angle2<0:
            diff_angle2+=360
        nums_list[pointer_index]=round((diff_angle1*nums_list[pointer_index+1]+diff_angle2*nums_list[pointer_index-1])/(diff_angle1+diff_angle2),3)

# 处理遮挡无框情况
def process_occlusion(nums,resorted_angle,resorted_polygons,resorted_points,pointer_index,pointer_angle,center_x,center_y,img):
    diffs = [resorted_angle[i + 1] - resorted_angle[i] for i in range(len(resorted_angle) - 1)]      
    diffs = [diff + 360 if diff < 0 else diff for diff in diffs]
    median = np.median(diffs)
    fake_polygon=[[0, 0], [10, 0], [10, 10], [0, 10]]                                              
    nums=correct_nan(nums)
    index_list=range(len(nums))
    pointlist = [list(x) for x in zip(index_list, nums)]
    (k,b)=getLine(pointlist)
    add_cnt=0
    # 将不在直线上的点更改为直线上的对应值
    for i in range(len(nums)):
        y_pred = k * (i-add_cnt) + b
        if '.' in str(y_pred):
            y_pred=round(y_pred,2)
        if y_pred==round(y_pred):
            y_pred=int(y_pred)           
        if nums[i] != y_pred :
            if(i==0 and nums[i]!=0 and nums[i]>b and nums[i]<b+k):#45
                continue  
            if (nums[i]-nums[i-1])<k and (nums[i]-nums[i-1])>0 and i==len(nums)-1:#最后一个不一定是等差
                continue
            if i==0 and nums[i]==0:#
                # if diffs[i]>1.5*median:#漏点了
                    
                #     nums.insert(i+1,y_pred)#读数
                #     # resorted_angle.insert(pointer_index,occlusion_point_angle)
                #     diffs.insert(i+1,diffs[i]/2)
                #     resorted_polygons.insert(i+1,fake_polygon)
                #     add_cnt+=1
                continue


                # continue     
            elif i>0 and nums[i]==0 or nums[i]==-1:
                nums[i] = y_pred
            else:
                #重新检测这个框框
                res=TextRecognizer.re_rec(resorted_polygons[i],img,k,nums[i-1])
                if res!=-1 and res!=nums[i]:
                    nums[i]=res
                else:
                    nums[i] = y_pred
    # correct_irregularNums(nums,resorted_angle,pointer_index,1)
    # nums=correct_int_nums(nums)#4.0->4
    if pointer_index!=len(resorted_angle)-1:
    
        diff=resorted_angle[pointer_index+1]-resorted_angle[pointer_index-1]

        
        
        if(diff<0):
            diff+=360
        occlusion_point_angle=resorted_angle[pointer_index-1]+diff/2
        if occlusion_point_angle>=360:
            occlusion_point_angle-=360

        fake_polygon=[[0, 0], [10, 0], [10, 10], [0, 10]]

        mag1, theta1 = cv2.cartToPolar(float(resorted_points[pointer_index+1][0])-center_x, float(resorted_points[pointer_index+1][1])-center_y, angleInDegrees=1)
        mag2, theta2 = cv2.cartToPolar(float(resorted_points[pointer_index-1][0])-center_x, float(resorted_points[pointer_index-1][1])-center_y, angleInDegrees=1)
        occlusion_point_mag=(mag1+mag2)/2
        occlusion_x, occlusion_y = cv2.polarToCart(occlusion_point_mag, occlusion_point_angle, angleInDegrees=1)
        if occlusion_point_angle>=pointer_angle:
            # 指针在遮挡读数前
            nums.insert(pointer_index,0)
            resorted_angle.insert(pointer_index+1,occlusion_point_angle)
            resorted_polygons.insert(pointer_index+1,fake_polygon)
            resorted_points.insert(pointer_index+1,(int(occlusion_x[0][0]+center_x), int(occlusion_y[0][0]+center_y)))
            get_pointer_num(nums,resorted_angle,pointer_index)
        else:
            # 指针在遮挡读数后
            nums.insert(pointer_index+1,0)#读数
            resorted_angle.insert(pointer_index,occlusion_point_angle)
            resorted_polygons.insert(pointer_index,fake_polygon)
            resorted_points.insert(pointer_index,(occlusion_x[0][0]+center_x, occlusion_y[0][0]+center_y))
            get_pointer_num(nums,resorted_angle,pointer_index+1)
        idx,ang=find_pointer(resorted_polygons,resorted_angle)
        temppolygons=copy.deepcopy(resorted_polygons)
        del temppolygons[idx]
        return nums,temppolygons
    else:
        fake_polygon=[[0, 0], [10, 0], [10, 10], [0, 10]]
        nums.insert(pointer_index+1,nums[pointer_index])
        mag1, theta1 = cv2.cartToPolar(float(resorted_points[pointer_index-2][0])-center_x, float(resorted_points[pointer_index-2][1])-center_y, angleInDegrees=1)
        mag2, theta2 = cv2.cartToPolar(float(resorted_points[pointer_index-1][0])-center_x, float(resorted_points[pointer_index-1][1])-center_y, angleInDegrees=1)
        occlusion_point_angle=resorted_angle[pointer_index]
        occlusion_point_mag=(mag1+mag2)/2
        occlusion_x, occlusion_y = cv2.polarToCart(occlusion_point_mag, occlusion_point_angle, angleInDegrees=1)
        resorted_points.insert(pointer_index,(occlusion_x[0][0]+center_x, occlusion_y[0][0]+center_y))
        resorted_polygons.insert(pointer_index,fake_polygon)
        resorted_angle.insert(pointer_index,occlusion_point_angle)
        get_pointer_num(nums,resorted_angle,pointer_index+1)
        idx,ang=find_pointer(resorted_polygons,resorted_angle)
        temppolygons=copy.deepcopy(resorted_polygons)
        del temppolygons[idx]
        return nums,temppolygons

# # 检查非递增数字
# def correct_irregularNums(nums,resorted_angle,pointer_index,occlusion):
#     """
#     如果第一个或最后一个差异角度小于平均值的0.7倍，
#     那么使用线性插值的方法来修正相应的数字，以使它们满足预期的角度序列。
#     """
#     diff_angle=[]
#     grad_angle=np.array(resorted_angle)
#     if(occlusion in {0,2}):
#         grad_angle=np.delete(grad_angle, pointer_index)

#     for i in range(len(grad_angle)-1):
#         diff=grad_angle[i+1]-grad_angle[i]
#         if diff<0:
#             diff+=360
#         diff_angle.append(diff)
#     mean_angle=np.mean(diff_angle)

#     for i in range(len(diff_angle)):
#         if(i==0 and diff_angle[i]<mean_angle*0.7):
#             diff_angle1=diff_angle[0]
#             diff_angle2=diff_angle[1]
#             nums[0]=nums[1]-(diff_angle1/diff_angle2)*(nums[2]-nums[1])
#         elif(i==len(diff_angle)-1 and diff_angle[i]<mean_angle*0.7):
#             diff_angle1=diff_angle[i-1]
#             diff_angle2=diff_angle[i]
#             nums[-1]=nums[-2]+(diff_angle2/diff_angle1)*(nums[-2]-nums[-3])
#         else:
#             continue
def process_occlusion_woutP(nums,resorted_angle,resorted_polygons,pointer_index,img):
            


    resorted_polygons_nopointer=copy.deepcopy(resorted_polygons)
    resorted_angle_nopointer=copy.deepcopy(resorted_angle)
    del resorted_angle_nopointer[pointer_index]  
    del resorted_polygons_nopointer[pointer_index]                            
    diffs = [resorted_angle_nopointer[i + 1] - resorted_angle_nopointer[i] for i in range(len(resorted_angle_nopointer) - 1)]      
    diffs = [diff + 360 if diff < 0 else diff for diff in diffs]
    median = np.median(diffs)
    fake_polygon=[[0, 0], [10, 0], [10, 10], [0, 10]]


    nums=correct_nan(nums)#NAN转成-1
    index_list=range(len(nums))
    pointlist = [list(x) for x in zip(index_list, nums)]
    k,b=getLine(pointlist)
    add_cnt=0
    for i in range(len(nums)):
        y_pred = k * (i-add_cnt) + b
        if '.' in str(y_pred):
            y_pred=round(y_pred,2)
        if y_pred==round(y_pred):
            y_pred=int(y_pred)        
        if nums[i] != y_pred :  
            if (nums[i]-nums[i-1])<k and (nums[i]-nums[i-1])>0 and i==len(nums)-1:#最后一个数不一定等差
                continue
            
            if i==0 and nums[i]==0:
                if diffs[i]>1.5*median:#漏点了
                    
                    nums.insert(i+1,y_pred)#读数
                    # resorted_angle.insert(pointer_index,occlusion_point_angle)
                    diffs.insert(i+1,diffs[i]/2)
                    resorted_polygons_nopointer.insert(i+1,fake_polygon)
                    add_cnt+=1
                    continue                                                        
                                                                                
                # if nums[i]!=0 and diffs[i]>1.5*median
                
            elif i>0 and nums[i]==0 or nums[i]==-1:#指针
                nums[i] = y_pred
            else:
                #重新检测这个框框
                res=TextRecognizer.re_rec(resorted_polygons_nopointer[i],img,k,nums[i-1])
                if res!=-1 and res!=nums[i]:
                                                                
                    nums[i]=res
                else:
                    nums[i] = y_pred
                                                                        
    nums.insert(pointer_index,0)
    get_pointer_num(nums,resorted_angle,pointer_index)
    return nums,resorted_polygons_nopointer
      
    
# 纠正数列，处理遮挡、缺省




def find_closest_list(input_list, nums_list):
    if len(input_list)<2:
        return []
                    # input_list = [float(x) for x in input_list]
    input_counter = Counter(input_list)

    closest_list = None
    closest_count = 0
    max_score=0
    differences1 = [round(input_list[i+1] - input_list[i],3) for i in range(len(input_list)-1)]
    # 统计差值出现的次数
    d_counts1 = Counter(differences1)
    d1 = d_counts1.most_common(1)[0][0]
    if len(input_list)<4:d1=999
    




    for nums in nums_list:

        differences2 = [round(nums[i + 1] - nums[i],3) for i in range(len(nums) - 1)]
        # 统计差值出现的次数
        d_counts2 = Counter(differences2)
        d2 = d_counts2.most_common(1)[0][0]
        if d2*d1<0 or len(nums)<len(input_list):
            continue
        nums_counter = Counter(nums)
        common_elements = set(input_counter) & set(nums_counter)
        count = sum(min(input_counter[key], nums_counter[key]) for key in common_elements)

        score=count/len(nums)
        if score > max_score:
            max_score = score
            closest_list = nums
            if d1==d2 and len(nums_list)==len(input_list):
                                                                    
                return closest_list

    return closest_list
def is_element_majority(nums, element):
    count = 0
    for num in nums:
        if num == element:
            count += 1
    return count > len(nums) / 2



def check_list(nums_list,gt_list,resorted_angle,resorted_polygons,resorted_points,center_x,center_y,img):
    # print(f"nums_list:  {nums_list}\n")
    # print(f"resorted_angle:     {resorted_angle}\n")
    # print(f"resorted_polygons:    {resorted_polygons}\n")
    # print(f"resorted_points:    {resorted_points}\n")
    # print(f"center:   {center_x}   ,{center_y}\n")

    pointer_index,pointer_angle=find_pointer(resorted_polygons,resorted_angle)
    resorted_polygons_nopointer=copy.deepcopy(resorted_polygons)
    resorted_angle_nopointer=copy.deepcopy(resorted_angle)
    resorted_points_nopointer=copy.deepcopy(resorted_points)
    del resorted_angle_nopointer[pointer_index]  
    del resorted_polygons_nopointer[pointer_index]  
    del resorted_points_nopointer[pointer_index]  
    diffs = [resorted_angle_nopointer[i + 1] - resorted_angle_nopointer[i] for i in range(len(resorted_angle_nopointer) - 1)]      
    diffs = [diff + 360 if diff < 0 else diff for diff in diffs]
    median = np.median(diffs)
    fake_polygon=[[0, 0], [10, 0], [10, 10], [0, 10]]


    nums=correct_nan(nums_list)
    if nums[0]==0:
        nums[0]=0
    # innerList,k = getInnerList(nums)  
    innerList = find_closest_list(nums, gt_list)#找到最接近的列表
    # print("innerList",innerList)
    if innerList==None or len(innerList)==0:
        return [],[],[],[]  
    if innerList!=None:
                
        correct_nums = []

        for item in nums:#遍历nums元素，如果不在innerList列表里设为-1，这一步也是预处理，-1即是NaN，有点，但是number未知

            if item in innerList:
                index = innerList.index(item)
                correct_nums.append(innerList[index])
            else:
                correct_nums.append(-1)
        for i in range(len(correct_nums)):#含有相同元素
            if correct_nums.count(correct_nums[i]) > 1 and correct_nums[i]!=-1:
                first_index = i
                try:
                    second_index = correct_nums.index(correct_nums[i], first_index + 1)
                except ValueError:
                    break
                # second_index = correct_nums.index(correct_nums[i], first_index + 1)
                innerIdx = innerList.index(correct_nums[i])
                dist_first = abs(innerIdx - first_index)
                dist_second = abs(innerIdx - second_index)

                if dist_first < dist_second:
                    # selected_index = first_index
                    correct_nums[second_index]=-1
                else:
                    correct_nums[first_index] = -1
                    # selected_index = second_index
        count = 0
        for i in range(min(len(correct_nums),len(innerList))):
            if i < len(correct_nums) and correct_nums[i] == innerList[i]:
                count += 1
                break
        if count!=0 and len(correct_nums)==len(innerList):
                    
            for i in range(len(correct_nums)):
                if correct_nums[i]!=-1 and correct_nums[i]!=-2 and correct_nums[i]!=innerList[i] :
                    correct_nums[i]=-1
        pp=-1
        ap=-1
        # for i in range(len(correct_nums)):#[0, 500, 750, 1000, 250, 1450]  # 递增时，突然有个数字变小或者变大，设为-1
        for i in range(1, len(correct_nums) - 1):
            if correct_nums[i] > abs(correct_nums[i - 1]) and correct_nums[i] > correct_nums[i + 1] and correct_nums[i - 1]<correct_nums[i + 1]:#[-0.1, 0.08, 0.04, 0.02, 0.0]
                correct_nums[i] = -1
            elif correct_nums[i] < correct_nums[i - 1] and correct_nums[i] < correct_nums[i + 1]and correct_nums[i - 1]<correct_nums[i + 1]:
                correct_nums[i] = -1



        # print(correct_nums)

        # breakwww

        # print(correct_nums)
        new_list = [-2] * len(innerList)#创建初始化列表，元素全是-2
        new_points=[-2] * len(innerList)
        new_polygons=[-2] * len(innerList)
        new_angles=[-2] * len(innerList)
        pre=-1
        aft=-1
        for i, item in enumerate(correct_nums):#遍历需要更新的列表
            if pre == len(correct_nums)-1 and correct_nums[pre]==-1:
                break
            if i < pre:#pre指向的是处理完的元素，这里如果小于pre需要跳过
                continue
            # if correct_nums.count(correct_nums[i]) == 1 and correct_nums[i] in innerList:
            if correct_nums[i] in innerList:#如果元素在innerlist里，直接找到对应的下标innerIdx，在new_list中更新该数字
                innerIdx = innerList.index(correct_nums[i])
                new_list[innerIdx]=correct_nums[i]
                new_points[innerIdx]=resorted_points_nopointer[i]
                new_polygons[innerIdx]=resorted_polygons_nopointer[i]
                new_angles[innerIdx]=resorted_angle_nopointer[i]


                pre=i#更新完后，pre指向这个数字
            if correct_nums[i]==-1:#如果该元素是-1，需要找到这个元素之后还有多少个-1，用cnt_neg记录，[6, -1, -1, 30, 5, 6]比如这里cnt_neg=2
                cnt_neg=0
                while(correct_nums[i]==-1):#找连续-1的个数
                    cnt_neg+=1
                    i+=1
                    aft=i#aft指向-1的后一个，循环结束时，在[0, -1, -1, 3, 5, 6]应该是指向3
                    if i ==len(correct_nums):#如果是结尾[0, 1,2，-1, -1]，不能指向后一个了，所以只能指向最后一个-1，所以这里要减1
                        aft-=1
                        break


                if correct_nums[aft]!=-1:#不是末尾，aft指向3，这里要做的就是pre到aft之间的数字都放到new_list
                    if pre!=-1:
                        innerIdxPre = innerList.index(correct_nums[pre])#pre指向的元素在innerlist中的坐标不一定一样，用innerIdxPre表示【4,5,0，-1,2,3】【0,1,2,3,4,5】这里pre=2，innerIdxPre=0
                    else:innerIdxPre=pre
                    innerIdxAft = innerList.index(correct_nums[aft])
                    # print(pre, aft, cnt_neg)
                    if innerIdxAft-innerIdxPre-1!=cnt_neg:
                        diff_angle1=[]
                        if cnt_neg==1 and innerIdxAft-innerIdxPre-1==2:
                            #两空插一个
                            for i in range(len(resorted_angle_nopointer)-1):
                                diff=resorted_angle_nopointer[i+1]-resorted_angle_nopointer[i]
                                if diff<0:
                                    diff+=360
                                diff_angle1.append(diff)
                            avg=np.min(diff_angle1)
                            idx=pre+1#第一个-1的下标
                            if idx==0:#如果是开头，就只能用diff_angle[idx]判断
                                if diff_angle1[idx]>1.5*avg:#idx距离下一个点角度过大，认为中间缺了一个点
                                    correct_nums.insert(idx+1,-2)  #correctnums里面在idx之后插入一个点，默认-2
                                    resorted_points_nopointer.insert(idx+1,-2)  
                                    resorted_polygons_nopointer.insert(idx+1,-2)  
                                    resorted_angle_nopointer.insert(idx+1,-2)  
                                else:
                                    correct_nums.insert(idx,-2) #idx距离下一个点角度正好，那么缺的点就在它前面，
                                    resorted_points_nopointer.insert(idx,-2)  
                                    resorted_polygons_nopointer.insert(idx,-2)  
                                    resorted_angle_nopointer.insert(idx,-2) 
                            else:
                                insert_idx=-100
                                idx=pre+1
                                if abs(diff_angle1[idx-1])>abs(diff_angle1[idx]):#idx=pre+1
                                            
                                    insert_idx=idx
                                else:
                                    insert_idx=idx+1
                                correct_nums.insert(insert_idx,-2)  #correctnums里面在idx之后插入一个点，默认-2
                                resorted_points_nopointer.insert(insert_idx,-2)  
                                resorted_polygons_nopointer.insert(insert_idx,-2)  
                                resorted_angle_nopointer.insert(insert_idx,-2)  
   
                            aft+=1#因为插入了一个，所以aft要移动
                            cnt_neg+=1                          
                            # elif idx==len(correct_nums)-1:#可以用diff_angle[idx-1]和 diff_angle[idx]末尾
                                
                            # else:中间情况 
                        # 三个插两个-1，1个-2
                        elif cnt_neg==2 and innerIdxAft-innerIdxPre-1==3:
                            for i in range(len(resorted_angle_nopointer)-1):
                                diff=resorted_angle_nopointer[i+1]-resorted_angle_nopointer[i]
                                if diff<0:
                                    diff+=360
                                diff_angle1.append(diff)
                            avg=np.min(diff_angle1)
                            idx=pre+1#第一个-1的下标
                            if idx!=len(resorted_angle_nopointer)-2:# 第二个-1不是最后一个
                                if diff_angle1[idx]>1.5*avg and diff_angle1[idx+1]<1.5*avg:# -1, -2, -1, 3, 4, 5,
                                    correct_nums.insert(idx+1,-2)  #correctnums里面在idx之后插入一个点，默认-2
                                    resorted_points_nopointer.insert(idx+1,-2)  
                                    resorted_polygons_nopointer.insert(idx+1,-2)  
                                    resorted_angle_nopointer.insert(idx+1,-2)  
                                elif diff_angle1[idx]<1.5*avg and diff_angle1[idx+1]>1.5*avg :# -1, -1, -2, 3, 4, 5
                                    correct_nums.insert(idx+2,-2) 
                                    resorted_points_nopointer.insert(idx+2,-2)  
                                    resorted_polygons_nopointer.insert(idx+2,-2)  
                                    resorted_angle_nopointer.insert(idx+2,-2) 
                                elif diff_angle1[idx]<1.5*avg and diff_angle1[idx+1]<1.5*avg:#-2, -1, -1, 3, 4, 5
                                    correct_nums.insert(idx,-2) 
                                    resorted_points_nopointer.insert(idx,-2)  
                                    resorted_polygons_nopointer.insert(idx,-2)  
                                    resorted_angle_nopointer.insert(idx,-2)                                                                        
                            aft+=1#因为插入了一个，所以aft要移动
                            cnt_neg+=1


                    if innerIdxAft-innerIdxPre-1==cnt_neg:#pre到aft全部放入newlist
                        if pre==-1:
                            pre=0
                            innerIdxPre=0

                        for j in range(0,aft+1-pre):#【4,5,0,-1,2,3】【0,1,2,3,4,5】把第一个list下标为2~4的数字移动到innerList对应的下标0~2
                            if innerIdxPre+j>=len(new_list) or pre+j>=len(correct_nums):
                                                            
                                return [],[],[],[]
                            new_list[innerIdxPre+j] = correct_nums[pre+j]
                            new_points[innerIdxPre+j]=resorted_points_nopointer[pre+j]
                            new_polygons[innerIdxPre+j]=resorted_polygons_nopointer[pre+j]
                            new_angles[innerIdxPre+j]=resorted_angle_nopointer[pre+j]
                        pre=aft
                else:#aft指向了最后一个元素，且是-1
                    # print(pre, aft)
                    if pre!=-1:
                        innerIdxPre = innerList.index(correct_nums[pre])
                    else:innerIdxPre=pre

                    if (len(innerList)-innerIdxPre-1)!=aft-pre:
                        diff_angle1=[]
                        if cnt_neg==1 and len(innerList)-innerIdxPre-1==2:
                            #两空插一个
                            for i in range(len(resorted_angle_nopointer)-1):
                                diff=resorted_angle_nopointer[i+1]-resorted_angle_nopointer[i]
                                if diff<0:
                                    diff+=360
                                diff_angle1.append(diff)
                            insert_idx=-100
                            avg=np.min(diff_angle1)
                            idx=pre+1
                           
                            if diff_angle1[idx-1]>1.5*avg:
                                            
                                insert_idx=idx
                            else:
                                insert_idx=idx+1
                            correct_nums.insert(insert_idx,-2)  #correctnums里面在idx之后插入一个点，默认-2
                            resorted_points_nopointer.insert(insert_idx,-2)  
                            resorted_polygons_nopointer.insert(insert_idx,-2)  
                            resorted_angle_nopointer.insert(insert_idx,-2)  
                            aft+=1
                            cnt_neg+=1
                        # 三个插两个-1，1个-2    
                        if cnt_neg==2 and len(innerList)-innerIdxPre-1==3:
                            for i in range(len(resorted_angle_nopointer)-1):
                                diff=resorted_angle_nopointer[i+1]-resorted_angle_nopointer[i]
                                if diff<0:
                                    diff+=360
                                diff_angle1.append(diff)
                            avg=np.min(diff_angle1)
                            idx=pre+1#第一个-1的下标
                            if diff_angle1[idx]>1.5*avg and diff_angle1[idx-1]<1.5*avg:# 0，1，2，-1，-2，-1
                                correct_nums.insert(idx+1,-2)  #correctnums里面在idx之后插入一个点，默认-2
                                resorted_points_nopointer.insert(idx+1,-2)  
                                resorted_polygons_nopointer.insert(idx+1,-2)  
                                resorted_angle_nopointer.insert(idx+1,-2)  
                            elif diff_angle1[idx]<1.5*avg and diff_angle1[idx-1]>1.5*avg :#0，1，2，-2，-1，-1
                                correct_nums.insert(idx,-2) 
                                resorted_points_nopointer.insert(idx,-2)  
                                resorted_polygons_nopointer.insert(idx,-2)  
                                resorted_angle_nopointer.insert(idx+2,-2) 
                            elif diff_angle1[idx]<1.5*avg and diff_angle1[idx-1]<1.5*avg:#0，1，2，-1，-1，-2
                                correct_nums.insert(idx+2,-2) 
                                resorted_points_nopointer.insert(idx+2,-2)  
                                resorted_polygons_nopointer.insert(idx+2,-2)  
                                resorted_angle_nopointer.insert(idx+2,-2)
                            aft+=1
                            cnt_neg+=1

                    if (len(innerList)-innerIdxPre-1)==aft-pre:
                        for j in range(0, aft+1-pre):
                            if innerIdxPre+j>=len(new_list) or pre+j>=len(correct_nums):
                                                                
                                return [],[],[],[]
                            new_list[innerIdxPre+j] = correct_nums[pre+j]
                            new_points[innerIdxPre+j]=resorted_points_nopointer[pre+j]
                            new_polygons[innerIdxPre+j]=resorted_polygons_nopointer[pre+j]
                            new_angles[innerIdxPre+j]=resorted_angle_nopointer[pre+j]
        result = is_element_majority(new_list, -2)
        if result:
            return [],[],[],[]

        # Correct -1 & -2
        # print("new_points",new_points)

        for i in range(len(new_list)):
            
            if new_list[i] is -1:
                new_list[i]=innerList[i]
            if new_list[i] !=-1 and new_list[i]!=-2 and new_list[i]!=innerList[i]:
                new_list[i]=-2
        for i in range(len(new_list)):
            
            if new_list[i] is -2:
                
                aft_neg2=i

                for j in range(i,len(new_list)):
                    if j==len(new_list)-1:
                        aft_neg2=j
                        break     
                    if new_list[j+1] is not -2:
                        aft_neg2=j+1
                        break
                
                if aft_neg2==len(new_list)-1 and new_list[aft_neg2]==-2:
                    for q in range(0,aft_neg2-i+1):
                        new_list[i+q]=innerList[i+q]
                        new_polygons[i+q]=fake_polygon
                else:
                    for q in range(0,aft_neg2-i):
                        new_list[i+q]=innerList[i+q]
                        new_polygons[i+q]=fake_polygon

                if i ==0 and aft_neg2!=len(new_list)-1:   
                    aft_aft_neg2=i              
                    for j in range(aft_neg2+1,len(new_list)):
                        if new_list[j] is not -2:
                            aft_aft_neg2=j
                            break
                    
                    if aft_aft_neg2==i:
                        return [],[],[],[]
    
                    # -2, -2, -2, 3, -2, -2, 6
                            
                    diff=new_list[aft_aft_neg2]-new_list[aft_neg2]

                    diff_angle=new_angles[aft_aft_neg2]-new_angles[aft_neg2]
                    if diff_angle<0:
                        diff_angle+=360

                    mag1, theta1 = cv2.cartToPolar(float(new_points[aft_neg2][0])-center_x, float(new_points[aft_neg2][1])-center_y, angleInDegrees=1)
                    mag2, theta2 = cv2.cartToPolar(float(new_points[aft_aft_neg2][0])-center_x, float(new_points[aft_aft_neg2][1])-center_y, angleInDegrees=1)
                    miss_point_mag=(mag1+mag2)/2
                    

                    for p in range(1,aft_neg2-i+1):
                        temp_angle=new_angles[aft_neg2]-diff_angle*((new_list[aft_neg2]-new_list[aft_neg2-p])/diff)
                        if temp_angle < 0 :
                            temp_angle+=360                        
                        new_angles[aft_neg2-p]=temp_angle
                        miss_x, miss_y = cv2.polarToCart(miss_point_mag, temp_angle, angleInDegrees=1)
                        new_points[aft_neg2-p]=(miss_x[0][0]+center_x, miss_y[0][0]+center_y)
                
                elif i ==0 and aft_neg2==len(new_list)-1:    
                    return [],[],[],[]   
                            
                elif aft_neg2==len(new_list)-1 and i > 1 and new_points[aft_neg2]==-2:              #1, 2, -2, -2
                    diff = new_list[i-1]-new_list[i-2]
                    diff_angle = new_angles[i-1]-new_angles[i-2]
                    if diff_angle<0:
                        diff_angle+=360

                    mag1, theta1 = cv2.cartToPolar(float(new_points[i-1][0])-center_x, float(new_points[i-1][1])-center_y, angleInDegrees=1)
                    mag2, theta2 = cv2.cartToPolar(float(new_points[i-2][0])-center_x, float(new_points[i-2][1])-center_y, angleInDegrees=1)
                    miss_point_mag=(mag1+mag2)/2

                    for p in range(0,aft_neg2-i+1):
                        temp_angle=new_angles[i-1]+diff_angle*((new_list[i+p]-new_list[i-1])/diff)
                        if temp_angle > 360 :
                            temp_angle-=360                        
                        new_angles[i+p]=temp_angle
                        miss_x, miss_y = cv2.polarToCart(miss_point_mag, temp_angle, angleInDegrees=1)
                        new_points[i+p]=(miss_x[0][0]+center_x, miss_y[0][0]+center_y)

                else:                   # 1, -2, -2, 4
                    diff = new_list[aft_neg2]-new_list[i-1]
                    diff_angle = new_angles[aft_neg2]-new_angles[i-1] 
                    if diff_angle<0:
                        diff_angle+=360

                    mag1, theta1 = cv2.cartToPolar(float(new_points[i-1][0])-center_x, float(new_points[i-1][1])-center_y, angleInDegrees=1)
                    mag2, theta2 = cv2.cartToPolar(float(new_points[aft_neg2][0])-center_x, float(new_points[aft_neg2][1])-center_y, angleInDegrees=1)
                    miss_point_mag=(mag1+mag2)/2

                    for p in range(0,aft_neg2-i):
                        temp_angle=new_angles[i-1]+diff_angle*((new_list[i+p]-new_list[i-1])/diff)
                        if temp_angle > 360 :
                            temp_angle-=360                        
                        new_angles[i+p]=temp_angle
                        miss_x, miss_y = cv2.polarToCart(miss_point_mag, temp_angle, angleInDegrees=1)
                        new_points[i+p]=(miss_x[0][0]+center_x, miss_y[0][0]+center_y)
                                                                                

        for i in range(len(new_list)-1):
                    
            diff1=new_angles[i]-new_angles[0]
            diff2=new_angles[i+1]-new_angles[0]
            diff3=pointer_angle-new_angles[0]
            if diff1 < 0:
                diff1 += 360
            if diff2 < 0:
                diff2 += 360
            if diff3 < 0:
                diff3+=360
                        
            if diff3 > diff1 and diff3 < diff2 or diff3>355 :
                                                                                                                                                                                    
                new_angles.insert(i+1,pointer_angle)
                new_polygons.insert(i+1,resorted_polygons[pointer_index])

                new_points.insert(i+1,resorted_points[pointer_index])

                new_list.insert(i+1,-1)
                pointer_index,pointer_angle=find_pointer(new_polygons,new_angles)
                get_pointer_num(new_list,new_angles,pointer_index)
                return new_list,new_points,new_polygons,new_angles  
            elif i==len(new_list)-2 and diff3-diff2<5:
                new_angles.insert(i+2,pointer_angle)
                new_polygons.insert(i+2,resorted_polygons[pointer_index])

                new_points.insert(i+2,resorted_points[pointer_index])

                new_list.insert(i+2,-1)
                pointer_index,pointer_angle=find_pointer(new_polygons,new_angles)
                get_pointer_num(new_list,new_angles,pointer_index)
                return new_list,new_points,new_polygons,new_angles  
        return [],[],[],[]
def correct_nums(nums_list,resorted_angle,resorted_polygons,resorted_points,center_x,center_y,img):
                                                                            
    pointer_index,pointer_angle=find_pointer(resorted_polygons,resorted_angle)
    nums=copy.deepcopy(nums_list)
    
    flag_reverse=False


    if pointer_index==0:#有待完善
        df=resorted_angle[1]-resorted_angle[0]
        if df<0:
            df+=360        
        if df>4:#没有0点，加一个
            nums.insert(0,'0')
        
            nums,polygons_nopointer=process_occlusion(nums,resorted_angle,resorted_polygons,resorted_points,pointer_index,pointer_angle,center_x,center_y,img)
        
            nums=correct_float_nums(nums)
            if nums[-1]==0:
                flag_reverse=True

        
            if flag_reverse:
                nums[0]=-1*nums[0]
    
            return nums,1,polygons_nopointer
        else:
            occlusion=2      
            nums,polygons_nopointer=process_occlusion_woutP(nums,resorted_angle,resorted_polygons,pointer_index,img)
 
            nums=correct_float_nums(nums)

            if nums[-1]==0:
                    flag_reverse=True

        
            if flag_reverse:
                nums[0]=-1*nums[0]
            return nums,occlusion,polygons_nopointer

    elif pointer_index==(len(resorted_polygons)-1):
        df=resorted_angle[-1]-resorted_angle[-2]
        if df<0:
            df+=360 
        if df>4:
            nums.append('0')
        
            nums,polygons_nopointer=process_occlusion(nums,resorted_angle,resorted_polygons,resorted_points,pointer_index,pointer_angle,center_x,center_y,img)
            nums=correct_float_nums(nums)
            if nums[-1]==0:
                    flag_reverse=True

        
            if flag_reverse:
                nums[0]=-1*nums[0]

            return nums,1,polygons_nopointer
        else:
            occlusion=2   
            nums,polygons_nopointer=process_occlusion_woutP(nums,resorted_angle,resorted_polygons,pointer_index,img)
 
            nums=correct_float_nums(nums)

            if nums[-1]==0:
                    flag_reverse=True

        
            if flag_reverse:
                nums[0]=-1*nums[0]
            return nums,occlusion,polygons_nopointer
    else:
        occlusion=is_occlusion(resorted_angle,resorted_polygons)


        if occlusion==0 or occlusion==2:
                                                   
            nums,polygons_nopointer=process_occlusion_woutP(nums,resorted_angle,resorted_polygons,pointer_index,img)
            nums=correct_float_nums(nums)

            if nums[-1]==0:
                    flag_reverse=True

        
            if flag_reverse:
                nums[0]=-1*nums[0]
            return nums,occlusion,polygons_nopointer

        if(occlusion==1):
            # 指针加入OCR识别
            nums.insert(pointer_index,0)

            #处理遮挡
            nums,polygons_nopointer=process_occlusion(nums,resorted_angle,resorted_polygons,resorted_points,pointer_index,pointer_angle,center_x,center_y,img)
            nums=correct_float_nums(nums)

            if nums[-1]==0:
                    flag_reverse=True

        
            if flag_reverse:
                nums[0]=-1*nums[0]
            return nums,occlusion,polygons_nopointer




# 等差数列对应有序点，还有顺序
def get_arithmetic_points(nums,resorted_points,resorted_polygons,resorted_angle,occlusion,polygons_nopointer):
                                                    
    # pointer_index,pointer_angle=find_pointer(resorted_polygons,reordered_angle)
    # grad_points=copy.deepcopy(resorted_points)
    # del grad_points[pointer_index]


    # return grad_points
    pointer_index,pointer_angle=find_pointer(resorted_polygons,resorted_angle)
    
    grad_points=copy.deepcopy(resorted_points)
    nums_list=copy.deepcopy(nums)
    temp_polygons =copy.deepcopy(resorted_polygons)
    del grad_points[pointer_index]
    del nums_list[pointer_index]
    # del temp_polygons[pointer_index]

    arithmetic_points=[]
    if occlusion!=1:
    
        j=0
        for i in range(len(nums_list)):
            
            if isinstance(polygons_nopointer[i], np.ndarray):
                arithmetic_points.append((i,grad_points[j]))
                j+=1
    else:
        for i in range(len(nums_list)):
                
            if isinstance(polygons_nopointer[i], np.ndarray):
                arithmetic_points.append((i,grad_points[i]))

    return arithmetic_points




# 仿射变换
def PerspectiveTransform(polygon,img):
    polygon = np.array(polygon)                                                                               
    rect = cv2.minAreaRect(polygon)

    # 计算旋转矩阵
    center, size, angle = rect
    
    M = cv2.getRotationMatrix2D(center, angle, 1)

    # 根据旋转矩阵进行旋转
    rotated_image = cv2.warpAffine(img, M, (img.shape[1],img.shape[0]))

    # 将旋转矩形区域裁剪出来
    crop = cv2.getRectSubPix(rotated_image, tuple(map(lambda x:int(x),size)), tuple(map(lambda x:int(x),center)))

    return crop

def draw_grad_kpt(x,y,img):
    cv2.circle(crop, (int(x),int(y)), 5, (0, 168, 255), -2)  

def draw_polygon(polygon:np.array,img):
    for i in range(len(polygon)):
        p1 = tuple(map(lambda x:int(x),polygon[i]))
        p2 = tuple(map(lambda x:int(x),polygon[(i + 1) % len(polygon)]))
        cv2.line(img, p1, p2, (0, 255, 0), 2)
                                            
def draw_polygon_label(polygon,text,img):
            
    rect = cv2.minAreaRect(polygon)
    box = np.int0(cv2.boxPoints(rect))
    box = np.int0(box)
    cv2.putText(img, f'{text}', box[2], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

def draw_grad_kpt_lable(points,text,img):
    cv2.putText(img, f'{text}', (int(points[0]),int(points[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 255), 1)                            
def draw_grad_kpt_lable(points,text,img):
    cv2.circle(img, (int(points[0]),int(points[1])), 5, (0, 255, 0), -3)  
    cv2.putText(img, f'{text}', (int(points[0]),int(points[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 255), 1)                            
    
def draw_measure(text,text2,img,i):
    cv2.putText(img, f'measure_{i}: {text};GT:{text2}', (10,i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
def draw_measure1(text,img,i):
    cv2.putText(img, f'measure_{i}: {text}', (10,i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    
# get the graduation point closest to the textdet_bbox
def getPoint( gra,px,py,vis):
            
    min_dist=10000
    nearest_point=(-1,-1)
    flag=False
    
    if len(gra)>0:
        for i in range(len(gra)):
            key=gra[i][0]
            dist = math.sqrt((px - key[0]) ** 2 + (py - key[1]) ** 2)
            if dist < min_dist:
                index_min=i
                min_dist = dist
                nearest_point = (key[0], key[1])

    if vis[index_min]==False:
        vis[index_min]=True
    else:
        flag=True

    return  nearest_point,flag


def get_reshape_point(M,x,y):
    point = np.array([x, y, 1.0])

    # Calculate the transformed point
    transformed_point = np.dot(M, point)
    u = transformed_point[0] / transformed_point[2]
    v = transformed_point[1] / transformed_point[2]
    return u,v





def find_index(nums_list, given_list):
    for idx, num_list in enumerate(nums_list):
        pointer = 0
        indices = []
        for i, num in enumerate(num_list):
            if pointer < len(given_list) and given_list[pointer] == num:
                indices.append(i)
                pointer += 1
        if pointer == len(given_list):
            return num_list, indices
    print( "No matching list found in nums_list.")
    return [],[]

def locate_pointer(lst, x):
    
    if lst[0]-x>0 and lst[0]-x<10:
                                    
        return 0
    if x>350 and lst[0]<20:
        if lst[0]-x+360>0:
            return 0

                # 选取第一个元素作为标记
    marker = lst[0]
    # 对除marker之外的所有元素进行从小到大排序
    sorted_lst = sorted(lst[:])
    # 使用二分查找算法找到第一个比x大的元素的位置
    left, right = 0, len(sorted_lst) - 1
    index = len(sorted_lst)
    while left <= right:
        mid = (left + right) // 2
        if sorted_lst[mid] >= x:
            index = mid
            right = mid - 1
        else:
            left = mid + 1
    # 在index位置插入x
    sorted_lst.insert(index, x)
    # 如果marker不是列表的第一个元素，则将marker之前的元素全部移动到列表末尾
    index_start = sorted_lst.index(marker)
    print(sorted_lst)
    left_part = sorted_lst[:index_start]
    right_part = sorted_lst[index_start:]
    # 将左边部分全部移动到列表末尾，成为新的列表
    result = right_part + left_part
    index_pointer = result.index(x)
    return index_pointer
def get_avgR(arithmetic_points_in,x1,y1):
                
    sum_of_distances = 0
    distances = []
    # 遍历每个元素，计算距离并将其添加到距离列表中
    for element in arithmetic_points_in:
        cx2= element[1][0]
        cy2 = element[1][1]
        distance = math.sqrt((cx2 - x1)**2 + (cy2 - y1)**2)
        distances.append(distance)
        sum_of_distances += distance

    # 计算平均距离
    avg_R = sum_of_distances / len(distances)
    return avg_R
def get_avgR2(points,x1,y1):
                                
    sum_of_distances = 0
    distances = []
    # 遍历每个元素，计算距离并将其添加到距离列表中
    for element in points:
        cx2,cy2= element
        distance = math.sqrt((cx2 - x1)**2 + (cy2 - y1)**2)
        distances.append(distance)
        sum_of_distances += distance

    # 计算平均距离
    avg_R = sum_of_distances / len(distances)
    return avg_R
    







def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

def export_trt(model_path):
    f = Path(model_path).with_suffix('.engine')  # TensorRT engine file
    onnx = f.with_suffix(".onnx")
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    workspace = 4
    config.max_workspace_size = workspace * 1 << 30
    # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 30)  # fix TRT 8.4 deprecation notice

    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx)):
        raise RuntimeError(f'failed to load ONNX file: {onnx}')

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    for inp in inputs:
        LOGGER.info(f' input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    for out in outputs:
        LOGGER.info(f' output "{out.name}" with shape{out.shape} {out.dtype}')

    # if builder.platform_has_fast_fp16 and half:
    #     config.set_flag(trt.BuilderFlag.FP16)
    with builder.build_engine(network, config) as engine, open(f, 'wb') as t:
        t.write(engine.serialize())
    return str(f)


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords









class yolo():
    def __init__(self, model_path, name_path, device = torch.device('cpu'), confThreshold=0.5, iouThreshold=0.5, objThreshold=0.5):
        self.classes = ['gauge']
        import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download
        if device.type == 'cpu':
            device = torch.device('cuda:0')
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        if model_path.endswith(".onnx"):
            model_path = export_trt(model_path)
        logger = trt.Logger(trt.Logger.INFO)
        with open(model_path, 'rb') as f, trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(f.read())
        self.context = model.create_execution_context()
        self.bindings = OrderedDict()
        fp16 = False  # default updated below
        dynamic = False
        for index in range(model.num_bindings):
            name = model.get_binding_name(index)
            dtype = trt.nptype(model.get_binding_dtype(index))
            if model.binding_is_input(index):
                if -1 in tuple(model.get_binding_shape(index)):  # dynamic
                    dynamic = True
                    self.context.set_binding_shape(index, tuple(model.get_profile_shape(0, index)[2]))
                if dtype == np.float16:
                    fp16 = True
            shape = tuple(self.context.get_binding_shape(index))
            im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
            self.bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
        self.img_size = [640,640]
        self.stride = 32
        self.confThresh = confThreshold
        self.iouThresh = iouThreshold

    def getClassName(self):
        return self.classes

    def from_numpy(self, x):
        return torch.from_numpy(x).to("cpu") if isinstance(x, np.ndarray) else x

    def detect(self, img):
        im = letterbox(img, self.img_size, stride=self.stride, auto=False)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im).to("cuda")
        im = im.float()
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]

        # im = im.cpu().numpy()  # torch to numpy
        s = self.bindings['images'].shape
        assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
        self.binding_addrs['images'] = int(im.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        y = self.bindings['output'].data
        if isinstance(y, (list, tuple)):
            prediction = self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            prediction = self.from_numpy(y)
        # print("prediton",prediction.shape)#[1,25200,6]
        bs = prediction.shape[0]
        nc = prediction.shape[2] - 5  # number of classes
        xc = prediction[..., 4] > self.confThresh  # candidates
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence
            # print(x.shape)
            x[:, 5:] *= x[:, 4:5]
            box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > self.confThresh]
            n = x.shape[0]  # number of boxes
            
            if not n:  # no boxes
                return [], im
            # print(x.shape)
            x = x[x[:, 4].argsort(descending=True)]
            # print(x.shape)
            boxes, scores = x[:, :4] , x[:, 4]  # boxes (offset by class), scores
            # print(boxes.shape)
            # print("boxes",boxes)
            i = torchvision.ops.nms(boxes, scores, self.iouThresh)

            return x[i], im



point_det_model = PointRecognizer()
one_point_model = OnePoint()
# textdet=TextDetInferencer(model='/project/train/src_repo/mmocr/configs/textdet/dbnetpp/my_dbnetpp_resnet50-oclip_fpnc_60e_icdar2015.py', 
#                         weights='/project/train/models/my_dbnetpp_resnet50-oclip_fpnc_60e_icdar2015/best_icdar_precision_epoch_60.pth',device='cuda:0')
# textdet=TextDetInferencer(model='/project/train/src_repo/mmocr/configs/textdet/dbnetpp/my_dbnetpp_resnet50_fpnc_60e_icdar2015.py', 
#                         weights='/project/train/models/my_dbnetpp_resnet50_fpnc_60e_icdar2015/best_icdar_precision_epoch_50.pth',device='cuda:0')
textdet=TextDetInferencer(model='/project/train/src_repo/mmocr/configs/textdet/dbnetpp/my_dbnetpp_resnet50-dcnv2_fpnc_90e_icdar2015.py', 
                        weights='/usr/local/ev_sdk/model/my_dbnetpp_resnet50-dcnv2_fpnc_90e_icdar2015/best_icdar_precision_epoch_78.pth',device='cuda:0')



def init():
    """Initialize model

    Returns: model

    """
    # model_path = "/usr/local/ev_sdk/model/best.onnx"
    name_path = '/usr/local/ev_sdk/src/labels.txt'
    model_path = "/usr/local/ev_sdk/model/exp/weights/best.onnx"
    thresh = 0.2
    session = yolo(model_path, name_path, confThreshold=thresh)
    return session

TextRecognizer=TextRec()
def process_image(net, input_image, args=None):
    det, resizeIm = net.detect(input_image)
    className = net.getClassName()
    padr = 100
    detect_objs, target_info = [], []
    if len(det):
        det[:, :4] = scale_coords(resizeIm.shape[2:], det[:, :4], input_image.shape).round()
        for *xyxy, conf, cls in reversed(det):
            xmin = max(int(xyxy[0]), 0)
            ymin = max(int(xyxy[1]), 0)
            xmax = min(int(xyxy[2]), input_image.shape[1])
            ymax = min(int(xyxy[3]), input_image.shape[0])
            if max(ymax-ymin,xmax-xmin)/min(ymax-ymin,xmax-xmin)>2:
                print("仪表长宽不对等,continue")
                continue
            if (xmax-xmin)<5:
                print("xmax-xmin)<5,continue")
                continue
            if conf<0.5:
                continue
            crop_img = input_image[ymin:ymax, xmin:xmax, :]
            cx, cy = crop_img.shape[1] * 0.5, crop_img.shape[0] * 0.5

            keypoints, polygons = [], []
            x1,y1,x2,y2,box=point_det_model.predict_text(crop_img)
            h,w=crop_img.shape[:2]
            # timg=crop_img.copy()
            # gray_image = cv2.cvtColor(timg, cv2.COLOR_BGR2GRAY)
            # timage = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
            graduations,gra_points=one_point_model.predict_text(crop_img)
            print("len(graduations)",len(graduations))
            print("len(gra_points[i][0])",len(gra_points))
            
            i=0

            keys_in=[]#存储内关键点，暂时的
            keys_out=[]#存储外关键点
            inner_list = [[0, 0.02, 0.04, 0.06, 0.08, 0.1],
                    [0, 2, 4, 6, 8, 10],
                    [0, 10, 20, 30, 40, 50, 60],
                    [0, 1, 2, 3, 4],
                    [0, 1, 2, 3, 4, 5, 6],
                    [0, 150, 300, 450],
                    [45, 46, 48, 50, 52, 54, 55],
                    [0, 100, 200, 300],
                    [0, 20, 40, 60, 80, 100],
                    [0, 500, 1000, 1500, 2000, 2300],
                    [0, 250, 500, 750, 1000, 1250, 1450],
                    [0, 25, 50, 75, 100, 125, 145],
                    [0, 5, 10, 15, 20, 25],
                    [0, 4, 8, 12, 16],
                    [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    [0, 0.5, 1, 1.5, 2, 2.5],
                    [0, 0.4, 0.8, 1.2, 1.6],
                    [-0.1, 0.08, 0.06, 0.04, 0.02, 0]]
            outer_list = [
                    [0, 4, 8, 12, 16],
                    [0, 2, 4, 6, 8, 10],
                    [0, 1,2,3,4,5,6],
                    [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                    [0, 2, 4, 6, 8, 10, 12, 14],
                    [32, 50, 100, 150, 200],
                    [0, 100, 200, 300, 400, 500, 600, 700, 800, 860],
                    [0, 10, 20, 30, 40, 50, 60],
                    [0, 1000, 2000, 3000, 4000, 5000, 6000]
                ]
            # try:
            measures=[]  
            polygons_out=[]
            polygons_in=[]
            points_out=[]#存储外关键点,要进行排序
            points_in=[]
            result_angle_out=[]
            result_angle_in=[]
            occlusion=-1
            occlusion_out=-1
            arithmetic_points_in=[]
            arithmetic_points_out=[]
            gt_nums_in=[]
            gt_nums_out=[]
            gt_angle_out=[]
            gt_angle_in=[]
            partial_polygons1 =[]
            partial_nums1 = []
            partial_polygons2 =[]
            partial_nums2 = []
            resorted_polygons_out=[]
            resorted_polygons_in=[]
            gt_polygons_out=[]
            gt_polygons_in=[]
            t_keypoints=[]
            t_measure=[]
            pointer_in=-1
            pointer_out=-1
            if len(graduations) > 0 and len(gra_points)>0:
                                                                        
                for i in range(len(graduations)):
                    # print(graduations[i][0],graduations[i][1],graduations[i][2],graduations[i][3])
                    # print("刻度点坐标：",gra_points[i][0][0],gra_points[i][0][1])
                    poly_cx=(graduations[i][0]+graduations[i][2])/2
                    poly_cy=(graduations[i][1]+graduations[i][3])/2
                    rc = math.sqrt((poly_cx - x1) ** 2 + (poly_cy - y1) ** 2)
                    rk = math.sqrt((gra_points[i][0][0] - x1) ** 2 + (gra_points[i][0][1] - y1) ** 2)
                    if rc>rk:
                        keys_out.append((gra_points[i][0][0],gra_points[i][0][1]))
                    else:
                        keys_in.append((gra_points[i][0][0],gra_points[i][0][1]))
                print("keys_in",keys_in)
                print("keys_out",keys_out)
                # print("keys个数：",len(keys_in)+len(keys_out))

                vis= np.zeros(len(gra_points), dtype=bool)

                result=textdet(crop_img)

                if(len(result['predictions'][0]['polygons'])>0):
                    for polygon in result['predictions'][0]['polygons']:
                                                
                        # a = np.array(polygon1)
                        # a = a.astype(int)
                        # a = [int(xmin + a[i]) if i % 2 == 0 else int(ymin + a[i]) for i in range(len(a))]
                        # graduation = {"name": "graduation", "points": a, "number": str(i), "confidence": 0.85}
                        # polygons.append(graduation)
                        # i+=1
                        polygon=np.array(polygon,dtype=np.float32)
                        polygon=polygon.reshape(4,2)


                        x_range = np.max(polygon[:, 0]) - np.min(polygon[:, 0])#new.dis->getpoint to refine the accuracy
                        y_range = np.max(polygon[:, 1]) - np.min(polygon[:, 1])
                        dis=max(x_range,y_range)

                        tri1 = np.array([polygon[0], polygon[1], polygon[2]])
                        area1 = 0.5 * np.abs(np.cross(tri1[1] - tri1[0], tri1[2] - tri1[0]))

                        # 计算第二个三角形的面积
                        tri2 = np.array([polygon[2], polygon[3], polygon[0]])
                        area2 = 0.5 * np.abs(np.cross(tri2[1] - tri2[0], tri2[2] - tri2[0]))

                        # 计算四边形的面积
                        area = int(area1 + area2)
                        print(area)
                        if(area<150):
                            continue
                        center = np.mean(polygon, axis=0)
                        point,flag = getPoint( gra_points, int(center[0]), int(center[1]),vis)

                        if flag:
                
                            if point in points_in:
                                index = [i for i, p in enumerate(points_in) if p == point]
                                center2 = np.mean(polygons_in[index[0]], axis=0)
                                dist2 = math.sqrt((center2[0] - point[0]) ** 2 + (center2[1] - point[1]) ** 2)
                                dist1 = math.sqrt((center[0] - point[0]) ** 2 + (center[1] - point[1]) ** 2)
                                if dist2<dist1:

                                    continue
                                else:
                                    del polygons_in[index[0]]
                                    # polygons_in = np.delete(polygons_in, index, axis=0)
                                    del points_in[index[0]]
                                    # points_in = np.delete(points_in, index, axis=0)
                                    # result_angle_in = np.delete(result_angle_in, index, axis=0)
                                    del result_angle_in[index[0]]

                            else:
                                index = [i for i, p in enumerate(points_out) if p == point]
                                # print(index)
                                center2 = np.mean(polygons_out[index[0]], axis=0)
                                dist2 = math.sqrt((center2[0] - point[0]) ** 2 + (center2[1] - point[1]) ** 2)
                                dist1 = math.sqrt((center[0] - point[0]) ** 2 + (center[1] - point[1]) ** 2)
                                if dist2<dist1:

                                    continue
                                else:
                                    del polygons_out[index[0]]
                                    # polygons_in = np.delete(polygons_in, index, axis=0)
                                    del points_out[index[0]]
                                    # points_in = np.delete(points_in, index, axis=0)
                                    # result_angle_in = np.delete(result_angle_in, index, axis=0)
                                    del result_angle_out[index[0]]
                        if point in keys_out:
                                                
                            polygons_out.append(polygon)
                            points_out.append(point)
                            # xp = np.array([point[0] - x1,point[1] - y1])
                            # print("xp",xp)
                            magnitude, angle = cv2.cartToPolar(float(point[0]) - float(x1),float(point[1]) - float(y1), angleInDegrees=True)
                            result_angle_out.append(angle[0][0])
                        else:
                            polygons_in.append(polygon)
                            points_in.append(point)
                            # xp = np.array([point[0] - x1,point[1] - y1])
                            magnitude, angle = cv2.cartToPolar(float(point[0]) - float(x1),float(point[1]) - float(y1), angleInDegrees=True)
                            result_angle_in.append(angle[0][0])
                    
                    fake_polygon = [[0, 0], [1, 0], [1, 1], [0, 1]]
                    polygons_out.append(fake_polygon)
                    points_out.append((x2,y2))
                    
                    # xp = np.array([x2 - x1,y2 - y1])
                    magnitude, angle = cv2.cartToPolar(x2 - x1,y2 - y1, angleInDegrees=True)
                    result_angle_in.append(angle[0][0])


                    polygons_in.append(fake_polygon)
                    points_in.append((x2,y2))
                    
                    result_angle_out.append(angle[0][0])
                    if len(result_angle_in)>1:
                        combined = list(zip(result_angle_in,points_in, polygons_in,))
                        sorted_combined = sorted(combined, key=lambda x: x[0])
                        sorted_angle_in, sorted_points_in,sorted_polygons_in = zip(*sorted_combined)
                        angle_diff = [sorted_angle_in[i + 1] - sorted_angle_in[i] for i in range(len(sorted_angle_in) - 1)] + [
                            sorted_angle_in[0] - sorted_angle_in[-1] + 360]

                        # 找到最大差值的下一个元素的索引
                        max_diff_index = angle_diff.index(max(angle_diff)) + 1

                        # 对angle和num_in数组重新排序
                        reordered_angle_in = sorted_angle_in[max_diff_index:] + sorted_angle_in[:max_diff_index]
                        resorted_points_in = sorted_points_in[max_diff_index:] + sorted_points_in[:max_diff_index]
                        resorted_polygons_in = sorted_polygons_in[max_diff_index:] + sorted_polygons_in[:max_diff_index]
                        reordered_angle_in=list(reordered_angle_in)
                        resorted_points_in=list(resorted_points_in)
                        resorted_polygons_in=list(resorted_polygons_in)

                    if len(result_angle_out)>1:
                        combined = list(zip(result_angle_out,points_out, polygons_out,))
                        sorted_combined = sorted(combined, key=lambda x: x[0])
                        sorted_angle_out, sorted_points_out,sorted_polygons_out = zip(*sorted_combined)
                        angle_diff = [sorted_angle_out[i + 1] - sorted_angle_out[i] for i in range(len(sorted_angle_out) - 1)] + [
                            sorted_angle_out[0] - sorted_angle_out[-1] + 360]

                        # 找到最大差值的下一个元素的索引
                        max_diff_index = angle_diff.index(max(angle_diff)) + 1

                        # 对angle和num_in数组重新排序
                        reordered_angle_out = sorted_angle_out[max_diff_index:] + sorted_angle_out[:max_diff_index]
                        resorted_points_out = sorted_points_out[max_diff_index:] + sorted_points_out[:max_diff_index]
                        resorted_polygons_out = sorted_polygons_out[max_diff_index:] + sorted_polygons_out[:max_diff_index]
                        reordered_angle_out=list(reordered_angle_out)
                        resorted_points_out=list(resorted_points_out)
                        resorted_polygons_out=list(resorted_polygons_out)

                      

                    if len(result_angle_out)>1 and len(result_angle_in)==1:#内径没有点，外径有，是特殊情形
                        cnt=0
                        nums_list=[]
                        #a=[]
                        
                        for i in range(len(resorted_polygons_out)):#polygon放入polygons
                            if np.array_equal(resorted_polygons_out[i], [[0, 0], [1, 0], [1, 1], [0, 1]]):


                                continue
                            crop=PerspectiveTransform(resorted_polygons_out[i],crop_img)

                            text_pred,confi=TextRecognizer.recog4direc(crop)
                        
                            nums_list.append(text_pred)
                            
                            # draw_polygon(resorted_polygons_out[i],img_label)


                        return_value=check_list(nums_list,outer_list,reordered_angle_out,resorted_polygons_out,resorted_points_out,x1,y1,crop_img)
                        
                        if return_value is not None:
                            new_list, new_points, new_polygons, new_angles = return_value
                            # 继续处理解包后的变量
                            # ...
                        else:
                            # 返回值为 None 的情况
                            # 在这里可以添加你想要执行的操作，或者直接使用 pass 关键字跳过
                            new_list, new_points, new_polygons, new_angles =  [], [], [], []
                        
                        if len(new_list)!=0:
                            corrected_nums =new_list
                            resorted_polygons_out=new_polygons
                            resorted_points_out=new_points
                            reordered_angle_out=new_angles
                            occlusion_out=1


                        else:
                            corrected_nums,occlusion_out,polygons_nopointer=correct_nums(nums_list,reordered_angle_out,resorted_polygons_out,resorted_points_out,x1,y1,crop_img)
            
                        for i in range(len(resorted_points_out)):
                            if np.array_equal(resorted_polygons_out[i], [[0, 0], [1, 0], [1, 1], [0, 1]]):
                                continue
                            a = np.array(resorted_polygons_out[i])
                            a = a.astype(int)
                            for k in range(len(a)):
                                a[k][0] += xmin
                                a[k][1] += ymin
                            keypoints.append({"name": "long_graduation" + str(cnt), "point": [float(resorted_points_out[i][0]+xmin), float(resorted_points_out[i][1]+ymin)], "confidence": 0.87})                                       
                            if not np.array_equal(resorted_polygons_out[i], [[0, 0], [10, 0], [10, 10], [0, 10]]):                                          
                                graduation = {"name": "graduation", "points": a.flatten().tolist(), "number": str(corrected_nums[i]), "confidence": 0.85}
                                polygons.append(graduation)

                        
                            cnt+=1
                        
                        

                        # res=(reordered_angle_out[point_index]-reordered_angle_out[point_index-1])/(reordered_angle_out[point_index+1]-reordered_angle_out[point_index-1])*(float(aft_num)-float(pre_num))+float(pre_num)
                        pointer_index,pointer_angle=find_pointer(resorted_polygons_out,reordered_angle_out)
                        res=corrected_nums[pointer_index]
                        measures.append(res)
                    if len(result_angle_out)>2 and len(result_angle_in)>1:#内径有点，外径也有，正常情形
                        cnt=0
                        nums_list=[]
                        # a=[]
                        for i in range(len(resorted_polygons_out)):#polygon放入polygons
                            if np.array_equal(resorted_polygons_out[i], [[0, 0], [1, 0], [1, 1], [0, 1]]):
                                # a = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
                                # a.append(a)
                                continue
                            crop=PerspectiveTransform(resorted_polygons_out[i],crop_img)
                            text_pred,confi=TextRecognizer.recog4direc(crop)
                            nums_list.append(text_pred)
                            

                            
                            # draw_polygon(resorted_polygons_out[i],img_label)

                        # arithmetic_points_out=get_arithmetic_points(nums_list,reordered_angle_out,resorted_polygons_out,resorted_points_out)
                        print("OCR识别外径:nums_list",nums_list)
                        return_value = check_list(nums_list,outer_list,reordered_angle_out,resorted_polygons_out,resorted_points_out,x1,y1,crop_img)
                        if return_value is not None:
                            new_list, new_points, new_polygons, new_angles = return_value
                            # 继续处理解包后的变量
                            # ...
                        else:
                            # 返回值为 None 的情况
                            # 在这里可以添加你想要执行的操作，或者直接使用 pass 关键字跳过
                            new_list, new_points, new_polygons, new_angles =  [], [], [], []
                        
                        
                        if len(new_list)!=0:
                            corrected_nums =new_list
                            resorted_polygons_out=new_polygons
                            resorted_points_out=new_points
                            reordered_angle_out=new_angles
                            occlusion_out=1
                            pointer_index,pointer_angle=find_pointer(resorted_polygons_out,reordered_angle_out)
                            polygons_nopointer =copy.deepcopy(resorted_polygons_out)
                            del polygons_nopointer[pointer_index]
                        else:
                            corrected_nums,occlusion_out,polygons_nopointer = correct_nums(nums_list,reordered_angle_out,resorted_polygons_out,resorted_points_out,x1,y1,crop_img)
                        print("修正后的外径",corrected_nums)

                            
                        arithmetic_points_out = get_arithmetic_points(corrected_nums,resorted_points_out,resorted_polygons_out,reordered_angle_out,occlusion_out,polygons_nopointer)
                        print("外径点个数",len(resorted_points_out))
                        print("外径点",resorted_points_out)
                        for i in range(len(resorted_points_out)):
                            if np.array_equal(resorted_polygons_out[i], [[0, 0], [1, 0], [1, 1], [0, 1]]):
                                continue

                            a = np.array(resorted_polygons_out[i])
                            a = a.astype(int)
                            for k in range(len(a)):
                                a[k][0] += xmin
                                a[k][1] += ymin
                            keypoints.append({"name": "outer_long_graduation" + str(cnt), "point": [float(resorted_points_out[i][0]+xmin), float(resorted_points_out[i][1]+ymin)], "confidence": 0.87})
                            if not np.array_equal(resorted_polygons_out[i], [[0, 0], [10, 0], [10, 10], [0, 10]]):                                          
                                graduation = {"name": "graduation", "points": a.flatten().tolist(), "number": str(corrected_nums[i]), "confidence": 0.85}
                                polygons.append(graduation)
                            # if np.array_equal(resorted_polygons_out[i], [[0, 0], [10, 0], [10, 10], [0, 10]]):
                            #     draw_grad_kpt_lable(resorted_points_out[i], corrected_nums[i], img_label)
                            # else:
                            #     draw_polygon_label(resorted_polygons_out[i],corrected_nums[i],img_label)
                            # draw_grad_kpt(resorted_points_out[i][0], resorted_points_out[i][1], img_label)
                        
                            cnt+=1
                        # res=(reordered_angle_out[point_index]-reordered_angle_out[point_index-1])/(reordered_angle_out[point_index+1]-reordered_angle_out[point_index-1])*(float(aft_num)-float(pre_num))+float(pre_num)
                        pointer_index,pointer_angle=find_pointer(resorted_polygons_out,reordered_angle_out)
                        
                        
                        gt_nums_out=copy.deepcopy(corrected_nums)
                        gt_angle_out=copy.deepcopy(reordered_angle_out)
                        gt_polygons_out=copy.deepcopy(polygons_nopointer)
                        del gt_nums_out[pointer_index]
                        del gt_angle_out[pointer_index]
                        # del gt_polygons_out[pointer_index]
                        print("传入变换的nopointer的gt_nums_out",gt_nums_out)
                        pointer_out=pointer_index

                        
                        res=corrected_nums[pointer_index]
                        measures.append(res)
                        partial_polygons2 = [x for x in resorted_polygons_out if isinstance(x, np.ndarray)]
                        # Remove corresponding elements from lst2
                        partial_nums2= [x for i, x in enumerate(gt_nums_out) if isinstance(polygons_nopointer[i], np.ndarray)]
                    if len(result_angle_in)>1:#内径有点
                        cnt=0
                        nums_list=[]
                        
                        for i in range(len(resorted_polygons_in)):#polygon放入polygons
                            if np.array_equal(resorted_polygons_in[i], [[0, 0], [1, 0], [1, 1], [0, 1]]):
                                

                                continue
                            crop=PerspectiveTransform(resorted_polygons_in[i],crop_img)
                            text_pred,confi=TextRecognizer.recog4direc(crop)
                            nums_list.append(text_pred)
                            

                            # draw_polygon(resorted_polygons_in[i],img_label)
                            # draw_polygon_label(resorted_polygons_in[i],text_pred,img_label,i)
                            
                            # a = np.array(resorted_polygons_in[i])
                            # a = a.astype(int)
                            # for i in range(len(a)):
                            #     a[i][0] += xmin
                            #     a[i][1] += ymin
                            # a = [int(xmin + a[i][0]) if i % 2 == 0 else int(ymin + a[i]) for i in range(len(a))]
                            # a.append(a)
                        # arithmetic_points_in=get_arithmetic_points(nums_list,reordered_angle_in,resorted_polygons_in,resorted_points_in)
                        # print("arithmetic_points_out",arithmetic_points_out)
                        # print("arithmetic_points_in",arithmetic_points_in)


                        print("OCR识别内径:nums_list",nums_list)
                        
                        return_value = check_list(nums_list,inner_list,reordered_angle_in,resorted_polygons_in,resorted_points_in,x1,y1,crop_img)
                        if return_value is not None:
                            new_list, new_points, new_polygons, new_angles = return_value
                            # 继续处理解包后的变量
                            # ...
                        else:
                            # 返回值为 None 的情况
                            # 在这里可以添加你想要执行的操作，或者直接使用 pass 关键字跳过
                            new_list, new_points, new_polygons, new_angles =  [], [], [], []
                        if len(new_list)!=0:
                            corrected_nums =new_list
                            resorted_polygons_in=new_polygons
                            resorted_points_in=new_points
                            reordered_angle_in=new_angles
                            occlusion=1
                            pointer_index,pointer_angle=find_pointer(resorted_polygons_in,reordered_angle_in)
                            polygons_nopointer =copy.deepcopy(resorted_polygons_in)
                            del polygons_nopointer[pointer_index]
                        else:

                            corrected_nums,occlusion,polygons_nopointer = correct_nums(nums_list,reordered_angle_in,resorted_polygons_in,resorted_points_in,x1,y1,crop_img)
                        
                        if len(result_angle_out)>2 and corrected_nums[-1]==1:
                            corrected_nums[-1]==1.0
                        if len(result_angle_out)<2 and corrected_nums[-1]==1:
                            corrected_nums[-1]==1
                                                    


                        print("修正后的内径",corrected_nums)
                        arithmetic_points_in = get_arithmetic_points(corrected_nums,resorted_points_in,resorted_polygons_in,reordered_angle_in,occlusion,polygons_nopointer)
                        
                        print("arithmetic_points_out",arithmetic_points_out)
                        print("arithmetic_points_in",arithmetic_points_in)
                        print("内径occlusion",occlusion)

                        print("外径occlusion",occlusion_out)


                        for i in range(len(resorted_points_in)):
                            if np.array_equal(resorted_polygons_in[i], [[0, 0], [1, 0], [1, 1], [0, 1]]):
                                continue

                            a = np.array(resorted_polygons_in[i])
                            a = a.astype(int)
                            for k in range(len(a)):
                                a[k][0] += xmin
                                a[k][1] += ymin
                            keypoints.append({"name": "long_graduation" + str(cnt), "point": [float(resorted_points_in[i][0]+xmin), float(resorted_points_in[i][1]+ymin)], "confidence": 0.87})
                            if not np.array_equal(resorted_polygons_in[i], [[0, 0], [10, 0], [10, 10], [0, 10]]):
                                                                    
                                graduation = {"name": "graduation", "points": a.flatten().tolist(), "number": str(corrected_nums[i]), "confidence": 0.85}
                                polygons.append(graduation)
                            # if np.array_equal(resorted_polygons_in[i], [[0, 0], [10, 0], [10, 10], [0, 10]]):
                            #     draw_grad_kpt_lable(resorted_points_in[i], corrected_nums[i], img_label)
                            # else:
                            #     draw_polygon_label(resorted_polygons_in[i],corrected_nums[i],img_label)
                            # draw_grad_kpt(resorted_points_in[i][0], resorted_points_in[i][1], img_label)
                        
                            
                            cnt+=1

                        
                        # res=(reordered_angle_in[point_index]-reordered_angle_in[point_index-1])/(reordered_angle_in[point_index+1]-reordered_angle_in[point_index-1])*(float(aft_num)-float(pre_num))+float(pre_num)
                        pointer_index,pointer_angle=find_pointer(resorted_polygons_in,reordered_angle_in)
                        res=corrected_nums[pointer_index]
                        gt_polygons_in=copy.deepcopy(polygons_nopointer)
                        gt_nums_in=copy.deepcopy(corrected_nums)
                        gt_angle_in=copy.deepcopy(reordered_angle_in)
                        del gt_nums_in[pointer_index]
                        del gt_angle_in[pointer_index]
                        # del gt_polygons_in[pointer_index]
                        print("传入变换的nopointer的gt_nums_in",gt_nums_in)
                        measures.append(res)
                        pointer_in=pointer_index
                        # print(res)
                    # print(keypoints)
                    # print(polygons)
                        partial_polygons1 = [x for x in resorted_polygons_in if isinstance(x, np.ndarray)]
                        # Remove corresponding elements from lst2
                        partial_nums1 = [x for i, x in enumerate(gt_nums_in) if isinstance(polygons_nopointer[i], np.ndarray)]

                    nums_list= [[0, 1, 2, 3, 4], 
                                [0, 4, 8, 12, 16],
                                [0, 0.4, 0.8, 1.2, 1.6],
                                [0, 1, 2, 3, 4, 5], 
                                [0, 20, 40, 60, 80, 100], 
                                [0, 2, 4, 6, 8, 10], 
                                [0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                                [0, 0.02, 0.04, 0.06, 0.08, 0.1], 
                                [0, 1, 2, 3, 4, 5, 6], 
                                [0, 10, 20, 30, 40, 50, 60],
                                [0,5,10,15,20,25],
                                [0,0.5,1,1.5,2,2.5]]

                    arithmetic_points=[]
                    gt_nums=[]
                    ari_occ=-1
                    tpointer=-1
                    tpolygons=[]
                    flag=0#

                    tnums1=[]
                    tidx1=[]
                    tnums2=[]
                    tidx2=[]

                    tnums=[]
                    tidx=[]
                    tpolygons_other=[]
                    reverse_flag=-1
                    if len(gt_nums_in)>0:
                                    
                        tnums1,tidx1=find_index(nums_list,partial_nums1)
                    if len(gt_nums_out)>0:
                                                    
                        tnums2,tidx2=find_index(nums_list,partial_nums2)
                    print("partial_nums1",partial_nums1)   
                    print("partial_nums2",partial_nums2)   
                    print("tnums1",tnums1)
                    print("tnums2",tnums2)
                    # 进行特例修补
                    if len(tnums1)>0 and len(tnums2)>0:
                        if tnums1==[0, 0.2, 0.4, 0.6, 0.8, 1.0] and tnums2==[0, 2, 4, 6, 8, 10]:#特例1_*
                            if tidx1[0]!=0:
                                    
                                tidx1.insert(0,0)#补一个0，需要外径的角度，如果外径有0
                                avg_R=get_avgR(arithmetic_points_in,x1,y1)    
                                if gt_nums_out[0]==0:
                    
                                    xr, yr = cv2.polarToCart(avg_R, gt_angle_out[0], angleInDegrees=1)#di一个点
                                    xr[0][0] += x1
                                    yr[0][0] += y1
                                    arithmetic_points_in.insert(0,(0,(xr[0][0],yr[0][0])))
                                else:
                                    tidx2.insert(0,0)
                                    avg_R2=get_avgR(arithmetic_points_out,x1,y1)   
                                    xr, yr = cv2.polarToCart(avg_R, 135, angleInDegrees=1)#di一个点
                                    xr[0][0] += x1
                                    yr[0][0] += y1
                                    arithmetic_points_in.insert(0,(0,(xr[0][0],yr[0][0])))
                                    xr, yr = cv2.polarToCart(avg_R2, 135, angleInDegrees=1)#di一个点
                                    xr[0][0] += x1
                                    yr[0][0] += y1
                                    arithmetic_points_out.insert(0,(0,(xr[0][0],yr[0][0])))
                                        
                    #######TODO 8_*、12_*、28_*








                    tflag=0
                    if len(tnums1)>0:
                        if len(tidx1)>3:
                            if len(tidx1)==4 and tidx1[0]==0:
                                pass
                            elif len(tidx1)==4 and tidx1[0]!=0:
                                if tidx1[-1]<len(tnums1)-1 and len(tnums2)>0:
                                    pass
                            else:
                                            
                                arithmetic_points=arithmetic_points_in
                                ari_occ=occlusion
                                gt_nums=gt_nums_in   
                                tpointer=pointer_in  
                                tnums=tnums1
                                tidx=tidx1
                                tpolygons=gt_polygons_in
                                tflag=1

                                arithmetic_points_other=arithmetic_points_out
                                ari_occ_other=occlusion_out
                                gt_nums_other=gt_nums_out   
                                tpointer_other=pointer_out
                                tnums_other=tnums2
                                tidx_other=tidx2
                                tpolygons_other=gt_polygons_out
                                
                    
                    if len(tnums2)>0 and tflag==0:
                    # if len(tnums2)>0 :
                                    
                        if len(tidx2)>3:
                            if len(tidx2)==4 and tidx2[0]==0:
                                pass
                            else:
                                arithmetic_points=arithmetic_points_out
                                ari_occ=occlusion_out
                                gt_nums=gt_nums_out 
                                tpointer=pointer_out                                        
                                tnums=tnums2 
                                tidx=tidx2      
                                tpolygons=gt_polygons_out

                                arithmetic_points_other=arithmetic_points_in
                                ari_occ_other=occlusion
                                gt_nums_other=gt_nums_in 
                                tpointer_other=pointer_in
                                tnums_other=tnums1
                                tpolygons_other=gt_polygons_in
                                tidx_other=tidx1
                                reverse_flag=1
                    # if len(gt_nums_in)==0 or pointer_in==0:
                                                        
                    #     print("forget")
                    # # elif len(reordered_angle_in)>0:
                    # #     difang=reordered_angle_in[-1]-reordered_angle_in[0]
                    # #     if difang<0 and difang+360<180:
                    # #         difang+=360
                    # #     if difang<180:#方块样例
                    # #         pass
                    # elif occlusion!=1 and gt_nums_in[0]!=0 or occlusion!=1 and occlusion_out==-1 and pointer_in==0:#1_71特例、22_7特例

                    #     print("forget")
                                                                        
                    # elif len(arithmetic_points_in)>3:
                            
                
                    #     if len(arithmetic_points_in)==4 and arithmetic_points_in[0][0]==0:
                    #         pass                                                   
                    #     elif len(arithmetic_points_in)==5 and occlusion==1 and arithmetic_points_in[0][0]==0:
                    #         pass
                                                                                                                                                                                                                                                                                                                                                                        
                    #     else:
                    #         arithmetic_points=arithmetic_points_in
                    #         ari_occ=occlusion
                    #         gt_nums=gt_nums_in   
                    #         tpointer=pointer_in    

                    # elif len(arithmetic_points_out)>0:
                    #     if len(arithmetic_points_out)==4 and arithmetic_points_out[0][0]==0 or len(arithmetic_points_out)<4:
                    #         pass
                    #     else:                                                
                    #         arithmetic_points=arithmetic_points_out
                    #         ari_occ=occlusion_out
                    #         gt_nums=gt_nums_out
                    #         tpointer=pointer_out
                    print(tnums)
                    print(tidx)


                    if len(arithmetic_points)>0:#透视变换
                        print("进入变换")
                        stride=270/(len(tnums)-1)
                        start_angle=135           


                        # 遍历每个元素，计算距离并将其添加到距离列表中
                        avg_R = get_avgR(arithmetic_points,x1,y1)

                        pts1=[]
                        pts2=[]

                        print(gt_nums)

                        l=0
                        r=len(tidx)-1
                        if tidx[0]==0: 
                            l=1
                        degree = start_angle+float(tidx[l])*stride
                        if degree>360:
                            degree-=360
                        xr, yr = cv2.polarToCart(avg_R, degree, angleInDegrees=1)#di一个点
                        xr[0][0] += x1
                        yr[0][0] += y1
                        tx,ty=arithmetic_points[l][1]

                        pts1.append([float(tx), float(ty)])
                        pts2.append([xr[0][0], yr[0][0]])
                        degree = start_angle+float(tidx[r])*stride
                        if degree>360:
                            degree-=360
                        xr, yr = cv2.polarToCart(avg_R, degree, angleInDegrees=1)#最后一个点
                        xr[0][0] += x1
                        yr[0][0] += y1

                        tx,ty=arithmetic_points[r][1]
                        pts1.append([float(tx), float(ty)])
                        pts2.append([xr[0][0], yr[0][0]])
                        # print(avg_R)


                        
                        print(l,r)
                        for i in range(2):

                            l=math.floor((l+r)/2)

                            print("cur",l)
                            degree = start_angle+float(tidx[l])*stride
                            if degree>360:
                                degree-=360
                            xr, yr = cv2.polarToCart(avg_R, degree, angleInDegrees=1)#最后一个点是固定位置
                            xr[0][0] += x1
                            yr[0][0] += y1

                            tx,ty=arithmetic_points[l][1]
                            pts1.append([float(tx), float(ty)])
                            pts2.append([xr[0][0], yr[0][0]])

                        pts1 = np.float32(pts1)
                        pts2 = np.float32(pts2)
                        M = cv2.getPerspectiveTransform(pts1, pts2)



                        # xr, yr = cv2.polarToCart(avg_R, 135, angleInDegrees=1)#零点
                        # xr[0][0] += x1
                        # yr[0][0] += y1
                        # dst = cv2.warpPerspective(img, M, (max(new_height,new_width),max(new_height,new_width)))
                        dst = cv2.warpPerspective(crop_img, M, (w,h))
                        # cv2.circle(dst, (int(x1), int(y1)), 5, (255, 206, 135), -1)
                        # tx1,ty1=get_reshape_point(M,x1,y1)#####center也需要转换(其实不用)
                        # cv2.circle(dst, (int(tx1), int(ty1)), 5, (255, 255, 0), -1)
                        # cv2.circle(dst, (int(xr[0][0]), int(yr[0][0])), 5, (255, 0, 255), -1)
                        

                        # expected_num=0
                        # for element in arithmetic_points:
                        #     # 获取元素中的数字
                        #     num = element[0]
                            
                        #     # 如果数字不等于预期数字，说明出现了缺失
                        #     if num != expected_num:
                        #         break
                        #     expected_num = num + 1

                    
                        tpoints=[]
                        tangles=[]
                        tpoints_other=[]
                        tangles_other=[]
                        for i in range(len(tidx)):
                                                    
                            element=arithmetic_points[i][1]

                        # for element in arithmetic_points:

                            tx,ty=element
                            tx,ty=get_reshape_point(M,tx,ty)
                            magnitude, angle = cv2.cartToPolar(float(tx)-x1,float(ty)-y1, angleInDegrees=True)
                            
                            tpoints.append([tx,ty])
                            # if tidx[i]==0:
                                                        
                            #     tangles.append(135)
                            # else:
                            tangles.append(angle[0][0])
                            # cv2.circle(dst, (int(tx), int(ty)), 5, (255, 0, 255), -1)
                        # cv2.circle(dst, (int(x1), int(y1)), 5, (255, 0, 255), -1)
                        nums_list2= [[0,1000,2000,3000,4000,5000,6000],
                                    [0, 10, 20, 30, 40, 50, 60]
                                    ]

                        if len(tnums)!=len(tidx):#补充被遮挡点位

                            for i in range(len(tnums)):
                                if len(tidx)>i and tidx[i]!=i or len(tidx)==i:#or缺失最后一个点
                                                                                                                                                                                                
                                    degree = start_angle+float(i)*stride
                                    if degree>360:
                                        degree-=360
                                    xr, yr = cv2.polarToCart(avg_R, degree, angleInDegrees=1)#最后一个点是固定位置
                                    xr[0][0] += x1
                                    yr[0][0] += y1
                                    magnitude, angle = cv2.cartToPolar(float(xr[0][0])-x1,float(yr[0][0])-y1, angleInDegrees=True)
                                    tidx.insert(i,i)
                                    tpoints.insert(i,[xr[0][0],yr[0][0]])#遮住的点补充了
                                    tangles.insert(i,angle[0][0])
                                    # cv2.circle(dst, (int(xr[0][0]), int(yr[0][0])), 5, (255, 0, 0), -1)
                                # if len(tidx)==i:#缺失最后一个点

                                #     degree = start_angle+float(i)*stride
                                #     if degree>360:
                                #         degree-=360
                                #     xr, yr = cv2.polarToCart(avg_R, degree, angleInDegrees=1)#最后一个点是固定位置
                                #     xr[0][0] += x1
                                #     yr[0][0] += y1
                                #     magnitude, angle = cv2.cartToPolar(float(xr[0][0])-x1,float(yr[0][0])-y1, angleInDegrees=True)
                                #     tidx.insert(i,i)
                                #     tpoints.insert(i,[xr[0][0],yr[0][0]])#没检测到的点补充了
                                #     tangles.insert(i,angle[0][0])
                                #     cv2.circle(dst, (int(xr[0][0]), int(yr[0][0])), 5, (255, 0, 0), -1)


                        if tnums1==[0, 10, 20, 30, 40, 50, 60] and len(tnums2)==0 and len(partial_nums2)>0:
                                                                                            
                            # print("partial_nums2",partial_nums2)
                            tnums2,tidx2=find_index(nums_list2,partial_nums2)
                            if len(tnums2)>0:
                                # print("tnums2",tnums2)
                                print("进入opencv的partial_polygons2",partial_polygons2)
                                out_points=get_out_point_opencv(crop_img,x1,y1,partial_polygons2,x2,y2)
                                print("opencvvvvvvvvvvvvvv返回点的个数",len(out_points))
                                for i in range(len(out_points)):                                                        
                                    tx,ty=out_points[i]
                                    tx,ty=get_reshape_point(M,tx,ty)
                                    magnitude, angle = cv2.cartToPolar(float(tx)-x1,float(ty)-y1, angleInDegrees=True)
                                    # cv2.circle(dst, (int(tx), int(ty)), 5, (255, 0, 255), -1)
                                    tpoints_other.append([tx,ty])
                                    tangles_other.append(angle[0][0])
                                avg_R2=get_avgR2(tpoints_other,x1,y1)
                                if tidx2[0]!=0:#加点最好等
                                                
                                    tidx2.insert(0,0)#补一个0，需要外径的角度，如果外径有0
                                        
                                    if tidx[0]==0:
                                
                                        xr, yr = cv2.polarToCart(avg_R2, tangles[0], angleInDegrees=1)#di一个点
                                        xr[0][0] += x1
                                        yr[0][0] += y1
                                        tpoints_other.insert(0,(xr[0][0],yr[0][0]))
                                        tangles_other.insert(0,tangles[0])
                                    # for i in range(len(tpoints_other)):
                                        # cv2.circle(dst, (int(tpoints_other[i][0]), int(tpoints_other[i][1])), 5, (255, 255, 0), -1)
                                stride2=270/(len(tnums2)-1)
                                if len(tnums2)!=len(tidx2):    
                                    for i in range(len(tnums2)):
                                        if len(tidx2)>i and tidx2[i]!=i or len(tidx2)==i:#or缺失最后一个点
                                                                                                                                                                                                        
                                            degree2 = start_angle+float(i)*stride2
                                            if degree2>360:
                                                degree2-=360
                                            xr, yr = cv2.polarToCart(avg_R2, degree2, angleInDegrees=1)#最后一个点是固定位置
                                            xr[0][0] += x1
                                            yr[0][0] += y1
                                            magnitude, angle = cv2.cartToPolar(float(xr[0][0])-x1,float(yr[0][0])-y1, angleInDegrees=True)
                                            tidx2.insert(i,i)
                                            tpoints_other.insert(i,[xr[0][0],yr[0][0]])#遮住的点补充了
                                            tangles_other.insert(i,angle[0][0])
                                            # cv2.circle(dst, (int(xr[0][0]), int(yr[0][0])), 5, (255, 255, 0), -1)
                                tnums2=[0,1000,2000,3000,4000,5000,6000]
                            # arithmetic_points_other=out_points
                            # ari_occ_other=occlusion_out
                            # gt_nums_other=gt_nums_out   
                            # tpointer_other=pointer_out
                            tnums_other=tnums2
                            tidx_other=tidx2
                        elif tnums1==[0, 20, 40, 60, 80, 100] :#特例，此时
                                                                                    
                            print("partial_nums2",partial_nums2)
                            # tnums2,tidx2=find_index(nums_list2,partial_nums2)
                            tnums2=[32,50,100,150,200]
                            tidx2=[0,1,2,3,4]

                            temp_points=[]                   
                            for i in range(len(arithmetic_points_out)):                                             
                                element=arithmetic_points_out[i][1]
                                tx,ty=element
                                tx,ty=get_reshape_point(M,tx,ty)
                                temp_points.append([tx,ty])
                            if len(temp_points)==0:
                                avg_R2=avg_R*1.3
                            else:
                                avg_R2=get_avgR2(temp_points,x1,y1)

                            for i in range(len(tnums2)):
                                C = (tnums2[i] - 32) * 5/9                                                                                                                                                                           
                                degree2 = start_angle+C*2.7
                                if degree2>360:
                                    degree2-=360
                                xr, yr = cv2.polarToCart(avg_R2, degree2, angleInDegrees=1)#最后一个点是固定位置
                                xr[0][0] += x1
                                yr[0][0] += y1
                                magnitude, angle = cv2.cartToPolar(float(xr[0][0])-x1,float(yr[0][0])-y1, angleInDegrees=True)
                                tpoints_other.insert(i,[xr[0][0],yr[0][0]])#遮住的点补充了
                                tangles_other.insert(i,angle[0][0])
                                # cv2.circle(dst, (int(xr[0][0]), int(yr[0][0])), 5, (255, 255, 0), -1)
                            tnums_other=tnums2
                            tidx_other=tidx2


                        elif len(tnums_other)>0:
                            for i in range(len(tidx_other)):
                                                                
                                element=arithmetic_points_other[i][1]

                            # for element in arithmetic_points:

                                tx,ty=element
                                tx,ty=get_reshape_point(M,tx,ty)
                                magnitude, angle = cv2.cartToPolar(float(tx)-x1,float(ty)-y1, angleInDegrees=True)
                                
                                tpoints_other.append([tx,ty])
                                if tidx_other[i]==0:
                                                                
                                    tangles_other.append(135)
                                else:
                                    tangles_other.append(angle[0][0])
                                # cv2.circle(dst, (int(tx), int(ty)), 5, (255, 0, 255), -1)
                            # cv2.circle(dst, (int(x1), int(y1)), 5, (255, 0, 255), -1)

                            avg_R2=get_avgR2(tpoints_other,x1,y1)
                            if len(tnums_other)!=len(tidx_other):#补充被遮挡点位

                                for i in range(len(tnums_other)):
                                    if len(tidx_other)>i and tidx_other[i]!=i or len(tidx_other)==i:#or缺失最后一个点
                                                                                                                                                                                                    
                                        degree = start_angle+float(i)*stride
                                        if degree>360:
                                            degree-=360
                                        xr, yr = cv2.polarToCart(avg_R2, degree, angleInDegrees=1)#最后一个点是固定位置
                                        xr[0][0] += x1
                                        yr[0][0] += y1
                                        magnitude, angle = cv2.cartToPolar(float(xr[0][0])-x1,float(yr[0][0])-y1, angleInDegrees=True)
                                        tidx_other.insert(i,i)
                                        tpoints_other.insert(i,[xr[0][0],yr[0][0]])#遮住的点补充了
                                        tangles_other.insert(i,angle[0][0])
                                        # cv2.circle(dst, (int(xr[0][0]), int(yr[0][0])), 5, (255, 0, 0), -1)
                
                        ##不在打表里的数组，先做个初版，把点移动到仿射变换后的图，计算新的角度和度数。
                        elif len(arithmetic_points_other)!=0:#normal
                        # else:
                
                            avg_R=get_avgR(arithmetic_points_other,x1,y1)
                            for i in range(len(arithmetic_points_other)):
                                temp_idx=arithmetic_points_other[i][0]
                                # if i == temp_idx:                                                      
                                tx,ty=arithmetic_points_other[i][1]#识别出刻度点的
                                                
                                tx,ty=get_reshape_point(M,tx,ty)
                                magnitude, angle = cv2.cartToPolar(float(tx)-x1,float(ty)-y1, angleInDegrees=True)
                                # cv2.circle(dst, (int(tx), int(ty)), 5, (135,206, 255), -1)
                                tpoints_other.append([tx,ty])
                                if i==0:
                                    tangles_other.append(135)
                                else:
                                    tangles_other.append(angle[0][0])
                 
                            for i, item in enumerate(arithmetic_points_other):
                                temp_idx=arithmetic_points_other[i][0]
                                if i != temp_idx:    
                                    if i!=0 and i<len(arithmetic_points_other)-1 or i!=0 and i<temp_idx:#遮挡点在中间时
                                            
                                        #获取前后点
                                        pre_ang=tangles_other[arithmetic_points_other[i-1][0]]
                                        aft_ang=tangles_other[arithmetic_points_other[i-1][0]+1]#默认取中间值
                                        if aft_ang<90 and pre_ang>270:
                                            aft_ang+=360
                                            mid_ang=((aft_ang+pre_ang)/2)%360
                                        else:
                                            mid_ang=((aft_ang+pre_ang)/2)
                                        avg_R=get_avgR2(tpoints_other,x1,y1)

                                        xr, yr = cv2.polarToCart(avg_R, mid_ang, angleInDegrees=1)#最后一个点是固定位置
                                        xr[0][0] += x1
                                        yr[0][0] += y1
                                        magnitude, angle = cv2.cartToPolar(float(xr[0][0])-x1,float(yr[0][0])-y1, angleInDegrees=True)
                                        # tidx2.insert(i,i)
                                        tpoints_other.insert(i,[xr[0][0],yr[0][0]])#遮住的点补充了
                                        tangles_other.insert(i,angle[0][0])
                                        arithmetic_points_other.insert(i,(i,(xr[0][0],yr[0][0])))
                                        # magnitude, angle = cv2.cartToPolar(float(tx)-tx1,float(ty)-ty1, angleInDegrees=True)
                                        # cv2.circle(dst, (int(xr[0][0]), int(yr[0][0])), 5, (255,206, 135), -1)
                                        # tpoints_other.append([tx,ty])
                                        # tangles_other.append(angle[0][0])
                            tnums_other = gt_nums_other
                            if len(tnums_other)!= len(arithmetic_points_other):
                                avg_R=get_avgR2(tpoints_other,x1,y1)
                                for k in range(len(arithmetic_points_other),len(tnums_other)):

                                    pre_ang=tangles_other[k-1]-tangles_other[k-2]
                                    if pre_ang<0:
                                        pre_ang+=360
                                    pre_diff=gt_nums_other[k-1]-gt_nums_other[k-2]
                                    now_diff=gt_nums_other[k]-gt_nums_other[k-1]
                                    if pre_diff==0:
                                        r=0.5
                                    else:
                                        r=now_diff/pre_diff
                                    new_angle=tangles_other[k-1]+pre_ang*(r)
                                    if new_angle>360:
                                        new_angle-=360
                                    xr, yr = cv2.polarToCart(avg_R, new_angle, angleInDegrees=1)#最后一个点是固定位置
                                    xr[0][0] += x1
                                    yr[0][0] += y1
                                    magnitude, angle = cv2.cartToPolar(float(xr[0][0])-x1,float(yr[0][0])-y1, angleInDegrees=True)
                                    # tidx2.insert(i,i)
                                    tpoints_other.insert(k,[xr[0][0],yr[0][0]])#遮住的点补充了
                                    tangles_other.insert(k,angle[0][0])
                                    arithmetic_points_other.insert(k,(k,(xr[0][0],yr[0][0])))


                            


                        tx,ty=get_reshape_point(M,x2,y2)#添加指针
                        # cv2.circle(dst, (int(tx), int(ty)), 5, (255, 0, 255), -1)
                        magnitude, angle = cv2.cartToPolar(float(tx)-x1,float(ty)-y1, angleInDegrees=True)
                        # if ari_occ==1:
                        #     difangle=tangles[expected_num]-angle[0][0]
                        #     if difangle<0 and difangle+360<180:
                        #         difangle+=360
                        #     if difangle>0:
                        #         tpointer=expected_num
                        #     else:
                        #         tpointer=expected_num+1
                        # if tpointer<len(gt_nums)-1:
                        #     if tangles[tpointer]<angle[0][0] and tangles[tpointer+1]<angle[0][0]:
                        tpointer = locate_pointer(tangles,angle[0][0])
                        tangles.insert(tpointer,angle[0][0])
                        tpoints.insert(tpointer,[tx,ty])
                        tnums.insert(tpointer,0)
                        get_pointer_num(tnums,tangles,tpointer)
                        M_inv = np.linalg.inv(M)

                        if len(tnums_other)>0:
                            tpointer_other = locate_pointer(tangles_other,angle[0][0])
                            tangles_other.insert(tpointer_other,angle[0][0])
                            tpoints_other.insert(tpointer_other,[tx,ty])
                            tnums_other.insert(tpointer_other,0)
                            get_pointer_num(tnums_other,tangles_other,tpointer_other)
                            print("measure_other2",tnums_other[tpointer_other])
                            print("pointer_index",tpointer_other)
                            t_measure.append(tnums_other[tpointer_other])

                            cnt=0
                            print("tnums_other",tnums_other,len(tnums_other))
                            print("tpoints_other",tpoints_other,len(tnums_other))
                            for i in range(len(tnums_other)):
                                if i==tpointer_other:
                                    continue
                                if i==len(tpoints_other):
                                    break
                                ttx,tty=tpoints_other[i]
                                homogeneous_coords = np.dot(M_inv, np.array([ttx, tty, 1]))
                                x = homogeneous_coords[0] / homogeneous_coords[2]
                                y = homogeneous_coords[1] / homogeneous_coords[2]
                                if reverse_flag==1:
                                    t_keypoints.append({"name": "long_graduation" + str(cnt), "point": [int(x+xmin), int(y+ymin)], "confidence": 0.87})
                                else:
                                    t_keypoints.append({"name": "outer_long_graduation" + str(cnt), "point": [int(x+xmin), int(y+ymin)], "confidence": 0.87})
                                cnt+=1


                        cnt=0
                        print("!!!!!!!!输出测试!!!!!!!!!!!!")
                        print(tnums)
                        print(tpointer)
                        print(tpoints)
                        print("!!!!!!!!!!!!!!!!!!!!")
                        for i in range(len(tnums)):
                            if i==tpointer:
                                continue
                            if i==len(tpoints):
                                break
                            ttx,tty=tpoints[i]
                            homogeneous_coords = np.dot(M_inv, np.array([ttx, tty, 1]))
                            x = homogeneous_coords[0] / homogeneous_coords[2]
                            y = homogeneous_coords[1] / homogeneous_coords[2]
                            if reverse_flag==1:
                                t_keypoints.append({"name": "outer_long_graduation" + str(cnt), "point": [int(x+xmin), int(y+ymin)], "confidence": 0.87})
                            else:
                                t_keypoints.append({"name": "long_graduation" + str(cnt), "point": [int(x+xmin), int(y+ymin)], "confidence": 0.87})
                            cnt+=1
                        t_measure.append(tnums[tpointer])
                        print(tnums[tpointer])
                        print("pointer_index",tpointer)


            if x1 >= 0 and y1 >= 0 and x2 >= 0 and y2 >= 0:
                                        
                cx=x1
                cy=y1
                center_pt = {"name": "center", "point": [x1+xmin, y1+ymin], "confidence": 0.87}
                tip_pt = {"name": "pointer_tip", "point": [x2+xmin, y2+ymin], "confidence": 0.87}
                if len(t_keypoints)>0:
                    t_keypoints.append(center_pt)
                    t_keypoints.append(tip_pt)  
                else:  
                    keypoints.append(center_pt)
                    keypoints.append(tip_pt)

            else:
                

                if len(t_keypoints)>0:
                    t_keypoints.append({"name": "center", "point": [50, 50], "confidence": 0.87})
                else:  
                    keypoints.append({"name": "center", "point": [50, 50], "confidence": 0.87})

                    
            measurements_dict = {}
            if len(measures)>0 and len(t_measure)==0:
                                                
                for i in range(len(measures)):
                    measurement_key = "measurement"+str(i+1)
                    measurement_value = measures[i]
                    measurements_dict[measurement_key] = measurement_value
            elif len(t_measure)>0:
                for i in range(len(t_measure)):
                    if reverse_flag==-1:    
    
                        measurement_key = "measurement"+str(i+1)
                        measurement_value = t_measure[i]
                        measurements_dict[measurement_key] = measurement_value    
                    else:
                        measurement_key = "measurement"+str(i+1)
                        measurement_value = t_measure[-1-i]
                        measurements_dict[measurement_key] = measurement_value    

            if len(t_keypoints)==0:
                                                                    
                detect_objs.append({
                    "name": className[int(cls)],
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    'confidence':float(conf),
                    'measurements':measurements_dict,
                    'keypoints':keypoints,
                    'polygons':polygons})

                target_info.append({
                    'name': className[int(cls)],
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax, 'confidence': float(conf),
                    "name": "gauge",
                    'measurements':measurements_dict
                    })
            else:
                                                            
                detect_objs.append({
                    "name": className[int(cls)],
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    'confidence':float(conf),
                    'measurements':measurements_dict,
                    'keypoints':t_keypoints,
                    'polygons':polygons})

                target_info.append({
                    'name': className[int(cls)],
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax, 'confidence': float(conf),
                    "name": "gauge",
                    'measurements':measurements_dict
                    })
            

    target_count = len(target_info)
    is_alert = True if target_count>0 else False
    return json.dumps({'algorithm_data':{'is_alert':is_alert, 'target_count':target_count, 'target_info':target_info}, 'model_data': {"objects": detect_objs}})


if __name__ == '__main__':
    """Test python api,注意修改图片路径
    """
    predictor = init()
    img = cv2.imread(
       "/project/train/src_repo/2116/ZDSappearance20230404_V2_sample_office_1_71.jpg")
    
    result = process_image(predictor, img)
    print(result)
