import numpy as np
import os
from PIL import Image as PImage
import numpy as np
from pylab import *
import copy
import argparse


def count(path):
    """
    Return the number of images in the path
    
    Examples:
    _________

    >>> path = "./Robbin_data/dataset2400/whole/images"
    >>> count(path)
    2400
    """
    count = 0
    for file in os.listdir(path): 
            count = count+1
    return count

def compute_IOU(rec1,rec2):
    """
    Return the IOU between two boxes 

    Parameters: 
    ____________
    :param rec1: (x0,y0,x1,y1)     
    :param rec2: (x0,y0,x1,y1)
    :return: IOU.

    Examples
    __________
    >>>r1=(2,3,4,5)
    >>>r2=(2,4,3,6)
    >>>compute_IOU(r1,r2)
    0.2

    """
    left_column_max  = max(rec1[0],rec2[0])
    right_column_min = min(rec1[2],rec2[2])
    up_row_max       = max(rec1[1],rec2[1])
    down_row_min     = min(rec1[3],rec2[3])

    if left_column_max>=right_column_min or down_row_min<=up_row_max:
        return 0

    else:
        S1 = (rec1[2]-rec1[0])*(rec1[3]-rec1[1])
        S2 = (rec2[2]-rec2[0])*(rec2[3]-rec2[1])
        S_cross = (down_row_min-up_row_max)*(right_column_min-left_column_max)
        return S_cross/(S1+S2-S_cross)
    
def label_rescale(array,img_size):
    """
    Convert the label from YOLO format into pixel coordinates
    Returns the X, Y, W, H in pixel coordinates

    Parameters
    ___________
    param:array: array loaded from the yolo label text file 
    param:img_size: shape of the image
    return:list of x,y,w,h for all craters 

    Example
    ________
    >>>x,y,w,h,txt = convert_label(array,img.size)

    """
    txtall = []
    xall = []
    yall = []
    wall = []
    hall = []
    txt_conf = []
    for n in range(len(array)):
        conf = []
        l = len(array[n])
        x = []
        y = []
        w = []
        h = []
        c = []
        txt = []
        #if one label in the label file
        if array[n].size == 5 or array[n].size == 6:
            x.append(array[n][1]*img_size[0])
            y.append(array[n][2]*img_size[0])
            w.append(array[n][3]*img_size[0])
            h.append(array[n][4]*img_size[0])
            # append the confidence score into the label file
            if n!=0:
                c.append(array[n][5])
            # Assign the confidence score to 1 if ground truth
            elif n==0:
                c.append(1)
                
            txt = [x,y,w,h,c]
            txt = np.reshape(txt,(1,5))
        # If more than 1 label in the label file
        else:
            for i in range(l):
                for j in range(5):
                    if j == 0 or j == 2:
                        txt.append(array[n][i][j+1]*img_size[1])
                    elif j == 1 or j == 3:
                        txt.append(array[n][i][j+1]*img_size[0])  
                    elif j == 4 and n!= 0:
                        txt.append(array[n][i][5])
                    elif j == 4 and n== 0:
                        txt.append(1)
        
            txt = np.reshape(txt, (l,5))
#         
            for i in range(l):
                x.append(txt[i][0])
                y.append(txt[i][1])
                w.append(txt[i][2])
                h.append(txt[i][3])    
        
        xall.append(x)
        yall.append(y)
        wall.append(w)
        hall.append(h)
        txtall.append(txt)

    return xall,yall,wall,hall,txtall
    
def iou_convert(l,txt):
    """
    Convert the label into IOU execuable format
    (x1,y1,x2,y2)

    Parameters:
    ___________
    param: l: len of array
    param: txt: label arrays of x,y,w,h
    return:(x1,y1,x2,y2)

    Example:
    ________
    >>>txt = [[2,3,1,1]]
    >>> iou_convert(1,txt)
   [[2, 3, 3, 4]]

    """
    txt_iou = []
    for j in range(len(txt)):
        txt_iou_single = txt[j].copy()
        for n in range(len(txt[j])):
            
            txt_iou_single[n][2] = txt[j][n][2]+txt[j][n][0]
            txt_iou_single[n][3] = txt[j][n][3]+txt[j][n][1]
        txt_iou.append(txt_iou_single)
    return txt_iou
    
def iou_label_convert(l,txt):
    """
    Convert the label into IOU execuable format
    (x1,y1,x2,y2)

    Parameters:
    ___________
    param: l: len of array
    param: txt: label arrays of x,y,w,h
    return:(x1,y1,x2,y2)

    Example:
    ________
    >>>txt = [[2,3,1,1]]
    >>> iou_convert(1,txt)
   [[2, 3, 3, 4]]

    """
    txt_iou = txt.copy()
        
    for i in range(l):
        txt_iou[i][2] = txt[i][2]+txt[i][0]
        txt_iou[i][3] = txt[i][3]+txt[i][1]
    
    return txt_iou

def size_array(arrayall):
    """
    Return a list that consists of len of each labels in the label file

    Parameters:
    ___________
    param: all arrays from the label file
    return: [1,2,3]

    Examples:
    _________
    >>>txt = [array([[2,3,1,1],[3,1,5,2]]),array([[2,4,5,2]])]
    >>>size_array(txt)
    [2,1]
    """
    l = []
    for i in range(len(arrayall)):
        if arrayall[i].size == 5:
            l.append(1)
        else:
            l.append(len(arrayall[i]))
    return l
    
def yolo_labels_convert(labels,img_size):
    """
    Convert the label format from pixel range into yolo format

    Parameters:
    ___________
    labels: results labels
    img_sie: image shape

    Examples:
    __________
    >>>labels = [[2, 3, 1, 1], [3, 1, 1, 2], [4, 4, 1, 1]]
    >>>img_size = [20,20]
    >>>yolo_labels_convert(labels,img_size)
    array([[0.1 , 0.15, 0.05, 0.05],
       [0.15, 0.05, 0.05, 0.1 ],
       [0.2 , 0.2 , 0.05, 0.05]])
    """
    labels_conv = []
    for i in range(len(labels)):
        for j in range(4):
            if j == 0 or j == 2:
                labels_conv.append(labels[i][j]/img_size[1])
            elif j == 1 or j == 3:
                labels_conv.append(labels[i][j]/img_size[0])  

    labels_conv = np.reshape(labels_conv, (len(labels),4))
    return labels_conv


def IOU_between_models(l1,l2,txt1,txt2,txt1_iou,txt2_iou):
    """
    Return a list of common labels between two models determined by
    their IoU

    Parameters:
    ___________
    param: l1, l2: array length of two label files from two models
    param: txt1, txt2: lists of labels from two models
    param: txt1_iou, txt_iou: lists of IoU exeutable format
    return: lists of common labels between two models 

    Example:
    _________
    >>>txt1 = [array([[2,3,1,1],[3,1,1,2]]),array([[2,3,1,1]])]
    >>>txt1_iou = iou_convert(l1,txt1)
    >>>IOU_between_models(len(txt1[0]),len(txt1[1]),txt1[0],txt1[1],txt1_iou[0],txt1_iou[1])
    [array([2, 3, 1, 1])]
    """
    labels = []
    txt_new = []
    counter1 = 0

    if l1 > l2:
        txtnew = txt1
    else:
        txtnew = txt2
        
    # If there is 1 label in the label file
    if l1 == 1 and l2 == 1:
        iou = compute_IOU(txt1_iou[0], txt2_iou[0])
        if iou > 0.5:
            labels.append(txtnew[0])

    # If there are multiple labels in the label file
    else:
        for i in range(max(l1,l2)):
            for j in range(min(l1,l2)):
                if l1 > l2:
                    iou = compute_IOU(txt1_iou[i], txt2_iou[j])
                    if iou > 0.50:
                        labels.append(txtnew[counter1])
                else:
                    iou = compute_IOU(txt1_iou[j], txt2_iou[i])
                    if iou > 0.50:
                        labels.append(txtnew[counter1])

            counter1 += 1

    return labels

def IOU_gt(txt_iou, l, l_label, labels_iou, labels,txt):
    """
    Return the result labels with blended undetected ground truth labels
    
    Parameters:
    ___________
    param: txt: labels from ground truth
    param: l: length of labeled from ground truth
    param: txt_iou: iou exeutable format of ground truth labels
    param: labels: result labels from IoU-based ensemble learning
    param: labels_iou: iou exeutable format of result labels from ensemble learning

    Examples:
    _________
    >>>labels = [[2,3,1,1],[3,1,1,2]]
    >>>txt = [[4,4,1,1]]
    >>>txt_iou = [[4,4,5,5]]
    >>>l = 1
    >>>l_label = 2
    >>>labels_iou = [[2,3,3,4],[3,1,4,3]]
    >>>IOU_gt(txt_iou, l, l_label, labels_iou, labels,txt)
    [[2, 3, 1, 1], [3, 1, 1, 2], [4, 4, 1, 1]]
    """
    counter3 = 0
    signal = np.zeros(len(txt_iou))

    #Compute  the IoU between ensemble results and ground truth
    for i in range(l):
        for j in range(l_label):
            iou = compute_IOU(txt_iou[i], labels_iou[j])
            if iou > 0.0:
                # Set the signal to 1 if the groudtruth and ensemble results overlap
                signal[counter3] = 1
        counter3 += 1
    
    for i in range(len(signal)):
        if signal[i] == 0:
            #If the ground truth and ensemble results does not overlap, save the corresponding labels into results
            labels.append(txt[i])      
    # print(signal)

    return labels

def conf_thres(labels):
    """
    Filter the result labels using confidence score on three crater scales 
    if area < 500:
        conf_threshold = 0.15
    elif 500<area<5000:
        conf_threshold = 0.4
    elif area > 5000
        conf_threshold = 0.55

    Parameters:
    ___________
    param: labels: resulting labels which include common label from two models and ground truth appended

    Examples:
    _________
    >>>labels= [[0.2, 0.3, 0.1, 0.1,0.3], [0.3, 0.1, 0.1, 0.2,0.5], [0.4, 0.4, 0.1, 0.1,0.1]]
    >>>conf_thres(labels)
    [[0.2, 0.3, 0.1, 0.1, 0.3], [0.3, 0.1, 0.1, 0.2, 0.5]]
    """
    labels_save = []

    for l in range(len(labels)):
        if labels[l][2]*labels[l][2]<500:
            if labels[l][4] > 0.15:
                labels_save.append(labels[l])
        elif 500<labels[l][2]*labels[l][2]<5000:
            if labels[l][4] > 0.4:
                labels_save.append(labels[l])
        elif labels[l][2]*labels[l][2]>5000:
            if labels[l][4] > 0.55:
                labels_save.append(labels[l])
    return labels_save

def savefile(labels_conv,counter2,pathsave):
    """
    Save the ensemble results into txt files

    Parameters:
    __________
    param: labels_conv: resulting ensemble labels in yolo format
    param: counter2: Naming index of the saved file
    param: pathsave: saving path of the ensemble results
    return: A result folder including all ensemble results
    
    Example:
    ________
    >>>labels_conv = [[0.2, 0.3, 0.1, 0.1], [0.3, 0.1, 0.1, 0.2], [0.4, 0.4, 0.1, 0.1]]
    >>>counter2=1
    >>>pathsave= './Robbin_data/ensemble/loop2/merged'
    savefile(labels_conv,counter2,pathsave)
    """

    path_save = pathsave+str(counter2)+'.txt'
    # Open the text file from path_save
    fileObject = open(path_save,'a')
    # empty the txt file before modification
    fileObject.truncate(0)
    # write the ensemble results into the text file
    for m in range(len(labels_conv)):
        list = []
        list = np.append(0,labels_conv[m])
        for ip in list:
            fileObject.write(str(ip))
            fileObject.write(' ')
        fileObject.write('\n')
    fileObject.close()

def ensemble_models(len_imgs, pathimg, pathtxtall, pathsave, pathgt,n_models):
    """
    This is the main function of the ensemble module. 
    This modules takes in the ground truth path and the detected file path generated from two models
    This module return a folder of resulting labels which is the ensembling of two models and filtered with the confidence score

    Parameters:
    ___________
    param: len_imgs: number of images
    param: pathimg: image path
    param: pathtxtall: array of label paths of two model's detection results
    param: pathsave: path to save the result labels
    param: pathgt: ground truth path
    param: n_models: number of models

    Example:
    ________
    >>>pathimg = "./Robbin_data/dataset2400/whole/images"
    >>>pathtxt1 = "./Robbin_data/ensemble/loop2/model1/labels/crater"
    >>>pathtxt2 = "./Robbin_data/ensemble/loop2/model2/labels/crater"
    >>>pathgt = "./Robbin_data/dataset2400/whole/labels"
    >>>pathsave = './Robbin_data/ensemble/loop2/merged'
    >>>pathtxtall = pathtxt1+' '+pathtxt2
    >>>n_models = 2
    >>>!python ensemble.py --gt {pathgt} --nargs {pathtxtall} --pathimg {pathimg} --n_models 2  --pathsave {pathsave}
    """
    counter2 = 0
    for i in range(len_imgs):
        pathtxtgt =  pathgt+str(i)+".txt"
        path_img = pathimg+str(i)+".jpg"
        signal = 0
        for n in range(n_models):
            # print('txt',pathtxtall)
            # print('txt',pathtxtall[n]+str(i)+".txt")
            # print('gt',pathtxtgt)
            if os.path.exists(pathtxtall[n]+str(i)+".txt") and os.path.exists(pathtxtgt):
                signal+=1
        if signal == (n_models):
            
            img = PImage.open(path_img)
            img_size = img.size
            
            #***********************************generate iou labels***********************************
            arrayall = []
            l = []
            arrayall.append(np.loadtxt(pathtxtgt,dtype = float))
            for n in range(n_models):

                arrayall.append(np.loadtxt(pathtxtall[n]+str(i)+".txt",dtype = float))
            x,y,w,h,txt = label_rescale(arrayall, img_size)
    
            l = size_array(arrayall)
            txt_iou = iou_convert(l,txt)

            print('gt:',len(txt[0]),txt[0])
            print('model1:',len(txt[1]),txt[1])
            print('model2: ',len(txt[2]),txt[2])
#           #**********************************compute IOU*********************************************
            #IOU between model 1 and model 2
            labels = IOU_between_models(len(txt[1]),len(txt[2]),txt[1],txt[2],txt_iou[1],txt_iou[2])
            print('iou labels: ',len(labels),labels)
            l_label = np.shape(labels)[0]
            labels_copy = copy.deepcopy(labels)
            labels_iou = iou_label_convert(l_label,labels_copy)
            
            #***************************5*Compute IOU for multiple models********************************
            if n_models >= 3:      
                l_label = np.shape(labels)[0]
                labels_copy = copy.deepcopy(labels)
                labels_iou = iou_label_convert(l_label,labels_copy)
                labels = IOU_between_models(l_label,len(txt[3]),labels,txt[3],labels_iou,txt_iou[3])
                
            if n_models >= 4:
                l_label = np.shape(labels)[0]
                labels_copy = copy.deepcopy(labels)
                labels_iou = iou_label_convert(l_label,labels_copy) 
                labels = IOU_between_models(l_label,len(txt[4]),labels,txt[4],labels_iou,txt_iou[4])   
                
            if n_models >= 5:
                l_label = np.shape(labels)[0]
                labels_copy = copy.deepcopy(labels)
                labels_iou = iou_label_convert(l_label,labels_copy) 
                labels = IOU_between_models(l_label,len(txt[4]),labels,txt[4],labels_iou,txt_iou[4])
                
            if n_models >= 6:
                l_label = np.shape(labels)[0]
                labels_copy = copy.deepcopy(labels)
                labels_iou = iou_label_convert(l_label,labels_copy6) 
                labels = IOU_between_models(l_label,len(txt[4]),labels,txt[4],labels_iou,txt_iou[4])
                
            labels = IOU_gt(txt_iou[0], l[0], l_label, labels_iou, labels,txt[0])
            print('gt labels:',len(labels),labels)
            labels = conf_thres(labels)
            print('conf labels: ',len(labels),labels)
#           #**********convert the label back to yolo format******************
            # labels = np.reshape(labels, (len(labels),4))
            labels_conv = yolo_labels_convert(labels,img_size)

            
#           #*********save the file*********************
            savefile(labels_conv,counter2,pathsave)

        else:
            print('File doesn''t exist')
        print('Pocessing file '+ str(counter2))
        counter2 += 1
                                        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, default='/content/yolov5/Training_set-1/train/labels')
    parser.add_argument('--nargs',nargs='+')
    parser.add_argument('--pathimg', type=str, default='/content/yolov5/Training_set-1/train/images')
    parser.add_argument('--pathsave', type=str, default="./Robbin_data/ensemble/ensemble01/labels/crater")
    parser.add_argument('--iou_models', type=float, default=0.6)
    parser.add_argument('--iou_gt', type=float, default=0.20)
    parser.add_argument('--n_iters', type=int, default=3)
    parser.add_argument('--n_models', type=int, default=2)
    parser.add_argument('--conf', type=float, default=1)

    opt = parser.parse_args()
    pathtxt = opt.nargs
    pathsave = opt.pathsave
    pathgt = opt.gt
    n_models = opt.n_models
    pathimg = opt.pathimg

    path = pathimg
    pathimg = pathimg+'/crater'
    len_imgs = count(path)

    #count the number of images
    len_imgs = count(path)

    pathsave = pathsave+'/crater'
    pathgt = pathgt+'/crater'

    character = [',','[',']']
    for i in range(len(pathtxt)):
      for c in range(len(character)):
        pathtxt[i] = pathtxt[i].replace(character[c],'')

    n = int(n_models)
    # print('pathtxt',pathtxt)
    # print('pathsave',pathsave)
    # print('pathgt',pathgt)
    ensemble_models(len_imgs, pathimg, pathtxt, pathsave, pathgt,n)