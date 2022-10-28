from pytesseract import pytesseract
import cv2 as cv
import numpy as np
import re
import keras
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator 

pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

def rescaleFrame(frame):
    frame = cv.imread(frame)
    h,w,c = frame.shape
    if h+w <65:
        scale = 25.0
        width = int(frame.shape[1]*scale)
        height = int(frame.shape[0]*scale)
        dimensions = (width,height)
        return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)
    elif h+w <100 and h+w>= 65:
        scale = 15.0
        width = int(frame.shape[1]*scale)
        height = int(frame.shape[0]*scale)
        dimensions = (width,height)
        return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)
    elif h+w <150 and h+w>= 100:
        scale = 10.0
        width = int(frame.shape[1]*scale)
        height = int(frame.shape[0]*scale)
        dimensions = (width,height)
        return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)
    elif h+w <215 and h+w >= 150:
        scale = 7.0
        width = int(frame.shape[1]*scale)
        height = int(frame.shape[0]*scale)
        dimensions = (width,height)
        return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)
    elif h+w <300 and h+w>= 215:
        scale = 5.0
        width = int(frame.shape[1]*scale)
        height = int(frame.shape[0]*scale)
        dimensions = (width,height)
        return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)
    elif h+w <600 and h+w>= 430:
        scale = 2.5
        width = int(frame.shape[1]*scale)
        height = int(frame.shape[0]*scale)
        dimensions = (width,height)
        return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)
    elif h+w <900 and h+w>= 600:
        scale =1.75 
        width = int(frame.shape[1]*scale)
        height = int(frame.shape[0]*scale)
        dimensions = (width,height)
        return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)
    elif h+w < 1000 and h+w>= 900:
        scale = 1.25
        width = int(frame.shape[1]*scale)
        height = int(frame.shape[0]*scale)
        dimensions = (width,height)
        return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)
    elif h+w <1250 and h+w>= 1000:
        scale = 1.15
        width = int(frame.shape[1]*scale)
        height = int(frame.shape[0]*scale)
        dimensions = (width,height)
        return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)
    elif h+w <1500 and h+w>= 1250:
        scale = 1.0
        width = int(frame.shape[1]*scale)
        height = int(frame.shape[0]*scale)
        dimensions = (width,height)
        return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)
    elif h+w <1800 and h+w >= 1500:
        scale = 0.8
        width = int(frame.shape[1]*scale)
        height = int(frame.shape[0]*scale)
        dimensions = (width,height)
        return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)
    elif h+w <2000 and h+w>= 1800:
        scale = 0.75
        width = int(frame.shape[1]*scale)
        height = int(frame.shape[0]*scale)
        dimensions = (width,height)
        return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)
    elif h+w <2500 and h+w>= 2000:
        scale = 0.6
        width = int(frame.shape[1]*scale)
        height = int(frame.shape[0]*scale)
        dimensions = (width,height)
        return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)
    elif h+w <3000 and h+w >= 2500:
        scale =0.5 
        width = int(frame.shape[1]*scale)
        height = int(frame.shape[0]*scale)
        dimensions = (width,height)
        return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)
    elif h+w < 3500  and h+w>= 3000:
        scale = 0.45
        width = int(frame.shape[1]*scale)
        height = int(frame.shape[0]*scale)
        dimensions = (width,height)
        return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)
    elif h+w <4000 and h+w>= 3500:
        scale = 0.4
        width = int(frame.shape[1]*scale)
        height = int(frame.shape[0]*scale)
        dimensions = (width,height)
        return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)
    elif h+w <4500 and h+w >= 4000:
        scale =0.35 
        width = int(frame.shape[1]*scale)
        height = int(frame.shape[0]*scale)
        dimensions = (width,height)
        return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)
    elif  h+w>= 5000 and h+w>= 4500:
        scale = 0.30
        width = int(frame.shape[1]*scale)
        height = int(frame.shape[0]*scale)
        dimensions = (width,height)
        return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)
    elif  h+w>= 5500 and h+w>= 5000:
        scale = 0.275
        width = int(frame.shape[1]*scale)
        height = int(frame.shape[0]*scale)
        dimensions = (width,height)
        return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)
    elif  h+w>= 6000 and h+w>= 5500:
        scale = 0.25
        width = int(frame.shape[1]*scale)
        height = int(frame.shape[0]*scale)
        dimensions = (width,height)
        return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)
    elif  h+w>= 6500 and h+w>= 6000:
        scale = 0.225
        width = int(frame.shape[1]*scale)
        height = int(frame.shape[0]*scale)
        dimensions = (width,height)
        return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)
    elif  h+w>= 7000 and h+w>= 6500:
        scale = 0.2
        width = int(frame.shape[1]*scale)
        height = int(frame.shape[0]*scale)
        dimensions = (width,height)
        return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)
    elif  h+w>= 7500 and h+w>= 7000:
        scale = 0.175
        width = int(frame.shape[1]*scale)
        height = int(frame.shape[0]*scale)
        dimensions = (width,height)
        return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)
    elif  h+w>= 10000 and h+w>= 7500:
        scale = 0.15
        width = int(frame.shape[1]*scale)
        height = int(frame.shape[0]*scale)
        dimensions = (width,height)
        return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)
    elif  h+w>= 13000 and h+w>= 10000:
        scale = 0.125
        width = int(frame.shape[1]*scale)
        height = int(frame.shape[0]*scale)
        dimensions = (width,height)
        return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)
    elif  h+w>= 15000 and h+w>= 13000:
        scale = 0.1
        width = int(frame.shape[1]*scale)
        height = int(frame.shape[0]*scale)
        dimensions = (width,height)
        return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)
    elif  h+w>= 21500 and h+w>= 15000:
        scale = 0.075
        width = int(frame.shape[1]*scale)
        height = int(frame.shape[0]*scale)
        dimensions = (width,height)
        return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)
    elif  h+w>= 21500:
        scale = 0.05
        width = int(frame.shape[1]*scale)
        height = int(frame.shape[0]*scale)
        dimensions = (width,height)
        return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)
    
def prepro(text):
    text = text.replace("\n"," ")
    text = text.upper()
    text = text.replace("NOMBRE","")
    text = text.replace("DOMICILIO","")
    text = text.replace("BARR","")
    text = text.replace("CLAVE","")
    text = text.replace("DE","")
    text = text.replace("ELECTOR","")
    text = text.replace("CURP","")  
    text = text.replace("SEXO","")
    return text

def INE_D(image):
    mrz_dict = dict()
    per = 25
    img = cv.imread("C:/Users/52473/Desktop/TRATO/Code/INE/reversos/reverso_0.jpg")
    img = cv.resize(img,(672,424),interpolation=cv.INTER_AREA)
    h,w,c = img.shape
    orb = cv.ORB_create(2650)
    kp1,des1 = orb.detectAndCompute(img,None)
       
    img_2 = rescaleFrame(image)
    kp2,des2 = orb.detectAndCompute(img_2,None)
    bf = cv.BFMatcher(cv.NORM_HAMMING)
    matches = list(bf.match(des2,des1))
    matches.sort(key = lambda x:x.distance)
    matches = tuple(matches)
    good = matches[:int(len(matches)*(per/100))]
    
    srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dtsPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    
    M, _= cv.findHomography(srcPoints,dtsPoints,cv.RANSAC,5.0)
    scan = cv.warpPerspective(img_2,M,(w,h))
    scan = cv.resize(scan,(w,h),interpolation=cv.INTER_AREA)
    
    pre_per = 25
    pre = img
    pre_h,pre_w,pre_c = pre.shape
    pre_orb = cv.ORB_create(2500)
    pre_kp1,pre_des1 = pre_orb.detectAndCompute(pre,None)
    
    pre_kp2,pre_des2 = pre_orb.detectAndCompute(scan,None)
    pre_matches = list(bf.match(pre_des2,pre_des1))
    pre_matches.sort(key = lambda x:x.distance)
    pre_matches = tuple(pre_matches)
    pre_good = pre_matches[:int(len(pre_matches)*(pre_per/100))]

    pre_srcPoints = np.float32([pre_kp2[m.queryIdx].pt for m in pre_good]).reshape(-1,1,2)
    pre_dtsPoints = np.float32([pre_kp1[m.trainIdx].pt for m in pre_good]).reshape(-1,1,2)    

    pre_M, _= cv.findHomography(pre_srcPoints,pre_dtsPoints,cv.RANSAC,5.0)
    pre_scan = cv.warpPerspective(scan,pre_M,(pre_w,pre_h))
    pre_scan = cv.resize(pre_scan,(pre_w,pre_h),interpolation=cv.INTER_AREA)
    roi = (2, 266, 667, 135)
    pre_scan = pre_scan[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
    mrz = pytesseract.image_to_string(pre_scan, lang='eng', config='--psm 6')
    mrz = re.findall(r'\d+', mrz)
    if len(mrz[0]) <= 1:
        mrz_dict['cic'] = mrz[1][0:9]
        mrz_dict['ocr'] = mrz[2]
    elif len(mrz[0]) > 1:
        mrz_dict['cic'] = mrz[0][0:9]
        mrz_dict['ocr'] = mrz[1]
     
    return mrz_dict
  
def INE_E(image):
    mrz_dict = dict()
    per = 25
    img = cv.imread("C:/Users/52473/Desktop/TRATO/Code/INE/reversos/reverso_1.jpg")
    img = cv.resize(img,(672,424),interpolation=cv.INTER_AREA)
    h,w,c = img.shape
    orb = cv.ORB_create(2500)
    kp1,des1 = orb.detectAndCompute(img,None)
       
    img_2 = rescaleFrame(image)
    kp2,des2 = orb.detectAndCompute(img_2,None)
    bf = cv.BFMatcher(cv.NORM_HAMMING)
    matches = list(bf.match(des2,des1))
    matches.sort(key = lambda x:x.distance)
    matches = tuple(matches)
    good = matches[:int(len(matches)*(per/100))]
    
    srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dtsPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    
    M, _= cv.findHomography(srcPoints,dtsPoints,cv.RANSAC,5.0)
    scan = cv.warpPerspective(img_2,M,(w,h))
    scan = cv.resize(scan,(w,h),interpolation=cv.INTER_AREA)
    
    pre_per = 25
    pre = img
    pre_h,pre_w,pre_c = pre.shape
    pre_orb = cv.ORB_create(2500)
    pre_kp1,pre_des1 = pre_orb.detectAndCompute(pre,None)
    
    pre_kp2,pre_des2 = pre_orb.detectAndCompute(scan,None)
    pre_matches = list(bf.match(pre_des2,pre_des1))
    pre_matches.sort(key = lambda x:x.distance)
    pre_matches = tuple(pre_matches)
    pre_good = pre_matches[:int(len(pre_matches)*(pre_per/100))]

    pre_srcPoints = np.float32([pre_kp2[m.queryIdx].pt for m in pre_good]).reshape(-1,1,2)
    pre_dtsPoints = np.float32([pre_kp1[m.trainIdx].pt for m in pre_good]).reshape(-1,1,2)    

    pre_M, _= cv.findHomography(pre_srcPoints,pre_dtsPoints,cv.RANSAC,5.0)
    pre_scan = cv.warpPerspective(scan,pre_M,(pre_w,pre_h))
    pre_scan = cv.resize(pre_scan,(pre_w,pre_h),interpolation=cv.INTER_AREA)
    roi = (0, 289, 665, 122)
    pre_scan = pre_scan[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
    
    mrz = pytesseract.image_to_string(pre_scan, lang='eng', config='--psm 6')
    mrz = re.findall(r'\d+', mrz)
    if len(mrz[0]) <= 1:
        mrz_dict['cic'] = mrz[1][0:9]
        mrz_dict['ocr'] = mrz[2][4:]
    elif len(mrz[0]) > 1:
        mrz_dict['cic'] = mrz[0][0:9]
        mrz_dict['ocr'] = mrz[1][4:]
     
    return mrz_dict
           
def INE_D_Data(cara):
    data = dict()
    per = 25
    img = cv.imread("C:/Users/52473/Desktop/TRATO/Code/INE/caras/cara_0.jpg")
    img = cv.resize(img,(672,424),interpolation=cv.INTER_AREA)
    h,w,c = img.shape
    orb = cv.ORB_create(2650)
    kp1,des1 = orb.detectAndCompute(img,None)
    
    img_2 = rescaleFrame(cara)
    kp2,des2 = orb.detectAndCompute(img_2,None)
    bf = cv.BFMatcher(cv.NORM_HAMMING)
    matches = list(bf.match(des2,des1))
    matches.sort(key = lambda x:x.distance)
    matches = tuple(matches)
    good = matches[:int(len(matches)*(per/100))]
    
    srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dtsPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    
    M, _= cv.findHomography(srcPoints,dtsPoints,cv.RANSAC,5.0)
    scan = cv.warpPerspective(img_2,M,(w,h))
    scan = cv.resize(scan,(w,h),interpolation=cv.INTER_AREA)
    
    pre_per = 25
    pre = img
    pre_h,pre_w,pre_c = pre.shape
    pre_orb = cv.ORB_create(2500)
    pre_kp1,pre_des1 = pre_orb.detectAndCompute(pre,None)
    
    pre_kp2,pre_des2 = pre_orb.detectAndCompute(scan,None)
    pre_matches = list(bf.match(pre_des2,pre_des1))
    pre_matches.sort(key = lambda x:x.distance)
    pre_matches = tuple(pre_matches)
    pre_good = pre_matches[:int(len(pre_matches)*(pre_per/100))]

    pre_srcPoints = np.float32([pre_kp2[m.queryIdx].pt for m in pre_good]).reshape(-1,1,2)
    pre_dtsPoints = np.float32([pre_kp1[m.trainIdx].pt for m in pre_good]).reshape(-1,1,2)    

    pre_M, _= cv.findHomography(pre_srcPoints,pre_dtsPoints,cv.RANSAC,5.0)
    pre_scan = cv.warpPerspective(scan,pre_M,(pre_w,pre_h))
    pre_scan = cv.resize(pre_scan,(pre_w,pre_h),interpolation=cv.INTER_AREA)
    names_list = []
    
    list_of_rois = [[40,116,176,266],[205,98,177,88],[20,186,354,87],[585,140,84,31],[529,97,143,48],[208,274,335,27],[209,302,248,27],[469,303,192,27],[215,331,100,27],[336,330,137,27],[469,332,125,26],[216,358,132,32],[350,358,124,29],[474,358,117,26]]
    #true_face = face[int(list_of_rois[0][1]):int(list_of_rois[0][1]+list_of_rois[0][3]), int(list_of_rois[0][0]):int(list_of_rois[0][0]+list_of_rois[0][2])]
    
    for rect in list_of_rois:
        x1 = rect[0]
        y1 = rect[1]
        x2 = rect[2]
        y2 = rect[3]
        face_crop = pre_scan[y1:y1+y2,x1:x1+x2]
        names_list.append(pytesseract.image_to_string(face_crop, lang='spa', config='--psm 6'))
        
    for i in range(len(names_list)):
        names_list[i] = prepro(names_list[i])
        
    data['nombre'] = names_list[1].lstrip()
    data['domicilio'] = names_list[2].lstrip()
    data['nacimiento'] = re.findall(r'\d+', names_list[4])
    data['sexo'] = names_list[3].replace(" ","")
    data['clave'] = names_list[5].replace(" ","")
    data['curp'] = names_list[6].replace(" ","")
    data['registro'] = re.findall(r'\d+', names_list[7])
    data['estado'] = re.findall(r'\d+', names_list[8])
    data['municipio'] = re.findall(r'\d+', names_list[9])
    data['seccion'] = re.findall(r'\d+', names_list[10])
    data['localidad'] = re.findall(r'\d+', names_list[11])
    data['emision'] = re.findall(r'\d+', names_list[12])
    data['vigencia'] = re.findall(r'\d+', names_list[13])
    
    return data

def INE_E_Data(cara):
    data = dict()
    per = 25
    img = cv.imread("C:/Users/52473/Desktop/TRATO/Code/INE/caras/cara_1.jpg")
    img = cv.resize(img,(672,424),interpolation=cv.INTER_AREA)
    h,w,c = img.shape
    orb = cv.ORB_create(2650)
    kp1,des1 = orb.detectAndCompute(img,None)
    
    img_2 = rescaleFrame(cara)
    kp2,des2 = orb.detectAndCompute(img_2,None)
    bf = cv.BFMatcher(cv.NORM_HAMMING)
    matches = list(bf.match(des2,des1))
    matches.sort(key = lambda x:x.distance)
    matches = tuple(matches)
    good = matches[:int(len(matches)*(per/100))]
    
    srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dtsPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    
    M, _= cv.findHomography(srcPoints,dtsPoints,cv.RANSAC,5.0)
    scan = cv.warpPerspective(img_2,M,(w,h))
    scan = cv.resize(scan,(w,h),interpolation=cv.INTER_AREA)
    
    pre_per = 25
    pre = img
    pre_h,pre_w,pre_c = pre.shape
    pre_orb = cv.ORB_create(2500)
    pre_kp1,pre_des1 = pre_orb.detectAndCompute(pre,None)
    
    pre_kp2,pre_des2 = pre_orb.detectAndCompute(scan,None)
    pre_matches = list(bf.match(pre_des2,pre_des1))
    pre_matches.sort(key = lambda x:x.distance)
    pre_matches = tuple(pre_matches)
    pre_good = pre_matches[:int(len(pre_matches)*(pre_per/100))]

    pre_srcPoints = np.float32([pre_kp2[m.queryIdx].pt for m in pre_good]).reshape(-1,1,2)
    pre_dtsPoints = np.float32([pre_kp1[m.trainIdx].pt for m in pre_good]).reshape(-1,1,2)    

    pre_M, _= cv.findHomography(pre_srcPoints,pre_dtsPoints,cv.RANSAC,5.0)
    pre_scan = cv.warpPerspective(scan,pre_M,(pre_w,pre_h))
    pre_scan = cv.resize(pre_scan,(pre_w,pre_h),interpolation=cv.INTER_AREA)
    names_list = []
    
    list_of_rois = [[56,120,168,166],[221,118,131,92],[226,216,279,88],[569,116,84,42],[225,300,312,29],[225,329,212,38],[459,321,138,45],[221,366,163,48],[381,366,87,44],[465,364,105,46]]
    #true_face = face[int(list_of_rois[0][1]):int(list_of_rois[0][1]+list_of_rois[0][3]), int(list_of_rois[0][0]):int(list_of_rois[0][0]+list_of_rois[0][2])]   
    for rect in list_of_rois:
        x1 = rect[0]
        y1 = rect[1]
        x2 = rect[2]
        y2 = rect[3]
        face_crop = pre_scan[y1:y1+y2,x1:x1+x2]
        names_list.append(pytesseract.image_to_string(face_crop, lang='spa', config='--psm 6'))
        
    for i in range(len(names_list)):
        names_list[i] = prepro(names_list[i])
        
    data['nombre'] = names_list[1].lstrip()
    data['domicilio'] = names_list[2].lstrip()
    data['sexo'] = names_list[3].replace(" ","")
    data['clave'] = names_list[4].replace(" ","")
    data['curp'] = names_list[5].replace(" ","")
    data['registro'] = re.findall(r'\d+', names_list[6])
    data['nacimiento'] = re.findall(r'\d+', names_list[7])
    data['seccion'] = re.findall(r'\d+', names_list[8])
    data['vigencia'] = re.findall(r'\d+', names_list[9])
    
    return data


classifier = keras.models.load_model('C:/Users/52473/Desktop/TRATO/Code/INE/classifier.h5')

def preprocessing(img):
    img = image.load_img(img, target_size = (106,168,3))
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis = 0)
    return img

def INE_MASTER(cara,reverso): 
    img = preprocessing(reverso)
    result = classifier.predict(img)
    if result[0][0] == 0:
       try:
           return {**INE_D_Data(cara), **INE_D(reverso)}
       except:
           dict("Error. Medidas incorrectas.")
    else:
        try:
            return {**INE_E_Data(cara), **INE_E(reverso)}
        except:
            dict("Error. Medidas incorrectas.")

INE_MASTER('C:/Users/52473/Desktop/TRATO/Code/INE/caras/cara_0.jpg','C:/Users/52473/Desktop/TRATO/Code/INE/reversos/reverso_0.jpg')

