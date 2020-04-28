import cv2 as cv
import argparse
import numpy as np
import os.path
from matplotlib import pyplot as plt
import math
import time

# Initialize the parameters
confThreshold = 0.5  #Confidence threshold; 컨피던스가 이 값 미만이면 무시할 것임; 관행적으로 0.5를 쓴다.
nmsThreshold = 0.4   #Non-maximum suppression threshold; 주변보다 크거나 작은 값만 살림; 
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image

# Load names of classes
classesFile = "coco.names"    #클래스명을 정리한 텍스트파일
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "yolov3.cfg"     #네트워크를 구성하는 레이어들의 파라미터를 정리한 텍스트파일
modelWeights = "yolov3.weights"       #모델 학습 결과 도출된 weight값을 정리한 파일

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)    #`readNetFromDarknet`: 외부 config 정보와 weight 정보로 네트워크 구축
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)    #백엔드 지정
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)




# Get the names of the output layers;
#아웃풋 레이어명을 리턴하는 함수
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Draw the predicted bounding box;
#바운딩 박스 그리고 클래스 레이블/확률값 쓰는 함수
def drawPred(frame, classId, conf, left, top, right, bottom):
    # Draw a bounding box.;
    #박스 그리기
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    
    label = '%.2f' % conf
        
    # Get the label for the class name and its confidence
    #클래스 레이블/확률값 구하기
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    #Display the label at the top of the bounding box;
    #클래스 레이블/확률값 쓸 위치 지정
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)

# Remove the bounding boxes with low confidence using non-maxima suppression;
#
def postprocess(frame, outs):
    frameHeight = frame.shape[0]   #행 == 높이
    frameWidth = frame.shape[1]    #열 == 폭

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:     #yolo layer 출력값을 1개씩 취함; 3번 돈다.
        for detection in out:     #바운딩박스 1개씩 취함; 507, 2028, 8112번 돈다.
            scores = detection[5:]    #80개 클래스에 대한 확률값
            classId = np.argmax(scores)     #가장 높은 확률값의 인덱스
            confidence = scores[classId]    #가장 높은 확률
            if confidence > confThreshold:    #확률이 미리 설정해둔 파라미터보다 높을 경우에만 실행된다.; confidence만으로도 많이 걸러짐
                center_x = int(detection[0] * frameWidth)   #detection 값이 0~1로 정규화되기 때문에 원래 값으로 되돌리기 위해서 폭/높이를 곱해준다.
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)


def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, bboxes

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load network
ageNet = cv.dnn.readNet(ageModel, ageProto)
genderNet = cv.dnn.readNet(genderModel, genderProto)
faceNet = cv.dnn.readNet(faceModel, faceProto)

padding = 25

def yolo3(img):
    cap = cv.VideoCapture(img)
    hasFrame, frame = cap.read()
    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
    net.setInput(blob)
    outs = net.forward(getOutputsNames(net))
    postprocess(frame, outs)
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    return frame


def detectFace(img):
    frame = cv.imread(img)
    frameFace, bboxes = getFaceBox(faceNet, frame)
    for bbox in bboxes:
        face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
               max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]

        blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        label = "{},{}".format(gender, age)
        cv.putText(frameFace, label, (bbox[0], bbox[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv.LINE_AA)
    return frameFace