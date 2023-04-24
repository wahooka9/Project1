#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 20:18:54 2023

@author: andrewriznyk
"""

import cv2 
import numpy as np 
from tensorflow import keras
import tensorflow as tf 
from tensorflow.keras.applications.inception_v3 import preprocess_input

LABEL_TRAFFIC_LIGHT = 10

"""
Remove Duplicate Bounding Boxes
"""
def accept_box(boxes, box_index, tolerance):
  box = boxes[box_index]
  for idx in range(box_index):
    other_box = boxes[idx]
    if abs(center(other_box, "x") - center(box, "x")) < tolerance and abs(center(other_box, "y") - center(box, "y")) < tolerance:
      return False
  return True

"""
Download a pretrained object detection model, and save it to your hard drive.
"""
def load_model(model_name):
  url = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/' + model_name + '.tar.gz'
     
  # Download a file from a URL that is not already in the cache
  model_dir = tf.keras.utils.get_file(fname=model_name, untar=True, origin=url)
 
  print("Model path: ", str(model_dir))
   
  model_dir = str(model_dir) + "/saved_model"
  model = tf.saved_model.load(str(model_dir))
 
  return model
 
def load_ssd_coco():
  return load_model("ssd_resnet50_v1_fpn_640x640_coco17_tpu-8")
 
def center(box, coord_type):
  return (box[coord_type] + box[coord_type + "2"]) / 2

     
def perform_object_detection_video(model, video_frame, model_traffic_lights=None):
  img_rgb = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
  input_tensor = tf.convert_to_tensor(img_rgb)
  input_tensor = input_tensor[tf.newaxis, ...]
 
  output = model(input_tensor)
 
  num_detections = int(output.pop('num_detections'))
  output = {key: value[0, :num_detections].numpy()
            for key, value in output.items()}
  output['num_detections'] = num_detections
    # this is for multiple classes -  we will jsut do "traffic lights"
  output['detection_classes'] = output['detection_classes'].astype(np.int64)
  output['boxes'] = [
    {"y": int(box[0] * img_rgb.shape[0]), "x": int(box[1] * img_rgb.shape[1]), "y2": int(box[2] * img_rgb.shape[0]),
     "x2": int(box[3] * img_rgb.shape[1])} for box in output['detection_boxes']]
 
  for idx in range(len(output['boxes'])):
 
    # Extract the type of the object that was detected
    obj_class = output["detection_classes"][idx]
     
    # How confident the object detection model is on the object's type
    score = int(output["detection_scores"][idx] * 100)
         
    # Extract the bounding box
    box = output["boxes"][idx]
 
    color = None
    label_text = ""
 

    if obj_class == LABEL_TRAFFIC_LIGHT:
      color = (255, 255, 255)
      label_text = "Traffic Light " + str(score)
             
      if model_traffic_lights:
        img_traffic_light = img_rgb[box["y"]:box["y2"], box["x"]:box["x2"]]
        img_inception = cv2.resize(img_traffic_light, (299, 299))
         
        img_inception = np.array([preprocess_input(img_inception)])
 
        prediction = model_traffic_lights.predict(img_inception)
        label = np.argmax(prediction)
        score_light = str(int(np.max(prediction) * 100))
        if label == 0:
          label_text = "Green " + score_light
        elif label == 1:
          label_text = "Yellow " + score_light
        elif label == 2:
          label_text = "Red " + score_light
        else:
          label_text = False  # This is not a traffic light
 
    if color and label_text and accept_box(output["boxes"], idx, 5.0) and score > 20:
      cv2.rectangle(img_rgb, (box["x"], box["y"]), (box["x2"], box["y2"]), color, 2)
      cv2.putText(img_rgb, label_text, (box["x"], box["y"]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
 
  output_frame = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
  return output_frame



file_size = (1280,720)
scale_ratio = 1 

filename = './00beeb02-ba0790aa.mp4'
output_filename = './DayTimeLights.mp4'
output_frames_per_second = 30.0
 

model_ssd = load_ssd_coco()
model_traffic_lights_nn = keras.models.load_model("traffic.h5")
 
def main():
 
  cap = cv2.VideoCapture(filename)

  fourcc = cv2.VideoWriter_fourcc(*'avc1')
  result = cv2.VideoWriter(output_filename,  
                           fourcc, 
                           output_frames_per_second, 
                           file_size) 
  """
  Process the video one frame at a time
  """
  while cap.isOpened():
    success, frame = cap.read() 

    if success:
      width = int(frame.shape[1] * scale_ratio)
      height = int(frame.shape[0] * scale_ratio)
      frame = cv2.resize(frame, (width, height))

      output_frame = perform_object_detection_video(
        model_ssd, frame, model_traffic_lights=model_traffic_lights_nn)

      result.write(output_frame)
    else:
      break
             
  cap.release()
  result.release()
  cv2.destroyAllWindows() 
     
main()