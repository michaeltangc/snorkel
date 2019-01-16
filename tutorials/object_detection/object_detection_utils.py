import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import csv

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

sys.path.append("google_object_detection_utils/")
from utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util
from object_detection_utils import *

def load_models_from_Google_API(model_names, download_base):
    detection_graphs = {}
    for name in model_names:
        model_file = name + '.tar.gz'
        if os.path.isfile(model_file):
            print(model_file + " is already downloaded.")
        else:
            opener = urllib.request.URLopener()
            opener.retrieve(download_base + model_file, model_file)
            tar_file = tarfile.open(model_file)
            tar_file.extractall()
            tar_file.close()

        path_to_frozen_graph = name + '/frozen_inference_graph.pb'
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            try:
                with tf.gfile.GFile(path_to_frozen_graph, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')
            except FileNotFoundError:
                print("The forzen graph is not found. Please try to export the graph from the checkpoint.")
        detection_graphs[model_file] = detection_graph
    print("All models loaded")
    return detection_graphs

def load_image_into_numpy_array(image):
    # Function originally found in Tensorflow tutorial
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

def load_images_as_np_array(paths):
    image_nps = []
    for image_path in paths:
        image = Image.open(image_path)
        try:
            image_np = load_image_into_numpy_array(image)
        except ValueError:
            print("The image cannot be reshaped correctly")
            continue
        image_nps.append(image_np)
    return image_nps

def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
              'num_detections', 'detection_boxes', 'detection_scores',
              'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                    tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                     feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

def detect_by_model(image_nps, detection_graph):
    output_dicts = []
    counter = 0
    for image_np in image_nps:
        image_np_expanded = np.expand_dims(image_np, axis=0)
        output_dict = run_inference_for_single_image(image_np, detection_graph)
        output_dicts.append(output_dict)
        counter += 1
        print("Predicted:" + str(counter))
    return output_dicts

def display_results(image_np, output_dict, category_index, min_score_thresh=0.1, image_size=(12,9), multiple_label=False):
    copy_image = np.copy(image_np)
    if multiple_label:
        vis_util.visualize_boxes_and_labels_on_image_array_2(
          copy_image,
          output_dict['detection_boxes'],
          output_dict['detection_classes'],
          output_dict['detection_scores'],
          category_index,
          instance_masks=output_dict.get('detection_masks'),
          use_normalized_coordinates=True,
          line_thickness=8,
          min_score_thresh=min_score_thresh)
    else:
        vis_util.visualize_boxes_and_labels_on_image_array(
          copy_image,
          output_dict['detection_boxes'],
          output_dict['detection_classes'],
          output_dict['detection_scores'],
          category_index,
          instance_masks=output_dict.get('detection_masks'),
          use_normalized_coordinates=True,
          line_thickness=8,
          min_score_thresh=min_score_thresh)
    
    plt.figure(figsize=image_size)
    plt.imshow(copy_image)
    plt.axis('off')

def non_max_suppression_fast(boxes, overlapThresh):
    # Modified version
    # original by Rosebrock, A. (2015). (Faster) Non-Maximum Suppression in Python. https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
 
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
 
    # initialize the list of picked indexes 
    pick = []
 
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
 
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1) * (y2 - y1)
    idxs = np.arange(len(boxes))[::-1]
 
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
 
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
 
        # compute the ratio of overlap
        overlap = (w * h) / (area[idxs[:last]] + area[i] - w * h)
 
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
        
 
    # return only the bounding boxes that were picked using the
    # integer data type
    return pick

def box_overlap(box, boxes):
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    area = (x2 - x1) * (y2 - y1)
    
    xx1 = np.maximum(box[0], x1)
    yy1 = np.maximum(box[1], y1)
    xx2 = np.minimum(box[2], x2)
    yy2 = np.minimum(box[3], y2)

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)
    
    # compute the ratio of overlap
    overlap = (w * h) / (area + (box[2] - box[0]) * (box[3] - box[1]) - w * h)
    return overlap

def combine_with_IoU(outputs, iou_threshold=0.4, score_threshold=0.5):
    combined_lables = []
    num_images = len(outputs[next(iter(outputs))])
    num_models = len(outputs)
    all_boxes = []
    all_classes = []
    
    for i in range(num_images):
        detection_boxes = None
        detection_classes = None
        for k in outputs:
            model_output = outputs[k][i]
            this_detection_boxes = model_output['detection_boxes'][:model_output['num_detections']]
            this_detection_scores = model_output['detection_scores'][:model_output['num_detections']]
            this_detection_classes = model_output['detection_classes'][:model_output['num_detections']]
            
            score_pick = np.where(this_detection_scores >= score_threshold)
            this_detection_boxes, this_detection_scores, this_detection_classes = this_detection_boxes[score_pick], this_detection_scores[score_pick], this_detection_classes[score_pick]
            
            maximal_pick = non_max_suppression_fast(this_detection_boxes, iou_threshold)
            
            this_detection_boxes, this_detection_scores, this_detection_classes = this_detection_boxes[maximal_pick], this_detection_scores[maximal_pick], this_detection_classes[maximal_pick]

            if detection_boxes is None:
                detection_boxes = this_detection_boxes
                detection_classes = np.expand_dims(this_detection_classes, axis=1)
            else:
                detection_classes = np.append(detection_classes, np.zeros((detection_classes.shape[0], 1)), axis=1)
                for j in range(len(this_detection_boxes)):
                    this_box = this_detection_boxes[j]
                    this_class = this_detection_classes[j]
                    IoUs = box_overlap(this_box, detection_boxes)

                    if len(IoUs) != 0 and np.max(IoUs) >= iou_threshold:
                        box_id = np.argmax(IoUs)
                        detection_boxes[box_id] = np.average(np.stack([this_box, detection_boxes[box_id]]), axis=0)
                        detection_classes[box_id, -1] = this_class
                    else:
                        detection_boxes = np.append(detection_boxes, np.array([this_box]), axis=0)
                        detection_classes = np.append(detection_classes, np.zeros((1, detection_classes.shape[1])), axis=0)
                        detection_classes[-1, -1] = this_class

        all_boxes.append(detection_boxes)
        all_classes.append(detection_classes)
    return all_boxes, all_classes
                