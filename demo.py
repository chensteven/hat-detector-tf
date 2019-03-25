import tensorflow as tf
import numpy as np
from utils import label_map_util
from utils import visualization_utils as vis_util
import cv2
import time
from skvideo.io import FFmpegWriter
import helper

from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50


from keras.layers import Dense, Activation, Flatten, Dropout
from keras import backend as K

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

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


PATH_TO_FROZEN_GRAPH  = '/Users/schen/hat-detector/output_inference_graph_v1.pb/frozen_inference_graph.pb'
PATH_TO_LABELS = '/Users/schen/hat-detector/label_map.pbtxt'

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('test.avi')
output_video = FFmpegWriter("output_final.mp4")

class_list_file = "ResNet50_data_class_list.txt"
class_list = helper.load_class_list(class_list_file)

from keras.applications.resnet50 import preprocess_input
preprocessing_function = preprocess_input
WIDTH = 300
HEIGHT = 300
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
finetune_model = helper.build_finetune_model(base_model, 0.003, [1024, 1024], len(class_list))
finetune_model.load_weights("ResNet50_model_weights.h5")


#while(True):
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  image_np = cv2.resize(frame, (800, 600))
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  image_np_expanded = np.expand_dims(image_np, axis=0)
  # Actual detection.
  start_time = time.time()
  output_dict = run_inference_for_single_image(image_np, detection_graph)
  elapsed_time = time.time() - start_time
  print('Inference time cost: {}'.format(elapsed_time))
  # Our operations on the frame come here
  # Display the resulting frame
  coords = vis_util.visualize_boxes_and_labels_on_image_array(
  image_np,
  output_dict['detection_boxes'],
  output_dict['detection_classes'],
  output_dict['detection_scores'],
  category_index,
  instance_masks=output_dict.get('detection_masks'),
  use_normalized_coordinates=True,
  line_thickness=8)
  # if coords:
  #   target = coords[0]
  #   img = image_np[target[2]:target[3], target[0]:target[1]]
  #   height, width = img
  #   img = preprocess_input(img.reshape(1, HEIGHT, WIDTH, 3))
  #   st = time.time()
  #   out = finetune_model.predict(img)
  #   confidence = out[0]
  #   class_prediction = list(out[0]).index(max(out[0]))
  #   class_name = class_list[class_prediction]
  #   run_time = time.time()-st

  #   print("Predicted class = ", class_name)
  #   print("Confidence = ", confidence)
  #   print("Run time = ", run_time)

  #   cv2.imwrite("preds/" + class_name[0] + ".jpg", image_np)
  output_video.writeFrame(np.flip(image_np, 2))  
  # cv2.imshow('frame',img)
  # if cv2.waitKey(1) & 0xFF == ord('q'):
  #     break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()