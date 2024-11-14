import sys
import time
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2

class FROZEN_GRAPH_HEAD():
    def __init__(self, PATH_TO_CKPT):
        self.inference_list = []
        self.count = 0

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        with self.detection_graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(graph=self.detection_graph, config=config)
            self.windowNotSet = True

    def expand_bbox(self, left, top, right, bottom, expand_ratio, im_width, im_height):
        """Mở rộng bbox với tỉ lệ cho trước và đảm bảo không vượt quá kích thước ảnh"""
        width = right - left
        height = bottom - top
        
        # Tính toán padding
        padding_x = int(width * expand_ratio)
        padding_y = int(height * expand_ratio)
        
        # Mở rộng bbox
        new_left = max(0, left - padding_x)
        new_top = max(0, top - padding_y)
        new_right = min(im_width, right + padding_x)
        new_bottom = min(im_height, bottom + padding_y)
        
        return new_left, new_top, new_right, new_bottom

    def draw_bounding_box(self, image, scores, boxes, classes, im_width, im_height):
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        heads = list()
        idx = 1

        for score, box, name in zip(scores, boxes, classes):
            if name == 1 and score > 0.6:
                # Tính toán bbox ban đầu
                left = int((box[1])*im_width)
                top = int((box[0])*im_height)
                right = int((box[3])*im_width)
                bottom = int((box[2])*im_height)

                # Mở rộng bbox với tỉ lệ 0.2 (20%)
                expanded_left, expanded_top, expanded_right, expanded_bottom = self.expand_bbox(
                    left, top, right, bottom, 0.15, im_width, im_height
                )

                # Cắt vùng ảnh với bbox đã mở rộng
                cropped_head = np.array(image[expanded_top:expanded_bottom, expanded_left:expanded_right])

                width = expanded_right - expanded_left
                height = expanded_bottom - expanded_top
                bottom_mid = (expanded_left + int(width / 2), expanded_top + height)
                confidence = score
                label = name

                mydict = {
                    "head_id": idx,
                    "width": width,
                    "height": height,
                    "cropped": cropped_head,
                    "left": expanded_left,
                    "right": expanded_right,
                    "top": expanded_top,
                    "bottom": expanded_bottom,
                    "confidence": confidence,
                    "label": None,
                    "bottom_mid": bottom_mid,
                    "model_type": 'FROZEN_GRAPH'
                }
                heads.append(mydict)
                idx += 1

                # Vẽ bbox mở rộng
                cv2.rectangle(image, (expanded_left, expanded_top), 
                            (expanded_right, expanded_bottom), (0, 255, 0), 2, 8)
                
                # Vẽ score
                score_text = 'score: {:.2f}%'.format(score)
                cv2.putText(image, score_text, (expanded_left-5, expanded_top-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,255), 3)

        return image, heads

    def run(self, image, im_width, im_height):
        """image: bgr image
        return (boxes, scores, classes, num_detections)
        """
        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        start_time = time.time()
        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        elapsed_time = time.time() - start_time
        self.inference_list.append(elapsed_time)
        self.count = self.count + 1
        average_inference = sum(self.inference_list)/self.count

        image, heads = self.draw_bounding_box(image, scores, boxes, classes, im_width, im_height)

        return image, heads