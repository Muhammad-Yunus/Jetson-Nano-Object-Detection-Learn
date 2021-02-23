import json
import numpy as np
import tensorflow as tf

class Detector():
    def __init__(self):
        self.tf_sess = None 
        self.tf_input = None 
        self.tf_scores = None 
        self.tf_boxes = None 
        self.tf_classes = None 
        self.tf_num_detections = None 

    def load_tf_graph(self, graph_name):
        # load frozen inference graph
        frcnn_graph = tf.Graph()
        with frcnn_graph.as_default():
            trt_graph = tf.GraphDef()
            
        with tf.gfile.GFile(graph_name, 'rb') as f:
            serialized_graph = f.read()
            trt_graph.ParseFromString(serialized_graph)
            tf.import_graph_def(trt_graph, name='')

            tf_config = tf.ConfigProto()
            tf_config.gpu_options.allow_growth = True

            self.tf_sess = tf.Session(config=tf_config)

            self.tf_input = self.tf_sess.graph.get_tensor_by_name('image_tensor:0')
            self.tf_scores = self.tf_sess.graph.get_tensor_by_name('detection_scores:0')
            self.tf_boxes = self.tf_sess.graph.get_tensor_by_name('detection_boxes:0')
            self.tf_classes = self.tf_sess.graph.get_tensor_by_name('detection_classes:0')
            self.tf_num_detections = self.tf_sess.graph.get_tensor_by_name('num_detections:0')
        return True

    def detect(self, img):
        # detection 
        scores, boxes, classes, num_detections = self.tf_sess.run([self.tf_scores, 
                                                            self.tf_boxes, 
                                                            self.tf_classes, 
                                                            self.tf_num_detections], 
                                                            feed_dict={
                                                                self.tf_input: img[None, ...]
                                                            })

        tmp = boxes[0]  # y1, x1, y2, x1
        boxes = np.zeros_like(tmp)
        boxes[:,0] = tmp[:,1]
        boxes[:,1] = tmp[:,0]
        boxes[:,2] = tmp[:,3]
        boxes[:,3] = tmp[:,2]
        scores = scores[0]
        classes = classes[0]
        pad = np.zeros_like(scores)

        outs = np.hstack((np.expand_dims(pad, axis=1), 
                          np.expand_dims(classes, axis=1), 
                          np.expand_dims(scores, axis=1), 
                          boxes))
        return outs.reshape(1, 1, 1, outs.shape[0], outs.shape[1])