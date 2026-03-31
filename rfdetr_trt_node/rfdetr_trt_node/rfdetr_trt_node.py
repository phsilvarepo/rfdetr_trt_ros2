#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import os

class RfdetrTRTNode(Node):
    def __init__(self):
        super().__init__('rfdetr_trt_node')

        # 1. Fetch values from Environment Variables
        env_model = os.environ.get('MODEL_PATH', '/ros_ws/src/rfdetr_trt_node/rfdetr_trt_node/models/rfdetr_trt.engine')
        env_input = os.environ.get('INPUT_TOPIC', '/camera/image_raw')
        env_output = os.environ.get('OUTPUT_TOPIC', '/rfdetr/image_annotated')
        env_conf = os.environ.get('CONFIDENCE_THRESHOLD', '0.5')
        self.img_size = int(os.getenv('IMAGE_RESOLUTION', '576'))

        # 2. Declare ROS parameters
        self.declare_parameter('engine_path', env_model)
        self.declare_parameter('input_topic', env_input)
        self.declare_parameter('output_topic', env_output)
        self.declare_parameter('conf_threshold', float(env_conf))

        # 3. Get values
        self.engine_path = self.get_parameter('engine_path').value
        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.conf = self.get_parameter('conf_threshold').value

        self.get_logger().info(f"--- RF-DETR TRT Initialized ---")
        self.get_logger().info(f"Engine: {self.engine_path} | Conf: {self.conf}")
        self.get_logger().info(f"Subscribed to: {self.input_topic}")
        self.get_logger().info(f"Publishing to: {self.output_topic}")

        self.bridge = CvBridge()
        self.load_engine()

        # 4. Subscriber and Publisher
        self.subscription = self.create_subscription(
            Image, self.input_topic, self.image_callback, 10)
        self.publisher = self.create_publisher(
            Image, self.output_topic, 10)

    def load_engine(self):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(None, "")
        
        if not os.path.exists(self.engine_path):
            self.get_logger().error(f"Engine file not found: {self.engine_path}")
            return

        with open(self.engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context_trt = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = [], [], [], cuda.Stream()

        for i in range(self.engine.num_io_tensors):
            binding_name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(binding_name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding_name))
            device_mem = cuda.mem_alloc(trt.volume(shape) * np.dtype(dtype).itemsize)
            self.bindings.append(int(device_mem))
            
            if self.engine.get_tensor_mode(binding_name) == trt.TensorIOMode.INPUT:
                self.inputs.append({'binding': binding_name, 'device_mem': device_mem, 'dtype': dtype, 'shape': shape})
            else:
                self.outputs.append({'binding': binding_name, 'device_mem': device_mem, 'dtype': dtype, 'shape': shape})

    def preprocess(self, img):
        # Use dynamic resolution from params
        resized = cv2.resize(img, (self.img_size, self.img_size))
        img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0
        img_transposed = np.transpose(img_norm, (2, 0, 1))
        input_tensor = np.expand_dims(img_transposed, axis=0).astype(np.float32)
        return np.ascontiguousarray(input_tensor)

    def image_callback(self, msg):
        self.get_logger().info("Processing image...")
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        input_tensor = self.preprocess(cv_img)

        cuda.memcpy_htod(self.inputs[0]['device_mem'], input_tensor)
        self.context_trt.execute_v2(self.bindings)

        # Retrieve outputs
        boxes_shape = self.engine.get_tensor_shape(self.outputs[0]['binding'])
        labels_shape = self.engine.get_tensor_shape(self.outputs[1]['binding'])

        boxes_host = np.empty(trt.volume(boxes_shape), dtype=np.float32)
        labels_host = np.empty(trt.volume(labels_shape), dtype=np.float32)

        cuda.memcpy_dtoh(boxes_host, self.outputs[0]['device_mem'])
        cuda.memcpy_dtoh(labels_host, self.outputs[1]['device_mem'])

        boxes = boxes_host.reshape(1, 300, 4)[0]
        logits = labels_host.reshape(1, 300, 91)[0]

        probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
        scores = np.max(probs, axis=-1)
        class_ids = np.argmax(probs, axis=-1)

        H, W = cv_img.shape[:2]
        detections = []

        for i in range(300):
            if scores[i] < self.conf: # Use dynamic threshold
                continue

            cx, cy, bw, bh = boxes[i]
            x1 = int((cx - bw / 2) * W)
            y1 = int((cy - bh / 2) * H)
            x2 = int((cx + bw / 2) * W)
            y2 = int((cy + bh / 2) * H)
            detections.append([x1, y1, x2, y2, scores[i], class_ids[i]])
        
        detections = self.nms(detections, iou_threshold=0.5)
        annotated = self.draw_boxes(cv_img, detections)

        out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        out_msg.header = msg.header
        self.publisher.publish(out_msg)


    def draw_boxes(self, image, detections):
        for det in detections:  # detections format: [x1, y1, x2, y2, conf, class_id]
            x1, y1, x2, y2, conf, cls = det
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"{cls}: {conf:.2f}"
            cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        return image

    def nms(self, detections, iou_threshold=0.5):
        """
        Apply Non-Maximum Suppression to avoid overlapping boxes.
        
        detections: list of [x1, y1, x2, y2, score, class_id]
        iou_threshold: float, IoU threshold to suppress boxes
        """
        if len(detections) == 0:
            return []

        # Convert to array for easier computation
        dets = np.array(detections)
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]
        class_ids = dets[:, 5]

        keep_boxes = []

        # Process each class separately
        unique_classes = np.unique(class_ids)
        for cls in unique_classes:
            idxs = np.where(class_ids == cls)[0]
            cls_boxes = dets[idxs]

            # Sort by confidence score descending
            order = np.argsort(cls_boxes[:, 4])[::-1]
            cls_boxes = cls_boxes[order]

            while len(cls_boxes) > 0:
                # Pick the box with highest score
                box = cls_boxes[0]
                keep_boxes.append(box.tolist())

                if len(cls_boxes) == 1:
                    break

                # Compute IoU of the remaining boxes with the first box
                xx1 = np.maximum(box[0], cls_boxes[1:, 0])
                yy1 = np.maximum(box[1], cls_boxes[1:, 1])
                xx2 = np.minimum(box[2], cls_boxes[1:, 2])
                yy2 = np.minimum(box[3], cls_boxes[1:, 3])

                w = np.maximum(0, xx2 - xx1)
                h = np.maximum(0, yy2 - yy1)
                inter = w * h

                area_box = (box[2] - box[0]) * (box[3] - box[1])
                area_cls_boxes = (cls_boxes[1:, 2] - cls_boxes[1:, 0]) * (cls_boxes[1:, 3] - cls_boxes[1:, 1])
                iou = inter / (area_box + area_cls_boxes - inter)

                # Keep boxes with IoU less than threshold
                cls_boxes = cls_boxes[1:][iou < iou_threshold]

        return keep_boxes

def main(args=None):
    rclpy.init(args=args)
    node = RfdetrTRTNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
