# rfdetr_trt_node

A configuration-driven ROS 2 (Humble) node that runs RF-DETR object detection inference using a TensorRT engine on any image topic and publishes annotated images and/or bounding box detections — all without recompiling.

## Topics

### Subscribed

| Topic | Type | Description |
|-------|------|-------------|
| `/image_raw` (default) | `sensor_msgs/Image` | Input image stream. Override with `INPUT_TOPIC`. |

### Published (all optional, enabled by setting the env var)

| Env Variable | Topic Type | Description |
|---|---|---|
| `OUTPUT_TOPIC_IMAGE` | `sensor_msgs/Image` | Annotated frame with bounding boxes drawn |
| `OUTPUT_TOPIC_BB` | `vision_msgs/Detection2DArray` | Bounding boxes with class ID and confidence score |

> Publishers are only created when the corresponding env variable is set. Set only the outputs you need.

---

## Configuration (Environment Variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `/ros_ws/src/rfdetr_trt_node/rfdetr_trt_node/models/rfdetr_trt.engine` | Path to the TensorRT `.engine` file inside the container |
| `CONFIDENCE_THRESHOLD` | `0.5` | Minimum detection confidence (0.0 – 1.0) |
| `IMAGE_RESOLUTION` | `547` | Inference resolution (square) passed to the preprocessor |
| `INPUT_TOPIC` | `/image_raw` | ROS 2 topic to subscribe for input images |
| `OUTPUT_TOPIC_IMAGE` | *(unset)* | Publish annotated debug images to this topic |
| `OUTPUT_TOPIC_BB` | *(unset)* | Publish bounding box detections to this topic |

---

## Quick Start (Docker)

### 1. Build the image

```bash
docker build -t rfdetr_trt_node .
```

### 2. Run — detections + annotated image

```bash
docker run --rm --network host \
  --gpus all \
  -e INPUT_TOPIC=/camera/image_raw \
  -e OUTPUT_TOPIC_IMAGE=/rfdetr/image_annotated \
  -e OUTPUT_TOPIC_BB=/rfdetr/detections \
  -e CONFIDENCE_THRESHOLD=0.4 \
  rfdetr_trt_node
```

### 3. Run — bounding boxes only

```bash
docker run --rm --network host \
  --gpus all \
  -e INPUT_TOPIC=/camera/image_raw \
  -e OUTPUT_TOPIC_BB=/rfdetr/detections \
  rfdetr_trt_node
```

> `--network host` is required so the container can communicate with other ROS 2 nodes via Fast DDS UDP multicast.
> `--gpus all` is required for GPU/CUDA access inside the container.

---

## Using a Custom Engine

Mount your own TensorRT `.engine` file into the container:

```bash
docker run --rm --network host \
  --gpus all \
  -v /path/to/my_model.engine:/models/my_model.engine \
  -e MODEL_PATH=/models/my_model.engine \
  -e INPUT_TOPIC=/camera/image_raw \
  -e OUTPUT_TOPIC_BB=/rfdetr/detections \
  rfdetr_trt_node
```

> The TensorRT engine must be built on the **same GPU architecture** as the target machine. Engines are not portable across different GPU types.

---

## Building from Source (without Docker)

**Prerequisites:** ROS 2 Humble, Python 3, CUDA toolkit, TensorRT, `cv_bridge`, `vision_msgs`

```bash
# Install Python dependencies
pip3 install pycuda opencv-python tensorrt==10.14.1.48.post1
pip3 install "numpy<2.0"

# Clone into your workspace
cd ~/ros_ws/src
git clone <repo-url> rfdetr_trt_node

# Build
cd ~/ros_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select rfdetr_trt_node

# Source and run
source install/setup.bash
export INPUT_TOPIC=/image_raw
export MODEL_PATH=/path/to/rfdetr_trt.engine
export OUTPUT_TOPIC_BB=/rfdetr/detections
export OUTPUT_TOPIC_IMAGE=/rfdetr/image
ros2 run rfdetr_trt_node rfdetr_trt_node
```

---

## Output Message Details

### Detection2DArray (bounding boxes)

Each `Detection2D` in the array contains:

- `bbox.center.position.x/y` — box centre in pixels
- `bbox.size_x/size_y` — box width and height in pixels
- `results[0].hypothesis.class_id` — COCO class name string (e.g. `"person"`)
- `results[0].hypothesis.score` — confidence score (0.0 – 1.0)

The node uses the **COCO 91-class** label set. An empty `detections` array is published when no objects pass the confidence threshold, so downstream nodes always receive a message every frame.

---

## Pipeline Details

### Preprocessing

Input frames are resized to `IMAGE_RESOLUTION × IMAGE_RESOLUTION`, converted to RGB, normalised to `[0, 1]`, and transposed to `(C, H, W)` before being copied to GPU memory.

### Inference

The node uses a TensorRT execution context with pre-allocated CUDA device buffers. `execute_v2` is called synchronously per frame.

### Postprocessing

Raw outputs are:
- **Boxes** — `(1, 300, 4)` in normalised `cx, cy, w, h` format, scaled to pixel coordinates
- **Logits** — `(1, 300, 91)` class scores, converted to probabilities via softmax

Detections above `CONFIDENCE_THRESHOLD` are passed through per-class **Non-Maximum Suppression** (IoU threshold `0.5`) before publishing.

---

## Dependencies

| Package | Source |
|---------|--------|
| ROS 2 Humble | ros.org |
| `ros-humble-cv-bridge` | apt |
| `ros-humble-vision-msgs` | apt |
| CUDA 13 | nvidia/cuda Docker base |
| TensorRT 10 | pip / apt (`python3-libnvinfer`) |
| PyCUDA | pip |
| OpenCV | pip |
| numpy < 2 | pip |
