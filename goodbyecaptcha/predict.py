from io import BytesIO

import os
import cv2
import numpy as np
from PIL import Image

from goodbyecaptcha import package_dir

__all__ = [
    'is_marked',
    "get_output_layers",
    "draw_prediction",
    "predict",
    "ultralytics_yolo",
    "darknet_yolo",
]


def is_marked(img_path):
    """Detect specific color for detect marked"""
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    for i in range(w):
        for j in range(h):
            r, g, b = img.getpixel((i, j))
            if r == 0 and g == 0 and b == 254:  # Detect Blue Color
                return True
    return False


def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


def draw_prediction(img, x, y, x_plus_w, y_plus_h):
    """Paint Rectangle Blue for detect prediction"""
    color = 256  # Blue
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, cv2.FILLED)

"""
reference: https://alimustoofaa.medium.com/how-to-load-model-yolov8-onnx-cv2-dnn-3e176cde16e6
"""

def check_model_exist(model_pairs):
    for m, c in model_pairs:
        if os.path.isfile(m) and os.path.isfile(c):
            return m, c
    return None

# yolo5l
def ultralytics_yolo(model_name = "yolov8l-cls.pt"):
    filename = os.path.splitext(model_name)[0]
    onnx_file = f"{filename}.onnx"
    class_file = f"{filename}.txt"

    base_model = os.path.basename(model_name)
    base_onnx = os.path.basename(onnx_file)
    base_class = os.path.basename(class_file)

    pair = check_model_exist( \
        [(onnx_file, class_file),
         (f"{package_dir}/models/{base_onnx}",
          f"{package_dir}/models/{base_class}")])

    if pair:
        weight_file, class_file = pair
    else:
        class_file = base_class
        from ultralytics.yolo.engine.model import YOLO
        model = YOLO(base_model)
        results = model.predict(source="https://ultralytics.com/images/bus.jpg")[0]
        model.export(format="onnx",opset=12)  # export the model to ONNX format
        weight_file = f"{package_dir}/models/{base_onnx}"
        # print(f"rename {base_onnx} to {weight_file}")
        os.rename(base_onnx, weight_file)
        os.remove(base_model)
        os.remove("bus.jpg")
        class_file = f"{package_dir}/models/{base_class}"
        classes = model.names
        with open(class_file, 'w') as f:
            print('\n'.join([classes[k] for k in classes]), file = f)

    # Load Model
    net = cv2.dnn.readNet(weight_file)
    # Load Class
    with open(class_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    return net, classes


async def darknet_yolo(model_name = "yolov3"):
    weight_file = f"{package_dir}/models/yolov3.weights"
    class_file = f"{package_dir}/models/yolov3.txt"
    file_cfg = f"{package_dir}/models/yolov3.cfg"

    # Import YoloV3
    try:
        net = cv2.dnn.readNet(weight_file, file_cfg)
    except Exception:
        # Download YoloV3
        yolo_url = 'https://pjreddie.com/media/files/yolov3.weights'

        import urllib3
        from tqdm import tqdm

        with urllib3.PoolManager() as http:
            # Get data from url.
            data = http.request('GET', yolo_url, preload_content=False)

            try:
                total_length = int(data.headers['content-length'])
            except (KeyError, ValueError, AttributeError):
                total_length = 0

            process_bar = tqdm(total=total_length)

            # 10 * 1024
            _data = BytesIO()
            for chunk in data.stream(10240):
                _data.write(chunk)
                process_bar.update(len(chunk))
            process_bar.close()
        # Save weights matrix
        with open(weight_file, 'wb') as f:
            f.write(_data.getvalue())
        return await darknet_yolo(model_name)  # Reload method

    with open(class_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
        return net, classes


async def predict(file, obj=None):
    """Predict Object on image"""
    image = cv2.imread(file)
    width = image.shape[1]
    height = image.shape[0]
    scale = 0.00392

    if '.' in file:
        net, classes = ultralytics_yolo(file)
    else:
        net, classes = darknet_yolo(file)

    if obj is None:
        blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(get_output_layers(net))
        classes_names = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = scores[class_id]
                if confidence > 0.5:
                    classes_names.append(classes[class_id])
        return classes_names  # Return all names object in the images
    else:
        out_path = f"{package_dir}/tmp/{hash(file)}.jpg"
        blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(get_output_layers(net))

        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        out = False
        for i in indices:
            if classes[int(class_ids[int(i)])] == obj or (obj == 'vehicles' and (
                    classes[int(class_ids[int(i)])] == 'car' or classes[int(class_ids[int(i)])] == 'truck')):
                out = out_path
                box = boxes[i[0]]
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]
                draw_prediction(image, round(x), round(y), round(x + w), round(y + h))
            # Save Image
        if out:
            cv2.imwrite(out_path, image)
        return out  # Return path of images or False if not found object
