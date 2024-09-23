# ONNXモデルによる推論(SIGNATE上で動作させるものと同等)
import onnxruntime as ort
from pathlib import Path
import cv2
import numpy as np
import datetime

input_image_dir = Path("dataset/validation/validation/0.25x")
output_image_dir = Path("output_ESPCN")
output_image_dir.mkdir(exist_ok=True, parents=True)

sess = ort.InferenceSession("model.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
input_images = []
output_images = []
output_paths = []

print("load image")
for image_path in input_image_dir.iterdir():
    output_iamge_path = output_image_dir / image_path.relative_to(input_image_dir)
    input_image = cv2.imread(str(image_path))
    input_image = np.array([cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB).transpose((2,0,1))], dtype=np.float32)/255
    input_images.append(input_image)
    output_paths.append(output_iamge_path)

print("inference")
start_time = datetime.datetime.now()
for input_image in input_images:
    output_images.append(sess.run(["output"], {"input": input_image})[0])
end_time = datetime.datetime.now()

print("save image")
for output_path, output_image in zip(output_paths, output_images):
    output_image = cv2.cvtColor((output_image.transpose((0,2,3,1))[0]*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), output_image)

print(f"inference time: {(end_time - start_time).total_seconds() / len(input_images)}[s/image]")
