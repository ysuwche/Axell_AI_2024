# PSNR計算(従来手法との比較付き)

# PSNRを計算する関数
def calc_psnr(image1: torch.Tensor, image2: torch.Tensor) -> float:
    to_image = transforms.ToPILImage()
    image1 = np.array(to_image(image1).convert('RGB'))
    image2 = np.array(to_image(image2).convert('RGB'))
    return cv2.PSNR(image1, image2)

original_image_dir = Path("dataset/validation/validation/original")
output_label = ["ESPCN", "NEAREST", "BILINEAR", "BICUBIC"]
output_psnr = [0.0, 0.0, 0.0, 0.0]
original_image_paths = list(original_image_dir.iterdir())
for image_path in tqdm(original_image_paths):
    input_image_path = input_image_dir / image_path.relative_to(original_image_dir)
    output_iamge_path = output_image_dir / image_path.relative_to(original_image_dir)
    input_image = cv2.imread(str(input_image_path))
    original_image = cv2.imread(str(image_path))
    espcn_image = cv2.imread(str(output_iamge_path))
    output_psnr[0] += cv2.PSNR(original_image, espcn_image)
    h, w = original_image.shape[:2]
    output_psnr[1] += cv2.PSNR(original_image, cv2.resize(input_image, (w, h), interpolation=cv2.INTER_NEAREST))
    output_psnr[2] += cv2.PSNR(original_image, cv2.resize(input_image, (w, h), interpolation=cv2.INTER_LINEAR))
    output_psnr[3] += cv2.PSNR(original_image, cv2.resize(input_image, (w, h), interpolation=cv2.INTER_CUBIC))

# 拡大結果を表示
from PIL import Image
display(Image.fromarray(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)))
display(Image.fromarray(cv2.cvtColor(espcn_image, cv2.COLOR_BGR2RGB)))
display(Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)))

for label, psnr in zip(output_label, output_psnr):
    print(f"{label}: {psnr / len(original_image_paths)}")
