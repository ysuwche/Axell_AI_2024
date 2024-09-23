import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import trange, tqdm
from torchvision import transforms

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ESPCN4x()
model.to(device)

# チェックポイントの保存先ディレクトリ
checkpoint_dir = "/content/drive/My Drive/checkpoints"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# TensorBoardのライターを作成
writer = SummaryWriter("log_ESPCN")

# オプティマイザー、スケジューラー、損失関数の設定
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)
criterion = nn.MSELoss()
awp = AWP(model, criterion, optimizer, adv_param="weight", adv_lr=1e-3, adv_eps=1e-2)

# チェックポイントのロード関数
def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        print(f"Checkpoint loaded. Resuming from epoch {epoch}")
        return epoch
    else:
        print("No checkpoint found. Starting from scratch.")
        return 0

# 学習の再開に必要な設定
start_epoch = load_checkpoint(model, optimizer, scheduler, os.path.join(checkpoint_dir, "last_checkpoint.pth"))

# トレーニングループの設定
num_epoch = 100  # トータルエポック数
for epoch in trange(start_epoch, num_epoch, desc="EPOCH"):
    try:
        # 学習
        model.train()
        train_loss = 0.0
        validation_loss = 0.0
        train_psnr = 0.0
        validation_psnr = 0.0

        if epoch <= 59:
           for idx, (low_resolution_image, high_resolution_image) in tqdm(enumerate(train_data_loader), desc=f"EPOCH[{epoch}] TRAIN", total=len(train_data_loader)):
               low_resolution_image = low_resolution_image.to(device)
               high_resolution_image = high_resolution_image.to(device)
               optimizer.zero_grad()
               output = model(low_resolution_image)
               loss = criterion(output, high_resolution_image)
               loss.backward()

               train_loss += loss.item() * low_resolution_image.size(0)
               for image1, image2 in zip(output, high_resolution_image):
                   train_psnr += calc_psnr(image1, image2)
               optimizer.step()

           scheduler.step()

        else:
            for idx, (low_resolution_image, high_resolution_image) in tqdm(enumerate(train_NoAug_data_loader), desc=f"EPOCH[{epoch}] TRAIN", total=len(train_NoAug_data_loader)):
                low_resolution_image = low_resolution_image.to(device)
                high_resolution_image = high_resolution_image.to(device)
                optimizer.zero_grad()
                output = model(low_resolution_image)
                loss = criterion(output, high_resolution_image)
                loss.backward()

                if epoch >= 80:
                    awp.attack_backward(low_resolution_image, high_resolution_image)

                train_loss += loss.item() * low_resolution_image.size(0)
                for image1, image2 in zip(output, high_resolution_image):
                    train_psnr += calc_psnr(image1, image2)
                optimizer.step()

            scheduler.step()

        # 検証
        model.eval()
        with torch.no_grad():
            for idx, (low_resolution_image, high_resolution_image) in tqdm(enumerate(validation_data_loader), desc=f"EPOCH[{epoch}] VALIDATION", total=len(validation_data_loader)):
                low_resolution_image = low_resolution_image.to(device)
                high_resolution_image = high_resolution_image.to(device)
                output = model(low_resolution_image)
                loss = criterion(output, high_resolution_image)
                validation_loss += loss.item() * low_resolution_image.size(0)
                for image1, image2 in zip(output, high_resolution_image):
                    validation_psnr += calc_psnr(image1, image2)

            # 10エポックごとに画像を表示
            if epoch < 10 or epoch % 10 == 0:
                to_image = transforms.ToPILImage()
                display(to_image(low_resolution_image[0].cpu()))
                display(to_image(high_resolution_image[0].cpu()))
                display(to_image(output[0].cpu()))

        # TensorBoardにログを記録
        writer.add_scalar("train/loss", train_loss / len(train_dataset), epoch)
        writer.add_scalar("train/psnr", train_psnr / len(train_dataset), epoch)
        writer.add_scalar("validation/loss", validation_loss / len(validation_dataset), epoch)
        writer.add_scalar("validation/psnr", validation_psnr / len(validation_dataset), epoch)
        writer.add_image("output_ES", output[0], epoch)

        # チェックポイントの保存
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, checkpoint_path)

        # 最新のチェックポイントも保存
        torch.save({
            'epoch': epoch + 1,  # 次回のエポック用
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, os.path.join(checkpoint_dir, "last_checkpoint.pth"))

    except Exception as ex:
        print(f"EPOCH[{epoch}] ERROR: {ex}")

# 最後にTensorBoardをクローズ
writer.close()

# モデルの保存（ONNXフォーマットで）
model.to(torch.device("cpu"))
dummy_input = torch.randn(1, 3, 128, 128, device="cpu")
torch.onnx.export(model, dummy_input, "model.onnx",
                  opset_version=17,
                  input_names=["input"],
                  output_names=["output"],
                  dynamic_axes={"input": {2: "height", 3: "width"}})
