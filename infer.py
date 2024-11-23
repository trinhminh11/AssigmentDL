import os
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import os

import torch
import segmentation_models_pytorch as smp
import argparse

val_transformation = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

color_dict= {0: (0, 0, 0),
             1: (255, 0, 0),
             2: (0, 255, 0)}

def mask_to_rgb(mask):
    output = np.zeros((mask.shape[0], mask.shape[1], 3))

    for k in color_dict.keys():
        output[mask==k] = color_dict[k]

    return np.uint8(output)    

def predict(model: smp.UnetPlusPlus, img_path: str, device = torch.device('cuda' if torch.cuda.is_available() else "cpu")):
    ori_img = cv2.imread(img_path)
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    ori_w = ori_img.shape[0]
    ori_h = ori_img.shape[1]
    img = cv2.resize(ori_img, (256, 256))
    transformed = val_transformation(image=img)
    input_img = transformed["image"]
    input_img = input_img.unsqueeze(0).to(device)
    with torch.no_grad():
        output_mask = model.forward(input_img).squeeze(0).cpu().numpy().transpose(1,2,0)
    mask = cv2.resize(output_mask, (ori_h, ori_w))
    mask = np.argmax(mask, axis=2)
    mask_rgb = mask_to_rgb(mask)
    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)

    cv2.imwrite(f"{os.path.splitext(img_path)[0]}_result.jpeg", mask_rgb)

def main():
    if not os.path.exists("checkpoints/model.pth"):
        import zipfile
        print("UnZipping checkpoints ...")
        with zipfile.ZipFile("checkpoints/model.pth.zip", 'r') as zip_ref:
            zip_ref.extractall("checkpoints/")
            

    parser = argparse.ArgumentParser(description='Inference')

    parser.add_argument('--image_path', type=str, help='input image path', default="image.jpeg")

    args = parser.parse_args()

    image_path = args.image_path

    model = smp.UnetPlusPlus(
        encoder_name="resnet34",        
        encoder_weights="imagenet",     
        in_channels=3,                  
        classes=3     
    )

    checkpoint = torch.load('checkpoints/model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model.to(device)

    predict(model, image_path, device)


if __name__ == "__main__":
    main()





