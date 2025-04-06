import cv2 
import torch
import numpy as np
import torch.nn.functional as F
from basicsr.models import create_model
from basicsr.utils.options import dict2str, parse
import torch.nn as nn


def self_ensemble(x, model):
    def forward_transformed(x, hflip, vflip, rotate, model):
        if hflip:
            x = torch.flip(x, (-2,))
        if vflip:
            x = torch.flip(x, (-1,))
        if rotate:
            x = torch.rot90(x, dims=(-2, -1))
        x = model(x)
        if rotate:
            x = torch.rot90(x, dims=(-2, -1), k=3)
        if vflip:
            x = torch.flip(x, (-1,))
        if hflip:
            x = torch.flip(x, (-2,))
        return x
    t = []
    for hflip in [False, True]:
        for vflip in [False, True]:
            for rot in [False, True]:
                t.append(forward_transformed(x, hflip, vflip, rot, model))
    t = torch.stack(t)
    return torch.mean(t, dim=0)

def process_video(input_video_path, output_video_path, model_restoration, factor = 4, self_assemble = True, skip_frames = 3):
    # Read the input video 
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not read the video.")
        return 
    
    # Get the video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4 files
    out_fps = fps/(skip_frames+1)
    out = cv2.VideoWriter(output_video_path, fourcc, out_fps, (frame_width, frame_height))

    with torch.inference_mode():
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames 
            if frame_count % (skip_frames + 1) != 0:
                frame_count += 1
                continue
            img = np.float32(frame) / 255.0 
            input_ = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()

            # Padding in case the image is not divisible by 4 
            b, c, h, w = input_.shape 
            H, W = (h + factor) // factor * factor, (w + factor) // factor * factor
            padh = H - h if h % factor != 0 else 0 
            padw = W - w if w % factor != 0 else 0 
            input_ = F.pad(input_, (0, padw, 0, padh), mode='reflect')

            if h < 3000 and w < 3000: 
                if self_assemble: 
                    restored = self_ensemble(input_, model_restoration)
                else:
                    restored = model_restoration(input_)

            else: 
                # SPLIT 
                input_1 = input_[:, :, :, 1::2]
                input_2 = input_[:, :, :, 0::2]
                if self_assemble: 
                    restored_1 = self_ensemble(input_1, model_restoration)
                    restored_2 = self_ensemble(input_2, model_restoration)
                else:
                    restored_1 = model_restoration(input_1)
                    restored_2 = model_restoration(input_2)
                restored = torch.zeros_like(input_)
                restored[:, :, :, 1::2] = restored_1 
                restored[:, :, :, 0::2] = restored_2 

            # Unpad the images to the original size 
            restored = restored[:, :, :h, :w]
            restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).numpy()
            restored = (restored * 255.0).astype(np.uint8)

            # Write the frame to the output video 
            out.write(restored[0])

    cap.release()
    out.release()

video_path = 'videos/input_video.mp4'
output_path = 'videos/speed-up_output_video.mp4'

yaml_path = 'Options/RetinexFormer_LOL_v1.yml'
opt = parse(yaml_path, is_train=False)
opt['dist'] = False
model_restoration = create_model(opt).net_g 

weights = 'experiments/RetinexFormer_LOL_v1/best_psnr_23.06_93000.pth'
checkpoint = torch.load(weights, weights_only=True)

try: 
    model_restoration.load_state_dict(checkpoint['params'], strict = True)
except:
    new_checkpoint = {}
    for k in checkpoint['params']:
        new_checkpoint['module.' + k] = checkpoint['params'][k]
    model_restoration.load_state_dict(new_checkpoint)

print("===>Testing using weights: ", weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()
process_video(video_path, output_path, model_restoration, factor = 4, self_assemble = True)
print("===>Video processing completed.")
            



