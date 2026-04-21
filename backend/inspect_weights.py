import torch
import sys

path = r'C:\Users\ankit\OneDrive\Desktop\InlandRoute\ml\models\saved\ensemble\tft_swin_ensemble.pt'
try:
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    print("Checkpoint Type:", type(checkpoint))
    if isinstance(checkpoint, dict):
        print("Keys:", checkpoint.keys())
        if 'model_state_dict' in checkpoint:
            sd = checkpoint['model_state_dict']
            print("Total parameters:", len(sd))
            for key in list(sd.keys()):
                if 'temporal_vsn' in key and 'fc1.weight' in key:
                    print(f"{key}: {sd[key].shape}")
                if 'static_vsn' in key and 'fc1.weight' in key:
                    print(f"{key}: {sd[key].shape}")
                if 'tft.encoder' in key and 'weight' in key:
                     print(f"{key}: {sd[key].shape}")
        elif 'state_dict' in checkpoint:
             sd = checkpoint['state_dict']
             print("State Dict Keys (sample):", list(sd.keys())[:10])
    else:
        print("Model Object Type:", type(checkpoint))
except Exception as e:
    print("Error:", e)
