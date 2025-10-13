import os, glob, random
from PIL import Image
import numpy as np

IN_IMG  = r'unified_dataset\images\train'
IN_LBL  = r'unified_dataset\labels\train'
OVR     = r'overlays'
OUT_IMG = r'synth\images'
OUT_LBL = r'synth\labels'
os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_LBL, exist_ok=True)

CLASS_ID = {'stamp':6,'checkbox':10,'keyvalue':8}

def paste_overlay(bg, ov, scale_range=(0.4,1.2)):
    W,H = bg.size
    s   = random.uniform(*scale_range)
    ow,oh = ov.size
    nw,nh = max(6,int(ow*s)), max(6,int(oh*s))
    ov2 = ov.resize((nw,nh), Image.LANCZOS)
    x1 = random.randint(0, max(0,W-nw))
    y1 = random.randint(0, max(0,H-nh))
    bg.alpha_composite(ov2, (x1,y1))
    # YOLO xywh norm
    cx,cy = (x1+nw/2)/W, (y1+nh/2)/H
    w,h   = nw/W, nh/H
    return cx,cy,w,h

targets = list(glob.glob(os.path.join(IN_IMG,'*.jpg')))+list(glob.glob(os.path.join(IN_IMG,'*.png')))
ov_paths = {k:list(glob.glob(os.path.join(OVR,k,'*.png'))) for k in CLASS_ID}

N = min(300, len(targets))  # jusqu'à 300 synth
random.shuffle(targets)
for img_path in targets[:N]:
    im = Image.open(img_path).convert('RGBA')
    boxes = []
    for k,paths in ov_paths.items():
        if not paths: continue
        for _ in range(random.randint(0,2)):   # 0,1 ou 2 overlays par classe
            ov = Image.open(random.choice(paths)).convert('RGBA')
            cx,cy,w,h = paste_overlay(im, ov)
            boxes.append((CLASS_ID[k],cx,cy,w,h))

    if not boxes: continue
    stem = os.path.splitext(os.path.basename(img_path))[0]+'_synth'
    out_img = os.path.join(OUT_IMG, stem+'.png')
    out_lbl = os.path.join(OUT_LBL, stem+'.txt')
    im.convert('RGB').save(out_img)
    with open(out_lbl,'w') as f:
        for cid,x,y,w,h in boxes:
            f.write(f'{cid} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n')

print('Synth done.')
