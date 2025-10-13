import os, glob
from PIL import Image

BASE_IMG = r'unified_dataset\images\train'
BASE_LBL = r'unified_dataset\labels\train'
OUT_DIR  = r'overlays'  # contiendra stamp/, checkbox/, keyvalue/
os.makedirs(os.path.join(OUT_DIR,'stamp'), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR,'checkbox'), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR,'keyvalue'), exist_ok=True)

def save_crop(img_path, cls, xywh, idx):
    im = Image.open(img_path).convert('RGBA')
    W,H = im.size
    x,y,w,h = xywh
    cx,cy = int(x*W), int(y*H)
    bw,bh = int(w*W), int(h*H)
    x1,y1 = max(0, cx-bw//2), max(0, cy-bh//2)
    x2,y2 = min(W, x1+bw),  min(H, y1+bh)
    crop = im.crop((x1,y1,x2,y2))
    sub = {6:'stamp',10:'checkbox',8:'keyvalue'}.get(cls,None)
    if sub and crop.size[0]>5 and crop.size[1]>5:
        base = os.path.join(OUT_DIR,sub,f'{os.path.splitext(os.path.basename(img_path))[0]}_{idx}.png')
        crop.save(base)

for lbl in glob.glob(os.path.join(BASE_LBL,'*.txt')):
    img = lbl.replace(r'labels\\train', r'images\\train').rsplit('.',1)[0]+'.jpg'
    if not os.path.exists(img):
        img = img[:-4]+'.png' if os.path.exists(img[:-4]+'.png') else None
    if not img: continue
    with open(lbl,'r') as f:
        for i,line in enumerate(f):
            sp = line.split()
            if len(sp)!=5: continue
            cls = int(sp[0]); x,y,w,h = map(float,sp[1:])
            if cls in (6,10,8):  # stamp, checkbox, keyvalue
                save_crop(img, cls, (x,y,w,h), i)
print('Done.')
