# -*- coding: utf-8 -*-
"""
MedSAM training script — simplified Dice handling.
Dice is now computed via utils.SurfaceDice.compute_dice_coefficient just like the demo you shared.
Other logic (model, loss, plotting, checkpointing) stays unchanged.
"""

# %% Imports & environment setup
import os
import glob
import random
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import monai
from segment_anything import sam_model_registry
import argparse


from utils.SurfaceDice import compute_dice_coefficient

ORGAN_IDS = {
    1: "liver", 2: "right_kidney", 3: "spleen", 4: "pancreas",
    5: "aorta", 6: "IVC", 7: "right_adrenal", 8: "left_adrenal",
    9: "gallbladder", 10: "esophagus", 11: "stomach", 12: "duodenum",
    13: "left_kidney",
}

torch.manual_seed(2023)
torch.cuda.empty_cache()

class NpyDataset(Dataset):
    """Return binary mask **and** its organ_id so we can report per‑organ Dice."""
    def __init__(self, data_root, bbox_shift=20):
        self.img_root = os.path.join(data_root, "imgs")
        self.gt_root  = os.path.join(data_root, "gts")
        self.gt_files = sorted(glob.glob(os.path.join(self.gt_root, "**/*.npy"), recursive=True))
        self.gt_files = [f for f in self.gt_files if os.path.isfile(os.path.join(self.img_root, os.path.basename(f)))]
        self.bbox_shift = bbox_shift
        print(f"Loaded {len(self.gt_files)} samples from {data_root}")

    def __len__(self):
        return len(self.gt_files)

    def __getitem__(self, idx):
        gt_path = self.gt_files[idx]
        img_path = os.path.join(self.img_root, os.path.basename(gt_path))
        img = np.load(img_path, mmap_mode="r", allow_pickle=True).transpose(2,0,1).astype(np.float32)
        gt  = np.load(gt_path, mmap_mode="r", allow_pickle=True)

        label_ids = np.unique(gt)[1:]
        organ_id = int(random.choice(label_ids))
        gt_bin = (gt == organ_id).astype(np.uint8)

        y,x = np.where(gt_bin)
        x0,x1,y0,y1 = x.min(),x.max(),y.min(),y.max()
        H,W = gt_bin.shape; s=self.bbox_shift
        x0=max(0,x0-random.randint(0,s)); x1=min(W,x1+random.randint(0,s))
        y0=max(0,y0-random.randint(0,s)); y1=min(H,y1+random.randint(0,s))
        bbox = np.array([x0,y0,x1,y1],dtype=np.float32)

        return (
            torch.from_numpy(img),
            torch.from_numpy(gt_bin[None]),
            torch.from_numpy(bbox),
            organ_id,
        )




class MedSAM(nn.Module):
    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        for p in self.prompt_encoder.parameters():
            p.requires_grad = False

    def forward(self, imgs, boxes):
        emb = self.image_encoder(imgs)
        with torch.no_grad():
            box_t = torch.as_tensor(boxes,dtype=torch.float32,device=imgs.device)
            if box_t.ndim==2: box_t = box_t[:,None]
            sparse,dense = self.prompt_encoder(points=None,boxes=box_t,masks=None)
        low,_ = self.mask_decoder(
            image_embeddings=emb,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        return F.interpolate(low,size=imgs.shape[-2:],mode="bilinear",align_corners=False)

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--tr_npy_path","-i", default="data/npy/CT_Abd")
parser.add_argument("--task_name", default="MedSAM-ViT-B")
parser.add_argument("--model_type", default="vit_b")
parser.add_argument("--checkpoint", default="work_dir/SAM/sam_vit_b_01ec64.pth")
parser.add_argument("--work_dir", default="./work_dir")
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-2)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--resume", default="")
parser.add_argument("--use_amp", action="store_true")
args = parser.parse_args()

run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_dir = os.path.join(args.work_dir, f"{args.task_name}-{run_id}")
os.makedirs(model_dir, exist_ok=True)

device = torch.device(args.device)

sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
model = MedSAM(sam.image_encoder, sam.mask_decoder, sam.prompt_encoder).to(device)
model.train()

opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True)
ce_loss  = nn.BCEWithLogitsLoss()


train_ds = NpyDataset(args.tr_npy_path)
train_loader = DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,pin_memory=True)

start_epoch = 0
if args.resume and os.path.isfile(args.resume):
    ckpt = torch.load(args.resume,map_location=device)
    model.load_state_dict(ckpt["model"]); opt.load_state_dict(ckpt["optimizer"])
    start_epoch = ckpt["epoch"]+1

scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

loss_curve, dice_curve = [], []

for epoch in range(start_epoch, args.num_epochs):
    model.train()
    epoch_loss, epoch_dice, n_batch = 0., 0., 0
    organ_sum = {oid: 0.0 for oid in ORGAN_IDS}
    organ_cnt = {oid: 0   for oid in ORGAN_IDS}

    for imgs, gts, boxes, organ_ids in tqdm(train_loader, desc=f"Epoch {epoch}"):
        imgs, gts = imgs.to(device), gts.to(device)
        boxes_np = boxes.numpy()

        with torch.cuda.amp.autocast(enabled=args.use_amp):
            preds = model(imgs, boxes_np)
            loss  = seg_loss(preds, gts.float()) + ce_loss(preds, gts.float())
        if args.use_amp:
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:
            loss.backward(); opt.step()
        opt.zero_grad()

        dice = compute_dice_coefficient(gts.cpu().numpy(), (preds > 0.5).cpu().numpy())
        epoch_loss += loss.item(); epoch_dice += dice; n_batch += 1

        for oid in organ_ids.tolist():
            organ_sum[oid] += dice
            organ_cnt[oid] += 1

    epoch_loss /= n_batch; epoch_dice /= n_batch
    loss_curve.append(epoch_loss); dice_curve.append(epoch_dice)

    print(f"[Epoch {epoch}] Loss: {epoch_loss:.4f} | Dice: {epoch_dice:.4f}")
    print("--- Per‑organ Dice ---")
    for oid, name in ORGAN_IDS.items():
        if organ_cnt[oid]:
            print(f"{name:<15s}: {organ_sum[oid] / organ_cnt[oid]:.4f}")
        else:
            print(f"{name:<15s}: N/A")
    print("-----------------------")

    ckpt = {"model": model.state_dict(), "optimizer": opt.state_dict(), "epoch": epoch}
    torch.save(ckpt, os.path.join(model_dir, "medsam_latest.pth"))

    plt.figure(); plt.plot(loss_curve); plt.title("Train Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.savefig(os.path.join(model_dir, "loss_curve.png")); plt.close()

print("Training complete.")

