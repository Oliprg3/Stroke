import os
import torch
from fastapi import FastAPI, UploadFile, File
from typing import Dict
import numpy as np
import SimpleITK as sitk
from src.models.classifier import SliceEncoder2p5D, AttentionAggregator
from src.data.slice_utils import make_2p5d_stacks

app = FastAPI(title="Stroke CT ICH Triage")

CKPT = os.getenv("CKPT_PATH", "artifacts/best_classifier.pt")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
bundle = torch.load(CKPT, map_location=device)
cfg = bundle['cfg']
enc = SliceEncoder2p5D(backbone=cfg['model']['backbone'], k=cfg['data']['num_slices_stack'], pretrained=False).to(device)
agg = AttentionAggregator(in_dim=256, hidden=cfg['model']['hidden_dim'], num_classes=2).to(device)
enc.load_state_dict(bundle['enc']); enc.eval()
agg.load_state_dict(bundle['agg']); agg.eval()

@app.post("/infer")

async def infer_ct(nifti: UploadFile = File(...)) -> Dict:
      import tempfile, gzip, shutil
    suffix = ".nii.gz" if nifti.filename.endswith(".gz") else ".nii"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await nifti.read()
        tmp.write(content)
        tmp_path = tmp.name
    vol = sitk.ReadImage(tmp_path)
    arr = sitk.GetArrayFromImage(vol).astype(np.float32)  
    stacks, idxs = make_2p5d_stacks(arr, k=cfg['data']['num_slices_stack'], stride=cfg['data']['slice_stride'])
    x = torch.from_numpy(stacks).to(device)  
    with torch.no_grad():
        feats = enc(x)
        feats = feats.unsqueeze(0) 
        logits, attn = agg(feats, [feats.shape[1]])
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy().tolist()
        attn = attn.squeeze(0).cpu().numpy().tolist()
    return {"prob_no_ich": probs[0], "prob_ich": probs[1], "attention": attn, "slice_indices": idxs.tolist()}
