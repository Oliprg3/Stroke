import streamlit as st
import nibabel as nib
import numpy as np
import requests
import tempfile
import matplotlib.pyplot as plt

st.set_page_config(page_title="CT ICH Triage", layout="wide")
st.title("Head CT ICH Triage (Prototype)")

api_url = st.text_input("Inference API URL", value="http://localhost:8000/infer")

up = st.file_uploader("Upload NIfTI (.nii or .nii.gz)", type=["nii", "nii.gz"])
if st.button("Run Inference", disabled=(up is None)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=up.name) as tmp:
        tmp.write(up.read())
        tmp_path = tmp.name
    with open(tmp_path, "rb") as f:
        files = {"nifti": (up.name, f, "application/octet-stream")}
        with st.spinner("Calling API..."):
            r = requests.post(api_url, files=files, timeout=120)
    if r.ok:
        res = r.json()
        st.success(f"p(ICH) = {res['prob_ich']:.3f}")
       
        img = nib.load(tmp_path).get_fdata().astype(np.float32)
        attn = np.array(res["attention"])
        idxs = np.array(res["slice_indices"])
   
        attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-6)
     
        order = np.argsort(attn)[::-1][:8]
        cols = st.columns(4)
        for i, o in enumerate(order):
            sl = idxs[o]
            ax_img = img[sl]
            with cols[i%4]:
                st.write(f"Slice {sl} | attn {attn[o]:.2f}")
                fig, ax = plt.subplots()
                ax.imshow(ax_img, cmap="gray")
                ax.axis("off")
                st.pyplot(fig)
    else:
        st.error(f"API Error: {r.status_code} - {r.text}")
