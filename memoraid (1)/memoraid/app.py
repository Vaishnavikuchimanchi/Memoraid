import streamlit as st
import torch
from PIL import Image
import torch.nn.functional as F

from model import load_model
from utils.preprocess import transform


# ---------------- Page Config ----------------
st.set_page_config(
    page_title="MemorAid AI",
    layout="wide"
)


# ---------------- Custom Styling ----------------
st.markdown("""
<style>

/* Center and reduce page width */
.block-container {
    max-width: 900px;
    padding-top: 1.5rem !important;
    padding-bottom: 1rem !important;
}

/* Font sizes */
h1 {
    font-size:30px !important;
}

h3 {
    font-size:19px !important;
}

p, label, div {
    font-size:15px !important;
}


/* Calm Button */
div.stButton > button:first-child {
    background-color:#5A8F7B;
    color:white;
    border-radius:8px;
    height:40px;
    width:210px;
    font-size:14px;
    border:none;
}

.custom-divider {
    margin-top:8px;
    margin-bottom:8px;
    border:0;
    border-top:1px solid #dcdcdc;
}

div.stButton > button:hover {
    background-color:#4a7b69;
    color:white;
}

</style>
""", unsafe_allow_html=True)


# ---------------- Class Labels ----------------
class_names = [
    "Non Demented",
    "Very Mild Demented",
    "Mild Demented",
    "Moderate Demented"
]


# ---------------- Load Model ----------------
model = load_model()

model.load_state_dict(
    torch.load("dementia_resnet18_model.pth", map_location=torch.device("cpu"))
)

model.eval()


# ---------------- Session State ----------------
if "image_uploaded" not in st.session_state:
    st.session_state.image_uploaded = False


# ---------------- Header ----------------
st.markdown(
"""
<h1 style='text-align:center;'>MemorAid: Dementia Detection System</h1>
<p style='text-align:center;'>AI Powered MRI Brain Analysis</p>
""",
unsafe_allow_html=True
)

st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)


# ---------------- Upload Screen ----------------
if not st.session_state.image_uploaded:

    uploaded_file = st.file_uploader(
        "Upload MRI Brain Image",
        type=["jpg","png","jpeg"]
    )

    if uploaded_file:

        st.session_state.image = uploaded_file
        st.session_state.image_uploaded = True
        st.rerun()


# ---------------- Result Screen ----------------
else:

    img = Image.open(st.session_state.image).convert("RGB")

    col1, col2 = st.columns([1,1])

    with col1:

        st.subheader("MRI Brain Scan")

        st.image(img, width=240)


    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():

        output = model(img_tensor)

        probs = F.softmax(output, dim=1)

    pred = torch.argmax(probs).item()


    with col2:

        st.subheader("Diagnosis Result")

        st.success(f"{class_names[pred]}")

        st.subheader("Prediction Probabilities")

        for i, p in enumerate(probs[0]):

            st.write(f"{class_names[i]} : {p.item()*100:.2f}%")

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
    

    # Center Button
    c1, c2, c3 = st.columns([1,2,1])

    with c2:

        if st.button("Analyze Another Image"):

            st.session_state.image_uploaded = False
            st.session_state.image = None
            st.rerun()