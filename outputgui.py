import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
import pandas as pd

# Load YOLO model
model = YOLO("models/best.pt")
st.write("Model class names:", model.names)
print(model.names)
st.title("JKR Vehicle Detection & Counting")

uploaded_file = st.file_uploader("Upload a traffic image or video", type=["jpg","png","mp4"])

if uploaded_file is not None:
    file_extension = os.path.splitext(uploaded_file.name)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    # Run YOLO prediction
    results = model.predict(file_path, verbose=False)

    # Customized visualization
    plotted = results[0].plot(
        line_width=1,
        font_size=1000,
        labels=True,
        conf=True
    )

    st.image(plotted, caption="Detection Result (Smaller Labels)")

    # Count JKR classes + compute confidence
    counts = {}
    conf_scores = {}

    for box in results[0].boxes:
        label = model.names[int(box.cls)]       # Already "Class 1", "Class 2", etc.
        conf = float(box.conf)

        # Count
        counts[label] = counts.get(label, 0) + 1

        # Store confidence
        conf_scores.setdefault(label, []).append(conf)

    # Build data table
    table_data = []
    for cls in counts:
        avg_conf = sum(conf_scores[cls]) / len(conf_scores[cls])
        table_data.append([cls, counts[cls], round(avg_conf * 100, 2)])

    df = pd.DataFrame(table_data, columns=["JKR Class", "Count", "Avg Confidence (%)"])

    st.write("### 📊 Detection Summary Table")
    st.dataframe(df, use_container_width=True)
