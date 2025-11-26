import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
import pandas as pd
import numpy as np

# Load YOLO model
model = YOLO("models/best.pt") 

st.title("JKR Vehicle Detection & Counting")

# --- THE RESTRICTION IS HERE ---
# By passing only image extensions to 'type', videos are blocked.
uploaded_file = st.file_uploader(
    "Upload a traffic image", 
    type=["jpg", "png", "jpeg"] 
)

if uploaded_file is not None:
    # Create a temporary file to save the uploaded content
    file_extension = os.path.splitext(uploaded_file.name)[1]
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    # Run YOLO prediction
    results = model.predict(file_path, verbose=False)
    result = results[0] 

    # --- VISUALIZATION SETTINGS ---
    plotted_bgr = result.plot(
        line_width=2,      
        font_size=20,      
        labels=True,
        conf=True
    )

    # --- COLOR CORRECTION (BGR to RGB) ---
    plotted_rgb = cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB)

    st.image(plotted_rgb, caption="Detection Result", use_container_width=True)

    # Count JKR classes + compute confidence
    counts = {}
    conf_scores = {}

    for box in result.boxes:
        class_id = int(box.cls[0])
        label = model.names[class_id]
        conf = float(box.conf[0])

        counts[label] = counts.get(label, 0) + 1
        conf_scores.setdefault(label, []).append(conf)

    # Build data table
    table_data = []
    for cls in counts:
        avg_conf = sum(conf_scores[cls]) / len(conf_scores[cls])
        table_data.append([cls, counts[cls], round(avg_conf * 100, 2)])

    df = pd.DataFrame(table_data, columns=["JKR Class", "Count", "Avg Confidence (%)"])

    st.write("### 📊 Detection Summary Table")
    st.dataframe(df, use_container_width=True)
    
    # Clean up temp file
    os.unlink(file_path)