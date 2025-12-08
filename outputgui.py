import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
import pandas as pd
import numpy as np
from PIL import Image
import io
import datetime

# --- 1. PAGE CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="TrafficSense AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- THEME-ADAPTIVE CSS STYLING ---
st.markdown("""
    <style>
    /* 1. Center all Headings */
    h1, h2, h3, h4, h5, h6 {
        text-align: center; 
    }

    /* 2. Remove Top Padding (Compact Look) */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* 3. Metric Cards (Theme Adaptive) */
    div[data-testid="metric-container"] {
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    /* 4. File Uploader (Theme Adaptive) */
    div[data-testid="stFileUploader"] section {
        width: 100%;
        padding: 20px;
        border: 1px dashed rgba(128, 128, 128, 0.5);
    }
    
    /* 5. Footer Styling */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        text-align: center;
        padding: 10px;
        font-size: 12px;
        border-top: 1px solid rgba(128, 128, 128, 0.2);
        z-index: 100;
    }
    
    /* Hide the default Streamlit footer */
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- 2. ROBUST MODEL LOADING ---
@st.cache_resource
def load_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "models", "best.pt")
    return YOLO(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# --- 3. SIDEBAR CONTROLS ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063823.png", width=80) 
    st.title("Control Panel")
    
    st.write("### ⚙️ Detection Settings")
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05, help="Filter out weak detections.")
    iou_threshold = st.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.05, help="Overlap threshold.")
    
    st.write("### 👁️ Visuals")
    show_labels = st.toggle("Show Labels", value=True)
    show_conf = st.toggle("Show Confidence", value=True)
    
    st.divider()
    st.caption("MY-VID Project | Traffic Analytics")

# --- 4. MAIN INTERFACE ---

# Hero Section
st.markdown("<h1>🚦 TrafficSense AI</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='color: gray;'>Next-Gen Vehicle Detection & Analytics</h5>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 0.9rem; color: gray;'>Powered by YOLO11 & MY-VID JKR Dataset</p>", unsafe_allow_html=True)

# --- UPDATED: JKR CLASS LEGEND (3-COLUMN TABLE) ---
with st.expander("ℹ️ Guide: JKR Vehicle Class Classifications"):
    st.markdown("""
    The model detects vehicles based on the Malaysian Public Works Department (JKR) classification system:
    
    | **Class ID** | **Vehicle Type** | **Description** |
    | :--- | :--- | :--- |
    | **Class 1** | Cars & Taxis | Private passenger vehicles primarily designed for personal or small-group transport, typically with 4-5 seats. |
    | **Class 2** | Vans & Utilities | Medium-sized vehicles for light commercial or passenger transport, often used for logistics or service operations. |
    | **Class 3** | Medium Lorries | Medium-duty trucks used for goods transport within urban and suburban areas. |
    | **Class 4** | Heavy Lorries | Large commercial vehicles designed for bulk cargo transport over long distances. |
    | **Class 5** | Buses | Passenger-carrying vehicles with multiple rows of seating including public transport and private tour buses. |
    | **Class 6** | Motorcycles | Two-wheeled motorized vehicles widely used for personal and delivery purposes in Malaysia. |
    """)

st.divider()

# File Uploader
uploaded_file = st.file_uploader(
    "📂 Upload a traffic image (Drag & Drop supported)", 
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    # Processing Indicator
    file_extension = os.path.splitext(uploaded_file.name)[1]
    
    # Save temp file safely
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    try:
        # --- INFERENCE ---
        with st.spinner('🤖 AI is scanning the road...'):
            results = model.predict(
                file_path, 
                conf=conf_threshold, 
                iou=iou_threshold,
                verbose=False
            )
            result = results[0]

            # Plotting
            plotted_bgr = result.plot(
                line_width=2,
                font_size=15,
                labels=show_labels,
                conf=show_conf
            )
            plotted_rgb = cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB)
            
            # --- DATA EXTRACTION ---
            counts = {}
            conf_scores = {}
            for box in result.boxes:
                label = model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                counts[label] = counts.get(label, 0) + 1
                conf_scores.setdefault(label, []).append(conf)

        # --- 5. RESULTS DASHBOARD ---
        
        # Calculate KPI Metrics
        total_vehicles = sum(counts.values())
        most_common = max(counts, key=counts.get) if counts else "None"
        avg_conf_global = np.mean([c for sublist in conf_scores.values() for c in sublist]) if conf_scores else 0.0

        # KPI Row
        st.markdown("### 📊 Live Analytics")
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Total Vehicles", total_vehicles, delta="Detected")
        kpi2.metric("Dominant Class", most_common)
        kpi3.metric("Avg. Confidence", f"{avg_conf_global:.1%}", delta_color="normal")

        st.divider()

        # Image Comparison Row
        col_img1, col_img2 = st.columns(2)
        
        with col_img1:
            st.info("Original Feed")
            original_img = Image.open(file_path)
            st.image(original_img, use_container_width=True)

        with col_img2:
            st.success("AI Detection Output")
            st.image(plotted_rgb, use_container_width=True)
            
            # Download Button
            result_pil = Image.fromarray(plotted_rgb)
            buf = io.BytesIO()
            result_pil.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            st.download_button(
                label="⬇️ Download Result Image",
                data=byte_im,
                file_name="detected_traffic.png",
                mime="image/png",
                use_container_width=True
            )

        # Detailed Table
        if total_vehicles > 0:
            st.subheader("📋 Class Breakdown")
            table_data = []
            for cls in counts:
                avg_conf = sum(conf_scores[cls]) / len(conf_scores[cls])
                table_data.append([cls, counts[cls], f"{avg_conf:.1%}"])

            df = pd.DataFrame(table_data, columns=["Vehicle Class", "Count", "Confidence"])
            
            # Full width table with styling
            st.dataframe(
                df.style.background_gradient(cmap="Reds", subset=["Count"]),
                use_container_width=True
            )
        else:
            st.warning("⚠️ No vehicles detected. Try lowering the confidence threshold in the sidebar.")

    finally:
        # Cleanup temp file
        if os.path.exists(file_path):
            os.unlink(file_path)

else:
    st.info("👆 Upload an image above to see the AI in action.")

# --- COPYRIGHT FOOTER ---
current_year = datetime.datetime.now().year
st.markdown(f"""
    <div class='footer'>
        <p>© {current_year} TrafficSense AI | Developed by Lim Zi Xuan in collaboration with JKR Malaysia Data Standards.</p>
    </div>
    """, unsafe_allow_html=True)