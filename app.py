import os
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
from PIL import Image
from matplotlib.cm import get_cmap

# Page configuration
st.set_page_config(
    page_title="OSTEO VISION",
    page_icon="🦵",
    layout="wide"
)

# Helper functions
def make_gradcam_heatmap(grad_model, img_array, pred_index=None):
    """
    Generate Grad-CAM heatmap for the given image and model.
    """
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap_on_image(img, heatmap, alpha=0.4):
    """
    Overlay Grad-CAM heatmap on the input image.
    """
    heatmap = np.uint8(255 * heatmap)
    jet = get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * alpha + img
    return tf.keras.preprocessing.image.array_to_img(superimposed_img)

# Load the trained model
MODEL_PATH = os.getenv("MODEL_PATH", r"C:\Users\mithu\Desktop\osteo_vision\src\models\OSTEO_VISION_MODEL_6_ Knee Osteoarthritis Detection with Fine-Tuned ResNet152V2 on dataset 3 MODEL_6.h5")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    st.sidebar.error("Error loading model.")
    st.stop()

# Configure Grad-CAM model
grad_model = tf.keras.models.Model(
    inputs=[model.input],
    outputs=[model.get_layer("global_average_pooling2d").input, model.output]
)

# Sidebar with images
st.sidebar.image(r"C:\Users\mithu\Desktop\osteo_vision\app\img\jssate .png", caption="JSSATE-B", width=220)
st.sidebar.image(r"C:\Users\mithu\Desktop\osteo_vision\app\img\jssaher.jpg", caption="JSSAHER", width=220)

# Sidebar for user inputs
uploaded_file = st.sidebar.file_uploader("Upload an X-ray image", type=["png", "jpg", "jpeg"])

# Main section layout
st.title("OSTEO VISION")

target_size = (224, 224)
class_names = ["KL-GRADE 0", "KL-GRADE 1", "KL-GRADE 2", "KL-GRADE 3", "KL-GRADE 4"]

if uploaded_file:
    # Preprocess the image
    img = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.xception.preprocess_input(img_array)

    # Make predictions
    with st.spinner("Analyzing the image..."):
        predictions = model.predict(img_array)[0]
        predicted_class = class_names[np.argmax(predictions)]
        prediction_probabilities = 100 * predictions

    # Display prediction result
    st.subheader("Prediction Result ✅")
    st.metric(label="Predicted Severity Level", value=predicted_class, delta=f"Confidence: {np.max(prediction_probabilities):.2f}%")

    # Grad-CAM heatmap
    heatmap = make_gradcam_heatmap(grad_model, img_array)
    heatmap_overlay = overlay_heatmap_on_image(
        tf.keras.preprocessing.image.img_to_array(img),
        heatmap
    )

    # Display side-by-side layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📸 Input Image")
        st.image(uploaded_file, caption="Uploaded X-ray Image", use_column_width=True)

    with col2:
        st.subheader("📊 Explainability with Grad-CAM")
        st.image(heatmap_overlay, caption="Grad-CAM Heatmap", use_column_width=True)

    # Bar chart visualization of predictions
    st.subheader("📈 Prediction Confidence Levels")
    fig, ax = plt.subplots(figsize=(5, 2))  # Reduced chart size
    ax.barh(class_names, prediction_probabilities, color='skyblue')
    ax.set_xlim([0, 100])
    ax.set_xlabel("Confidence (%)")
    ax.set_title("Prediction Confidence Levels")
    st.pyplot(fig)

else:
    st.info("Please upload an X-ray image to begin analysis.")


