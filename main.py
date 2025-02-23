import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

def apply_filters(image, enabled_filters):
    intermediates = []
    current_image = image.copy()
    
    for filter_name, params in enabled_filters:
        if filter_name == 'grayscale':
            gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
            current_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            intermediates.append(('Grayscale', current_image.copy()))
        
        elif filter_name == 'gaussian_blur':
            ksize = params['ksize']
            sigma = params['sigma']
            current_image = cv2.GaussianBlur(current_image, (ksize, ksize), sigma)
            intermediates.append(('Gaussian Blur', current_image.copy()))
        
        elif filter_name == 'canny':
            gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, params['low'], params['high'])
            current_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            intermediates.append(('Canny Edge', current_image.copy()))
        
        elif filter_name == 'threshold':
            gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
            _, thresh_img = cv2.threshold(gray, params['threshold'], 255, cv2.THRESH_BINARY)
            current_image = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2BGR)
            intermediates.append(('Threshold', current_image.copy()))
        
        elif filter_name == 'sepia':
            sepia_filter = np.array([
                [0.272, 0.534, 0.131],
                [0.349, 0.686, 0.168],
                [0.393, 0.769, 0.189]
            ])
            current_image = cv2.transform(current_image, sepia_filter)
            current_image = np.clip(current_image, 0, 255).astype(np.uint8)
            intermediates.append(('Sepia', current_image.copy()))
        
        elif filter_name == 'sharpen':
            kernel = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
            current_image = cv2.filter2D(current_image, -1, kernel)
            intermediates.append(('Sharpen', current_image.copy()))
        
        elif filter_name == 'color_change':
            hsv = cv2.cvtColor(current_image, cv2.COLOR_BGR2HSV)
            hsv[:, :, 0] = (hsv[:, :, 0] + params['hue_shift']) % 180  # Hue shift
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * params['saturation'], 0, 255)  # Saturation
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * params['brightness'], 0, 255)  # Brightness
            current_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            intermediates.append(('Color Change', current_image.copy()))
        
        elif filter_name == 'pixelate':
            (h, w) = current_image.shape[:2]
            size = params['size']
            current_image = cv2.resize(current_image, (size, size), interpolation=cv2.INTER_LINEAR)
            current_image = cv2.resize(current_image, (w, h), interpolation=cv2.INTER_NEAREST)
            intermediates.append(('Pixelate', current_image.copy()))
        
        elif filter_name == 'rotate':
            (h, w) = current_image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, params['angle'], 1.0)
            current_image = cv2.warpAffine(current_image, M, (w, h))
            intermediates.append(('Rotate', current_image.copy()))
        
        elif filter_name == 'flip':
            flip_code = params['flip_code']
            current_image = cv2.flip(current_image, flip_code)
            intermediates.append(('Flip', current_image.copy()))
    
    return current_image, intermediates

def main():
    st.title("Advanced Image Processing Playground")
    st.markdown("Explore different image processing filters and transformations!")
    
    # Sidebar controls
    st.sidebar.header("Filter Controls")
    enabled_filters = []
    
    # Basic Filters
    with st.sidebar.expander("Basic Filters"):
        if st.checkbox("Grayscale"):
            enabled_filters.append(('grayscale', {}))
        
        if st.checkbox("Gaussian Blur"):
            ksize = st.slider("Kernel Size", 1, 15, 3, 2, key='blur_ksize')
            sigma = st.slider("Sigma", 0.0, 10.0, 1.0, key='blur_sigma')
            enabled_filters.append(('gaussian_blur', {'ksize': ksize, 'sigma': sigma}))
        
        if st.checkbox("Thresholding"):
            thresh = st.slider("Threshold Value", 0, 255, 127, key='threshold')
            enabled_filters.append(('threshold', {'threshold': thresh}))
    
    # Edge Detection
    with st.sidebar.expander("Edge Detection"):
        if st.checkbox("Canny Edge"):
            low = st.slider("Low Threshold", 0, 255, 50, key='canny_low')
            high = st.slider("High Threshold", 0, 255, 150, key='canny_high')
            enabled_filters.append(('canny', {'low': low, 'high': high}))
    
    # Special Effects
    with st.sidebar.expander("Special Effects"):
        if st.checkbox("Sepia Tone"):
            enabled_filters.append(('sepia', {}))
        
        if st.checkbox("Sharpen"):
            enabled_filters.append(('sharpen', {}))
    
    # Color Manipulation
    with st.sidebar.expander("Color Manipulation"):
        if st.checkbox("Change Colors"):
            hue_shift = st.slider("Hue Shift", -180, 180, 0, key='hue_shift')
            saturation = st.slider("Saturation", 0.0, 2.0, 1.0, key='saturation')
            brightness = st.slider("Brightness", 0.0, 2.0, 1.0, key='brightness')
            enabled_filters.append(('color_change', {
                'hue_shift': hue_shift,
                'saturation': saturation,
                'brightness': brightness
            }))
    
    # Transformations
    with st.sidebar.expander("Transformations"):
        if st.checkbox("Pixelate"):
            size = st.slider("Pixelation Size", 10, 200, 50, key='pixelate_size')
            enabled_filters.append(('pixelate', {'size': size}))
        
        if st.checkbox("Rotate"):
            angle = st.slider("Rotation Angle", -180, 180, 0, key='rotate_angle')
            enabled_filters.append(('rotate', {'angle': angle}))
        
        if st.checkbox("Flip"):
            flip_code = st.radio("Flip Direction", ["Horizontal", "Vertical"], key='flip_direction')
            flip_code = 1 if flip_code == "Horizontal" else 0
            enabled_filters.append(('flip', {'flip_code': flip_code}))
    
    # Image input
    img_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    use_sample = st.checkbox("Use Sample Image")
    
    if img_file or use_sample:
        if use_sample:
            img_path = os.path.join("images", "sample.jpg")  # Add sample image path
            image_pil = Image.open(img_path)
        else:
            image_pil = Image.open(img_file)
        
        st.subheader("Original Image")
        st.image(image_pil, use_column_width=True)
        
        # Convert to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        
        if st.button("Apply Filters"):
            final_image, intermediates = apply_filters(opencv_image, enabled_filters)
            
            # Display results
            st.subheader("Final Result")
            final_image_rgb = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
            st.image(final_image_rgb, use_column_width=True)
            
            if intermediates:
                st.subheader("Processing Steps")
                cols = st.columns(len(intermediates))
                for col, (name, img) in zip(cols, intermediates):
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    col.image(img_rgb, caption=name, use_column_width=True)
    else:
        st.info("Please upload an image or use the sample image to begin.")

if __name__ == "__main__":
    main()