from flask import Flask, render_template, request, redirect, url_for
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils.saliency import generate_saliency_map
from utils.morph_ops import apply_morph_ops
from utils.metrics_loader import load_metrics_classification, load_metrics_segmentation
from werkzeug.utils import secure_filename
import tensorflow.keras.backend as K

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploaded/'
app.config['RESULT_FOLDER'] = 'static/results/'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

classification_model = load_model(r'N:\NewFYP\forgery_app\models\final_model.h5')

segmentation_model = load_model(
    r'N:\NewFYP\forgery_app\models\unet_forgery_segmentation.h5',
    custom_objects={'dice_coef': dice_coef},
    compile=False
)

def preprocess_image(img_path, target_size=(224, 224)):
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('image')
    if not file or file.filename == '':
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    base_filename = os.path.splitext(filename)[0]
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(upload_path)

    img_array = preprocess_image(upload_path)

    # === Classification Prediction ===
    class_pred = classification_model.predict(img_array)
    class_label_idx = np.argmax(class_pred, axis=1)[0]
    class_confidence = float(np.max(class_pred))

   # === Segmentation Prediction ===
    seg_mask = segmentation_model.predict(img_array)[0]
    seg_mask = (seg_mask > 0.7).astype(np.uint8).squeeze()

    orig_img = cv2.imread(upload_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    mask_resized = cv2.resize(seg_mask, (orig_img.shape[1], orig_img.shape[0]))

    morphed_mask = apply_morph_ops(mask_resized)

    # Check if forgery area is significant
    min_area = 0.01 * mask_resized.size  # 1% of image pixels
    seg_has_forgery = np.sum(morphed_mask) > min_area

    # === Decision ===
    label = 'Authentic' if (class_label_idx == 0 and not seg_has_forgery) else 'Tampered'
    
    saliency_map = generate_saliency_map(orig_img)

    # === Red Mask Overlay ===
    red_overlay = orig_img.copy()
    red_overlay[morphed_mask > 0] = [255, 0, 0]

    # Save results
    def save_result(img, suffix):
        path = os.path.join(app.config['RESULT_FOLDER'], f'{base_filename}_{suffix}.png')
        plt.imsave(path, img)
        return f'results/{base_filename}_{suffix}.png'

    result_paths = {
        'red_overlay': save_result(red_overlay, 'redmask'),
        'saliency': save_result(saliency_map, 'saliency'),
        'morph': save_result(morphed_mask * 255, 'morph'),
        'mask': save_result(mask_resized * 255, 'mask')
    }

    return render_template(
        'result.html',
        label=label,
        confidence=round(class_confidence * 100, 2),
        original_image=f'uploaded/{filename}',
        result_images=result_paths
    )

@app.route("/metrics")
def metrics():
    classification_report, training_metrics = load_metrics_classification()
    seg_metrics, _ = load_metrics_segmentation()

    class_metrics = {
        label: values for label, values in classification_report.items()
        if label not in ["accuracy", "macro avg", "weighted avg"]
    }

    return render_template(
        "metrics.html",
        classification_report=classification_report,
        class_metrics=class_metrics,
        seg_metrics=seg_metrics,
        overall_metrics=classification_report  # ✅ Add this line
    )

@app.route('/model_info')
def model_info():
    return render_template('model_info.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
