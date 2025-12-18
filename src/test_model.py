import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import _KerasLazyLoader as keras

# EfficientNet preprocessing (must match training)
from keras.applications.efficientnet import preprocess_input

# ==============================
# CONFIG
# ==============================
MODEL_PATH = "..." # your model path
IMAGE_PATH = "..." # your image path
IMG_SIZE = (224, 224)

# Default (must match training order)
CLASS_NAMES = ["Reject", "Ripe", "Unripe"]

# ==============================
# LOAD MODEL
# ==============================
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print(f"[INFO] Loaded model from: {MODEL_PATH}")

# ==============================
# FUNCTIONS
# ==============================
def load_and_prepare_image(img_path: str) -> tf.Tensor:
    """
    Loads image and applies the SAME preprocessing used in training.
    For EfficientNet + preprocess_input.
    Returns: (1, 224, 224, 3) float32
    """
    img_bytes = tf.io.read_file(img_path)

    # decode_image supports jpg/png; expand_animations=False avoids GIF shape weirdness
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)

    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)

    # IMPORTANT: EfficientNet preprocessing (NOT /255)
    img = preprocess_input(img)

    img = tf.expand_dims(img, axis=0)
    return img


def classify_image(img_path: str):
    img_tensor = load_and_prepare_image(img_path)

    preds = model.predict(img_tensor, verbose=0)[0]  # shape (num_classes,)
    pred_idx = int(np.argmax(preds))
    confidence = float(preds[pred_idx])
    predicted_class = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else str(pred_idx)

    print("===================================")
    print(f"Image: {img_path}")
    print(f"Predicted Class : {predicted_class}")
    print(f"Confidence      : {confidence*100:.2f}%")
    print("===================================\n")

    for i, c in enumerate(CLASS_NAMES):
        val = float(preds[i]) if i < len(preds) else 0.0
        print(f"{c}: {val*100:.2f}%")

    # Display image (for visualization only)
    # Use tf decode to ensure consistent reading
    raw = tf.image.decode_image(tf.io.read_file(img_path), channels=3, expand_animations=False).numpy()
    plt.imshow(raw)
    plt.title(f"{predicted_class} ({confidence*100:.2f}%)")
    plt.axis("off")
    plt.show()


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    if not os.path.exists(IMAGE_PATH):
        print(f"[ERROR] File not found -> {IMAGE_PATH}")
        raise SystemExit(1)

    classify_image(IMAGE_PATH)
