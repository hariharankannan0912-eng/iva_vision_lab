from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import cv2
import os, uuid, io, base64

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024
UPLOAD_DIR = os.path.join("static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
ALLOWED = {"png", "jpg", "jpeg", "webp"}

def read_image(file_storage):
    data = file_storage.read()
    file_storage.stream.seek(0)
    pil = Image.open(io.BytesIO(data)).convert("RGB")
    arr = np.array(pil)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def data_url(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

def check_upload(f):
    return f and "." in f.filename and f.filename.rsplit(".", 1)[1].lower() in ALLOWED

@app.route("/")
def index():
    return render_template("index.html")

@app.post("/api/template-match")
def template_match():
    image_file = request.files.get("image")
    template_file = request.files.get("template")
    if not check_upload(image_file) or not check_upload(template_file):
        return jsonify(error="Upload both a scene image and a template image (PNG, JPG, JPEG, or WEBP)."), 400
    scene = read_image(image_file)
    template = read_image(template_file)
    sh, sw = scene.shape[:2]
    th, tw = template.shape[:2]
    if th > sh or tw > sw:
        return jsonify(error="The template must be smaller than (or equal to) the scene image. Crop a smaller object from the scene and upload it as the template."), 400
    gray_scene = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(gray_scene, gray_template, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(result)
    x, y = loc
    cv2.rectangle(scene, (x, y), (x + tw, y + th), (60, 220, 120), 3)
    cv2.putText(scene, f"Match {score:.2%}", (x, max(24, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, .75, (60, 220, 120), 2, cv2.LINE_AA)
    return jsonify(image=data_url(scene), score=round(float(score), 4), location={"x": x, "y": y}, template_size={"width": tw, "height": th},
                   message="Higher score means a closer visual match. Template matching works best when scale, rotation, and lighting are similar.")

@app.post("/api/viola-jones")
def viola_jones():
    f = request.files.get("image")
    if not check_upload(f):
        return jsonify(error="Upload an image (PNG, JPG, JPEG, or WEBP)."), 400
    img = read_image(f)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        return jsonify(error="OpenCV's Haar cascade could not be loaded. Reinstall opencv-python."), 500
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for i, (x, y, w, h) in enumerate(faces, 1):
        cv2.rectangle(img, (x, y), (x+w, y+h), (40, 190, 255), 3)
        cv2.putText(img, f"Face {i}", (x, max(22, y-8)), cv2.FONT_HERSHEY_SIMPLEX, .7, (40, 190, 255), 2, cv2.LINE_AA)
    return jsonify(image=data_url(img), count=int(len(faces)), message="Uses OpenCV's Haar-cascade frontal-face detector, a practical implementation associated with the Viola–Jones approach.")

@app.post("/api/deepface")
def deepface():
    a, b = request.files.get("image1"), request.files.get("image2")
    if not check_upload(a) or not check_upload(b):
        return jsonify(error="Upload two face images (PNG, JPG, JPEG, or WEBP)."), 400
    try:
        # DeepFace is optional because its model weights are large and may need a first-run download.
        from deepface import DeepFace
        p1, p2 = save_temp(a), save_temp(b)
        try:
            result = DeepFace.verify(img1_path=p1, img2_path=p2, model_name="Facenet512", detector_backend="opencv", enforce_detection=True)
            return jsonify(verified=bool(result.get("verified")), distance=round(float(result.get("distance", 0)), 4),
                           threshold=result.get("threshold"), model=result.get("model", "Facenet512"),
                           message="DeepFace verification completed. A match is a similarity estimate, not proof of identity.")
        finally:
            for p in (p1, p2):
                try: os.remove(p)
                except OSError: pass
    except ImportError:
        return jsonify(error="DeepFace is not installed. Run: pip install deepface. Then restart the server. First use may download model weights."), 503
    except Exception as e:
        return jsonify(error="DeepFace could not complete verification: " + str(e)[:240]), 422

def save_temp(file_storage):
    ext = file_storage.filename.rsplit(".", 1)[1].lower()
    path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}.{ext}")
    file_storage.save(path)
    return path

@app.post("/api/facenet")
def facenet():
    a, b = request.files.get("image1"), request.files.get("image2")
    if not check_upload(a) or not check_upload(b):
        return jsonify(error="Upload two face images (PNG, JPG, JPEG, or WEBP)."), 400
    try:
        from keras_facenet import FaceNet
        from mtcnn import MTCNN
        embedder = FaceNet()
        detector = MTCNN()
        vectors = []
        for f in (a, b):
            rgb = cv2.cvtColor(read_image(f), cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(rgb)
            if not faces:
                return jsonify(error="No face detected in one of the images. Try a clear, front-facing portrait."), 422
            x, y, w, h = faces[0]["box"]
            x, y = max(0, x), max(0, y)
            crop = rgb[y:y+max(1,h), x:x+max(1,w)]
            if crop.size == 0:
                return jsonify(error="Could not crop a face. Try another image."), 422
            crop = cv2.resize(crop, (160, 160))
            vectors.append(embedder.embeddings([crop])[0])
        cosine = float(np.dot(vectors[0], vectors[1]) / (np.linalg.norm(vectors[0]) * np.linalg.norm(vectors[1]) + 1e-10))
        return jsonify(similarity=round(cosine, 4), message="Cosine similarity of FaceNet embeddings. Higher generally means more similar; there is no universal threshold for every dataset.")
    except ImportError:
        return jsonify(error="FaceNet dependencies are optional. Run: pip install keras-facenet mtcnn tensorflow. Then restart the server. This may require a compatible Python/TensorFlow setup."), 503
    except Exception as e:
        return jsonify(error="FaceNet could not complete comparison: " + str(e)[:240]), 422

@app.errorhandler(413)
def too_large(_):
    return jsonify(error="Image too large. Maximum upload size is 12 MB per request."), 413

if __name__ == "__main__":
    app.run(debug=True)
