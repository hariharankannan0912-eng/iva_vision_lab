# VisionLab — IVA Computer Vision Website

A responsive Python Flask website for an Image and Video Analytics (IVA) assignment. It demonstrates:

1. **Template Matching** — finds a smaller reference image inside a larger scene using OpenCV normalized correlation.
2. **Viola–Jones** — detects frontal faces with OpenCV's built-in Haar cascade.
3. **DeepFace** — optional real face verification using the DeepFace package with the Facenet512 model.
4. **FaceNet** — optional face embeddings and cosine similarity using `keras-facenet` and MTCNN.

## Requirements

- Python 3.10 or 3.11 recommended
- Internet connection for installing packages and first-time optional model downloads
- A modern browser

## Run locally

### Windows (PowerShell or Command Prompt)

```bash
cd iva_vision_lab
python -m venv .venv
.venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

### macOS / Linux

```bash
cd iva_vision_lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

## Enable DeepFace and FaceNet

These dependencies are intentionally optional because TensorFlow and model weights can be large and compatibility varies by operating system and Python version.

```bash
pip install -r requirements-optional.txt
```

Restart the server. The first run can download model weights and may take several minutes. If installation fails, use Python 3.10/3.11 in a fresh virtual environment and check the TensorFlow installation instructions for your OS.

The website shows a helpful error if an optional library is missing; Template Matching and Viola–Jones continue to work without it.

## How to use

- **Template Matching:** upload a larger scene image and a smaller cropped template of the object to find.
- **Viola–Jones:** upload a photo with one or more clear, front-facing faces.
- **DeepFace:** upload two portrait images to run verification.
- **FaceNet:** upload two portrait images to compare embeddings using cosine similarity.

Use images you have permission to process. Face similarity scores are probabilistic and should not be treated as proof of identity.

## Deployment

For a Python backend, use a Python-capable host such as Render or Railway (not a static-only hosting target). Set the build command to `pip install -r requirements.txt` and the start command to `gunicorn app:app` (add `gunicorn` to requirements first). Optional deep-learning dependencies can make deployments slow and memory-heavy; deploy the base features first.

For a production deployment, turn off Flask debug mode, add authentication/rate limits as appropriate, and configure upload cleanup and privacy controls.
