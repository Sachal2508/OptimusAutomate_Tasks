# Optimus AI Internship Tasks

This repository contains four demo applications and helper scripts illustrating common ML tasks and simple UIs to run them locally.

Summary:
- Task 1 — Image classification (CNN example using CIFAR-10).
- Task 2 — Chatbot / Intent recognition (TF-IDF + MultinomialNB with Streamlit UI).
- Task 3 — Object detection (OpenCV MobileNet SSD + YOLOv8 fallback, Streamlit UI).
- Task 4 — Movie recommender (MovieLens small dataset; content-based + collaborative examples, Streamlit UI and CLI).

Repository structure

- `data/` — sample datasets (CIFAR-10, MovieLens small). Large raw archives should be kept outside Git or in Git LFS.
- `src/` — Streamlit apps and helper scripts:
  - `src/task1_cnn.py` — CNN training/eval demo for CIFAR-10
  - `src/task2_chatbot.py` — Streamlit chatbot UI
  - `src/object_detection_app.py` — main object detection Streamlit app (supports MobileNet SSD and YOLOv8)
  - `src/task4_recommendation_app.py` — Streamlit recommender UI
  - `src/task4_recommender.py` — CLI demo utilities for MovieLens
  - `src/streamlit_app.py` — simple launcher to pick a demo
  - other helpers and model files (e.g. `src/MobileNetSSD_deploy.caffemodel`, `yolov8n.pt`)

Requirements & setup

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# or cmd
.\.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

2. If `requirements.txt` references `sklearn`, install `scikit-learn` instead:

```bash
python -m pip install scikit-learn
```

Running the Streamlit apps

General launcher (choose demo):

```bash
python -m streamlit run src/streamlit_app.py
```

Individual demos:

- Chatbot (Task 2):
  ```bash
  python -m streamlit run src/task2_chatbot.py
  ```
- Object detection (Task 3):
  ```bash
  python -m streamlit run src/object_detection_app.py
  ```
  Notes: Use the sidebar to pick the backend (OpenCV MobileNet SSD or YOLOv8). Place YOLO weights (`yolov8n.pt`) at the repo root or update the path in the app. MobileNet model files are in `src/` (`deploy.prototxt` + `MobileNetSSD_deploy.caffemodel`).
- Recommender (Task 4):
  ```bash
  python -m streamlit run src/task4_recommendation_app.py
  ```
  Or run the CLI helper:
  ```bash
  python src/task4_recommender.py --download True
  ```

Notes about data, models, and large files

- Large datasets and model weights (for example `data/cifar-10-python.tar.gz` and some pre-trained models) may exceed GitHub's file-size limits. Avoid committing large archives directly.
- Recommended approaches:
  - Use Git LFS to store large files (`git lfs install`, `git lfs track "data/*.tar.gz"`).
  - Keep raw datasets and heavy pretrained weights in external storage (Google Drive, S3) and add download helpers to the app.
  - If a large file has already been committed, remove it from history (or use BFG/git filter-repo) before pushing to GitHub.

Quick commands to remove a single large file from history (rewrites history):

```bash
# remove the file from all commits, then garbage-collect and force-push
git filter-branch --force --index-filter "git rm --cached --ignore-unmatch data/cifar-10-python.tar.gz" --prune-empty --tag-name-filter cat -- --all
git for-each-ref --format="%(refname)" refs/original/ | xargs -n 1 git update-ref -d
git reflog expire --expire=now --all
git gc --prune=now --aggressive
git push origin main --force
```

If you prefer Git LFS migration, install `git-lfs` and use BFG or `git lfs migrate import` to migrate existing blobs.

Testing

- Start any Streamlit app and open the local URL printed in the terminal (usually `http://localhost:8501`).
- For object detection, try different images and set the confidence threshold to 0.10-0.25 for initial testing; enable tiled detection for small-object scenarios.

Contributing

- Please avoid committing large binary datasets and trained model weights directly. Open an issue or add a download script if you want a file included as part of the demo.

License

This repository is provided as-is for demonstration and educational purposes. Add a license file if you want to publish under a specific license.

Contact

For questions or help running the demos, open an issue or contact the maintainer.
