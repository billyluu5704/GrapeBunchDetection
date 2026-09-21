# Grape Bunch Detection

Detect and count grape bunches (single class: `bunch`) in real vineyard photos. The repo holds two
independent pipelines that are trained on the same data and compared against each other:

| Pipeline | Notebook | Approach |
|---|---|---|
| **Deep learning** | [Deep Learning/Deep_Learning_GrapeBunchDetection.ipynb](Deep%20Learning/Deep_Learning_GrapeBunchDetection.ipynb) | YOLO11 (Ultralytics) with a domain-gap-aware training mix, SAHI tiled inference, an in-notebook manual annotation tool for a live-photo holdout, and real-farm fine-tuning |
| **Traditional ML** | [Machine Learning/Traditional_Image_Analysis_of_GrapeBunchDetection.ipynb](Machine%20Learning/Traditional_Image_Analysis_of_GrapeBunchDetection.ipynb) | HSV colour segmentation for region proposals, handcrafted features (CLAHE, Gabor, LBP, HOG, Canny, blob shape, Haralick), SVM / XGBoost / Random Forest / ensemble classifiers, NMS, SHAP explainability |

Both notebooks end with a real-farm field test on the same 43 unlabeled farm photos and export
annotated images plus a per-image bunch-count CSV.

## Directory layout

```
GrapeBunchDetection/                  <- repo root (this README)
├── README.md
├── requirements.txt
├── LiveImages/                       <- 43 real farm photos (.jpg/.png), shared by both notebooks
├── Deep Learning/
│   ├── Deep_Learning_GrapeBunchDetection.ipynb
│   ├── farm_detections/              <- annotated farm-photo outputs (tracked)
│   └── grape_bunch_dataset/          <- you supply this (see "Dataset"), NOT tracked
└── Machine Learning/
    ├── Traditional_Image_Analysis_of_GrapeBunchDetection.ipynb
    ├── farm_detections/              <- annotated farm-photo outputs (tracked)
    └── grape_bunch_dataset/          <- you supply this (see "Dataset"), NOT tracked
```

The notebooks locate everything relative to the folder they are run from, so **start Jupyter with the
working directory set to the notebook's own folder** (`Deep Learning/` or `Machine Learning/`). They
find the farm photos at `<notebook folder>/../LiveImages`.

Folders the notebooks create next to themselves as they run (none are tracked, and there is no
`.gitignore` yet, so avoid `git add .`):

- `Deep Learning/`: `data/processed/` (train/val/test split and mixed lists), `data/eval/`,
  `data/annotations/live_holdout/` (your hand-labeled holdout), `data/processed_farm_ft/`, `runs/`
  (YOLO training runs), `configs/data.yaml`, `saved_models/` (checkpoints and `*_meta.json`), and
  `farm_bunch_counts.csv`
- `Machine Learning/`: `saved_models/` (`*.joblib` plus `meta.json`), `cross_validation_cache/`,
  `learning_curve_cache/`, and `farm_bunch_counts.csv`

## Dataset

Training uses a flat YOLO-format pool, `grape_bunch_dataset/{images,labels}/`, with one
`<name>.txt` label per image and one `class cx cy w h` line (normalised 0-1) per bunch. All class ids
are collapsed to a single `bunch` class. The pool merges two CC BY 4.0 sources:

- **Zenodo Grapevine Bunch Detection Dataset**, about 10,010 images: close-up, mostly single-bunch, green-dominant. <https://zenodo.org/records/7717055>
- **CERTH Grape Dataset**, about 2,587 images (filenames prefixed `CERTH_`): in-vineyard iPhone photos, several small bunches per frame, converted from COCO to YOLO format. <https://zenodo.org/records/10777647>

The pool is not included in the repo. Place (or symlink) it as `grape_bunch_dataset/` inside **each**
notebook folder that you want to run past the environment-setup phase. The Embrapa WGISD dataset was
dropped from the project because of its non-commercial licence (CC BY-NC 4.0).

The 43 farm photos are unlabeled. In the deep-learning notebook, Phase 3.1 provides a click-to-box
annotation tool to label a subset as a live holdout; Phase 9 fine-tunes on the labeled farm photos.

## Setup

Tested with Python 3.13.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1          # Linux/macOS/WSL: source .venv/bin/activate

# GPU users: install a CUDA build of PyTorch first, using the command from https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

The default `pip install torch` gives a CPU-only build on Windows. Training the YOLO model on CPU is
impractical, so use a CUDA build (native Windows, or WSL with an NVIDIA GPU passed through). The base
weights (`yolo11m.pt`) are downloaded automatically by Ultralytics on first use.

## Running the notebooks

```powershell
cd "Deep Learning"                  # or: cd "Machine Learning"
jupyter lab
```

Each notebook's **Phase 0** has two mutually exclusive setup cells. Run exactly one:

- **Option B (local / WSL)**: the default for this layout. Reads the dataset and `LiveImages/` straight off disk.
- **Option A (Google Colab)**: mounts Google Drive. On Drive, expect `grape_bunch_dataset/` and the farm photos under `data/raw/`
  (upload the contents of `LiveImages/` there). The Colab-only `google.colab` import is not in `requirements.txt`.

### Deep learning notebook

- **Phases 0-2** need only the farm photos, so they are a cheap way to check the environment and the preprocessing (sharpness, CLAHE, denoising, HSV analysis, tiling).
- **Phases 1 and 3-9** need `grape_bunch_dataset/` and a GPU: split and mix the data, annotate the live holdout, train the domain-aware model, evaluate (single-pass vs. SAHI, confidence sweep, count MAE), explain (Grad-CAM, error analysis), export farm results, then fine-tune on the farm photos.
- Phase 3.1 uses the `ipympl` widget backend (`%matplotlib widget`), so run it in JupyterLab or Jupyter Notebook, not a plain script.

### Traditional ML notebook

Runs on CPU. It extracts positive/negative patches, builds the feature matrices, trains and
cross-validates the four classifiers, then runs the full detection-and-counting pipeline
(colour-segmentation proposals, classification, NMS) with SHAP explanations. Trained models are cached
in `saved_models/` so later runs can load them instead of retraining.

## Outputs committed to the repo

`Deep Learning/farm_detections/` and `Machine Learning/farm_detections/` hold each pipeline's
annotated versions of the 43 farm photos, for side-by-side comparison. They are regenerated by Phase 7
(deep learning) and Phase 8 (traditional ML) of the respective notebook.
