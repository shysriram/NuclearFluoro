# NuclearFluoro
NuclearFluoro is a reproducible pipeline for nuclear detection and per-nuclear fluorescence quantification in microscopy images. This project intends to standardize how raw fluorescence images are converted into analysis-ready quantitative outputs that are interpretable and reusable across datasets. 

While NuclearFluoro does not propose a new segmentaton algorithm, it treats segmentation as the backend within a broader measurement pipeline focused on reproducibility and downstream usability. 

The pipeline operates on any fluorescence micrscopy data containing nuclear stains. We demonstrate its use on colon cancer microscopy data as a representative application. 

---

## Key Features & Performance Metrics

### 1. **Classical Backend (--method 'otsu')**
     **Mechanics:** Employs global statisical thresholding via Otsu's method combined with morphological cleaning
     **Pros:** Extremely efficient for CPU-based machines; requires zero heavy deep learning dependencies
     **Cons:** Susceptible to incomplete segmentation; struggles to separate touching or high confluence cell clumps

### 2. **Deep Learning Backend: Cellpose SAM (--method 'cellpose_sam')**
    **Mechanics:** Adapts the Vision Transformer (ViT) encoder from Meta AI's **Segment Anything Model (SAM)** into Cellpose's vector-flow framework, allowing it to prioritize geometry and spatial boundary detection
    **Pros:** Exceptional edge localization and perimeter tracing, as well as being highly resilient to noisy background artifacts, out-of-focus blurs, or poor/dim imaging conditions
    **Cons:** Cellpose SAM has a strict tile-size constraint of internal 256 x 256 pixel patches due to SAM's fixed position embeddings

### 3. **Deep Learning Backend: Cellpose DINO (--method 'cellpose_dino')**
    **Mechanics:** Leverages a self-supervised foundation backbone (**DINOv3**) trained to recognize universal visual features, cellular textures, and subtle morphologic gradients without manual annotations
    **Pros:** Cellpose DINO excels at separate highly concentrated clumps of cells or varying cell phenotypes. Additionally, as opposed to Cellpose SAM, DINO features flexible resolution scaling, depending on the user's needs
    **Cons:** Computationally demanding; heavily relies on local hardware acceleration to prevent slow inference times



