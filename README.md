Sentinel-2 Super-Resolution Mapping (SRM) EngineNational Deep Learning Infrastructure for High-Resolution Earth ObservationUpscaling Sentinel-2 Multispectral Surface Reflectance from 10m to 2.5m Spatial Resolution with Physics-Constrained Uncertainty Quantification.Executive OverviewThe Sentinel-2 Super-Resolution Mapping (SRM) Engine is an end-to-end deep learning framework developed for Smart India Hackathon (SIH) Problem Statement 26142, under the aegis of the Ministry of Education (MoE) Innovation Cell, Government of India.High-resolution satellite imagery (0.5m–2.5m) from commercial constellations like PlanetScope or WorldView is cost-prohibitive for large-scale, continuous public sector governance. Conversely, free and open ESA Sentinel-2 imagery provides high temporal resolution (5-day revisit) but is spatially restricted to 10m spatial resolution across its key visible and near-infrared (VNIR) bands.+------------------------------------+        +------------------------------------+
|  Input: 10m Sentinel-2 Native      |        |  Output: 2.5m Super-Resolved (SR)  |
|  Bands: B2, B3, B4, B8             |  --->  |  Bands: B2, B3, B4, B8             |
|  Grid: 32 x 32 Spatial Resolution  |  4x SR |  Grid: 128 x 128 Spatial Resolution|
|  Uncertainty: N/A                  |        |  Uncertainty: Per-pixel sigma Map  |
+------------------------------------+        +------------------------------------+
The SRM Engine closes this spatial gap by executing 4x sub-pixel spatial upscaling (10m $\rightarrow$ 2.5m) directly on 4 surface reflectance bands: B2 (Blue), B3 (Green), B4 (Red), and B8 (NIR). To guarantee operational safety in critical governance applications, the system pairs high-resolution spatial reconstruction with Heteroscedastic Uncertainty Quantification, producing a pixel-level variance map ($\sigma$) that explicitly flags model hallucinations and out-of-distribution land features.Strategic National Impact for IndiaPM Fasal Bima Yojana (Crop Insurance): Delineates micro-agricultural field boundaries, smallholder land parcels, and intra-field vegetation stress.PM Gati Shakti & Urban Governance: Enables precise building footprint extraction, informal settlement monitoring, and road network mapping for Urban Local Bodies (ULBs).Disaster Management & Flood Hazard Mapping: Provides 2.5m Water Index (NDWI) mapping across vulnerable basins such as the Kosi and Brahmaputra rivers to identify breached embankments and submerged transportation links.Key Features & Technical Innovations4x Spatial Scaling Engine: Upscales 10m native spatial resolution down to 2.5m super-resolved spatial grid without losing radiometric calibration.Full 4-Band VNIR Pipeline: Operates simultaneously across [B2, B3, B4, B8], preserving multi-band spectral profiles and physical vegetation/water signatures.Heteroscedastic Uncertainty Estimation: Outputs dual-head predictions—reconstructed surface reflectance ($\hat{Y}$) and pixel-level noise log-variance ($s = \log\sigma^2$)—to prevent unverified AI hallucinations from influencing spatial policy.Physics-Guided & Anti-Hallucination Loss Function: Integrates Spectral Angle Mapper (SAM) Loss, NDVI Preservation Loss, and Downsampling Consistency Enforcement ($D_{4\times}(\text{SR}) \approx \text{LR}$).3-Tier Holistic Validation Framework: Evaluates predictions across classical image quality (PSNR/SSIM), physical spectral conservation ($\Delta\text{NDVI}$, SAM), and real-world downstream task performance (building segmentation mIoU).Interactive Digital India GIS Dashboard: A Streamlit web portal with pre-loaded Indian Area of Interest (AOI) presets, split-screen comparison tools, and uncertainty heatmaps.System Architecture & Monorepo LayoutData & Execution PipelineCode snippetflowchart TD
    STAC[STAC Fetcher / Sentinel-2 L2A] --> PRE[Preprocessing & Band Normalization]
    PRE --> M[Model Ladder Selector]
    
    M --> M0[M0: Bicubic Baseline 4x]
    M --> M1[M1: RRDB-L1 Deep Residual Net]
    M --> M2[M2: SwinIR-GAN Transformer]
    
    M0 & M1 & M2 --> HEAD[Heteroscedastic Uncertainty Head]
    
    HEAD --> TIER[3-Tier Validation Suite]
    
    TIER --> T1[Tier 1: Spatial & Spectral Metrics]
    TIER --> T2[Tier 2: Physical Consistency Enforcement]
    TIER --> T3[Tier 3: Downstream Utility Verification]
    
    T1 & T2 & T3 --> APP[Streamlit Interactive GIS Portal]
    T1 & T2 & T3 --> EXPORT[ONNX Runtime Exporter]
Directory StructurePlaintextsrm_project/
├── configs/                        # System & model configuration files
│   ├── default_config.yaml         # Dataset path & training hyperparameters
│   └── model_m2_swinir.yaml        # Transformer backbone architectural specs
├── srm/                            # Core Python package
│   ├── data/                       # Earth Observation data pipelines
│   │   ├── stac_fetcher.py         # Automated Copernicus STAC API client
│   │   ├── dataset.py              # PyTorch Dataset for 4-band patches
│   │   └── transforms.py           # Radiometric normalization & augmentations
│   ├── models/                     # Deep learning model ladder
│   │   ├── baseline_m0.py          # Bicubic baseline implementation
│   │   ├── rrdb_m1.py              # RRDB-L1 Residual Dense Network
│   │   ├── swinir_m2.py            # Shifted Window Transformer backbone
│   │   └── uncertainty_head.py     # Dual-head variance prediction module
│   ├── train/                      # Training loop & physics losses
│   │   ├── trainer.py              # PyTorch AMP distributed training loop
│   │   ├── losses.py               # Composite physics loss functions
│   │   └── amp_utils.py            # FP16 mixed-precision helpers
│   ├── eval/                       # 3-Tier evaluation framework
│   │   ├── metrics.py              # Tier 1: PSNR, SSIM, SAM, ERGAS
│   │   ├── physical_checks.py      # Tier 2: Delta-NDVI & Downsampling checks
│   │   └── downstream_eval.py      # Tier 3: Building mIoU evaluator
│   ├── apps/                       # Web applications
│   │   └── app.py                  # Streamlit Digital India GIS Portal
│   └── serve/                      # Model deployment modules
│       └── export_onnx.py          # ONNX conversion & engine optimization
├── tests/                          # Automated Pytest Suite (53 Passing)
│   ├── test_data.py                # Dataset & STAC fetcher unit tests
│   ├── test_models.py              # Tensor shape & forward pass tests
│   ├── test_losses.py              # Loss gradient & physics constraint tests
│   └── test_eval.py                # Metric verification tests
├── requirements.txt                # Fixed dependency versions
└── README.md                       # Repository documentation
Model Ladder & Loss FormulationsModel Ladder ArchitectureM0: Bicubic Baseline: Standard 4x bicubic interpolation serving as the lower-bound benchmark.M1: RRDB-L1 Network: Deep Residual-in-Residual Dense Block network optimized for spatial feature extraction with dense skip connections.M2: SwinIR-GAN Transformer: State-of-the-art Residual Swin Transformer backbone leveraging shifted-window self-attention ($8 \times 8$ window size) for non-local texture synthesis.Loss Function FormulationsTo prevent high-frequency generative artifacts and spectral distortion, models are optimized using a composite physics-constrained loss function:$$\mathcal{L}_{\text{total}} = \lambda_{\text{NLL}} \mathcal{L}_{\text{NLL}} + \lambda_{\text{SAM}} \mathcal{L}_{\text{SAM}} + \lambda_{\text{NDVI}} \mathcal{L}_{\text{NDVI}} + \lambda_{\text{consist}} \mathcal{L}_{\text{consist}}$$1. Heteroscedastic Gaussian Negative Log-Likelihood (NLL) LossMeasures spatial reconstruction error while jointly training the model to predict pixel-wise log-variance $s = \log(\sigma^2)$:$$\mathcal{L}_{\text{NLL}}(\hat{Y}, Y, s) = \frac{1}{2 C \cdot H \cdot W} \sum_{c,h,w} \left( e^{-s_{c,h,w}} \Vert{}\hat{y}_{c,h,w} - y_{c,h,w}\Vert{}_1 + s_{c,h,w} \right)$$2. Spectral Angle Mapper (SAM) LossCalculates the physical spectral angle across all 4 bands between the predicted spectrum vector $\hat{y}_i$ and target vector $y_i$:$$\mathcal{L}_{\text{SAM}}(\hat{Y}, Y) = \frac{1}{N} \sum_{i=1}^{N} \arccos \left( \frac{\hat{y}_i \cdot y_i}{\Vert{}\hat{y}_i\Vert{}_2 \Vert{}y_i\Vert{}_2 + \epsilon} \right)$$3. NDVI Preservation LossEnforces ecological consistency by minimizing the absolute difference in Normalized Difference Vegetation Index:$$\text{NDVI}(X) = \frac{X_{\text{NIR}} - X_{\text{Red}}}{X_{\text{NIR}} + X_{\text{Red}} + \epsilon}$$$$\mathcal{L}_{\text{NDVI}}(\hat{Y}, Y) = \frac{1}{H \cdot W} \Vert{}\text{NDVI}(\hat{Y}) - \text{NDVI}(Y)\Vert{}_1$$4. Downsampling Consistency Loss ($D_4$)Guarantees physical reversibility by penalizing discrepancies when average-downsampling the 2.5m super-resolved output back to the original 10m low-resolution grid:$$\mathcal{L}_{\text{consist}}(\hat{Y}, X) = \frac{1}{H \cdot W} \Vert{}\text{AvgPool}_{4\times}(\hat{Y}) - X\Vert{}_1$$Quantitative Benchmarks TableThe models were evaluated on an independent benchmark test set of 1,200 Sentinel-2 surface reflectance tiles across diverse Indian geographical terrain types.Model ArchitectureScale FactorPSNR (dB) ↑SSIM ↑SAM (°) ↓ERGAS ↓ΔNDVI ↓Building mIoU (%) ↑M0: Bicubic Baseline$4\times$28.140.7923.454.120.05862.4%M1: RRDB-L1 Network$4\times$30.820.8642.152.850.02471.8%M2: SwinIR-GAN Transformer$4\times$31.980.8981.822.310.01276.6%Net Improvement (M2 vs M0)—+3.84 dB+0.106-1.63°-1.81-0.046+14.2%Quick Start & Installation GuidePrerequisitesOperating System: Linux (Ubuntu 22.04+), macOS, or Windows 11Python: 3.11CUDA Toolkit: 12.1+ (optional, for GPU training)Step 1: Clone Repository & Setup Virtual EnvironmentBash# Clone the official repository
git clone https://github.com/DivyanshPrakashIIT/srm_project.git
cd srm_project

# Create a virtual environment
python -m venv venv

# Activate environment (Linux / macOS)
source venv/bin/activate

# Activate environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1
Step 2: Install Core DependenciesBashpip install --upgrade pip
pip install -r requirements.txt
Step 3: Run Full Pytest Verification Suite (53 Tests)Verify that the entire codebase, loss math, data transforms, and model forward passes function correctly:Bashpytest tests/ -v
Plaintext============================== test session starts ==============================
collected 53 items

tests/test_data.py .........................                             [ 47%]
tests/test_models.py ....................                                [ 84%]
tests/test_losses.py ....                                                [ 92%]
tests/test_eval.py ....                                                  [100%]

============================== 53 passed in 4.12s ===============================
Step 4: Launch Interactive GIS Web DashboardLaunch the Streamlit web app:Bashstreamlit run srm/apps/app.py
Or via the Python executable runner:Bashpy -m streamlit run srm/apps/app.py
Open your web browser and navigate to http://localhost:8501.Interactive Web Portal UsageThe integrated Streamlit GIS Portal provides an end-to-end environment for real-time inference and evaluation.+-------------------------------------------------------------------------------+
|  🛰️ Sentinel-2 Super-Resolution Mapping System (2.5m)                          |
|  Smart India Hackathon - Problem Statement 26142                              |
+-------------------------------------------------------------------------------+
|  CONTROL PANEL             |  SIDE-BY-SIDE VISUAL COMPARISON                  |
|                            |                                                  |
|  Target AOI Preset:        |  10m Native Sentinel-2    2.5m Super-Resolved    |
|  [ Punjab Agriculture  v ] |  +--------------------+  +--------------------+  |
|                            |  |                    |  |                    |  |
|  Model Architecture:       |  |   [10m Low-Res]    |  |  [2.5m High-Res]   |  |
|  [ M2: SwinIR Transformer v]| |                    |  |                    |  |
|                            |  +--------------------+  +--------------------+  |
|  Composite Mode:           |                                                  |
|  (o) True Color (B4,B3,B2) |  UNCERTAINTY MAP (sigma)                         |
|  ( ) False Color (B8,B4,B3) |  +--------------------------------------------+  |
|  ( ) NDVI Index Map        |  |  [Low Variance: Green | High Variance: Red] |  |
|  ( ) Uncertainty (sigma)   |  +--------------------------------------------+  |
+-------------------------------------------------------------------------------+
Features & WorkflowPreset Indian AOI Selection:Punjab Agriculture Belt: High NIR contrast patches for boundary testing.Delhi NCR Urban Region: High spatial frequency structural grids (buildings/roads).Kosi River Flood Basin: High NDWI contrast for water body mapping.Custom Patch Upload (.npy / .npz):Upload custom 4-channel surface reflectance NumPy arrays.Expected Shape: (4, H, W) or (1, 4, H, W) corresponding to [B2, B3, B4, B8] normalized to $[0.0, 1.0]$.Uncertainty Overlay:Toggle the Heteroscedastic Uncertainty Map ($\sigma$) to inspect model confidence across cloud shadows, building edges, and water boundaries.Model Card & Ethical AI SafeguardsIntended UsageTarget Applications: Agricultural land management, disaster recovery planning, infrastructure tracking, and land-use/land-cover (LULC) mapping.Target Users: Remote sensing analysts, GIS developers, urban planning bodies, and agricultural researchers.Anti-Hallucination & Safety SafeguardsUncertainty Masking: Pixels displaying uncertainty values above the confidence threshold ($\sigma > \text{threshold}$) are flagged in red on the web dashboard and masked out prior to downstream vectorization.Physical Reversibility Enforcement: Downsampling loss ensures that upscaled features remain anchored to real-world radiometric measurements.Operational LimitationsCloud & Snow Interference: Severe cloud cover ($>30\%$) or snow coverage distorts spectral reflectance values in Band 8 (NIR).Extreme Topography: Deep terrain shadows in alpine regions (e.g., High-Altitude Himalayas) may exhibit higher uncertainty levels.Contributors & CitationDeveloped for Smart India Hackathon (SIH) PS 26142 by Divyansh Prakash & Team (Indian Institute of Technology / Partner Institutions).Academic CitationCode snippet@software{Prakash_Sentinel2_Super_Resolution_2026,
  author       = {Prakash, Divyansh and Team},
  title        = {{Sentinel-2 Super-Resolution Mapping (SRM) Engine (10m to 2.5m)}},
  month        = sep,
  year         = 2026,
  publisher    = {GitHub},
  journal      = {GitHub Repository},
  howpublished = {\url{https://github.com/DivyanshPrakashIIT/srm_project}},
  note         = {Smart India Hackathon - Problem Statement 26142}
}