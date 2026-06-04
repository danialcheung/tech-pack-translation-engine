# Tech Pack Translation Engine

A containerized Python microservice designed to automate the translation of manufacturing tech packs. The engine utilizes **PaddleOCR** and **OpenCV** to extract a document's structural grid, segments key layout components (such as BOM tables and technical design sketches), and applies a dynamic column-filtering mask driven by a runtime JSON configuration. Approved text is routed to the **Google Cloud Translation API** and seamlessly re-rendered onto the document using **PIL (Pillow)** while maintaining pixel-perfect layout and line integrity.

---

## 📌 Table of Contents
* [🚀 Key Features](#-key-features)
* [🛠️ Project Structure](#️-project-structure)
* [⚙️ Configuration (`terms.json`)](#️-configuration-termsjson)
* [📦 Prerequisites](#-prerequisites)
* [⚙️ Local Deployment & Setup](#️-local-deployment--setup)
* [📊 Data Pipeline Flow](#-data-pipeline-flow)
* [🧠 Challenges & Engineering Solutions](#-challenges--engineering-solutions)
* [🎯 Production Roadmap: Phase 2 Architecture Evolution](#-production-roadmap-phase-2-architecture-evolution)

---

## 🚀 Key Features

* **Hybrid-Deterministic Architecture:** Combines deep-learning OCR text extraction with lightweight, high-performance local matrix mathematics.
* **Spatial Guillotine:** Automatically detects bounding boxes that mistakenly straddle table cell borders and algorithmically slices the text based on physical column line intersections.
* **Surgical Grid Restoration:** Isolates and extracts structural lines before clearing text, blending the original grid line mask back over the final canvas to ensure continuous cell borders.
* **Dynamic Column Masking (DNT):** Protects raw technical specifications, sizing markers, and material codes by mapping dynamic "Do Not Translate" corridors at runtime.


---

## 🛠️ Project Structure

```text
techpack-translator/
├── main.py               # Main pipeline orchestration script
├── terms.json            # Dynamic configuration dictionary for DNT mapping
├── Dockerfile            # Production multi-stage Debian-slim environment
├── .dockerignore         # Excludes local data and secrets from build context
├── fonts/
│   └── simhei.ttf        # TrueType font used for rendering CJK characters
├── input/                # Volume-mounted local directory for source images
├── output/               # Volume-mounted local directory for translated results
└── requirements.txt      # Required Python Libraries
```

## ⚙️ Configuration (`terms.json`)

To prevent the engine from translating critical technical data or factory codes, you can decouple business terminology from the source code. Update the `dnt_headers` array with lowercase strings of column titles that should remain untranslated:

```json
{
  "dnt_headers": [
    "color details",
    "quantity",
    "cost",
    "unit of measure",
    "item number"
  ]
}
```

## 📦 Prerequisites

* Docker Desktop installed and running.
* An active Google Cloud Project with the **Cloud Translation API** enabled.
* A service account key JSON file downloaded from Google Cloud IAM.

---

## ⚙️ Local Deployment & Setup

### 1. Build the Docker Image
Navigate to the root directory of your project and run the following command to package the application:
```powershell
docker build -t techpack-translator:v1.0 .
```

### 2. Configure Local Folders
Ensure you have created the following directories and placed your files accordingly:
* Place the target image files inside the `./input/` directory.
* Place your Google Cloud Service Account key file (`google-key.json`) in an accessible directory.

### 3. Execution

Run the following command to execute the container, mapping your local input/output folders and passing your Google Cloud credentials into the environment:

```bash
docker run --rm \
  -v "/path/to/your/input":/input \
  -v "/path/to/your/output":/output \
  -v "/path/to/your/google-key.json":/app/credentials.json \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json \
  -e GOOGLE_CLOUD_PROJECT="your-gcp-project-id" \
  techpack-translator:v1.0
  ```

## 📊 Data Pipeline Flow

1. **Image Loading & Asset Mapping:** Scans `/input` directory, determines target output paths, and initializes single-instance PaddleOCR engines.
2. **Structural Grid Profiling:** Isolates clothing blueprints and sketches into a "Forbidden Zone" while extracting horizontal and vertical cell borders using directional mathematical morphologies.
3. **Spatial Slicing:** Adjusts multi-column text intersections using an adaptive text-token boundary look-ahead buffer.
4. **DNT Filtering:** Iterates over row boundaries to locate matching `terms.json` strings and dynamically protects vertical data tunnels.
5. **Restoration & Localized Rendering:** Overlays an isolated layout grid mask on top of freshly rendered PIL typography to maintain original pixel integrity.

![algorithm diagram]("algorithm_diagram.png")
---

## 🧠 Challenges & Engineering Solutions

### 1. Paradigm Selection: Generative AI vs. Deterministic Engineering
* **Challenge:** While Vision-LLMs can be prompted to parse layouts, they introduce high computing costs, strict GPU hardware dependencies, slow runtime latency, and non-deterministic text alignment or hallucinations.
* **Solution:** I intentionally bypassed Generative AI for the structural parsing layer, favoring classic computer vision via OpenCV and localized math. This ensures the system operates with repeatable precision at near-zero computing cost, while making system edge cases and behavioral weaknesses entirely predictable. I believe Generative AI should serve as a rigorously tested fallback rather than a primary driver for rigid data structures.

### 2. Layout Segmentation: Identifying Design Art Without YOLO
* **Challenge:** Tech packs contain large fashion sketches and technical drawings. To prevent the engine from wiping out parts of the artwork or mistaking sketchy lines for table cells, these regions must be isolated. However, training a deep-learning object detection model like YOLO introduces massive deployment overhead, gigabytes of container bloat, and heavy dataset training demands.
* **Solution:** I built a lightweight structural profiling algorithm using OpenCV contours. By evaluating the area and aspect ratio of all closed bounding paths, the engine filters out massive sketches based on a spatial density threshold. These coordinates are locked into a temporary "Forbidden Zone" mask that completely shields the artwork from downstream text extraction in milliseconds, bypassing heavy neural network inference entirely.

### 3. Text Segmentation: Resolving Merged Cell Bounding Boxes
* **Challenge:** Text recognition models like PaddleOCR aggressively cluster localized text pixels. When columns are tightly formatted, the model frequently ignores column borders and merges text from two adjacent cells into a single, combined bounding box, corrupting downstream data filters.
* **Solution:** I engineered a "Spatial Guillotine" pipeline. Using directional morphological filtering, the script strips text noise to isolate a clean map of the vertical table lines. If a text box crosses a vertical line, the algorithm calculates the precise intersection ratio, applies a 3-character look-ahead safety buffer to snap to the nearest natural word break, and cleanly slices the text back into two distinct cell blocks.

### 4. Canvas Restoration: Preserving Structural Grid Integrity
* **Challenge:** Completely removing the original English text requires clearing the canvas within each bounding box. However, applying a blank white rectangle over the text inevitably erases the underlying cell borders and table lines, leaving the final document looking digitally fractured and unreadable.
* **Solution:** I implemented a multi-layered image restoration technique. The algorithm first isolates a dedicated pixel mask of the table's structural grid lines before any text clearing occurs. After the English text is masked out and the new translations are rendered on top using PIL, the engine blends the original grid line mask back over the canvas as a top layer, seamlessly healing all broken line intersections.

### 5. Domain Data Integrity: Mapping Dynamic Industry Terminology
* **Challenge:** Tech packs contain critical technical specifications (material codes, sizing markers, measurements) that must remain in their native format for production accuracy. Inferring what to protect based on table headers is difficult because terminology varies wildly across factories, and hardcoding these names would make the software fragile and rigid.
* **Solution:** I decoupled the business logic from the codebase by building a runtime configuration schema (`terms.json`). The engine dynamically matches scanned column headers against this external dictionary. Once a match is locked, it calculates that column's exact horizontal boundaries on the fly, automatically safeguarding all data nested directly beneath it. This shifts terminology control entirely to end-users (product managers or factory teams), who can update vocabulary at any time without needing code redeployments.

## 🎯 Production Roadmap: Phase 2 Architecture Evolution

### 1. Structural Segmentation: Multi-Tiered Layout Handling
* **The Constraint:** The current lightweight OpenCV contour profiling relies on spatial density thresholds. While highly effective at isolating massive, complex technical drawings, it can underperform on minimalist or overly simplistic artwork, occasionally failing to isolate the sketch area.
* **Phase 2 Evolution:** Phase 1 optimized for a low-compute, zero-GPU baseline. To support minimalist art styles without sacrificing this efficiency, Phase 2 will introduce a **hybrid fallback architecture**. The system will pass layouts through our lightweight OpenCV engine first; if layout confidence scores fall below a strict threshold, it will trigger an asynchronous fallback to a fine-tuned object detection model (like YOLO) to dynamically bound the canvas.

### 2. Column Data Confidentiality & Segmentation Fail-Safes
* **The Constraint:** The deterministic "Spatial Guillotine" is bounded by string-token spacing mechanics; highly non-standard text alignment can occasionally result in a stray word bleeding past a column wall. For enterprise clients, this introduces data alignment and confidentiality concerns if text crosses into protected corridors.
* **Phase 2 Evolution:** To achieve enterprise-grade security and 100% airtight data isolation, we plan to evaluate migrating the layout and extraction engine to **Azure Document Intelligence (AzureDI) containerized models**, which can be hosted completely locally within a secure private cloud. Furthermore, if a client’s budget supports high-performance compute, we will explore **Vision-Language Models (VLMs)** to perform semantic layout validation prior to API routing.

### 3. Fine-Grained Character Recognition (The Inches `"` Marker)
* **The Constraint:** Standard PaddleOCR models can occasionally drop or skip fine-grained punctuation, such as the double-quote symbol (`"`) used to denote inches in technical measurements. This can result in structural data losing its unit context.
* **Phase 2 Evolution:** We will address this granular data loss by introducing an **Image Pre-Processing Upscaling Pipeline** (using algorithms like Super-Resolution CNNs or Bilateral Filtering) to maximize character sharpness before OCR execution. Concurrently, migrating to a high-fidelity tabular OCR engine like AzureDI will provide the character-level confidence scores necessary to preserve microscopic engineering markers.

### 4. Vertical Layout Reading Order & Token Aggregation
* **The Constraint:** Vertically stacked text (such as stacked header labels or split words like "Design" on line one and "Pack" on line two) can sometimes be treated as completely independent blocks, leading to fragmented or contextually awkward translations.
* **Phase 2 Evolution:** We will implement an **Adaptive Spatial Proximity Grouping** algorithm. By analyzing the vertical bounding coordinates and measuring the precise pixel-distance gap between text fragments, the script will mathematically infer line continuations. These fragments will be merged into single, unified lines *before* translation, which maps directly to how advanced engines like AzureDI structure word coordinates to guarantee contextual translation accuracy.

### 5. Standardizing Industry Vocabulary Governance
* **The Constraint:** Relying solely on localized, factory-specific column names via `terms.json` assumes a level of layout standardization that doesn't always exist in global manufacturing supply chains.
* **Phase 2 Evolution:** We plan to scale the simple configuration file into a **Rigorous Industry-Standard Dictionary Service**. By partnering with production teams to catalog an exhaustive ontology of manufacturing, textile, and engineering terminology, the engine will no longer be limited to matching exact column strings. Instead, it will evaluate text on a per-word/per-line semantic basis, determining exactly what data corridor to mask based on a standardized, universal glossary of manufacturing logic.