# Radiology Desktop

An offline-capable AI radiology workstation built with **Tauri 2**, **React 19**, and **llama.cpp**. Upload DICOM or medical images, draw ROIs for segmentation, and generate structured radiology reports using a locally-running vision-language model -- no cloud dependency required.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                        Radiology Desktop                           │
│                                                                    │
│  ┌──────────────┐   ┌──────────────────────┐   ┌────────────────┐  │
│  │  StudyPanel  │   │   ViewerWorkspace    │   │ DecisionPanel  │  │
│  │              │   │                      │   │                │  │
│  │  - Upload    │   │  - Image viewport    │   │  - AI Report   │  │
│  │  - Study     │   │  - ROI drawing       │   │  - Findings    │  │
│  │    list      │   │  - Seg overlay       │   │  - Impression  │  │
│  │  - Accept/   │   │  - Zoom/Brightness   │   │  - Recommend.  │  │
│  │    Note      │   │  - 3D preview        │   │  - Seg masks   │  │
│  └──────────────┘   │  - Undo/Redo/Clear   │   │  - Confidence  │  │
│                     └──────────┬───────────┘   │  - Token usage │  │
│                                │               └───────┬────────┘  │
│                                │                       │           │
│                     ┌──────────▼───────────────────────▼────────┐  │
│                     │         React State (App.tsx)             │  │
│                     │  - studies[], apiSettings, localAi,       │  │
│                     │    undoStacks, redoStacks, llmStatus      │  │
│                     └──────────────────┬────────────────────────┘  │
│                                        │                           │
│              ┌─────────────────────────┼──────────────────────┐    │
│              │                         │                      │    │
│    ┌─────────▼──────────┐    ┌─────────▼────────┐             │    │
│    │  Tauri Commands    │    │  HTTP /v1 API    │             │    │
│    │  (invoke)          │    │  (fetch)         │             │    │
│    └─────────┬──────────┘    └────────┬─────────┘             │    │
│              │                        │                       │    │
└──────────────┼────────────────────────┼───────────────────────┘────┘
               │                        │                            
    ┌──────────▼──────────┐   ┌─────────▼─────────────┐              
    │   Rust Backend      │   │   llama-server        │              
    │   (src-tauri/)      │   │   (sidecar process)   │              
    │                     │   │                       │              
    │  - spawn/kill       │──▶│  /v1/chat/completions │              
    │  - health poll      │   │  /v1/models           │              
    │  - path resolution  │   │  GGUF + mmproj        │              
    │  - window lifecycle │   │  GPU layers (Metal)   │              
    └─────────────────────┘   └───────────────────────┘              
                                                                     
    ┌────────────────────────────────────────────────────────────────┐
    │  Bundled dylibs: libggml, libllama, libllama-common, libmtmd   │ 
    └────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
radiology-desktop/
├── index.html                    # Vite HTML entry point
├── package.json                  # Frontend deps & scripts
├── vite.config.ts                # Vite + React plugin config
├── tailwind.config.js            # Tailwind CSS config
├── postcss.config.js             # PostCSS config
├── tsconfig.json                 # TypeScript project refs
├── tsconfig.app.json             # App TS config
├── tsconfig.node.json            # Node TS config
│
├── src/
│   ├── main.tsx                  # React root (StrictMode)
│   ├── App.tsx                   # Main app -- all UI components (~1776 lines)
│   ├── styles.css                # Tailwind directives + clinical theme
│   └── vite-env.d.ts             # Vite type declarations
│
└── src-tauri/
    ├── Cargo.toml                # Rust deps (tauri 2.x, serde, dialog plugin)
    ├── Cargo.lock                # Rust lockfile
    ├── build.rs                  # Tauri build script
    ├── tauri.conf.json           # Tauri app config (window, bundle, resources)
    ├── src/
    │   ├── main.rs               # Rust entry point (calls app_lib::run)
    │   └── lib.rs                # Core logic: llama-server lifecycle
    ├── binaries/
    │   ├── llama-server-*        # Platform-specific llama-server binary
    │   ├── libggml*.dylib        # GGML compute libraries
    │   ├── libllama*.dylib       # LLaMA inference libraries
    │   ├── libllama-common*.dylib
    │   └── libmtmd*.dylib        # Multimodal (vision) encoder
    ├── icons/                    # App icons (icns, ico, png)
    └── gen/                      # Tauri-generated schemas & ACL
```

---

## Workflow

```
 1. LAUNCH
    │
    ├─ Tauri starts → Rust backend spawns llama-server sidecar
    │  (if model + mmproj paths are configured)
    │
    ├─ Frontend shows "Loading Embedded AI Engine" overlay
    │  while health-polling TCP port 8081
    │
    └─ Once port is live → emits "llama-server-ready" event
       → overlay clears, app is ready

 2. UPLOAD STUDY
    │
    ├─ Click "Upload DICOM / Image" in StudyPanel
    │  or drag-and-drop onto the viewport
    │
    ├─ File is read → preview URL created (blob URL for images)
    │
    └─ New Study added to list with "unreviewed" status

 3. SEGMENTATION (ROI)
    │
    ├─ Click "Medical AI" toolbar button (auto-centers ROI)
    │  OR draw a rectangle on the viewport (mouse drag)
    │
    ├─ ROI box is normalized to 0-1 coordinates
    │
    └─ Segmentation result (label, confidence, volume) appears
       as a cyan overlay on the image

 4. AI REPORT GENERATION
    │
    ├─ Click "Generate" in DecisionPanel
    │
    ├─ Frontend builds chat-completions request:
    │  - System prompt: radiology assistant
    │  - User content: modality, body part, patient, seg summary
    │  - If image available: base64-encoded as image_url content
    │
    ├─ POST to llama-server /v1/chat/completions
    │  (or external API in browser-only mode)
    │
    └─ Response parsed as JSON → report fields populated:
       summary, findings, impression, recommendation, confidence

 5. CLINICAL REVIEW
    │
    ├─ Review AI report in DecisionPanel
    │
    ├─ "Accept" → marks study as accepted
    │
    └─ "Note" → opens feedback modal → saves doctor feedback
       → marks study as "needs-review"

 6. 3D PREVIEW (optional)
    │
    ├─ Click "3D" toolbar button
    │
    └─ Three.js modal renders:
       - Image surface with displacement map, OR
       - Fallback sphere scaled to ROI dimensions
       - Wireframe overlay + ROI bounding box
       - Auto-rotation animation
```

---

## Key Components

| Component | File | Description |
|---|---|---|
| **App** | `src/App.tsx` | Root component. Manages all state: studies, settings, AI engine lifecycle, undo/redo stacks |
| **StudyPanel** | `src/App.tsx` | Left sidebar. Study list, file upload, Accept/Note actions |
| **ViewerWorkspace** | `src/App.tsx` | Center panel. Image viewport with ROI drawing, segmentation overlays, zoom/brightness controls, toolbar |
| **DecisionPanel** | `src/App.tsx` | Right sidebar. AI report generation, findings/impression/recommendation display, segmentation mask list, token usage chart |
| **ApiSettingsModal** | `src/App.tsx` | Settings dialog. Model paths (GGUF), mmproj path, GPU layers, API URL/key |
| **FeedbackModal** | `src/App.tsx` | Doctor feedback dialog for "needs-review" notes |
| **StructurePreviewModal** | `src/App.tsx` | Three.js 3D surface preview modal |
| **Rust Backend** | `src-tauri/src/lib.rs` | Tauri commands: start/stop llama-server, get server status, path resolution, health polling |

---

## Tauri Commands (Rust → Frontend)

| Command | Direction | Description |
|---|---|---|
| `start_llama_server` | Frontend → Rust | Spawns llama-server with model, mmproj, GPU layers. Returns immediately; signals readiness via event |
| `stop_llama_server` | Frontend → Rust | Kills the running llama-server child process |
| `get_server_status` | Frontend → Rust | Returns `{ running, port }` for the current server |
| `llama-server-ready` | Rust → Frontend (event) | Emitted when TCP port becomes reachable, or on timeout with error |

---

## Dual Mode: Desktop vs Browser

| Feature | Desktop (Tauri) | Browser (Vite dev) |
|---|---|---|
| AI engine | Embedded llama-server sidecar | External API (e.g. LM Studio) |
| Model config | File picker for .gguf paths | URL endpoint + API key |
| Settings | modelPath, mmprojPath, gpuLayers | modelApiUrl, modelApiKey, modelId |
| Server URL | Hardcoded `http://127.0.0.1:8081/v1` | Configurable (default `:1234/v1`) |
| File dialog | Native via `@tauri-apps/plugin-dialog` | Standard HTML file input |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Desktop shell | **Tauri 2** (Rust) |
| Frontend | **React 19** + **TypeScript** |
| Styling | **Tailwind CSS 3** (dark clinical theme) |
| 3D preview | **Three.js** (r184) |
| Icons | **Lucide React** |
| Build | **Vite 7** |
| AI inference | **llama.cpp** (llama-server sidecar, ~v0.9113) |
| Vision | **libmtmd** (multimodal encoder for MedGemma-style models) |

---

## Getting Started

### Prerequisites

- Node.js 18+
- Rust 1.77.2+ (with Cargo)
- Tauri CLI v2 (`npm install -g @tauri-apps/cli`)
- A GGUF vision-language model (e.g. MedGemma) + matching mmproj file

### Development

```bash
# Install frontend dependencies
npm install

# Run in browser-only mode (external API needed)
npm run dev

# Run as Tauri desktop app
npm run tauri:dev
```

On first Tauri launch, you'll be prompted to configure model paths via Settings.

### Production Build

```bash
npm run tauri:build
```

Outputs a `.dmg` (macOS). The bundle includes the llama-server binary and all required dylibs.

---

## Configuration

Settings are persisted in `localStorage` under key `radiology-api-settings-v2`.

| Setting | Desktop | Browser | Default |
|---|---|---|---|
| `modelPath` | Path to model .gguf | N/A | -- |
| `mmprojPath` | Path to mmproj .gguf | N/A | -- |
| `gpuLayers` | Number of GPU layers (0 = CPU) | N/A | 0 |
| `modelApiUrl` | Auto-set to localhost:8081 | Configurable | `http://127.0.0.1:1234/v1` |
| `modelApiKey` | Optional bearer token | Optional bearer token | -- |
| `modelId` | Auto-detected from server | Manual | `medical-model` |

Theme preference is stored under key `radiology-theme` (light/dark).

---

## Keyboard Shortcuts

| Shortcut | Action |
|---|---|
| `Ctrl/Cmd + Z` | Undo segmentation |
| `Ctrl/Cmd + Y` | Redo segmentation |
| `Ctrl/Cmd + Shift + Z` | Redo segmentation |

---

## License

MIT
