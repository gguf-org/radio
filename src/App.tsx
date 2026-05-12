import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ChangeEvent, MouseEvent, ReactNode } from "react";
import * as THREE from "three";
import {
  Brain,
  CheckCircle2,
  CircleDot,
  Cpu,
  FileImage,
  KeyRound,
  Layers,
  Loader2,
  MessageSquareText,
  Moon,
  PanelLeftClose,
  PanelLeftOpen,
  Redo2,
  Save,
  Settings,
  ShieldCheck,
  Sparkles,
  Sun,
  Trash2,
  Undo2,
  Upload,
  XCircle,
  ZoomIn,
  ZoomOut,
} from "lucide-react";

// ── Tauri detection ────────────────────────────────────────────────────────
const IS_TAURI = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
const LOCAL_SERVER_PORT = 8081;
const LOCAL_SERVER_URL = `http://127.0.0.1:${LOCAL_SERVER_PORT}/v1`;

type LocalAiPhase = "idle" | "starting" | "ready" | "error";

type LocalAiStatus = {
  phase: LocalAiPhase;
  error?: string;
  modelId?: string;
};

// ── Domain types ───────────────────────────────────────────────────────────
type ReviewDecision = "unreviewed" | "accepted" | "needs-review";
type StudyStatus = "ready" | "segmenting" | "reporting";
type ThemeMode = "light" | "dark";

type RoiBox = {
  x: number;
  y: number;
  width: number;
  height: number;
};

type Segmentation = {
  id: string;
  label: string;
  confidence: number;
  volumeMl: number;
  source: string;
  box: RoiBox;
};

type TokenUsage = {
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
  source: "api" | "estimated";
};

type Study = {
  id: string;
  patientName: string;
  patientDetail: string;
  modality: string;
  bodyPart: string;
  timestamp: string;
  series: string;
  slices: number;
  uploadedFileName?: string;
  previewUrl?: string;
  isDicom?: boolean;
  doctorFeedback?: string;
  status: StudyStatus;
  reviewDecision: ReviewDecision;
  segmentations: Segmentation[];
  report: {
    summary: string;
    findings: string;
    impression: string;
    recommendation: string;
    confidence: number;
    source: string;
    usage?: TokenUsage;
  };
};

type LlmStatus = {
  provider: "mock" | "medical";
  modelConfigured: boolean;
  model: string | null;
};

type ApiSettings = {
  modelApiUrl: string;
  modelApiKey: string;
  modelId: string;
  gpuLayers: number;
  modelPath: string;
  mmprojPath: string;
};

// ── Constants ──────────────────────────────────────────────────────────────
const API_SETTINGS_STORAGE_KEY = "radiology-api-settings-v2";
const THEME_STORAGE_KEY = "radiology-theme";

const defaultApiSettings: ApiSettings = {
  modelApiUrl: IS_TAURI ? LOCAL_SERVER_URL : (import.meta.env.VITE_MODEL_API_URL ?? "http://127.0.0.1:1234/v1"),
  modelApiKey: "",
  modelId: import.meta.env.VITE_MODEL_ID ?? "medical-model",
  gpuLayers: 0,
  modelPath: "",
  mmprojPath: "",
};

const initialStudy: Study = {
  id: "NO-DICOM",
  patientName: "No DICOM loaded",
  patientDetail: "Upload a study",
  modality: "DICOM",
  bodyPart: "Study",
  timestamp: "Waiting",
  series: "None",
  slices: 0,
  status: "ready",
  reviewDecision: "unreviewed",
  segmentations: [],
  report: {
    summary: "Awaiting DICOM upload",
    findings: "Upload a DICOM file to render the image and generate a medical AI draft report.",
    impression: "No imaging study is loaded.",
    recommendation: "Use the Upload DICOM control on the left panel.",
    confidence: 0,
    source: "mock",
  },
};

// ── Settings persistence ───────────────────────────────────────────────────
function loadApiSettings(): ApiSettings {
  try {
    const saved = localStorage.getItem(API_SETTINGS_STORAGE_KEY);
    if (!saved) return defaultApiSettings;
    const parsed = JSON.parse(saved) as Partial<ApiSettings>;
    return {
      ...defaultApiSettings,
      ...parsed,
    };
  } catch {
    return defaultApiSettings;
  }
}

function saveApiSettings(settings: ApiSettings) {
  localStorage.setItem(API_SETTINGS_STORAGE_KEY, JSON.stringify(settings));
}

function loadTheme(): ThemeMode {
  const saved = localStorage.getItem(THEME_STORAGE_KEY);
  if (saved === "light" || saved === "dark") return saved;
  return window.matchMedia?.("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

// ── Utilities ──────────────────────────────────────────────────────────────
function normalizeBaseUrl(url: string) {
  return url.trim().replace(/\/+$/, "");
}

function modelHeaders(settings: ApiSettings) {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (settings.modelApiKey.trim()) {
    headers.Authorization = `Bearer ${settings.modelApiKey.trim()}`;
  }
  return headers;
}

function estimateTokens(text: string) {
  return Math.max(1, Math.ceil(text.trim().length / 4));
}

function buildEstimatedUsage(prompt: string, completion: string): TokenUsage {
  const promptTokens = estimateTokens(prompt);
  const completionTokens = estimateTokens(completion);
  return { promptTokens, completionTokens, totalTokens: promptTokens + completionTokens, source: "estimated" };
}

function formatSegmentationSource(source: string) {
  return source === "mock" ? "Local" : "Medical AI";
}

function unwrapMarkdownTag(value: string) {
  return value
    .trim()
    .replace(/^<MD>\s*/i, "")
    .replace(/\s*<\/MD>$/i, "")
    .replace(/^```(?:md|markdown)?\s*/i, "")
    .replace(/\s*```$/i, "")
    .trim();
}

function renderInlineMarkdown(text: string) {
  const parts = text.split(/(\*\*[^*]+\*\*|__[^_]+__|\*[^*]+\*|_[^_]+_|`[^`]+`)/g);
  return parts.map((part, index) => {
    if ((part.startsWith("**") && part.endsWith("**")) || (part.startsWith("__") && part.endsWith("__"))) {
      return <strong key={index}>{part.slice(2, -2)}</strong>;
    }
    if ((part.startsWith("*") && part.endsWith("*")) || (part.startsWith("_") && part.endsWith("_"))) {
      return <em key={index}>{part.slice(1, -1)}</em>;
    }
    if (part.startsWith("`") && part.endsWith("`")) {
      return <code key={index}>{part.slice(1, -1)}</code>;
    }
    return part;
  });
}

function MarkdownReportText({ content }: { content: string }) {
  const lines = unwrapMarkdownTag(content).split(/\r?\n/);
  const blocks: ReactNode[] = [];
  let index = 0;

  while (index < lines.length) {
    const line = lines[index].trim();
    if (!line) { index += 1; continue; }

    const heading = line.match(/^(#{1,6})\s+(.+)$/);
    if (heading) {
      blocks.push(
        <h4 key={blocks.length} className="text-sm font-semibold text-slate-100">
          {renderInlineMarkdown(heading[2])}
        </h4>,
      );
      index += 1;
      continue;
    }

    if (/^[-*]\s+/.test(line)) {
      const items: string[] = [];
      while (index < lines.length && /^[-*]\s+/.test(lines[index].trim())) {
        items.push(lines[index].trim().replace(/^[-*]\s+/, ""));
        index += 1;
      }
      blocks.push(
        <ul key={blocks.length} className="list-disc space-y-1 pl-5">
          {items.map((item, i) => <li key={i}>{renderInlineMarkdown(item)}</li>)}
        </ul>,
      );
      continue;
    }

    if (/^\d+\.\s+/.test(line)) {
      const items: string[] = [];
      while (index < lines.length && /^\d+\.\s+/.test(lines[index].trim())) {
        items.push(lines[index].trim().replace(/^\d+\.\s+/, ""));
        index += 1;
      }
      blocks.push(
        <ol key={blocks.length} className="list-decimal space-y-1 pl-5">
          {items.map((item, i) => <li key={i}>{renderInlineMarkdown(item)}</li>)}
        </ol>,
      );
      continue;
    }

    const paragraph: string[] = [];
    while (
      index < lines.length &&
      lines[index].trim() &&
      !/^(#{1,6})\s+/.test(lines[index].trim()) &&
      !/^[-*]\s+/.test(lines[index].trim()) &&
      !/^\d+\.\s+/.test(lines[index].trim())
    ) {
      paragraph.push(lines[index].trim());
      index += 1;
    }
    blocks.push(<p key={blocks.length}>{renderInlineMarkdown(paragraph.join(" "))}</p>);
  }

  return <div className="space-y-2 text-sm leading-6 text-slate-300">{blocks}</div>;
}

// ── API calls ──────────────────────────────────────────────────────────────
async function requestReport(study: Study, settings: ApiSettings, detectedModel: string | null) {
  const segSummary =
    study.segmentations.length > 0
      ? study.segmentations
          .map((s) => `${s.label} (confidence ${(s.confidence * 100).toFixed(0)}%, volume ${s.volumeMl} ml)`)
          .join("; ")
      : "No segmentation masks available.";

  const userContent = `You are a radiology AI assistant. Generate a structured radiology report for the following imaging study.

Modality: ${study.modality}
Body Part: ${study.bodyPart}
Patient: ${study.patientName}
Segmentation Findings: ${segSummary}

Respond in this exact JSON format:
{
  "summary": "one-line clinical summary",
  "findings": "detailed findings paragraph",
  "impression": "clinical impression",
  "recommendation": "follow-up recommendation",
  "confidence": 0.0-1.0
}`;

  const messages: Array<{ role: string; content: string | Array<Record<string, unknown>> }> = [
    { role: "system", content: "You are a medical imaging AI. Respond only with valid JSON matching the requested schema." },
    { role: "user", content: userContent },
  ];

  if (study.previewUrl) {
    let imageUrl = study.previewUrl;
    try {
      const imgResp = await fetch(study.previewUrl);
      const imgBlob = await imgResp.blob();
      const base64 = await new Promise<string>((resolve) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result as string);
        reader.readAsDataURL(imgBlob);
      });
      imageUrl = base64;
    } catch {
      // fall through with original URL
    }
    messages[1].content = [
      { type: "text", text: userContent },
      { type: "image_url", image_url: { url: imageUrl } },
    ];
  }

  // Prefer the server-detected model ID so llama-server accepts the request
  const modelId = detectedModel || settings.modelId.trim() || "medical-model";

  const response = await fetch(`${normalizeBaseUrl(settings.modelApiUrl)}/chat/completions`, {
    method: "POST",
    headers: modelHeaders(settings),
    body: JSON.stringify({ model: modelId, messages, max_tokens: 1024, temperature: 0.3 }),
  });

  if (!response.ok) {
    let detail = "Medical model API is unavailable";
    try {
      const payload = (await response.json()) as { error?: { message?: string } };
      if (payload.error?.message) detail = payload.error.message;
    } catch { /* keep generic error */ }
    throw new Error(detail);
  }

  const completion = (await response.json()) as {
    choices: Array<{ message: { content: string } }>;
    usage?: { prompt_tokens?: number; completion_tokens?: number; total_tokens?: number };
  };

  const raw = completion.choices?.[0]?.message?.content ?? "";
  let parsed: Record<string, unknown>;
  try {
    const jsonMatch = raw.match(/\{[\s\S]*\}/);
    parsed = jsonMatch ? JSON.parse(jsonMatch[0]) : {};
  } catch {
    parsed = {};
  }

  const usage =
    typeof completion.usage?.prompt_tokens === "number" || typeof completion.usage?.completion_tokens === "number"
      ? {
          promptTokens: completion.usage.prompt_tokens ?? 0,
          completionTokens: completion.usage.completion_tokens ?? 0,
          totalTokens: completion.usage.total_tokens ?? (completion.usage.prompt_tokens ?? 0) + (completion.usage.completion_tokens ?? 0),
          source: "api" as const,
        }
      : buildEstimatedUsage(userContent, raw);

  return {
    summary: (parsed.summary as string) ?? "Medical AI draft generated",
    findings: (parsed.findings as string) ?? raw,
    impression: (parsed.impression as string) ?? "See findings above.",
    recommendation: (parsed.recommendation as string) ?? "Clinical correlation recommended.",
    confidence: typeof parsed.confidence === "number" ? Math.min(1, Math.max(0, parsed.confidence)) : 0.5,
    source: "medical",
    usage,
  };
}

async function requestLlmStatus(settings: ApiSettings) {
  const response = await fetch(`${normalizeBaseUrl(settings.modelApiUrl)}/models`, {
    headers: settings.modelApiKey.trim() ? { Authorization: `Bearer ${settings.modelApiKey.trim()}` } : undefined,
  });
  if (!response.ok) throw new Error();
  const data = (await response.json()) as { data: Array<{ id: string }> };
  const modelIds = data.data?.map((m) => m.id) ?? [];
  const configuredModel = settings.modelId.trim();
  const matchingModel = modelIds.find((id) => id === configuredModel) ?? modelIds[0];
  return {
    provider: Boolean(matchingModel) ? "medical" as const : "mock" as const,
    modelConfigured: Boolean(matchingModel),
    model: matchingModel ?? null,
  } satisfies LlmStatus;
}

// ── Custom hooks ───────────────────────────────────────────────────────────
function useLlmStatus(settings: ApiSettings, localAi: LocalAiStatus, setLlmStatus: (s: LlmStatus) => void) {
  useEffect(() => {
    // Don't poll until the embedded server is ready (in Tauri mode)
    if (IS_TAURI && localAi.phase !== "ready") return;

    let cancelled = false;
    requestLlmStatus(settings)
      .then((status) => { if (!cancelled) setLlmStatus(status); })
      .catch(() => { if (!cancelled) setLlmStatus({ provider: "mock", modelConfigured: false, model: null }); });
    return () => { cancelled = true; };
  }, [settings, localAi.phase, setLlmStatus]);
}


// ── App ────────────────────────────────────────────────────────────────────
function App() {
  const [studies, setStudies] = useState<Study[]>([]);
  const [activeStudyId, setActiveStudyId] = useState<string | null>(null);
  const [segmentVisible, setSegmentVisible] = useState(true);
  const [zoom, setZoom] = useState(100);
  const [brightness, setBrightness] = useState(100);
  const [llmStatus, setLlmStatus] = useState<LlmStatus>({ provider: "mock", modelConfigured: false, model: null });
  const [isDragging, setIsDragging] = useState(false);
  const [feedbackOpen, setFeedbackOpen] = useState(false);
  const [feedbackText, setFeedbackText] = useState("");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [studyPanelOpen, setStudyPanelOpen] = useState(true);
  const [structurePreviewOpen, setStructurePreviewOpen] = useState(false);
  const [apiSettings, setApiSettings] = useState<ApiSettings>(() => loadApiSettings());
  const [theme, setTheme] = useState<ThemeMode>(() => loadTheme());

  const [undoStacks, setUndoStacks] = useState<Record<string, Segmentation[][]>>({});
  const [redoStacks, setRedoStacks] = useState<Record<string, Segmentation[][]>>({});

  const study = studies.find((c) => c.id === activeStudyId) ?? studies[0] ?? initialStudy;

  // Embedded AI engine lifecycle
  const [localAi, setLocalAi] = useState<LocalAiStatus>({
    phase: IS_TAURI ? (apiSettings.modelPath && apiSettings.mmprojPath ? "starting" : "error") : "idle",
    error: IS_TAURI && !(apiSettings.modelPath && apiSettings.mmprojPath) ? "not-configured" : undefined,
  });

  useEffect(() => {
    if (!IS_TAURI) return;
    if (!apiSettings.modelPath || !apiSettings.mmprojPath) {
      setLocalAi({ phase: "error", error: "not-configured" });
      return;
    }

    setLocalAi({ phase: "starting" });
    let unlisten: (() => void) | undefined;

    const start = async () => {
      try {
        const { invoke } = await import("@tauri-apps/api/core");
        const { listen } = await import("@tauri-apps/api/event");
        unlisten = await listen<{ running?: boolean; modelId?: string; error?: string }>(
          "llama-server-ready",
          (event) => {
            if (event.payload.error) {
              setLocalAi({ phase: "error", error: event.payload.error });
            } else {
              const modelId = event.payload.modelId ?? null;
              setLocalAi({ phase: "ready", modelId: modelId ?? undefined });
              if (modelId) setApiSettings((prev) => ({ ...prev, modelId }));
            }
          },
        );
        await invoke("start_llama_server", {
          modelPath: apiSettings.modelPath,
          mmprojPath: apiSettings.mmprojPath,
          gpuLayers: apiSettings.gpuLayers,
        });
      } catch (err) {
        setLocalAi({ phase: "error", error: String(err) });
      }
    };

    start();
    return () => { unlisten?.(); };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [apiSettings.modelPath, apiSettings.mmprojPath, apiSettings.gpuLayers]);

  // Use local server URL automatically in Tauri mode
  const effectiveSettings = useMemo((): ApiSettings => {
    if (IS_TAURI) {
      return { ...apiSettings, modelApiUrl: LOCAL_SERVER_URL };
    }
    return apiSettings;
  }, [apiSettings]);

  useLlmStatus(effectiveSettings, localAi, setLlmStatus);

  const pushUndo = useCallback((studyId: string, segmentations: Segmentation[]) => {
    setUndoStacks((prev) => ({ ...prev, [studyId]: [...(prev[studyId] ?? []), segmentations] }));
    setRedoStacks((prev) => ({ ...prev, [studyId]: [] }));
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem(THEME_STORAGE_KEY, theme);
  }, [theme]);

  const updateStudy = (updater: (s: Study) => Study) => {
    setStudies((current) => current.map((c) => (c.id === study.id ? updater(c) : c)));
  };

  const buildFallbackStudy = (file: File): Study => {
    const isDicom = file.name.toLowerCase().endsWith(".dcm") || file.type === "application/dicom";
    const canPreview = file.type.startsWith("image/");
    const previewUrl = canPreview ? URL.createObjectURL(file) : undefined;
    return {
      id: `LOCAL-${Date.now()}`,
      patientName: "Uploaded Study",
      patientDetail: "Local file",
      modality: isDicom ? "DICOM" : "IMG",
      bodyPart: "Unspecified",
      timestamp: "Just now",
      series: isDicom ? "DICOM series" : "Image preview",
      slices: 1,
      uploadedFileName: file.name,
      previewUrl,
      isDicom,
      status: "ready",
      reviewDecision: "unreviewed",
      segmentations: [],
      report: {
        summary: "Awaiting AI review",
        findings: "The study is loaded. Run Medical AI report generation.",
        impression: "Pending AI draft and clinician review.",
        recommendation: "Select an ROI for segmentation if a suspicious region is present.",
        confidence: 0,
        source: "mock",
      },
    };
  };

  const addStudy = (nextStudy: Study) => {
    setStudies((current) => [nextStudy, ...current.filter((c) => c.id !== nextStudy.id)]);
    setActiveStudyId(nextStudy.id);
    setSegmentVisible(true);
  };

  const handleStudyFiles = async (files: File[]) => {
    for (const file of files) {
      addStudy(buildFallbackStudy(file));
    }
  };

  const runSegmentation = async (prompt: RoiBox = { x: 0.47, y: 0.34, width: 0.16, height: 0.22 }) => {
    pushUndo(study.id, study.segmentations);
    updateStudy((s) => ({ ...s, status: "segmenting" }));
    setSegmentVisible(true);

    // Simulated ROI segmentation (no separate segmentation backend needed)
    await new Promise((r) => setTimeout(r, 800));
    const segmentation: Segmentation = {
      id: `seg-${Date.now()}`,
      label: study.modality === "DICOM" ? "Prompted DICOM ROI" : "Prompted Image ROI",
      confidence: 0.79 + Math.random() * 0.12,
      volumeMl: Math.round(prompt.width * prompt.height * 1200) / 10,
      source: "mock",
      box: prompt,
    };
    updateStudy((s) => ({
      ...s,
      status: "ready",
      segmentations: [segmentation, ...s.segmentations],
    }));
  };

  const runReport = async () => {
    updateStudy((s) => ({ ...s, status: "reporting" }));
    try {
      const report = await requestReport(study, effectiveSettings, llmStatus.model);
      updateStudy((s) => ({ ...s, status: "ready", report }));
    } catch (error) {
      updateStudy((s) => ({
        ...s,
        status: "ready",
        report: {
          summary: "Medical AI draft unavailable",
          findings:
            s.segmentations.length > 0
              ? `AI draft based on ${s.segmentations.length} segmentation ROI(s). ${error instanceof Error ? error.message : "Embedded AI service is unavailable."}`
              : error instanceof Error
                ? error.message
                : "No segmentation mask has been generated yet.",
          impression: "Preliminary decision support only.",
          recommendation: IS_TAURI
            ? localAi.phase === "starting"
              ? "The embedded AI engine is still loading. Please wait and try again."
              : "The embedded AI engine encountered an error. Check the app is fully loaded."
            : `Ensure the medical AI model is running at ${apiSettings.modelApiUrl}, then generate again.`,
          confidence: s.segmentations.length > 0 ? 0.72 : 0.38,
          source: "medical",
          usage: buildEstimatedUsage("report request", error instanceof Error ? error.message : "error"),
        },
      }));
    }
  };

  const handleUndo = () => {
    const stack = undoStacks[study.id];
    if (!stack?.length) return;
    const previous = stack[stack.length - 1];
    setUndoStacks((prev) => ({ ...prev, [study.id]: stack.slice(0, -1) }));
    setRedoStacks((prev) => ({ ...prev, [study.id]: [...(prev[study.id] ?? []), study.segmentations] }));
    updateStudy((s) => ({ ...s, segmentations: previous }));
  };

  const handleRedo = () => {
    const stack = redoStacks[study.id];
    if (!stack?.length) return;
    const next = stack[stack.length - 1];
    setRedoStacks((prev) => ({ ...prev, [study.id]: stack.slice(0, -1) }));
    setUndoStacks((prev) => ({ ...prev, [study.id]: [...(prev[study.id] ?? []), study.segmentations] }));
    updateStudy((s) => ({ ...s, segmentations: next }));
  };

  const handleClear = () => {
    if (study.segmentations.length === 0) return;
    pushUndo(study.id, study.segmentations);
    updateStudy((s) => ({ ...s, segmentations: [] }));
  };

  const canUndo = (undoStacks[study.id]?.length ?? 0) > 0;
  const canRedo = (redoStacks[study.id]?.length ?? 0) > 0;

  return (
    <main className="clinical-shell h-screen overflow-hidden bg-[#05070b] text-slate-100" data-theme={theme}>
      {/* Loading overlay while embedded AI engine initialises */}
      {IS_TAURI && localAi.phase === "starting" && (
        <div className="fixed inset-0 z-50 grid place-items-center bg-[#05070b]/95 backdrop-blur-sm">
          <div className="text-center">
            <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full border border-cyan-400/25 bg-cyan-400/10">
              <Cpu className="h-8 w-8 animate-pulse text-cyan-300" />
            </div>
            <p className="text-base font-semibold text-white">Loading Embedded AI Engine</p>
            <p className="mt-2 text-sm text-slate-400">Initialising Med model…</p>
            <p className="mt-1 text-xs text-slate-600">This may take 20-60 seconds on first launch</p>
          </div>
        </div>
      )}

      {/* Error / setup overlay */}
      {IS_TAURI && localAi.phase === "error" && !settingsOpen && (
        <div className="fixed inset-0 z-50 grid place-items-center bg-[#05070b]/95 backdrop-blur-sm">
          {localAi.error === "not-configured" ? (
            <div className="w-full max-w-md rounded-md border border-cyan-400/30 bg-[#0c1420] p-6 text-center">
              <Cpu className="mx-auto mb-4 h-10 w-10 text-cyan-400" />
              <p className="text-base font-semibold text-white">Model paths not configured</p>
              <p className="mt-2 text-sm text-slate-400">
                Select the two GGUF files to start the embedded AI engine.
              </p>
              <button
                className="mt-5 inline-flex items-center gap-2 rounded-md border border-cyan-400/25 bg-cyan-400/10 px-5 py-2.5 text-sm font-semibold text-cyan-100 transition hover:bg-cyan-400/15"
                type="button"
                onClick={() => setSettingsOpen(true)}
              >
                <Settings className="h-4 w-4" />
                Open Settings
              </button>
            </div>
          ) : (
            <div className="w-full max-w-md rounded-md border border-red-400/30 bg-[#0c1420] p-6 text-center">
              <XCircle className="mx-auto mb-4 h-10 w-10 text-red-400" />
              <p className="text-base font-semibold text-white">AI Engine Failed to Start</p>
              <p className="mt-2 text-sm text-slate-400">{localAi.error}</p>
              <button
                className="mt-4 inline-flex items-center gap-2 rounded-md border border-slate-700 bg-slate-950 px-4 py-2 text-sm text-slate-200 transition hover:bg-slate-800"
                type="button"
                onClick={() => setSettingsOpen(true)}
              >
                <Settings className="h-4 w-4" />
                Settings
              </button>
            </div>
          )}
        </div>
      )}

      <div
        className={`relative grid h-full overflow-hidden transition-[grid-template-columns] duration-300 ease-out max-lg:grid-cols-1 max-lg:overflow-y-auto ${
          studyPanelOpen
            ? "grid-cols-[280px_minmax(0,1fr)_390px] max-xl:grid-cols-[240px_minmax(0,1fr)_360px]"
            : "grid-cols-[0_minmax(0,1fr)_390px] max-xl:grid-cols-[0_minmax(0,1fr)_360px]"
        }`}
      >
        <div className="relative min-h-0 overflow-visible">
          <StudyPanel
            activeStudyId={study.id}
            open={studyPanelOpen}
            studies={studies}
            onAccept={() => updateStudy((s) => ({ ...s, reviewDecision: "accepted" }))}
            onFiles={handleStudyFiles}
            onNeedsReview={() => {
              updateStudy((s) => ({ ...s, reviewDecision: "needs-review" }));
              setFeedbackText(study.doctorFeedback ?? "");
              setFeedbackOpen(true);
            }}
            onSelectStudy={setActiveStudyId}
            onToggleOpen={() => setStudyPanelOpen((o) => !o)}
          />
        </div>

        {!studyPanelOpen && (
          <button
            className="absolute left-3 top-3 z-30 grid h-9 w-9 place-items-center rounded-md border border-cyan-300/35 bg-slate-950/95 text-cyan-100 shadow-panel-soft transition hover:border-cyan-200 hover:bg-cyan-400/10"
            type="button"
            title="Show study panel"
            onClick={() => setStudyPanelOpen(true)}
          >
            <PanelLeftOpen className="h-4 w-4" />
          </button>
        )}

        <ViewerWorkspace
          brightness={brightness}
          canRedo={canRedo}
          canUndo={canUndo}
          isDragging={isDragging}
          segmentVisible={segmentVisible}
          study={study}
          zoom={zoom}
          theme={theme}
          onClear={handleClear}
          onDragStateChange={setIsDragging}
          onBrightnessChange={setBrightness}
          onRedo={handleRedo}
          onRunSegmentation={runSegmentation}
          onSegmentVisibleChange={setSegmentVisible}
          onStructurePreviewOpen={() => setStructurePreviewOpen(true)}
          onThemeChange={setTheme}
          onUndo={handleUndo}
          onZoomChange={setZoom}
        />

        <DecisionPanel
          llmStatus={llmStatus}
          localAi={localAi}
          study={study}
          onOpenSettings={() => setSettingsOpen(true)}
          onRunReport={runReport}
        />
      </div>

      {settingsOpen && (
        <ApiSettingsModal
          settings={apiSettings}
          localAi={localAi}
          onClose={() => setSettingsOpen(false)}
          onSave={(next) => {
            const normalized: ApiSettings = {
              modelApiUrl: IS_TAURI ? LOCAL_SERVER_URL : (normalizeBaseUrl(next.modelApiUrl) || defaultApiSettings.modelApiUrl),
              modelApiKey: next.modelApiKey.trim(),
              modelId: next.modelId.trim() || defaultApiSettings.modelId,
              gpuLayers: Math.max(0, Math.floor(next.gpuLayers)),
              modelPath: next.modelPath.trim(),
              mmprojPath: next.mmprojPath.trim(),
            };
            saveApiSettings(normalized);
            setApiSettings(normalized);
            setSettingsOpen(false);
          }}
        />
      )}

      {feedbackOpen && (
        <FeedbackModal
          feedbackText={feedbackText}
          study={study}
          onClose={() => { setFeedbackOpen(false); setFeedbackText(""); }}
          onFeedbackTextChange={setFeedbackText}
          onSave={() => {
            const saved = feedbackText.trim();
            updateStudy((s) => ({
              ...s,
              doctorFeedback: saved,
              reviewDecision: saved ? "needs-review" : s.reviewDecision,
            }));
            setFeedbackOpen(false);
            setFeedbackText("");
          }}
        />
      )}

      {structurePreviewOpen && <StructurePreviewModal study={study} onClose={() => setStructurePreviewOpen(false)} />}
    </main>
  );
}

// ── StudyPanel ─────────────────────────────────────────────────────────────
type StudyPanelProps = {
  studies: Study[];
  activeStudyId: string;
  open: boolean;
  onAccept: () => void;
  onFiles: (files: File[]) => void;
  onNeedsReview: () => void;
  onSelectStudy: (id: string) => void;
  onToggleOpen: () => void;
};

function StudyPanel({ studies, activeStudyId, open, onAccept, onFiles, onNeedsReview, onSelectStudy, onToggleOpen }: StudyPanelProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileInput = (event: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files ?? []);
    if (files.length > 0) { void onFiles(files); event.target.value = ""; }
  };

  return (
    <aside
      className={`flex h-full min-h-0 w-[280px] flex-col border-r border-slate-800 bg-[#080d15] transition-transform duration-300 ease-out max-xl:w-[240px] max-lg:min-h-[360px] ${
        open ? "translate-x-0" : "pointer-events-none -translate-x-full"
      }`}
      aria-hidden={!open}
    >
      <header className="border-b border-slate-800 px-4 py-4">
        <div className="mb-3 flex items-center justify-between gap-3">
          <p className="text-sm font-semibold text-white">Studies</p>
          <button
            className="grid h-8 w-8 place-items-center rounded-md border border-slate-700 bg-slate-950 text-slate-400 transition hover:border-cyan-300/45 hover:text-cyan-100"
            type="button"
            title="Hide study panel"
            onClick={onToggleOpen}
          >
            <PanelLeftClose className="h-4 w-4" />
          </button>
        </div>
        <input ref={fileInputRef} className="hidden" type="file" multiple accept=".dcm,image/png,image/jpeg,image/webp" onChange={handleFileInput} />
        <button
          className="flex w-full items-center justify-center gap-2 rounded-md border border-cyan-400/25 bg-cyan-400/10 px-3 py-2 text-sm font-semibold text-cyan-100 transition hover:bg-cyan-400/15"
          type="button"
          onClick={() => fileInputRef.current?.click()}
        >
          <FileImage className="h-4 w-4" />
          Upload DICOM / Image
        </button>
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto p-3">
        {studies.length === 0 ? (
          <article className="rounded-md border border-slate-800 bg-[#0c1420] p-3 text-sm leading-6 text-slate-400">
            No study uploaded yet.
          </article>
        ) : (
          <div className="space-y-2">
            {studies.map((s) => (
              <button
                key={s.id}
                className={`w-full rounded-md border p-3 text-left transition ${
                  activeStudyId === s.id
                    ? "border-cyan-300/45 bg-cyan-400/10 shadow-active-glow"
                    : "border-slate-800 bg-[#0c1420] hover:border-slate-600"
                }`}
                type="button"
                onClick={() => onSelectStudy(s.id)}
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <p className="truncate text-sm font-semibold text-white">{s.patientName}</p>
                    <p className="mt-1 truncate text-xs text-slate-500">{s.patientDetail}</p>
                  </div>
                  <StatusDot decision={s.reviewDecision} />
                </div>
                <div className="mt-3 flex items-center justify-between gap-2 text-xs">
                  <span className="rounded border border-slate-700 bg-slate-950 px-2 py-1 text-slate-300">{s.modality} {s.bodyPart}</span>
                  <span className="text-slate-500">{s.timestamp}</span>
                </div>
                <div className="mt-3 border-t border-slate-800 pt-3 text-xs leading-5 text-slate-400">
                  <p className="truncate">{s.series}</p>
                  <p className="truncate">{s.slices} slice{s.slices === 1 ? "" : "s"}</p>
                  <p className="truncate">{s.uploadedFileName ?? s.id}</p>
                </div>
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="grid shrink-0 gap-3 border-t border-slate-800 bg-[#090f18] p-4">
        <button className="flex w-full items-center justify-center gap-2 rounded-md bg-emerald-600 px-4 py-3 text-sm font-semibold text-white transition hover:bg-emerald-500" type="button" onClick={onAccept}>
          <CheckCircle2 className="h-4 w-4" />
          Accept
        </button>
        <button className="flex w-full items-center justify-center gap-2 rounded-md bg-red-600 px-4 py-3 text-sm font-semibold text-white transition hover:bg-red-500" type="button" onClick={onNeedsReview}>
          <XCircle className="h-4 w-4" />
          Note
        </button>
      </div>
    </aside>
  );
}

// ── ViewerWorkspace ────────────────────────────────────────────────────────
type ViewerWorkspaceProps = {
  study: Study;
  brightness: number;
  segmentVisible: boolean;
  isDragging: boolean;
  zoom: number;
  canUndo: boolean;
  canRedo: boolean;
  theme: ThemeMode;
  onSegmentVisibleChange: (v: boolean) => void;
  onDragStateChange: (d: boolean) => void;
  onThemeChange: (t: ThemeMode) => void;
  onBrightnessChange: (b: number) => void;
  onZoomChange: (z: number) => void;
  onRunSegmentation: (prompt?: RoiBox) => void;
  onStructurePreviewOpen: () => void;
  onUndo: () => void;
  onRedo: () => void;
  onClear: () => void;
};

function ViewerWorkspace({
  study, brightness, segmentVisible, isDragging, zoom, canUndo, canRedo, theme,
  onSegmentVisibleChange, onDragStateChange, onThemeChange, onBrightnessChange,
  onZoomChange, onRunSegmentation, onStructurePreviewOpen, onUndo, onRedo, onClear,
}: ViewerWorkspaceProps) {
  const viewportRef = useRef<HTMLDivElement>(null);
  const [draftRoi, setDraftRoi] = useState<RoiBox | null>(null);
  const [roiStart, setRoiStart] = useState<{ x: number; y: number } | null>(null);
  const windowLevel = useMemo(() => (study.modality === "CT" || study.modality === "DICOM" ? "W: 420 L: 38" : "Auto WL"), [study.modality]);

  const getPoint = (event: MouseEvent<HTMLDivElement>) => {
    const rect = viewportRef.current?.getBoundingClientRect();
    if (!rect) return { x: 0, y: 0 };
    return { x: clamp((event.clientX - rect.left) / rect.width), y: clamp((event.clientY - rect.top) / rect.height) };
  };

  const handleRoiStart = (event: MouseEvent<HTMLDivElement>) => {
    event.preventDefault();
    const point = getPoint(event);
    setRoiStart(point);
    setDraftRoi({ x: point.x, y: point.y, width: 0.01, height: 0.01 });
  };

  const handleRoiMove = (event: MouseEvent<HTMLDivElement>) => {
    if (!roiStart) return;
    event.preventDefault();
    setDraftRoi(normalizeRoi(roiStart, getPoint(event)));
  };

  const handleRoiEnd = () => {
    if (!draftRoi || !roiStart) return;
    setRoiStart(null);
    const roi = draftRoi.width < 0.025 || draftRoi.height < 0.025 ? null : draftRoi;
    setDraftRoi(null);
    if (roi) onRunSegmentation(roi);
  };

  useEffect(() => {
    if (!roiStart) return;
    const handleGlobalUp = () => handleRoiEnd();
    window.addEventListener("mouseup", handleGlobalUp);
    return () => window.removeEventListener("mouseup", handleGlobalUp);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [roiStart, draftRoi]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key === "z" && !event.shiftKey) {
        event.preventDefault();
        if (canUndo) onUndo();
      } else if ((event.ctrlKey || event.metaKey) && (event.key === "y" || (event.key === "z" && event.shiftKey))) {
        event.preventDefault();
        if (canRedo) onRedo();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [canUndo, canRedo, onUndo, onRedo]);

  return (
    <section className="flex min-h-0 flex-col bg-[#05070b]">
      <header className="flex shrink-0 flex-wrap items-center justify-between gap-3 border-b border-slate-800 bg-[#090f18] px-4 py-3">
        <div className="min-w-0">
          <p className="truncate text-sm font-semibold text-white">Clinical Imaging Console</p>
          <p className="mt-1 truncate text-xs text-slate-500">{study.patientName} / {study.modality} {study.bodyPart}</p>
          <p className="mt-1 truncate text-[11px] uppercase tracking-[0.12em] text-slate-500">
            {study.series} • {study.slices} slice{study.slices === 1 ? "" : "s"} • {study.uploadedFileName ?? study.id}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <ThemeSwitcher theme={theme} onThemeChange={onThemeChange} />
          <div className="mx-1 h-6 w-px bg-slate-700" />
          <ToolbarButton icon={study.status === "segmenting" ? Loader2 : Brain} label="Medical AI" loading={study.status === "segmenting"} onClick={onRunSegmentation} />
          <button
            className="inline-flex h-9 items-center gap-2 rounded-md border border-slate-700 bg-slate-950 px-3 text-sm font-medium text-slate-200 transition hover:border-cyan-300/50 hover:text-cyan-100"
            type="button"
            onClick={onStructurePreviewOpen}
          >
            <Layers className="h-4 w-4" />
            3D
          </button>
          <button
            className={`inline-flex h-9 items-center gap-2 rounded-md border px-3 text-sm font-medium transition ${
              segmentVisible ? "border-cyan-300/40 bg-cyan-400/10 text-cyan-100" : "border-slate-700 bg-slate-950 text-slate-400"
            }`}
            type="button"
            onClick={() => onSegmentVisibleChange(!segmentVisible)}
          >
            <Layers className="h-4 w-4" />
            Seg
          </button>
          <div className="mx-1 h-6 w-px bg-slate-700" />
          <button
            className={`inline-flex h-9 items-center gap-1.5 rounded-md border px-2.5 text-xs font-medium transition ${
              canUndo ? "border-slate-700 bg-slate-950 text-slate-200 hover:border-cyan-300/40" : "border-slate-800 bg-slate-950/50 text-slate-600 cursor-not-allowed"
            }`}
            type="button"
            disabled={!canUndo}
            title="Undo (Ctrl+Z)"
            onClick={onUndo}
          >
            <Undo2 className="h-3.5 w-3.5" /> Undo
          </button>
          <button
            className={`inline-flex h-9 items-center gap-1.5 rounded-md border px-2.5 text-xs font-medium transition ${
              canRedo ? "border-slate-700 bg-slate-950 text-slate-200 hover:border-cyan-300/40" : "border-slate-800 bg-slate-950/50 text-slate-600 cursor-not-allowed"
            }`}
            type="button"
            disabled={!canRedo}
            title="Redo (Ctrl+Y)"
            onClick={onRedo}
          >
            <Redo2 className="h-3.5 w-3.5" /> Redo
          </button>
          <button
            className={`inline-flex h-9 items-center gap-1.5 rounded-md border px-2.5 text-xs font-medium transition ${
              study.segmentations.length > 0 ? "border-red-400/30 bg-red-400/10 text-red-200 hover:border-red-300/50" : "border-slate-800 bg-slate-950/50 text-slate-600 cursor-not-allowed"
            }`}
            type="button"
            disabled={study.segmentations.length === 0}
            onClick={onClear}
          >
            <Trash2 className="h-3.5 w-3.5" /> Clear
          </button>
        </div>
      </header>

      <div className="flex shrink-0 flex-wrap items-center justify-between gap-3 border-b border-slate-800 bg-[#070b12] px-4 py-2">
        <div className="flex min-w-[220px] items-center gap-3 text-xs text-slate-300">
          <button className="grid h-7 w-7 place-items-center rounded-md border border-slate-800 bg-slate-950 text-slate-400 transition hover:border-amber-300/40 hover:text-amber-100" type="button" onClick={() => onBrightnessChange(Math.max(45, brightness - 10))}>
            <Moon className="h-4 w-4" />
          </button>
          <input className="h-1.5 w-full cursor-pointer accent-amber-300" type="range" min="45" max="180" step="5" value={brightness} onChange={(e) => onBrightnessChange(Number(e.target.value))} />
          <button className="grid h-7 w-7 place-items-center rounded-md border border-slate-800 bg-slate-950 text-slate-400 transition hover:border-amber-300/40 hover:text-amber-100" type="button" onClick={() => onBrightnessChange(Math.min(180, brightness + 10))}>
            <Sun className="h-4 w-4" />
          </button>
          <span className="w-11 text-right font-medium">{brightness}%</span>
        </div>
        <div className="flex min-w-[220px] items-center gap-3 text-xs text-slate-300">
          <ZoomOut className="h-4 w-4 text-slate-500" />
          <input className="h-1.5 w-full cursor-pointer accent-cyan-400" type="range" min="70" max="180" step="5" value={zoom} onChange={(e) => onZoomChange(Number(e.target.value))} />
          <ZoomIn className="h-4 w-4 text-slate-500" />
          <span className="w-10 text-right font-medium">{zoom}%</span>
        </div>
      </div>

      <div className="min-h-0 flex-1 p-4">
        <div
          ref={viewportRef}
          className={`relative h-full min-h-[520px] select-none overflow-hidden rounded-md border bg-black transition ${
            isDragging ? "border-cyan-300 ring-4 ring-cyan-400/20" : "border-slate-800"
          } cursor-crosshair`}
          onMouseDown={handleRoiStart}
          onMouseMove={handleRoiMove}
          onMouseUp={handleRoiEnd}
          onDragEnter={(e) => { e.preventDefault(); onDragStateChange(true); }}
          onDragLeave={() => onDragStateChange(false)}
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => { e.preventDefault(); onDragStateChange(false); }}
        >
          {study.previewUrl ? (
            <img
              className="pointer-events-none absolute inset-0 h-full w-full object-contain transition-transform duration-200"
              src={study.previewUrl}
              alt={`${study.modality} ${study.bodyPart} preview`}
              style={{ filter: `brightness(${brightness}%)`, transform: `scale(${zoom / 100})` }}
            />
          ) : (
            <div
              className="medical-scan pointer-events-none absolute inset-0 transition-transform duration-200"
              style={{ filter: `grayscale(1) contrast(1.08) brightness(${brightness * 0.9}%)`, transform: `scale(${zoom / 100})` }}
            />
          )}

          <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_center,transparent_43%,rgba(0,0,0,0.62)_100%)]" />
          <ViewportOverlay study={study} windowLevel={windowLevel} />

          {segmentVisible && study.segmentations.length > 0 && <SegmentationOverlay segmentations={study.segmentations} />}
          {draftRoi && <RoiBoxOverlay box={draftRoi} label="ROI prompt" className="border-amber-200 bg-amber-300/10 shadow-[0_0_28px_rgba(252,211,77,0.25)]" />}

          {isDragging && (
            <div className="absolute inset-0 grid place-items-center bg-cyan-950/45 backdrop-blur-sm">
              <div className="rounded-md border border-cyan-300/50 bg-slate-950/85 px-6 py-5 text-center shadow-active-glow">
                <Upload className="mx-auto h-7 w-7 text-cyan-200" />
                <p className="mt-2 text-sm font-semibold text-white">Drop DICOM or image study</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

// ── DecisionPanel ──────────────────────────────────────────────────────────
type DecisionPanelProps = {
  study: Study;
  llmStatus: LlmStatus;
  localAi: LocalAiStatus;
  onRunReport: () => void;
  onOpenSettings: () => void;
};

function DecisionPanel({ study, llmStatus, localAi, onRunReport, onOpenSettings }: DecisionPanelProps) {
  return (
    <aside className="flex min-h-0 flex-col border-l border-slate-800 bg-[#080d15] max-lg:min-h-[640px]">
      <header className="flex shrink-0 items-center justify-between border-b border-slate-800 bg-[#090f18] px-4 py-3">
        <div className="min-w-0">
          <p className="text-sm font-semibold text-white">AI Review</p>
          <p className="mt-1 truncate text-xs text-slate-500">Embedded Model · offline capable</p>
        </div>
        <div className="flex items-center gap-2">
          {IS_TAURI && (
            <div
              className={`flex h-7 items-center gap-1.5 rounded-md border px-2 text-xs font-medium ${
                localAi.phase === "ready"
                  ? "border-emerald-400/30 bg-emerald-400/10 text-emerald-300"
                  : localAi.phase === "starting"
                    ? "border-amber-400/30 bg-amber-400/10 text-amber-300"
                    : "border-red-400/30 bg-red-400/10 text-red-300"
              }`}
            >
              <Cpu className={`h-3 w-3 ${localAi.phase === "starting" ? "animate-pulse" : ""}`} />
              {localAi.phase === "ready" ? "AI Ready" : localAi.phase === "starting" ? "Loading…" : "AI Error"}
            </div>
          )}
          <button
            className="grid h-9 w-9 place-items-center rounded-md border border-slate-700 bg-slate-950 text-slate-300 transition hover:border-cyan-300/45 hover:text-cyan-100"
            title="Settings"
            type="button"
            onClick={onOpenSettings}
          >
            <Settings className="h-4 w-4" />
          </button>
        </div>
      </header>

      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-4">
        <ActionCard
          icon={study.status === "reporting" ? Loader2 : Sparkles}
          title="Medical AI Report"
          actionLabel={study.status === "reporting" ? "Generating" : "Generate"}
          loading={study.status === "reporting"}
          onAction={onRunReport}
        >
          <p className={`mb-3 text-xs font-medium ${llmStatus.modelConfigured ? "text-emerald-300" : "text-amber-300"}`}>
            {llmStatus.modelConfigured
              ? `Model: ${llmStatus.model}`
              : IS_TAURI
                ? localAi.phase === "starting"
                  ? "Embedded AI engine is loading, please wait…"
                  : "Embedded AI unavailable. Check engine status."
                : "No model connected. Check API settings."}
          </p>
          <p className="text-sm leading-6 text-slate-300">{study.report.summary}</p>
          <div className="mt-3">
            <div className="mb-2 flex items-center justify-between text-xs">
              <span className="text-slate-500">Draft confidence</span>
              <span className="font-semibold text-cyan-100">{Math.round(study.report.confidence * 100)}%</span>
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-slate-800">
              <div className="h-full rounded-full bg-gradient-to-r from-cyan-400 to-emerald-400" style={{ width: `${study.report.confidence * 100}%` }} />
            </div>
          </div>
          <TokenUsageChart usage={study.report.usage} />
        </ActionCard>

        <ReportSection icon={FileImage} title="Findings">
          <MarkdownReportText content={study.report.findings} />
        </ReportSection>
        <ReportSection icon={Brain} title="Impression">{study.report.impression}</ReportSection>
        <ReportSection icon={ShieldCheck} title="Recommendation">{study.report.recommendation}</ReportSection>

        <article className="rounded-md border border-slate-800 bg-[#0c1420] p-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-white">Segmentation Masks</h3>
            <span className="text-xs text-slate-500">{study.segmentations.length}</span>
          </div>
          <div className="mt-3 space-y-2">
            {study.segmentations.length === 0 ? (
              <p className="text-sm leading-6 text-slate-400">No mask yet. Drag on the image to choose an ROI, or click Medical AI.</p>
            ) : (
              study.segmentations.map((seg) => (
                <div key={seg.id} className="rounded-md border border-slate-800 bg-slate-950/45 px-3 py-2">
                  <div className="flex items-center justify-between gap-3">
                    <span className="text-sm font-medium text-slate-100">{seg.label}</span>
                    <span className="text-xs text-cyan-200">{Math.round(seg.confidence * 100)}%</span>
                  </div>
                  <p className="mt-1 text-xs text-slate-500">{seg.volumeMl} ml · {formatSegmentationSource(seg.source)}</p>
                </div>
              ))
            )}
          </div>
          {study.doctorFeedback?.trim() && (
            <div className="mt-4 rounded-md border border-amber-300/25 bg-amber-400/10 px-3 py-2">
              <h4 className="text-xs font-semibold uppercase tracking-[0.12em] text-amber-200">Doctor Feedback</h4>
              <p className="mt-2 whitespace-pre-wrap text-sm leading-6 text-slate-200">{study.doctorFeedback}</p>
            </div>
          )}
        </article>
      </div>
    </aside>
  );
}

// ── ApiSettingsModal ───────────────────────────────────────────────────────
function ApiSettingsModal({
  settings,
  localAi,
  onClose,
  onSave,
}: {
  settings: ApiSettings;
  localAi: LocalAiStatus;
  onClose: () => void;
  onSave: (s: ApiSettings) => void;
}) {
  const [draft, setDraft] = useState<ApiSettings>(settings);
  const update = (key: keyof ApiSettings, value: string | number) =>
    setDraft((d) => ({ ...d, [key]: value }));

  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-black/62 p-4 backdrop-blur-sm">
      <section className="w-full max-w-xl rounded-md border border-cyan-300/30 bg-[#0c1420] shadow-panel-soft">
        <header className="flex items-start justify-between gap-4 border-b border-slate-800 px-5 py-4">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <Settings className="h-5 w-5 text-cyan-200" />
              <h2 className="text-base font-semibold text-white">Settings</h2>
            </div>
            <p className="mt-1 text-xs text-slate-500">Configuration is saved locally in this app.</p>
          </div>
          <button className="grid h-8 w-8 place-items-center rounded-md border border-slate-700 bg-slate-950 text-slate-400 transition hover:border-slate-500 hover:text-white" type="button" onClick={onClose}>
            <XCircle className="h-4 w-4" />
          </button>
        </header>

        <div className="space-y-4 p-5">
          {IS_TAURI && (
            <div className="rounded-md border border-cyan-400/20 bg-cyan-400/5 p-3 space-y-3">
              <div className="flex items-center gap-2">
                <Cpu className="h-4 w-4 text-cyan-300" />
                <span className="text-xs font-semibold uppercase tracking-[0.12em] text-cyan-300">Embedded AI Engine</span>
                <span className={`ml-auto rounded px-2 py-0.5 text-xs font-medium ${
                  localAi.phase === "ready" ? "bg-emerald-400/15 text-emerald-300" : localAi.phase === "starting" ? "bg-amber-400/15 text-amber-300" : "bg-red-400/15 text-red-300"
                }`}>
                  {localAi.phase === "ready" ? "Running" : localAi.phase === "starting" ? "Starting…" : localAi.error === "not-configured" ? "Not configured" : "Error"}
                </span>
              </div>
              <PathBrowseField
                label="Model file (medgemma .gguf)"
                value={draft.modelPath}
                onChange={(v) => update("modelPath", v)}
              />
              <PathBrowseField
                label="mmproj file (vision encoder .gguf)"
                value={draft.mmprojPath}
                onChange={(v) => update("mmprojPath", v)}
              />
              <label className="block">
                <span className="mb-1 block text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">GPU Layers (0 = CPU only)</span>
                <input
                  className="h-10 w-full rounded-md border border-slate-700 bg-slate-950 px-3 text-sm text-slate-100 outline-none transition focus:border-cyan-300"
                  type="number"
                  min="0"
                  max="99"
                  value={draft.gpuLayers}
                  onChange={(e) => update("gpuLayers", Number(e.target.value))}
                />
                <p className="mt-1 text-xs text-slate-600">Changing GPU layers restarts the server.</p>
              </label>
            </div>
          )}

          {!IS_TAURI && (
            <SettingsField
              label="Model Endpoint URL"
              value={draft.modelApiUrl}
              placeholder="http://127.0.0.1:1234/v1"
              onChange={(v) => update("modelApiUrl", v)}
            />
          )}

          <SettingsField
            icon={KeyRound}
            label="API Key"
            value={draft.modelApiKey}
            placeholder="Optional bearer token"
            type="password"
            onChange={(v) => update("modelApiKey", v)}
          />
          <SettingsField
            label="Model Identifier"
            value={draft.modelId}
            placeholder="medical-model-id"
            onChange={(v) => update("modelId", v)}
          />

          <div className="flex justify-end gap-3 pt-1">
            <button className="rounded-md border border-slate-700 px-4 py-2 text-sm font-medium text-slate-200 transition hover:bg-slate-800" type="button" onClick={onClose}>
              Cancel
            </button>
            <button
              className="inline-flex items-center gap-2 rounded-md border border-cyan-400/25 bg-cyan-400/10 px-4 py-2 text-sm font-semibold text-cyan-100 transition hover:bg-cyan-400/15"
              type="button"
              onClick={() => onSave(draft)}
            >
              <Save className="h-4 w-4" /> Save
            </button>
          </div>
        </div>
      </section>
    </div>
  );
}

// ── FeedbackModal ──────────────────────────────────────────────────────────
function FeedbackModal({ study, feedbackText, onClose, onFeedbackTextChange, onSave }: {
  study: Study; feedbackText: string;
  onClose: () => void; onFeedbackTextChange: (v: string) => void; onSave: () => void;
}) {
  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-black/62 p-4 backdrop-blur-sm">
      <section className="w-full max-w-lg rounded-md border border-amber-300/35 bg-[#0c1420] shadow-panel-soft">
        <header className="flex items-start justify-between gap-4 border-b border-slate-800 px-5 py-4">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <MessageSquareText className="h-5 w-5 text-amber-200" />
              <h2 className="text-base font-semibold text-white">Doctor Feedback</h2>
            </div>
            <p className="mt-1 truncate text-xs text-slate-500">{study.patientName} · {study.modality} {study.bodyPart}</p>
          </div>
          <button className="grid h-8 w-8 place-items-center rounded-md border border-slate-700 bg-slate-950 text-slate-400 transition hover:text-white" type="button" onClick={onClose}>
            <XCircle className="h-4 w-4" />
          </button>
        </header>
        <div className="p-5">
          <textarea
            className="h-36 w-full resize-none rounded-md border border-slate-700 bg-slate-950 p-3 text-sm leading-6 text-slate-100 outline-none transition placeholder:text-slate-600 focus:border-amber-300"
            placeholder="Record what needs correction before this can be accepted…"
            value={feedbackText}
            autoFocus
            onChange={(e) => onFeedbackTextChange(e.target.value)}
          />
          <div className="mt-4 flex justify-end gap-3">
            <button className="rounded-md border border-slate-700 px-4 py-2 text-sm font-medium text-slate-200 transition hover:bg-slate-800" type="button" onClick={onClose}>Cancel</button>
            <button className="rounded-md border border-amber-300/35 bg-amber-400/15 px-4 py-2 text-sm font-semibold text-amber-100 transition hover:bg-amber-400/20" type="button" onClick={onSave}>Save Note</button>
          </div>
        </div>
      </section>
    </div>
  );
}

// ── 3D Structure Preview Modal ─────────────────────────────────────────────
function StructurePreviewModal({ study, onClose }: { study: Study; onClose: () => void }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const frameRef = useRef<HTMLDivElement>(null);
  const primarySegmentation = study.segmentations[0];

  useEffect(() => {
    const canvas = canvasRef.current;
    const frame = frameRef.current;
    if (!canvas || !frame) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#05070b");
    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
    camera.position.set(0, 0.35, 4.2);
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const key = new THREE.DirectionalLight(0x67e8f9, 1.6);
    key.position.set(3, 4, 5);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0x34d399, 0.8);
    fill.position.set(-4, -2, 3);
    scene.add(fill);

    const group = new THREE.Group();
    scene.add(group);
    const geometries: Array<{ dispose: () => void }> = [];
    const materials: Array<{ dispose: () => void }> = [];
    const textures: Array<{ dispose: () => void }> = [];

    const addFallbackShape = () => {
      const roi = primarySegmentation?.box;
      const w = roi ? Math.max(0.7, roi.width * 7) : 1.45;
      const h = roi ? Math.max(0.75, roi.height * 7.5) : 1.2;
      const d = primarySegmentation ? Math.max(0.65, Math.min(1.6, primarySegmentation.volumeMl / 55)) : 0.95;
      const geo = new THREE.SphereGeometry(1, 48, 32);
      const mat = new THREE.MeshStandardMaterial({ color: 0x22d3ee, emissive: 0x083344, metalness: 0.05, roughness: 0.38, transparent: true, opacity: 0.86 });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.scale.set(w, h, d);
      group.add(mesh);
      const outGeo = new THREE.SphereGeometry(1.015, 24, 16);
      const outMat = new THREE.MeshBasicMaterial({ color: 0xa7f3d0, wireframe: true, transparent: true, opacity: 0.35 });
      const out = new THREE.Mesh(outGeo, outMat);
      out.scale.copy(mesh.scale);
      group.add(out);
      geometries.push(geo, outGeo);
      materials.push(mat, outMat);
    };

    const addImageSurface = (texture: THREE.Texture) => {
      const img = texture.image as { width?: number; height?: number } | null;
      const iw = img?.width ?? 1;
      const ih = img?.height ?? 1;
      const aspect = iw / Math.max(1, ih);
      const sw = aspect >= 1 ? 3.4 : 3.4 * aspect;
      const sh = aspect >= 1 ? 3.4 / aspect : 3.4;
      const geo = new THREE.PlaneGeometry(sw, sh, 160, 160);
      const mat = new THREE.MeshStandardMaterial({ map: texture, displacementMap: texture, displacementScale: 0.48, displacementBias: -0.18, metalness: 0.02, roughness: 0.72, side: THREE.DoubleSide });
      group.add(new THREE.Mesh(geo, mat));
      const wGeo = new THREE.PlaneGeometry(sw, sh, 32, 32);
      const wMat = new THREE.MeshBasicMaterial({ color: 0xa7f3d0, wireframe: true, transparent: true, opacity: 0.16 });
      const wire = new THREE.Mesh(wGeo, wMat);
      wire.position.z = 0.03;
      group.add(wire);
      if (primarySegmentation) {
        const roi = primarySegmentation.box;
        const l = (roi.x - 0.5) * sw, r = (roi.x + roi.width - 0.5) * sw;
        const t = (0.5 - roi.y) * sh, b = (0.5 - roi.y - roi.height) * sh;
        const rGeo = new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(l, t, 0.34), new THREE.Vector3(r, t, 0.34),
          new THREE.Vector3(r, b, 0.34), new THREE.Vector3(l, b, 0.34), new THREE.Vector3(l, t, 0.34),
        ]);
        group.add(new THREE.Line(rGeo, new THREE.LineBasicMaterial({ color: 0xfbbf24 })));
        geometries.push(rGeo);
      }
      geometries.push(geo, wGeo);
      materials.push(mat, wMat);
    };

    if (study.previewUrl) {
      const loader = new THREE.TextureLoader();
      loader.load(study.previewUrl, (tex) => {
        tex.colorSpace = THREE.SRGBColorSpace;
        textures.push(tex);
        addImageSurface(tex);
      }, undefined, addFallbackShape);
    } else {
      addFallbackShape();
    }

    const grid = new THREE.GridHelper(4.5, 8, 0x334155, 0x1e293b);
    grid.position.y = -1.45;
    scene.add(grid);

    const resize = () => {
      const { width: fw, height: fh } = frame.getBoundingClientRect();
      renderer.setSize(fw, fh, false);
      camera.aspect = fw / Math.max(1, fh);
      camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(frame);
    resize();

    let frameId = 0;
    const render = () => {
      group.rotation.x = -0.35 + Math.sin(Date.now() * 0.001) * 0.06;
      group.rotation.y += 0.006;
      renderer.render(scene, camera);
      frameId = requestAnimationFrame(render);
    };
    render();

    return () => {
      cancelAnimationFrame(frameId);
      observer.disconnect();
      geometries.forEach((g) => g.dispose());
      materials.forEach((m) => m.dispose());
      textures.forEach((t) => t.dispose());
      renderer.dispose();
    };
  }, [primarySegmentation, study.previewUrl]);

  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-black/70 p-4 backdrop-blur-sm">
      <section className="flex h-[min(720px,88vh)] w-full max-w-4xl flex-col overflow-hidden rounded-md border border-cyan-300/30 bg-[#05070b] shadow-panel-soft">
        <header className="flex shrink-0 items-center justify-between gap-4 border-b border-slate-800 bg-[#090f18] px-5 py-4">
          <div>
            <h2 className="text-base font-semibold text-white">3D Structure Preview</h2>
            <p className="mt-1 text-xs text-slate-500">
              {study.previewUrl
                ? `Image-derived surface${primarySegmentation ? ` / ${primarySegmentation.label}` : ""}`
                : primarySegmentation
                  ? `${primarySegmentation.label} / ${primarySegmentation.volumeMl} ml`
                  : `${study.modality} ${study.bodyPart} preview`}
            </p>
          </div>
          <button className="grid h-8 w-8 place-items-center rounded-md border border-slate-700 bg-slate-950 text-slate-400 transition hover:text-white" type="button" onClick={onClose}>
            <XCircle className="h-4 w-4" />
          </button>
        </header>
        <div ref={frameRef} className="min-h-0 flex-1">
          <canvas ref={canvasRef} className="block h-full w-full" />
        </div>
      </section>
    </div>
  );
}

// ── Small components ───────────────────────────────────────────────────────
function ViewportOverlay({ study, windowLevel }: { study: Study; windowLevel: string }) {
  return (
    <>
      <div className="pointer-events-none absolute inset-x-0 top-0 flex items-center justify-between p-4 text-xs text-slate-400">
        <span>{study.uploadedFileName ?? study.id}</span>
        <span>{windowLevel}</span>
      </div>
      <div className="pointer-events-none absolute bottom-4 left-4 rounded border border-slate-700/80 bg-black/45 px-3 py-2 text-xs text-slate-300">
        {study.series} · Slice {Math.ceil(study.slices * 0.33)} / {study.slices}
      </div>
      <div className="pointer-events-none absolute bottom-4 right-4 rounded border border-slate-700/80 bg-black/45 px-3 py-2 text-xs text-slate-300">
        {study.isDicom ? "DICOM loaded" : "Preview mode"}
      </div>
    </>
  );
}

function SegmentationOverlay({ segmentations }: { segmentations: Segmentation[] }) {
  return (
    <>
      {segmentations.map((seg, i) => (
        <RoiBoxOverlay key={seg.id} box={seg.box} label={`${formatSegmentationSource(seg.source)} mask ${i + 1}`} className="border-cyan-300/90 bg-cyan-300/10 shadow-[0_0_30px_rgba(103,232,249,0.35)]" />
      ))}
    </>
  );
}

function RoiBoxOverlay({ box, label, className }: { box: RoiBox; label: string; className: string }) {
  return (
    <div
      className={`pointer-events-none absolute rounded-[42%] border-2 ${className}`}
      style={{ left: `${box.x * 100}%`, top: `${box.y * 100}%`, width: `${box.width * 100}%`, height: `${box.height * 100}%` }}
    >
      <div className="absolute inset-[12%] rounded-[48%] border border-white/35 bg-white/5" />
      <span className="absolute -right-3 top-2 translate-x-full whitespace-nowrap rounded border border-cyan-300/40 bg-cyan-950/85 px-2 py-1 text-[11px] font-medium text-cyan-50">
        {label}
      </span>
    </div>
  );
}

function StatusDot({ decision }: { decision: ReviewDecision }) {
  const className = { accepted: "text-emerald-300", "needs-review": "text-red-300", unreviewed: "text-amber-300" }[decision];
  return <CircleDot className={`mt-0.5 h-4 w-4 ${className}`} />;
}

function ThemeSwitcher({ theme, onThemeChange }: { theme: ThemeMode; onThemeChange: (t: ThemeMode) => void }) {
  return (
    <div className="theme-switcher inline-flex h-9 items-center rounded-md border border-slate-700 bg-slate-950 p-1">
      <ThemeOption icon={Sun} label="Clinical day" active={theme === "light"} onClick={() => onThemeChange("light")} />
      <ThemeOption icon={Moon} label="Diagnostic night" active={theme === "dark"} onClick={() => onThemeChange("dark")} />
    </div>
  );
}

function ThemeOption({ icon: Icon, label, active, onClick }: { icon: typeof Sun; label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      className={`grid h-7 w-8 place-items-center rounded-[5px] transition ${active ? "bg-cyan-400/15 text-cyan-100 shadow-sm" : "text-slate-500 hover:text-slate-200"}`}
      type="button" title={label} aria-pressed={active} onClick={onClick}
    >
      <Icon className="h-4 w-4" />
    </button>
  );
}

function ToolbarButton({ icon: Icon, label, loading, onClick }: { icon: typeof Upload; label: string; loading?: boolean; onClick: () => void }) {
  return (
    <button
      className="inline-flex h-9 items-center gap-2 rounded-md border border-slate-700 bg-slate-950 px-3 text-sm font-medium text-slate-200 transition hover:border-cyan-300/50 hover:text-cyan-100 disabled:cursor-wait disabled:opacity-70"
      type="button" disabled={loading} onClick={onClick}
    >
      <Icon className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
      {label}
    </button>
  );
}

function ActionCard({ icon: Icon, title, actionLabel, loading, children, onAction }: {
  icon: typeof Sparkles; title: string; actionLabel: string; loading?: boolean; children: ReactNode; onAction: () => void;
}) {
  return (
    <article className="rounded-md border border-slate-800 bg-[#0c1420] p-4">
      <div className="mb-3 flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <span className="grid h-8 w-8 place-items-center rounded-md border border-cyan-400/25 bg-cyan-400/10 text-cyan-200">
            <Icon className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
          </span>
          <h3 className="text-sm font-semibold text-white">{title}</h3>
        </div>
        <button
          className="rounded-md border border-cyan-400/25 bg-cyan-400/10 px-3 py-1.5 text-xs font-semibold text-cyan-100 transition hover:bg-cyan-400/15 disabled:cursor-wait disabled:opacity-70"
          type="button" disabled={loading} onClick={onAction}
        >
          {actionLabel}
        </button>
      </div>
      {children}
    </article>
  );
}

function ReportSection({ icon: Icon, title, children }: { icon: typeof Brain; title: string; children: ReactNode }) {
  return (
    <article className="rounded-md border border-slate-800 bg-[#0c1420] p-4">
      <div className="mb-3 flex items-center gap-2">
        <span className="grid h-7 w-7 place-items-center rounded-md border border-slate-700 bg-slate-950 text-slate-300">
          <Icon className="h-4 w-4" />
        </span>
        <h3 className="text-sm font-semibold text-white">{title}</h3>
      </div>
      <div className="text-sm leading-6 text-slate-300">{children}</div>
    </article>
  );
}

function PathBrowseField({ label, value, onChange }: { label: string; value: string; onChange: (v: string) => void }) {
  const browse = async () => {
    try {
      const { open } = await import("@tauri-apps/plugin-dialog");
      const result = await open({ multiple: false, filters: [{ name: "GGUF Model", extensions: ["gguf"] }] });
      if (typeof result === "string") onChange(result);
    } catch {
      // user cancelled
    }
  };
  return (
    <label className="block">
      <span className="mb-1 block text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">{label}</span>
      <div className="flex gap-2">
        <input
          className="h-10 min-w-0 flex-1 rounded-md border border-slate-700 bg-slate-950 px-3 text-sm text-slate-100 outline-none transition placeholder:text-slate-600 focus:border-cyan-300"
          type="text"
          value={value}
          placeholder="C:\path\to\model.gguf"
          onChange={(e) => onChange(e.target.value)}
        />
        <button
          className="h-10 shrink-0 rounded-md border border-slate-700 bg-slate-950 px-3 text-sm text-slate-300 transition hover:border-cyan-300/50 hover:text-cyan-100"
          type="button"
          onClick={browse}
        >
          Browse…
        </button>
      </div>
    </label>
  );
}

function SettingsField({ icon: Icon, label, value, placeholder, type = "text", onChange }: {
  icon?: typeof KeyRound; label: string; value: string; placeholder: string; type?: "text" | "password"; onChange: (v: string) => void;
}) {
  return (
    <label className="block">
      <span className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">
        {Icon && <Icon className="h-3.5 w-3.5" />}
        {label}
      </span>
      <input
        className="h-10 w-full rounded-md border border-slate-700 bg-slate-950 px-3 text-sm text-slate-100 outline-none transition placeholder:text-slate-600 focus:border-cyan-300 focus:ring-2 focus:ring-cyan-300/10"
        type={type} value={value} placeholder={placeholder} onChange={(e) => onChange(e.target.value)}
      />
    </label>
  );
}

function TokenUsageChart({ usage }: { usage?: TokenUsage }) {
  if (!usage || usage.totalTokens <= 0) {
    return (
      <div className="mt-4 rounded-md border border-slate-800 bg-slate-950/45 p-3">
        <div className="flex items-center justify-between gap-3">
          <h4 className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">Token Usage</h4>
          <span className="text-xs text-slate-600">No run data</span>
        </div>
        <div className="mt-3 h-2 rounded-full bg-slate-800" />
      </div>
    );
  }

  const total = Math.max(1, usage.totalTokens);
  const pp = Math.round((usage.promptTokens / total) * 100);
  const cp = Math.round((usage.completionTokens / total) * 100);
  const cs = (usage.promptTokens / total) * 100;

  return (
    <div className="mt-4 rounded-md border border-slate-800 bg-slate-950/45 p-3">
      <div className="flex items-center justify-between gap-3">
        <h4 className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">Token Usage</h4>
        <span className="rounded border border-slate-700 bg-[#0c1420] px-2 py-1 text-[11px] text-slate-400">
          {usage.source === "api" ? "API" : "Estimated"}
        </span>
      </div>
      <div className="mt-3 flex items-center gap-4">
        <div
          className="relative grid h-24 w-24 shrink-0 place-items-center rounded-full border border-slate-700"
          style={{ background: `conic-gradient(#22d3ee 0% ${cs}%, #34d399 ${cs}% ${cs + cp}%, #64748b ${cs + cp}% 100%)` }}
        >
          <div className="grid h-14 w-14 place-items-center rounded-full border border-slate-800 bg-[#0c1420] text-center">
            <span className="text-sm font-semibold text-white">{usage.totalTokens}</span>
            <span className="text-[10px] uppercase text-slate-500">tokens</span>
          </div>
        </div>
        <div className="min-w-0 flex-1 space-y-2">
          <TokenRow label="Prompt" value={usage.promptTokens} percent={pp} className="bg-cyan-400" />
          <TokenRow label="Output" value={usage.completionTokens} percent={cp} className="bg-emerald-400" />
        </div>
      </div>
    </div>
  );
}

function TokenRow({ label, value, percent, className }: { label: string; value: number; percent: number; className: string }) {
  return (
    <div>
      <div className="mb-1 flex items-center justify-between gap-3 text-xs">
        <span className="flex items-center gap-2 text-slate-300">
          <span className={`h-2.5 w-2.5 rounded-sm ${className}`} />
          {label}
        </span>
        <span className="font-medium text-slate-100">{value} <span className="text-slate-500">({percent}%)</span></span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-slate-800">
        <div className={`h-full rounded-full ${className}`} style={{ width: `${percent}%` }} />
      </div>
    </div>
  );
}

// ── Helpers ────────────────────────────────────────────────────────────────
function clamp(v: number) { return Math.max(0, Math.min(1, v)); }

function normalizeRoi(start: { x: number; y: number }, end: { x: number; y: number }): RoiBox {
  return {
    x: clamp(Math.min(start.x, end.x)),
    y: clamp(Math.min(start.y, end.y)),
    width: clamp(Math.abs(end.x - start.x)),
    height: clamp(Math.abs(end.y - start.y)),
  };
}

export default App;
