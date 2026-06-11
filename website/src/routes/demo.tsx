import { createFileRoute, Link } from "@tanstack/react-router";
import { useCallback, useRef, useState } from "react";
import {
  Upload,
  Image as ImageIcon,
  X,
  Sparkles,
  Loader2,
  CheckCircle2,
  AlertTriangle,
  Cpu,
  Brain,
  ArrowLeft,
  RotateCcw,
} from "lucide-react";
import { Navbar } from "@/components/site/Navbar";
import { Footer } from "@/components/site/Footer";

export const Route = createFileRoute("/demo")({
  head: () => ({
    meta: [
      { title: "MRI Analysis Demo — Brain Tumor Detection" },
      {
        name: "description",
        content:
          "Interactive demo of the Brain Tumor Detection research project: upload an MRI scan and compare CNN and Vision Transformer predictions side by side.",
      },
      { property: "og:title", content: "MRI Analysis Demo — Brain Tumor Detection" },
      {
        property: "og:description",
        content:
          "Experience the Brain Tumor Detection CNN vs ViT brain MRI classification pipeline in an interactive research demo.",
      },
    ],
  }),
  component: DemoPage,
});

type ClassName = "Glioma" | "Meningioma" | "Pituitary" | "No Tumor";
type Prediction = { label: ClassName; confidence: number; probs: Record<ClassName, number> };
type Result = { cnn: Prediction; vit: Prediction };

const CLASSES: ClassName[] = ["Glioma", "Meningioma", "Pituitary", "No Tumor"];

const CLASS_BLURBS: Record<ClassName, string> = {
  Glioma:
    "Gliomas originate from glial cells in the brain. The model identified imaging patterns commonly associated with this tumor category.",
  Meningioma:
    "Meningiomas typically arise from the meninges surrounding the brain. The model detected features characteristic of this class.",
  Pituitary:
    "Pituitary tumors form in the pituitary gland near the base of the brain. The model surfaced features aligned with this category.",
  "No Tumor":
    "The model did not detect imaging patterns consistent with the three tumor classes in this scan.",
};

function mockPredict(seed: number): Result {
  // Deterministic-ish mock using the seed (file size length)
  const idx = seed % CLASSES.length;
  const primary = CLASSES[idx];

  const makeProbs = (top: ClassName, base: number) => {
    const others = CLASSES.filter((c) => c !== top);
    const remainder = 100 - base;
    const r1 = Math.random() * remainder * 0.6;
    const r2 = Math.random() * (remainder - r1) * 0.7;
    const r3 = remainder - r1 - r2;
    const probs = {
      [top]: base,
      [others[0]]: Math.max(0.2, r1),
      [others[1]]: Math.max(0.2, r2),
      [others[2]]: Math.max(0.2, r3),
    } as Record<ClassName, number>;
    return probs;
  };

  const cnnConf = 92 + Math.random() * 6.5;
  const vitConf = 95 + Math.random() * 4.5;

  return {
    cnn: { label: primary, confidence: cnnConf, probs: makeProbs(primary, cnnConf) },
    vit: { label: primary, confidence: vitConf, probs: makeProbs(primary, vitConf) },
  };
}

function DemoPage() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [status, setStatus] = useState<"idle" | "loading" | "done">("idle");
  const [result, setResult] = useState<Result | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFiles = useCallback((f: File | null) => {
    if (!f) return;
    if (!f.type.startsWith("image/")) return;
    setFile(f);
    setResult(null);
    setStatus("idle");
    const url = URL.createObjectURL(f);
    setPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return url;
    });
  }, []);

  const runAnalysis = useCallback(() => {
    if (!file) return;
    setStatus("loading");
    setResult(null);
    const seed = file.name.length + Math.floor(file.size / 1024);
    setTimeout(() => {
      setResult(mockPredict(seed));
      setStatus("done");
    }, 2400);
  }, [file]);

  const reset = useCallback(() => {
    setFile(null);
    setResult(null);
    setStatus("idle");
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    if (inputRef.current) inputRef.current.value = "";
  }, [previewUrl]);

  return (
    <div id="main-content" className="min-h-dvh bg-background text-foreground">
      <Navbar />

      <main className="relative pt-32 pb-24">
        <div
          className="absolute inset-x-0 top-0 -z-10 h-[520px]"
          style={{ background: "var(--gradient-hero)" }}
          aria-hidden
        />
        <div className="absolute inset-0 -z-10 grid-bg" aria-hidden />

        <div className="mx-auto max-w-6xl px-4">
          <Link
            to="/"
            className="inline-flex items-center gap-2 text-xs text-muted-foreground transition-colors hover:text-foreground"
          >
            <ArrowLeft className="h-3.5 w-3.5" />
            Back to overview
          </Link>

          <div className="mt-6 max-w-3xl animate-fade-up">
            <div className="inline-flex items-center gap-2 rounded-full glass px-3 py-1 text-xs text-muted-foreground">
              <Sparkles className="h-3.5 w-3.5 text-primary" />
              Live Demo · Research Simulation
            </div>
            <h1 className="mt-5 font-display text-4xl font-semibold leading-[1.05] tracking-tight sm:text-5xl md:text-6xl">
              <span className="text-gradient">MRI Analysis</span> Pipeline
            </h1>
            <p className="mt-4 max-w-2xl text-sm text-muted-foreground sm:text-base">
              Upload an MRI scan and watch the CNN and Vision Transformer models classify it side
              by side, with confidence visualization and a plain-language summary.
            </p>
          </div>

          <div className="mt-10 grid gap-5 lg:grid-cols-5">
            <UploadPanel
              file={file}
              previewUrl={previewUrl}
              dragOver={dragOver}
              status={status}
              onPick={() => inputRef.current?.click()}
              onDragOver={(e) => {
                e.preventDefault();
                setDragOver(true);
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragOver(false);
                handleFiles(e.dataTransfer.files?.[0] ?? null);
              }}
              onAnalyze={runAnalysis}
              onReset={reset}
            />
            <input
              ref={inputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => handleFiles(e.target.files?.[0] ?? null)}
            />

            <ResultsPanel status={status} result={result} hasFile={!!file} />
          </div>

          <Disclaimer />
        </div>
      </main>

      <Footer />
    </div>
  );
}

/* ---------------- Upload Panel ---------------- */

function UploadPanel({
  file,
  previewUrl,
  dragOver,
  status,
  onPick,
  onDragOver,
  onDragLeave,
  onDrop,
  onAnalyze,
  onReset,
}: {
  file: File | null;
  previewUrl: string | null;
  dragOver: boolean;
  status: "idle" | "loading" | "done";
  onPick: () => void;
  onDragOver: (e: React.DragEvent) => void;
  onDragLeave: () => void;
  onDrop: (e: React.DragEvent) => void;
  onAnalyze: () => void;
  onReset: () => void;
}) {
  return (
    <section className="lg:col-span-2">
      <div className="rounded-3xl glass-strong p-6">
        <div className="flex items-center justify-between">
          <div>
            <div className="font-mono text-[10px] uppercase tracking-widest text-primary">
              Step 01
            </div>
            <h2 className="mt-1 font-display text-xl font-semibold">Upload MRI scan</h2>
          </div>
          {file && (
            <button
              onClick={onReset}
              className="inline-flex items-center gap-1.5 rounded-lg glass px-3 py-1.5 text-xs text-muted-foreground transition-colors hover:text-foreground"
            >
              <X className="h-3.5 w-3.5" />
              Clear
            </button>
          )}
        </div>

        {!previewUrl ? (
          <div
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onDrop={onDrop}
            onClick={onPick}
            role="button"
            tabIndex={0}
            className={`mt-5 group relative grid cursor-pointer place-items-center rounded-2xl border-2 border-dashed p-10 text-center transition-all ${
              dragOver
                ? "border-primary bg-primary/5"
                : "border-border hover:border-primary/50 hover:bg-foreground/[0.02]"
            }`}
          >
            <div className="absolute inset-0 -z-10 rounded-2xl opacity-0 transition-opacity group-hover:opacity-100" />
            <span className="grid h-14 w-14 place-items-center rounded-2xl bg-gradient-to-br from-primary/20 to-accent/20 ring-1 ring-primary/30">
              <Upload className="h-6 w-6 text-primary" />
            </span>
            <div className="mt-4 font-display text-base font-semibold">
              Drag & drop your MRI image
            </div>
            <div className="mt-1 text-xs text-muted-foreground">
              PNG, JPG, or WEBP · processed locally in this demo
            </div>
            <button
              onClick={(e) => {
                e.stopPropagation();
                onPick();
              }}
              className="mt-5 inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-primary to-accent px-4 py-2 text-xs font-medium text-primary-foreground shadow-glow transition-transform hover:scale-[1.03]"
            >
              <ImageIcon className="h-3.5 w-3.5" />
              Browse files
            </button>
          </div>
        ) : (
          <div className="mt-5 space-y-4">
            <div className="relative overflow-hidden rounded-2xl border border-border">
              <img
                src={previewUrl}
                alt="MRI preview"
                className="aspect-square w-full bg-black object-contain"
              />
              <div className="absolute left-3 top-3 inline-flex items-center gap-1.5 rounded-full glass px-2.5 py-1 text-[10px] font-mono uppercase tracking-widest text-foreground/80">
                <span className="h-1.5 w-1.5 rounded-full bg-primary animate-pulse-glow" />
                MRI Input
              </div>
            </div>

            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span className="truncate font-mono">{file?.name}</span>
              <span className="font-mono">
                {(file ? file.size / 1024 : 0).toFixed(1)} KB
              </span>
            </div>

            <div className="flex flex-wrap gap-2">
              <button
                onClick={onAnalyze}
                disabled={status === "loading"}
                className="group inline-flex flex-1 items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-primary to-accent px-4 py-3 text-sm font-medium text-primary-foreground shadow-glow transition-transform hover:scale-[1.02] disabled:opacity-60"
              >
                {status === "loading" ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Analyzing…
                  </>
                ) : status === "done" ? (
                  <>
                    <RotateCcw className="h-4 w-4" />
                    Re-run analysis
                  </>
                ) : (
                  <>
                    <Sparkles className="h-4 w-4" />
                    Run analysis
                  </>
                )}
              </button>
            </div>
          </div>
        )}
      </div>

      <div className="mt-4 rounded-2xl glass p-4 text-xs text-muted-foreground">
        <div className="flex items-start gap-2">
          <AlertTriangle className="mt-0.5 h-4 w-4 flex-none text-primary" />
          <span>
            This demo uses simulated model responses for the portfolio interface. Real inference
            runs in the underlying PyTorch / TensorFlow pipeline.
          </span>
        </div>
      </div>
    </section>
  );
}

/* ---------------- Results Panel ---------------- */

function ResultsPanel({
  status,
  result,
  hasFile,
}: {
  status: "idle" | "loading" | "done";
  result: Result | null;
  hasFile: boolean;
}) {
  return (
    <section className="lg:col-span-3 space-y-5">
      <div className="rounded-3xl glass-strong p-6">
        <div className="flex items-center justify-between">
          <div>
            <div className="font-mono text-[10px] uppercase tracking-widest text-primary">
              Step 02
            </div>
            <h2 className="mt-1 font-display text-xl font-semibold">Model predictions</h2>
          </div>
          {result && (
            <AgreementBadge agree={result.cnn.label === result.vit.label} />
          )}
        </div>

        {status === "idle" && !result && (
          <EmptyState hasFile={hasFile} />
        )}

        {status === "loading" && <LoadingState />}

        {status === "done" && result && (
          <div className="mt-6 grid gap-4 md:grid-cols-2">
            <ModelCard
              title="CNN Prediction"
              subtitle="MobileNetV2"
              icon={Cpu}
              pred={result.cnn}
              accent="primary"
            />
            <ModelCard
              title="ViT Prediction"
              subtitle="Vision Transformer"
              icon={Brain}
              pred={result.vit}
              accent="accent"
            />
          </div>
        )}
      </div>

      {result && status === "done" && <SummaryCard result={result} />}
    </section>
  );
}

function EmptyState({ hasFile }: { hasFile: boolean }) {
  return (
    <div className="mt-6 grid place-items-center rounded-2xl border border-dashed border-border p-12 text-center">
      <span className="grid h-12 w-12 place-items-center rounded-2xl glass">
        <Brain className="h-5 w-5 text-muted-foreground" />
      </span>
      <p className="mt-4 text-sm text-muted-foreground">
        {hasFile
          ? "Press “Run analysis” to send the scan through CNN and ViT."
          : "Upload an MRI scan to see CNN and ViT predictions appear here."}
      </p>
    </div>
  );
}

function LoadingState() {
  const stages = [
    "Preprocessing image · 224×224 · normalize",
    "Forward pass · MobileNetV2 backbone",
    "Forward pass · ViT-Base/16 attention",
    "Aggregating confidence distributions",
  ];
  return (
    <div className="mt-6 rounded-2xl border border-border p-8">
      <div className="flex items-center gap-3">
        <span className="relative grid h-12 w-12 place-items-center rounded-2xl bg-gradient-to-br from-primary/20 to-accent/20 ring-1 ring-primary/30">
          <Loader2 className="h-5 w-5 animate-spin text-primary" />
          <span className="absolute inset-0 rounded-2xl ring-1 ring-primary/40 animate-pulse-glow" />
        </span>
        <div>
          <div className="font-display text-base font-semibold">Running inference</div>
          <div className="text-xs text-muted-foreground">
            Routing scan through CNN and Transformer pipelines…
          </div>
        </div>
      </div>

      <ul className="mt-6 space-y-2.5">
        {stages.map((s, i) => (
          <li
            key={s}
            className="flex items-center gap-3 rounded-lg bg-foreground/[0.03] px-3 py-2 text-xs animate-fade-up"
            style={{ animationDelay: `${i * 220}ms` }}
          >
            <span className="h-1.5 w-1.5 rounded-full bg-primary animate-pulse-glow" />
            <span className="font-mono text-muted-foreground">{s}</span>
          </li>
        ))}
      </ul>

      <div className="relative mt-6 h-1 overflow-hidden rounded-full bg-foreground/5">
        <div
          className="absolute inset-y-0 left-0 w-1/3 rounded-full bg-gradient-to-r from-primary via-accent to-primary"
          style={{
            animation: "shimmer 1.6s linear infinite",
            backgroundSize: "200% 100%",
          }}
        />
      </div>
    </div>
  );
}

function AgreementBadge({ agree }: { agree: boolean }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-[11px] font-mono uppercase tracking-widest ${
        agree
          ? "bg-primary/10 text-primary ring-1 ring-primary/30"
          : "bg-destructive/10 text-destructive ring-1 ring-destructive/30"
      }`}
    >
      {agree ? <CheckCircle2 className="h-3.5 w-3.5" /> : <AlertTriangle className="h-3.5 w-3.5" />}
      {agree ? "Models agree" : "Models disagree"}
    </span>
  );
}

/* ---------------- Model card ---------------- */

function ModelCard({
  title,
  subtitle,
  icon: Icon,
  pred,
  accent,
}: {
  title: string;
  subtitle: string;
  icon: React.ComponentType<{ className?: string }>;
  pred: Prediction;
  accent: "primary" | "accent";
}) {
  const ring = accent === "primary" ? "ring-primary/30" : "ring-accent/40";
  const bar = accent === "primary"
    ? "from-primary to-primary/70"
    : "from-accent to-accent/70";

  return (
    <div className="group relative overflow-hidden rounded-2xl glass p-5 transition-all hover:-translate-y-0.5">
      <div className={`absolute -right-16 -top-16 h-40 w-40 rounded-full blur-3xl ${accent === "primary" ? "bg-primary/15" : "bg-accent/15"}`} />
      <div className="relative">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className={`grid h-9 w-9 place-items-center rounded-xl bg-foreground/5 ring-1 ${ring}`}>
              <Icon className="h-4 w-4 text-foreground" />
            </span>
            <div>
              <div className="text-sm font-semibold">{title}</div>
              <div className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
                {subtitle}
              </div>
            </div>
          </div>
        </div>

        <div className="mt-5 flex items-end justify-between">
          <div>
            <div className="text-xs text-muted-foreground">Predicted class</div>
            <div className="mt-1 font-display text-3xl font-semibold tracking-tight">
              {pred.label}
            </div>
          </div>
          <div className="text-right">
            <div className="text-xs text-muted-foreground">Confidence</div>
            <div className="mt-1 font-display text-3xl font-semibold tracking-tight text-gradient">
              {pred.confidence.toFixed(1)}%
            </div>
          </div>
        </div>

        <div className="mt-6 space-y-2.5">
          {CLASSES.map((c) => {
            const v = pred.probs[c] ?? 0;
            const isTop = c === pred.label;
            return (
              <div key={c}>
                <div className="flex items-center justify-between text-[11px]">
                  <span className={isTop ? "text-foreground" : "text-muted-foreground"}>{c}</span>
                  <span className={`font-mono ${isTop ? "text-foreground" : "text-muted-foreground"}`}>
                    {v.toFixed(1)}%
                  </span>
                </div>
                <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-foreground/5">
                  <div
                    className={`h-full rounded-full bg-gradient-to-r ${isTop ? bar : "from-foreground/20 to-foreground/10"}`}
                    style={{
                      width: `${Math.min(100, Math.max(2, v))}%`,
                      transition: "width 900ms cubic-bezier(0.16, 1, 0.3, 1)",
                    }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

/* ---------------- Summary ---------------- */

function SummaryCard({ result }: { result: Result }) {
  const agree = result.cnn.label === result.vit.label;
  const avgConf = (result.cnn.confidence + result.vit.confidence) / 2;
  const cls = result.cnn.label;

  return (
    <div className="rounded-3xl glass-strong p-6 animate-fade-up">
      <div className="flex items-start gap-4">
        <span className="grid h-11 w-11 flex-none place-items-center rounded-xl bg-gradient-to-br from-primary/20 to-accent/20 ring-1 ring-primary/30">
          <Sparkles className="h-5 w-5 text-primary" />
        </span>
        <div className="flex-1">
          <div className="font-mono text-[10px] uppercase tracking-widest text-primary">
            Step 03 — Summary
          </div>
          <h3 className="mt-1 font-display text-xl font-semibold">
            {agree
              ? `Both models predicted ${cls}`
              : `Models disagreed on the prediction`}
          </h3>
          <p className="mt-2 text-sm text-muted-foreground">
            {agree ? (
              <>
                MobileNetV2 and the Vision Transformer independently classified this scan as{" "}
                <span className="text-foreground font-medium">{cls}</span> with an average
                confidence of{" "}
                <span className="text-foreground font-medium">{avgConf.toFixed(1)}%</span>.{" "}
                {CLASS_BLURBS[cls]}
              </>
            ) : (
              <>
                MobileNetV2 predicted{" "}
                <span className="text-foreground font-medium">{result.cnn.label}</span> while the
                Vision Transformer predicted{" "}
                <span className="text-foreground font-medium">{result.vit.label}</span>. Cross-model
                disagreement is one of the signals this research project studies.
              </>
            )}
          </p>

          <div className="mt-5 grid gap-3 sm:grid-cols-3">
            <Stat label="CNN confidence" value={`${result.cnn.confidence.toFixed(1)}%`} />
            <Stat label="ViT confidence" value={`${result.vit.confidence.toFixed(1)}%`} />
            <Stat label="Agreement" value={agree ? "Yes" : "No"} />
          </div>
        </div>
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl bg-foreground/[0.03] p-4 ring-1 ring-border">
      <div className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
        {label}
      </div>
      <div className="mt-1 font-display text-xl font-semibold">{value}</div>
    </div>
  );
}

function Disclaimer() {
  return (
    <div className="mt-10 rounded-2xl border border-destructive/30 bg-destructive/5 p-5 text-sm">
      <div className="flex items-start gap-3">
        <AlertTriangle className="mt-0.5 h-5 w-5 flex-none text-destructive" />
        <p className="text-foreground/85">
          <span className="font-semibold">Research disclaimer.</span> This project is intended for
          educational and research purposes only and is not a medical diagnostic system. It must
          not be used to inform clinical decisions.
        </p>
      </div>
    </div>
  );
}
