import { createFileRoute, Link } from "@tanstack/react-router";
import { useState, useRef, useCallback } from "react";
import {
  ArrowLeft,
  Brain,
  Cpu,
  Eye,
  Layers,
  Lightbulb,
  AlertTriangle,
  ZoomIn,
  MoveHorizontal,
} from "lucide-react";
import { Navbar } from "@/components/site/Navbar";
import { Footer } from "@/components/site/Footer";

export const Route = createFileRoute("/explainability")({
  head: () => ({
    meta: [
      { title: "Grad-CAM Explainability — Brain Tumor Detection" },
      {
        name: "description",
        content:
          "Explore Grad-CAM visual explanations for the Brain Tumor Detection brain tumor classification model. Understand what the CNN looks at during inference.",
      },
      { property: "og:title", content: "Grad-CAM Explainability — Brain Tumor Detection" },
      {
        property: "og:description",
        content:
          "Interactive Grad-CAM visualization for the Brain Tumor Detection CNN brain tumor classification research project.",
      },
    ],
  }),
  component: ExplainabilityPage,
});

/* ────────────── Placeholder image components ────────────── */

function PlaceholderMRI() {
  return (
    <div className="relative grid h-full w-full place-items-center overflow-hidden rounded-2xl bg-black">
      {/* Simulated MRI scan using SVG */}
      <svg viewBox="0 0 400 400" className="h-full w-full opacity-80">
        <defs>
          <radialGradient id="brainGrad" cx="50%" cy="45%" r="50%">
            <stop offset="0%" stopColor="#b8c5d6" stopOpacity="0.9" />
            <stop offset="60%" stopColor="#7a8fa8" stopOpacity="0.7" />
            <stop offset="100%" stopColor="#4a5a6e" stopOpacity="0.5" />
          </radialGradient>
          <filter id="blur">
            <feGaussianBlur stdDeviation="3" />
          </filter>
        </defs>
        {/* Brain outline */}
        <ellipse cx="200" cy="180" rx="140" ry="130" fill="url(#brainGrad)" />
        {/* Sulci (folds) */}
        <path
          d="M120 150 Q160 130 200 140 Q240 150 280 145"
          fill="none"
          stroke="#9aacbf"
          strokeWidth="2"
          opacity="0.5"
        />
        <path
          d="M130 190 Q170 170 210 180 Q250 190 290 185"
          fill="none"
          stroke="#9aacbf"
          strokeWidth="2"
          opacity="0.4"
        />
        <path
          d="M140 230 Q180 210 220 220 Q260 230 300 225"
          fill="none"
          stroke="#9aacbf"
          strokeWidth="2"
          opacity="0.3"
        />
        {/* Asymmetry hint */}
        <ellipse cx="260" cy="160" rx="35" ry="30" fill="#6b7d94" opacity="0.6" filter="url(#blur)" />
        <ellipse cx="260" cy="160" rx="20" ry="18" fill="#8a9eb5" opacity="0.4" />
        {/* Ventricles */}
        <ellipse cx="200" cy="190" rx="8" ry="25" fill="#3a4a5e" opacity="0.5" />
        {/* Noise texture */}
        {Array.from({ length: 30 }).map((_, i) => (
          <circle
            key={i}
            cx={80 + Math.random() * 240}
            cy={80 + Math.random() * 220}
            r={0.5 + Math.random() * 1.5}
            fill="#c0cdd9"
            opacity={0.15 + Math.random() * 0.2}
          />
        ))}
      </svg>
      <div className="absolute left-3 top-3 inline-flex items-center gap-1.5 rounded-full glass px-2.5 py-1 text-[10px] font-mono uppercase tracking-widest text-foreground/80">
        <span className="h-1.5 w-1.5 rounded-full bg-muted-foreground" />
        T1-weighted MRI
      </div>
    </div>
  );
}

function PlaceholderHeatmap() {
  return (
    <div className="relative grid h-full w-full place-items-center overflow-hidden rounded-2xl bg-black">
      {/* Grad-CAM heatmap on top of MRI */}
      <svg viewBox="0 0 400 400" className="h-full w-full">
        <defs>
          <radialGradient id="camGrad" cx="65%" cy="40%" r="40%">
            <stop offset="0%" stopColor="#ff4444" stopOpacity="0.85" />
            <stop offset="30%" stopColor="#ff8800" stopOpacity="0.65" />
            <stop offset="60%" stopColor="#ffcc00" stopOpacity="0.4" />
            <stop offset="100%" stopColor="#ffcc00" stopOpacity="0" />
          </radialGradient>
          <radialGradient id="brainGrad2" cx="50%" cy="45%" r="50%">
            <stop offset="0%" stopColor="#b8c5d6" stopOpacity="0.4" />
            <stop offset="60%" stopColor="#7a8fa8" stopOpacity="0.3" />
            <stop offset="100%" stopColor="#4a5a6e" stopOpacity="0.2" />
          </radialGradient>
          <filter id="blur2">
            <feGaussianBlur stdDeviation="6" />
          </filter>
        </defs>
        {/* Faint brain outline */}
        <ellipse cx="200" cy="180" rx="140" ry="130" fill="url(#brainGrad2)" />
        {/* Heatmap activation */}
        <ellipse cx="260" cy="160" rx="55" ry="48" fill="url(#camGrad)" filter="url(#blur2)" />
        <ellipse cx="260" cy="160" rx="30" ry="26" fill="#ff4444" opacity="0.5" filter="url(#blur2)" />
        {/* Heatmap overlay noise */}
        {Array.from({ length: 20 }).map((_, i) => (
          <circle
            key={i}
            cx={220 + Math.random() * 80}
            cy={120 + Math.random() * 80}
            r={1 + Math.random() * 3}
            fill={["#ff4444", "#ff8800", "#ffcc00"][i % 3]}
            opacity={0.3 + Math.random() * 0.3}
          />
        ))}
      </svg>
      <div className="absolute left-3 top-3 inline-flex items-center gap-1.5 rounded-full glass px-2.5 py-1 text-[10px] font-mono uppercase tracking-widest text-foreground/80">
        <span className="h-1.5 w-1.5 rounded-full bg-destructive animate-pulse-glow" />
        Grad-CAM Activation
      </div>
    </div>
  );
}

/* ────────────── Comparison Slider ────────────── */

function ComparisonSlider() {
  const [sliderPos, setSliderPos] = useState(50);
  const containerRef = useRef<HTMLDivElement>(null);
  const isDragging = useRef(false);

  const updatePosition = useCallback(
    (clientX: number) => {
      if (!containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const x = clientX - rect.left;
      const pct = Math.max(0, Math.min(100, (x / rect.width) * 100));
      setSliderPos(pct);
    },
    []
  );

  const handlePointerDown = useCallback(
    (e: React.PointerEvent) => {
      isDragging.current = true;
      (e.target as HTMLElement).setPointerCapture(e.pointerId);
      updatePosition(e.clientX);
    },
    [updatePosition]
  );

  const handlePointerMove = useCallback(
    (e: React.PointerEvent) => {
      if (!isDragging.current) return;
      updatePosition(e.clientX);
    },
    [updatePosition]
  );

  const handlePointerUp = useCallback(() => {
    isDragging.current = false;
  }, []);

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <div className="font-mono text-[10px] uppercase tracking-widest text-primary">
            Interactive
          </div>
          <h3 className="mt-1 font-display text-lg font-semibold">Overlay Comparison</h3>
        </div>
        <div className="inline-flex items-center gap-1.5 rounded-full glass px-2.5 py-1 text-[10px] font-mono text-muted-foreground">
          <MoveHorizontal className="h-3 w-3" />
          Drag to compare
        </div>
      </div>

      <div
        ref={containerRef}
        className="relative aspect-square w-full cursor-ew-resize overflow-hidden rounded-2xl border border-border select-none sm:aspect-[16/10]"
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerLeave={handlePointerUp}
      >
        {/* Bottom layer: Original MRI */}
        <div className="absolute inset-0">
          <PlaceholderMRI />
        </div>

        {/* Top layer: Heatmap (clipped) */}
        <div
          className="absolute inset-0 overflow-hidden"
          style={{ clipPath: `inset(0 ${100 - sliderPos}% 0 0)` }}
        >
          <PlaceholderHeatmap />
        </div>

        {/* Divider line */}
        <div
          className="absolute inset-y-0 w-0.5 bg-white/80 shadow-lg"
          style={{ left: `${sliderPos}%` }}
        >
          {/* Handle */}
          <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2">
            <div className="grid h-10 w-10 place-items-center rounded-full bg-white/90 shadow-lg ring-2 ring-white/50 backdrop-blur-sm">
              <MoveHorizontal className="h-4 w-4 text-black" />
            </div>
          </div>
        </div>

        {/* Labels */}
        <div className="absolute bottom-3 left-3 rounded-full glass px-2.5 py-1 text-[10px] font-mono text-foreground/80">
          Original MRI
        </div>
        <div className="absolute bottom-3 right-3 rounded-full glass px-2.5 py-1 text-[10px] font-mono text-foreground/80">
          Grad-CAM Overlay
        </div>
      </div>
    </div>
  );
}

/* ────────────── Explanation Cards ────────────── */

const explanations = [
  {
    icon: Eye,
    title: "What is Grad-CAM?",
    body: "Gradient-weighted Class Activation Mapping produces a coarse localization map highlighting the important regions in the image for predicting a particular class. It uses the gradients flowing into the final convolutional layer.",
    accent: "primary",
  },
  {
    icon: Layers,
    title: "Last Convolutional Layer",
    body: "Grad-CAM is computed from the feature maps of the last convolutional layer in MobileNetV2. These maps retain spatial information while capturing high-level semantic concepts, making them ideal for localization.",
    accent: "accent",
  },
  {
    icon: Cpu,
    title: "Guided Backpropagation",
    body: "When combined with Guided Backpropagation, the result is Guided Grad-CAM, which creates high-resolution visualizations that show both where the network looks and what features it detects at each neuron.",
    accent: "brand-2",
  },
  {
    icon: Lightbulb,
    title: "Interpretation Notes",
    body: "Warmer colors (red, orange) indicate higher activation. The model focuses on abnormal tissue regions rather than healthy brain structures. This helps verify that predictions are grounded in clinically relevant regions.",
    accent: "primary",
  },
];

function ExplanationCards() {
  return (
    <div className="grid gap-4 sm:grid-cols-2">
      {explanations.map((card, i) => (
        <div
          key={card.title}
          className="group relative overflow-hidden rounded-2xl glass p-5 transition-all hover:-translate-y-0.5"
          style={{ animationDelay: `${i * 100}ms` }}
        >
          {/* Subtle gradient ring on hover */}
          <div className="absolute inset-0 rounded-2xl opacity-0 transition-opacity group-hover:opacity-100 ring-1 ring-inset ring-primary/20" />

          <div className="relative flex items-start gap-4">
            <span className="grid h-10 w-10 flex-none place-items-center rounded-xl bg-gradient-to-br from-primary/20 to-accent/20 ring-1 ring-primary/30">
              <card.icon className="h-4 w-4 text-primary" />
            </span>
            <div>
              <h4 className="font-display text-sm font-semibold">{card.title}</h4>
              <p className="mt-1.5 text-xs leading-relaxed text-muted-foreground">{card.body}</p>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

/* ────────────── Disclaimer ────────────── */

function Disclaimer() {
  return (
    <div className="rounded-2xl glass p-4">
      <div className="flex items-start gap-3">
        <AlertTriangle className="mt-0.5 h-4 w-4 flex-none text-primary" />
        <div>
          <p className="text-xs font-medium text-foreground">Research Use Only</p>
          <p className="mt-1 text-xs text-muted-foreground">
            These visualizations are generated from a research model and are intended for educational
            and explainability analysis only. They do not constitute a medical diagnosis or clinical
            interpretation. Always consult qualified medical professionals for diagnostic decisions.
          </p>
        </div>
      </div>
    </div>
  );
}

/* ────────────── Page ────────────── */

function ExplainabilityPage() {
  return (
    <div id="main-content" className="min-h-dvh bg-background text-foreground">
      <Navbar />

      <main className="relative pt-32 pb-24">
        {/* Background accents */}
        <div
          className="absolute inset-x-0 top-0 -z-10 h-[520px]"
          style={{ background: "var(--gradient-hero)" }}
          aria-hidden
        />
        <div className="absolute inset-0 -z-10 grid-bg" aria-hidden />

        <div className="mx-auto max-w-6xl px-4">
          {/* Breadcrumb */}
          <Link
            to="/"
            className="inline-flex items-center gap-2 text-xs text-muted-foreground transition-colors hover:text-foreground"
          >
            <ArrowLeft className="h-3.5 w-3.5" />
            Back to overview
          </Link>

          {/* Page header */}
          <div className="mt-6 max-w-3xl animate-fade-up">
            <div className="inline-flex items-center gap-2 rounded-full glass px-3 py-1 text-xs text-muted-foreground">
              <ZoomIn className="h-3.5 w-3.5 text-primary" />
              Explainable AI
            </div>
            <h1 className="mt-5 font-display text-4xl font-semibold leading-[1.05] tracking-tight sm:text-5xl md:text-6xl">
              <span className="text-gradient">Grad-CAM</span> Visualizations
            </h1>
            <p className="mt-4 max-w-2xl text-sm text-muted-foreground sm:text-base">
              Understand what the CNN looks at when classifying brain MRI scans. Grad-CAM highlights
              the regions that most influence the model's prediction.
            </p>
          </div>

          {/* Image panels */}
          <div className="mt-10 grid gap-5 lg:grid-cols-2">
            {/* Original MRI */}
            <section className="rounded-3xl glass-strong p-5">
              <div className="flex items-center justify-between">
                <div>
                  <div className="font-mono text-[10px] uppercase tracking-widest text-primary">
                    Input
                  </div>
                  <h2 className="mt-1 font-display text-xl font-semibold">Original MRI</h2>
                </div>
                <div className="inline-flex items-center gap-1.5 rounded-full glass px-2.5 py-1 text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
                  <Brain className="h-3 w-3" />
                  T1-weighted
                </div>
              </div>
              <div className="mt-4 aspect-square w-full overflow-hidden rounded-2xl border border-border">
                <PlaceholderMRI />
              </div>
              <p className="mt-3 text-xs text-muted-foreground">
                Axial view of a brain MRI scan showing the full structural context used by the
                model during inference.
              </p>
            </section>

            {/* Grad-CAM Heatmap */}
            <section className="rounded-3xl glass-strong p-5">
              <div className="flex items-center justify-between">
                <div>
                  <div className="font-mono text-[10px] uppercase tracking-widest text-destructive">
                    Activation
                  </div>
                  <h2 className="mt-1 font-display text-xl font-semibold">Grad-CAM Heatmap</h2>
                </div>
                <div className="inline-flex items-center gap-1.5 rounded-full glass px-2.5 py-1 text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
                  <Cpu className="h-3 w-3" />
                  MobileNetV2
                </div>
              </div>
              <div className="mt-4 aspect-square w-full overflow-hidden rounded-2xl border border-border">
                <PlaceholderHeatmap />
              </div>
              <p className="mt-3 text-xs text-muted-foreground">
                The model's attention is strongly focused on the right frontal region, consistent
                with the predicted Meningioma class.
              </p>
            </section>
          </div>

          {/* Overlay comparison slider */}
          <section className="mt-5 rounded-3xl glass-strong p-5">
            <ComparisonSlider />
          </section>

          {/* Explanation cards */}
          <section className="mt-10">
            <div className="mb-5">
              <div className="font-mono text-[10px] uppercase tracking-widest text-primary">
                Understanding
              </div>
              <h2 className="mt-1 font-display text-2xl font-semibold">How Grad-CAM Works</h2>
              <p className="mt-2 text-sm text-muted-foreground">
                A quick primer on the explainability technique powering these visualizations.
              </p>
            </div>
            <ExplanationCards />
          </section>

          {/* Disclaimer */}
          <div className="mt-10">
            <Disclaimer />
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
}
