import { createFileRoute } from "@tanstack/react-router";
import {
  Eye, Brain, Layers, Activity, Flame, Sparkles, ArrowDown,
  ShieldCheck, Lightbulb, SearchCheck, Microscope, AlertTriangle, ImageIcon,
} from "lucide-react";
import { Navbar } from "@/components/site/Navbar";
import { Footer } from "@/components/site/Footer";

export const Route = createFileRoute("/explainable-ai")({
  head: () => ({
    meta: [
      { title: "Explainable AI — Brain Tumor Detection" },
      { name: "description", content: "How Grad-CAM reveals which regions of an MRI scan influence the CNN's prediction." },
      { property: "og:title", content: "Explainable AI — Brain Tumor Detection" },
      { property: "og:description", content: "Grad-CAM visualizations, gradient flow, and why explainability matters in deep learning." },
    ],
  }),
  component: ExplainableAIPage,
});

function PlaceholderMRI() {
  return (
    <svg viewBox="0 0 200 200" className="h-full w-full">
      <defs>
        <radialGradient id="mri-bg" cx="50%" cy="50%" r="55%">
          <stop offset="0%" stopColor="hsl(220 15% 35%)" />
          <stop offset="70%" stopColor="hsl(220 20% 12%)" />
          <stop offset="100%" stopColor="hsl(220 30% 4%)" />
        </radialGradient>
        <filter id="mri-blur"><feGaussianBlur stdDeviation="1.2" /></filter>
      </defs>
      <rect width="200" height="200" fill="#000" />
      <ellipse cx="100" cy="100" rx="72" ry="86" fill="url(#mri-bg)" filter="url(#mri-blur)" />
      <ellipse cx="100" cy="95" rx="55" ry="68" fill="hsl(220 18% 22%)" opacity="0.85" />
      <ellipse cx="100" cy="95" rx="42" ry="52" fill="hsl(220 14% 30%)" opacity="0.7" />
      <path d="M100 50 Q 80 95 100 140 Q 120 95 100 50" fill="hsl(220 12% 18%)" opacity="0.6" />
      <ellipse cx="115" cy="85" rx="9" ry="11" fill="hsl(220 8% 55%)" opacity="0.55" />
    </svg>
  );
}

function PlaceholderHeatmap() {
  return (
    <svg viewBox="0 0 200 200" className="h-full w-full">
      <defs>
        <radialGradient id="heat-core" cx="58%" cy="45%" r="22%">
          <stop offset="0%" stopColor="#fde047" stopOpacity="1" />
          <stop offset="35%" stopColor="#f97316" stopOpacity="0.9" />
          <stop offset="70%" stopColor="#dc2626" stopOpacity="0.55" />
          <stop offset="100%" stopColor="#7c2d12" stopOpacity="0" />
        </radialGradient>
        <radialGradient id="heat-spread" cx="50%" cy="50%" r="60%">
          <stop offset="0%" stopColor="#2563eb" stopOpacity="0.4" />
          <stop offset="100%" stopColor="#0c0a09" stopOpacity="0" />
        </radialGradient>
      </defs>
      <rect width="200" height="200" fill="#020617" />
      <rect width="200" height="200" fill="url(#heat-spread)" />
      <circle cx="116" cy="90" r="60" fill="url(#heat-core)" />
    </svg>
  );
}

function Overlay() {
  return (
    <div className="relative h-full w-full">
      <div className="absolute inset-0"><PlaceholderMRI /></div>
      <div className="absolute inset-0 mix-blend-screen opacity-80"><PlaceholderHeatmap /></div>
    </div>
  );
}

function PanelCard({ title, badge, children }: { title: string; badge: string; children: React.ReactNode }) {
  return (
    <div className="group rounded-2xl border border-border/60 glass p-4 transition-all hover:-translate-y-1 hover:shadow-glow">
      <div className="mb-3 flex items-center justify-between">
        <div className="text-sm font-medium">{title}</div>
        <span className="rounded-full bg-secondary px-2 py-0.5 font-mono text-[10px] text-muted-foreground">{badge}</span>
      </div>
      <div className="aspect-square overflow-hidden rounded-xl border border-border/40 bg-background">
        {children}
      </div>
    </div>
  );
}

const flowSteps = [
  { icon: ImageIcon, label: "MRI Image", sub: "Input scan · 224 × 224" },
  { icon: Layers, label: "CNN Feature Maps", sub: "Final conv layer activations" },
  { icon: Activity, label: "Gradient Computation", sub: "∂y/∂A for target class" },
  { icon: Flame, label: "Heatmap Generation", sub: "Weighted sum · ReLU" },
  { icon: Sparkles, label: "Visual Explanation", sub: "Upsample · overlay" },
];

const whyCards = [
  { icon: Eye, title: "Transparency", body: "Reveal what visual cues the network relied on, instead of treating the model as an opaque black box." },
  { icon: ShieldCheck, title: "Trust", body: "Stakeholders can verify whether predictions align with reasonable visual evidence in the input." },
  { icon: Lightbulb, title: "Model Understanding", body: "Researchers gain intuition for what features each layer learns and how decisions form." },
  { icon: SearchCheck, title: "Error Analysis", body: "Misclassifications become traceable — spot when the model attends to artifacts or background." },
];

function ExplainableAIPage() {
  return (
    <div id="main-content" className="min-h-dvh bg-background">
      <Navbar />
      <main className="pt-28">
        {/* Hero */}
        <section className="mx-auto max-w-6xl px-4 pb-14">
          <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-border/60 bg-secondary/40 px-3 py-1 text-xs text-muted-foreground">
            <Microscope className="h-3.5 w-3.5 text-primary" /> Explainable AI
          </div>
          <h1 className="font-display text-4xl font-semibold tracking-tight md:text-6xl">
            Seeing what the <span className="bg-gradient-to-r from-primary via-accent to-primary bg-clip-text text-transparent">model sees</span>
          </h1>
          <p className="mt-4 max-w-2xl text-muted-foreground">
            Grad-CAM turns a deep network's internal reasoning into a heatmap — highlighting the
            regions of an MRI that pushed the prediction toward a given class.
          </p>
        </section>

        {/* Introduction */}
        <section className="mx-auto max-w-6xl px-4 py-10">
          <div className="grid gap-6 lg:grid-cols-[1.1fr_1fr]">
            <div className="rounded-2xl border border-border/60 glass p-6">
              <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-border/60 bg-secondary/40 px-3 py-1 text-xs text-muted-foreground">
                <Brain className="h-3.5 w-3.5 text-primary" /> 01 — Introduction
              </div>
              <h2 className="font-display text-2xl font-semibold">What is Grad-CAM?</h2>
              <p className="mt-3 text-sm text-muted-foreground">
                Grad-CAM, short for <span className="text-foreground">Gradient-weighted Class Activation Mapping</span>,
                is a technique that produces a coarse heatmap showing which parts of an image most
                influenced a CNN's prediction. It works by flowing gradients of the target class back
                into the final convolutional layer, then weighing the feature maps by how strongly
                each one contributed to that decision.
              </p>
              <p className="mt-3 text-sm text-muted-foreground">
                The result is an intuitive visualization — warmer colors mark regions the model
                "looked at" most, while cooler colors mark regions it largely ignored.
              </p>
            </div>
            <div className="rounded-2xl border border-border/60 glass p-6">
              <div className="text-xs uppercase tracking-wider text-muted-foreground">Quick intuition</div>
              <ul className="mt-4 space-y-3 text-sm">
                {[
                  ["Gradient", "How much the output changes with each activation"],
                  ["Activation", "What each feature map detected in the image"],
                  ["Heatmap", "Activation × gradient, summed and ReLU'd"],
                  ["Overlay", "Upsampled and blended over the original scan"],
                ].map(([k, v]) => (
                  <li key={k} className="flex items-start gap-3">
                    <span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-primary" />
                    <div>
                      <div className="font-medium">{k}</div>
                      <div className="text-muted-foreground">{v}</div>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </section>

        {/* Visualization layout */}
        <section className="mx-auto max-w-6xl px-4 py-10">
          <div className="mb-8 max-w-2xl">
            <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-border/60 bg-secondary/40 px-3 py-1 text-xs text-muted-foreground">
              <Eye className="h-3.5 w-3.5 text-primary" /> 02 — Visualization
            </div>
            <h2 className="font-display text-3xl font-semibold tracking-tight">Side-by-Side Inspection</h2>
            <p className="mt-2 text-sm text-muted-foreground">Compare the raw input, the pure Grad-CAM signal, and the blended overlay.</p>
          </div>
          <div className="grid gap-4 md:grid-cols-3">
            <PanelCard title="Original MRI" badge="input"><PlaceholderMRI /></PanelCard>
            <PanelCard title="Grad-CAM Heatmap" badge="cam"><PlaceholderHeatmap /></PanelCard>
            <PanelCard title="Overlay Visualization" badge="blend"><Overlay /></PanelCard>
          </div>
        </section>

        {/* How it works */}
        <section className="mx-auto max-w-6xl px-4 py-10">
          <div className="mb-8 max-w-2xl">
            <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-border/60 bg-secondary/40 px-3 py-1 text-xs text-muted-foreground">
              <Activity className="h-3.5 w-3.5 text-primary" /> 03 — How It Works
            </div>
            <h2 className="font-display text-3xl font-semibold tracking-tight">From Pixels to Explanation</h2>
          </div>
          <div className="mx-auto max-w-2xl space-y-2">
            {flowSteps.map((step, i) => (
              <div key={step.label}>
                <div className="group rounded-2xl border border-border/60 glass p-5 transition-all hover:border-primary/60 hover:shadow-glow">
                  <div className="flex items-center gap-4">
                    <div className={`grid h-11 w-11 place-items-center rounded-xl ${i === 0 || i === flowSteps.length - 1 ? "bg-gradient-to-br from-primary to-accent" : "bg-secondary"}`}>
                      <step.icon className={`h-5 w-5 ${i === 0 || i === flowSteps.length - 1 ? "text-primary-foreground" : "text-foreground"}`} />
                    </div>
                    <div className="flex-1">
                      <div className="font-medium">{step.label}</div>
                      <div className="text-xs text-muted-foreground">{step.sub}</div>
                    </div>
                    <span className="hidden font-mono text-xs text-muted-foreground/60 md:block">step {i + 1}</span>
                  </div>
                </div>
                {i < flowSteps.length - 1 && (
                  <div className="flex justify-center py-1">
                    <ArrowDown className="h-4 w-4 animate-pulse text-muted-foreground/60" />
                  </div>
                )}
              </div>
            ))}
          </div>
        </section>

        {/* Why it matters */}
        <section className="mx-auto max-w-6xl px-4 py-10">
          <div className="mb-8 max-w-2xl">
            <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-border/60 bg-secondary/40 px-3 py-1 text-xs text-muted-foreground">
              <Lightbulb className="h-3.5 w-3.5 text-primary" /> 04 — Why It Matters
            </div>
            <h2 className="font-display text-3xl font-semibold tracking-tight">Why Explainability Matters</h2>
          </div>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            {whyCards.map((c) => (
              <div key={c.title} className="group rounded-2xl border border-border/60 glass p-5 transition-all hover:-translate-y-1 hover:shadow-glow">
                <div className="mb-3 grid h-10 w-10 place-items-center rounded-xl bg-gradient-to-br from-primary/80 to-accent/80">
                  <c.icon className="h-5 w-5 text-primary-foreground" />
                </div>
                <h3 className="font-display text-base font-semibold">{c.title}</h3>
                <p className="mt-1.5 text-sm text-muted-foreground">{c.body}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Research Notes */}
        <section className="mx-auto max-w-6xl px-4 py-10">
          <div className="rounded-2xl border border-border/60 glass p-6">
            <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-border/60 bg-secondary/40 px-3 py-1 text-xs text-muted-foreground">
              <Microscope className="h-3.5 w-3.5 text-primary" /> 05 — Research Notes
            </div>
            <h3 className="font-display text-xl font-semibold">A qualitative tool, not a proof</h3>
            <p className="mt-3 text-sm text-muted-foreground">
              Grad-CAM provides intuitive, qualitative insight into where a network's attention lies —
              but a highlighted region is not evidence of medical reasoning. Heatmaps can be coarse,
              sensitive to layer choice, and may surface correlations rather than causes. Treat them
              as a debugging and exploration aid, not as clinical justification.
            </p>
          </div>
        </section>

        {/* Disclaimer */}
        <section className="mx-auto max-w-6xl px-4 py-12">
          <div className="rounded-2xl border border-amber-500/30 bg-amber-500/5 p-6">
            <div className="flex items-start gap-4">
              <div className="grid h-10 w-10 place-items-center rounded-xl bg-amber-500/20">
                <AlertTriangle className="h-5 w-5 text-amber-400" />
              </div>
              <div>
                <h3 className="font-display text-lg font-semibold text-amber-100">Disclaimer</h3>
                <p className="mt-2 text-sm text-amber-100/80">
                  This visualization is intended for research and educational purposes only.
                </p>
              </div>
            </div>
          </div>
        </section>
      </main>
      <Footer />
    </div>
  );
}
