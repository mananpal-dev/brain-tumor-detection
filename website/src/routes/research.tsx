import { createFileRoute } from "@tanstack/react-router";
import {
  Database, Layers, Sparkles, RotateCw, ZoomIn, Sun, ImageIcon,
  Cpu, Network, Grid3x3, Boxes, Activity, AlertTriangle, BookOpen,
  Microscope, Brain, Eye, Zap, ArrowDown, GitBranch, Hash,
} from "lucide-react";
import { Navbar } from "@/components/site/Navbar";
import { Footer } from "@/components/site/Footer";

export const Route = createFileRoute("/research")({
  head: () => ({
    meta: [
      { title: "Research & Methodology — Brain Tumor Detection" },
      { name: "description", content: "Technical depth: dataset, augmentation pipeline, CNN and Vision Transformer architectures, and research notes behind Brain Tumor Detection." },
      { property: "og:title", content: "Research & Methodology — Brain Tumor Detection" },
      { property: "og:description", content: "Inside the architecture: MobileNetV2, ViT, augmentation pipelines and dataset analysis." },
    ],
  }),
  component: ResearchPage,
});

const classes = [
  { name: "Glioma", count: "~1,621", color: "from-fuchsia-500/30 to-fuchsia-500/5", desc: "Tumors arising from glial cells in the brain and spinal cord." },
  { name: "Meningioma", count: "~1,645", color: "from-sky-500/30 to-sky-500/5", desc: "Typically benign tumors forming on meninges surrounding the brain." },
  { name: "Pituitary", count: "~1,757", color: "from-emerald-500/30 to-emerald-500/5", desc: "Growths on the pituitary gland at the base of the brain." },
  { name: "No Tumor", count: "~2,000", color: "from-amber-500/30 to-amber-500/5", desc: "Healthy MRI baseline scans with no detected tumor." },
];

const augmentations = [
  { icon: RotateCw, name: "Rotation", desc: "±15° random rotation" },
  { icon: ZoomIn, name: "Zoom", desc: "0.9–1.1× scale jitter" },
  { icon: Sun, name: "Brightness", desc: "±20% intensity shift" },
  { icon: ImageIcon, name: "Horizontal Flip", desc: "50% probability mirror" },
];

const cnnLayers = [
  { icon: ImageIcon, label: "Input MRI", sub: "224 × 224 × 3" },
  { icon: Cpu, label: "MobileNetV2 Backbone", sub: "Pre-trained on ImageNet" },
  { icon: Grid3x3, label: "Global Average Pooling", sub: "Spatial → 1280-d vector" },
  { icon: Layers, label: "Dense Layer", sub: "128 units · ReLU" },
  { icon: Activity, label: "Dropout", sub: "p = 0.3" },
  { icon: Sparkles, label: "Softmax Output", sub: "4 tumor classes" },
];

const vitLayers = [
  { icon: ImageIcon, label: "MRI Image", sub: "224 × 224 input" },
  { icon: Hash, label: "Patch Embedding", sub: "16 × 16 patches → tokens" },
  { icon: Boxes, label: "Transformer Encoder", sub: "12 stacked blocks" },
  { icon: Network, label: "Multi-Head Attention", sub: "12 heads · global context" },
  { icon: GitBranch, label: "Classification Head", sub: "[CLS] token → MLP" },
  { icon: Sparkles, label: "Prediction", sub: "4-way Softmax" },
];

function SectionTitle({ eyebrow, title, desc }: { eyebrow: string; title: string; desc?: string }) {
  return (
    <div className="mb-10 max-w-2xl">
      <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-border/60 bg-secondary/40 px-3 py-1 text-xs text-muted-foreground">
        <span className="h-1.5 w-1.5 rounded-full bg-primary" /> {eyebrow}
      </div>
      <h2 className="font-display text-3xl font-semibold tracking-tight md:text-4xl">{title}</h2>
      {desc && <p className="mt-3 text-muted-foreground">{desc}</p>}
    </div>
  );
}

function PipelineNode({ icon: Icon, label, sub, accent }: { icon: any; label: string; sub: string; accent?: boolean }) {
  return (
    <div className={`group relative w-full rounded-2xl border ${accent ? "border-primary/40" : "border-border/60"} glass p-5 transition-all hover:border-primary/60 hover:shadow-glow`}>
      <div className="flex items-center gap-4">
        <div className={`grid h-11 w-11 place-items-center rounded-xl ${accent ? "bg-gradient-to-br from-primary to-accent" : "bg-secondary"}`}>
          <Icon className={`h-5 w-5 ${accent ? "text-primary-foreground" : "text-foreground"}`} />
        </div>
        <div className="flex-1">
          <div className="font-medium">{label}</div>
          <div className="text-xs text-muted-foreground">{sub}</div>
        </div>
        <div className="hidden font-mono text-xs text-muted-foreground/60 md:block">layer</div>
      </div>
    </div>
  );
}

function Arrow() {
  return (
    <div className="flex justify-center py-1">
      <ArrowDown className="h-4 w-4 text-muted-foreground/60 animate-pulse" />
    </div>
  );
}

function ResearchPage() {
  return (
    <div id="main-content" className="min-h-dvh bg-background">
      <Navbar />
      <main className="pt-28">
        {/* Hero */}
        <section className="mx-auto max-w-6xl px-4 pb-16">
          <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-border/60 bg-secondary/40 px-3 py-1 text-xs text-muted-foreground">
            <Microscope className="h-3.5 w-3.5 text-primary" /> Research & Methodology
          </div>
          <h1 className="font-display text-4xl font-semibold tracking-tight md:text-6xl">
            Inside the <span className="bg-gradient-to-r from-primary via-accent to-primary bg-clip-text text-transparent">architecture</span>
          </h1>
          <p className="mt-4 max-w-2xl text-muted-foreground">
            A deep look at the dataset, augmentation pipeline, and dual-architecture approach used in Brain Tumor Detection —
            combining convolutional priors with transformer-scale attention.
          </p>
        </section>

        {/* Dataset */}
        <section className="mx-auto max-w-6xl px-4 py-12">
          <SectionTitle
            eyebrow="01 — Dataset"
            title="Kaggle Brain MRI Dataset"
            desc="A curated public dataset of T1-weighted MRI scans across four diagnostic categories, balanced for supervised training."
          />
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {classes.map((c) => (
              <div key={c.name} className={`group relative overflow-hidden rounded-2xl border border-border/60 glass p-5 transition-all hover:-translate-y-1 hover:shadow-glow`}>
                <div className={`pointer-events-none absolute inset-0 bg-gradient-to-br ${c.color} opacity-60`} />
                <div className="relative">
                  <div className="flex items-center justify-between">
                    <Brain className="h-5 w-5 text-foreground/80" />
                    <span className="font-mono text-xs text-muted-foreground">{c.count}</span>
                  </div>
                  <div className="mt-4 font-display text-lg font-semibold">{c.name}</div>
                  <div className="mt-1 text-xs text-muted-foreground">{c.desc}</div>
                </div>
              </div>
            ))}
          </div>

          {/* Class distribution */}
          <div className="mt-6 rounded-2xl border border-border/60 glass p-6">
            <div className="mb-4 flex items-center justify-between">
              <div className="flex items-center gap-2 text-sm font-medium"><Database className="h-4 w-4 text-primary" /> Class Distribution</div>
              <span className="font-mono text-xs text-muted-foreground">~7,023 samples</span>
            </div>
            <div className="space-y-3">
              {classes.map((c, i) => {
                const widths = [82, 84, 89, 100];
                return (
                  <div key={c.name} className="flex items-center gap-3">
                    <div className="w-24 text-xs text-muted-foreground">{c.name}</div>
                    <div className="h-3 flex-1 overflow-hidden rounded-full bg-secondary">
                      <div
                        className="h-full rounded-full bg-gradient-to-r from-primary to-accent transition-all"
                        style={{ width: `${widths[i]}%` }}
                      />
                    </div>
                    <div className="w-16 text-right font-mono text-xs text-muted-foreground">{widths[i]}%</div>
                  </div>
                );
              })}
            </div>
          </div>
        </section>

        {/* Augmentation Pipeline */}
        <section className="mx-auto max-w-6xl px-4 py-12">
          <SectionTitle
            eyebrow="02 — Preprocessing"
            title="Data Augmentation Pipeline"
            desc="Synthetic variation expands the effective dataset and improves generalisation across orientation and intensity."
          />
          <div className="rounded-2xl border border-border/60 glass p-6">
            <div className="grid items-center gap-4 md:grid-cols-[1fr_auto_1fr_auto_1fr_auto_1fr_auto_1fr]">
              {[
                { icon: ImageIcon, label: "MRI Image" },
                { icon: RotateCw, label: "Rotation" },
                { icon: ZoomIn, label: "Zoom" },
                { icon: Sun, label: "Brightness" },
                { icon: Sparkles, label: "Training Sample" },
              ].map((step, idx, arr) => (
                <div key={step.label} className="contents">
                  <div className="flex flex-col items-center gap-2 rounded-xl border border-border/60 bg-secondary/40 p-4 text-center">
                    <step.icon className="h-5 w-5 text-primary" />
                    <div className="text-xs font-medium">{step.label}</div>
                  </div>
                  {idx < arr.length - 1 && (
                    <div className="hidden md:flex justify-center">
                      <div className="h-px w-6 bg-gradient-to-r from-primary/60 to-accent/60" />
                    </div>
                  )}
                </div>
              ))}
            </div>

            <div className="mt-6 grid grid-cols-2 gap-3 md:grid-cols-4">
              {augmentations.map((a) => (
                <div key={a.name} className="rounded-xl border border-border/50 bg-background/40 p-4">
                  <a.icon className="h-4 w-4 text-accent" />
                  <div className="mt-2 text-sm font-medium">{a.name}</div>
                  <div className="text-xs text-muted-foreground">{a.desc}</div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* CNN + ViT side by side */}
        <section className="mx-auto max-w-6xl px-4 py-12">
          <SectionTitle
            eyebrow="03 — Architectures"
            title="Dual-Model Approach"
            desc="A convolutional backbone for local features, paired with a transformer for long-range attention — compared head-to-head."
          />
          <div className="grid gap-6 lg:grid-cols-2">
            {/* CNN */}
            <div className="rounded-2xl border border-border/60 glass p-6">
              <div className="mb-4 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Cpu className="h-5 w-5 text-primary" />
                  <h3 className="font-display text-xl font-semibold">CNN — MobileNetV2</h3>
                </div>
                <span className="rounded-full bg-secondary px-2 py-0.5 font-mono text-[10px] text-muted-foreground">~3.5M params</span>
              </div>
              <div className="space-y-2">
                {cnnLayers.map((l, i) => (
                  <div key={l.label}>
                    <PipelineNode icon={l.icon} label={l.label} sub={l.sub} accent={i === 0 || i === cnnLayers.length - 1} />
                    {i < cnnLayers.length - 1 && <Arrow />}
                  </div>
                ))}
              </div>
            </div>

            {/* ViT */}
            <div className="rounded-2xl border border-border/60 glass p-6">
              <div className="mb-4 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Network className="h-5 w-5 text-accent" />
                  <h3 className="font-display text-xl font-semibold">Vision Transformer (ViT-B/16)</h3>
                </div>
                <span className="rounded-full bg-secondary px-2 py-0.5 font-mono text-[10px] text-muted-foreground">~86M params</span>
              </div>
              <div className="space-y-2">
                {vitLayers.map((l, i) => (
                  <div key={l.label}>
                    <PipelineNode icon={l.icon} label={l.label} sub={l.sub} accent={i === 0 || i === vitLayers.length - 1} />
                    {i < vitLayers.length - 1 && <Arrow />}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* Comparison Table */}
        <section className="mx-auto max-w-6xl px-4 py-12">
          <SectionTitle
            eyebrow="04 — Comparison"
            title="Model Comparison"
            desc="An educational breakdown — strengths, trade-offs, and the why behind each architecture choice."
          />
          <div className="overflow-hidden rounded-2xl border border-border/60 glass">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-secondary/60 text-left text-xs uppercase tracking-wider text-muted-foreground">
                  <tr>
                    <th className="px-5 py-4">Aspect</th>
                    <th className="px-5 py-4">CNN (MobileNetV2)</th>
                    <th className="px-5 py-4">Vision Transformer</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/60">
                  {[
                    ["Parameters", "~3.5M (lightweight)", "~86M (heavy)"],
                    ["Strengths", "Local feature extraction, fast, mobile-friendly", "Global context, scales with data, attention maps"],
                    ["Weaknesses", "Limited receptive field, biased to texture", "Data-hungry, slower, harder to train from scratch"],
                    ["Explainability", "Grad-CAM, intuitive heatmaps", "Attention rollout, per-head attention"],
                    ["Inference Speed", "Fast (real-time on CPU)", "Slower (benefits from GPU)"],
                    ["Best For", "Edge deployment & rapid iteration", "Research & high-capacity reasoning"],
                  ].map((row) => (
                    <tr key={row[0]} className="transition-colors hover:bg-secondary/30">
                      <td className="px-5 py-4 font-medium">{row[0]}</td>
                      <td className="px-5 py-4 text-muted-foreground">{row[1]}</td>
                      <td className="px-5 py-4 text-muted-foreground">{row[2]}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </section>

        {/* Research Notes */}
        <section className="mx-auto max-w-6xl px-4 py-12">
          <SectionTitle
            eyebrow="05 — Notes"
            title="Research Observations"
            desc="Reflections on what each architecture brings — and what the medical-imaging domain demands."
          />
          <div className="grid gap-4 md:grid-cols-2">
            {[
              { icon: Cpu, title: "Why CNNs work well", body: "Convolutions encode strong spatial priors — locality and translation invariance match how lesions appear consistently across MRI slices, even with limited data." },
              { icon: Eye, title: "Why Vision Transformers are interesting", body: "Self-attention models long-range dependencies between distant regions, capturing global anatomical context that convolutions only see in deeper layers." },
              { icon: Zap, title: "Challenges in medical imaging", body: "Class imbalance, inter-scanner variability, small annotated cohorts, and patient privacy all complicate building robust, generalizable models." },
              { icon: BookOpen, title: "Dataset limitations", body: "Kaggle datasets are curated and not clinically validated. Real-world deployment would require multi-site cohorts, expert relabeling, and prospective trials." },
            ].map((n) => (
              <div key={n.title} className="group rounded-2xl border border-border/60 glass p-6 transition-all hover:-translate-y-0.5 hover:shadow-glow">
                <div className="mb-3 grid h-10 w-10 place-items-center rounded-xl bg-gradient-to-br from-primary/80 to-accent/80">
                  <n.icon className="h-5 w-5 text-primary-foreground" />
                </div>
                <h3 className="font-display text-lg font-semibold">{n.title}</h3>
                <p className="mt-2 text-sm text-muted-foreground">{n.body}</p>
              </div>
            ))}
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
                <h3 className="font-display text-lg font-semibold text-amber-100">Research Disclaimer</h3>
                <p className="mt-2 text-sm text-amber-100/80">
                  This project is an academic and research implementation and is not intended for clinical diagnosis.
                  All outputs are illustrative only and must not inform medical decisions.
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
