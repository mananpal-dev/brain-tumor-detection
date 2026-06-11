import { createFileRoute } from "@tanstack/react-router";
import {
  Database, Layers, Sparkles, ImageIcon, Cpu, Network, Grid3x3, Boxes,
  Activity, AlertTriangle, Hash, GitBranch, ArrowDown, Scale, Maximize,
  Wand2, GraduationCap, FlaskConical, Eye, Rocket, CheckCircle2,
  Cog, Microscope, ShieldAlert,
} from "lucide-react";
import { Navbar } from "@/components/site/Navbar";
import { Footer } from "@/components/site/Footer";

export const Route = createFileRoute("/methodology")({
  head: () => ({
    meta: [
      { title: "Research Methodology — Brain Tumor Detection" },
      { name: "description", content: "End-to-end engineering workflow behind Brain Tumor Detection: dataset, preprocessing, CNN and ViT pipelines, model comparison, and development journey." },
      { property: "og:title", content: "Research Methodology — Brain Tumor Detection" },
      { property: "og:description", content: "The technical workflow behind Brain Tumor Detection — built for recruiters and researchers." },
    ],
  }),
  component: MethodologyPage,
});

const classes = [
  { name: "Glioma", count: "~1,621", tint: "from-fuchsia-500/30 to-fuchsia-500/5", desc: "Tumors arising from glial cells in the brain and spinal cord." },
  { name: "Meningioma", count: "~1,645", tint: "from-sky-500/30 to-sky-500/5", desc: "Typically benign tumors forming on the meninges." },
  { name: "Pituitary", count: "~1,757", tint: "from-emerald-500/30 to-emerald-500/5", desc: "Growths on the pituitary gland at the base of the brain." },
  { name: "No Tumor", count: "~2,000", tint: "from-amber-500/30 to-amber-500/5", desc: "Healthy MRI baseline scans with no detected tumor." },
];

const prepSteps = [
  { icon: Database, label: "MRI Dataset", sub: "Raw T1 / T2 weighted scans" },
  { icon: Maximize, label: "Resize", sub: "224 × 224 standard input" },
  { icon: Scale, label: "Normalization", sub: "Pixel scaling [0, 1]" },
  { icon: Wand2, label: "Data Augmentation", sub: "Rotation · Zoom · Flip" },
  { icon: GraduationCap, label: "Training", sub: "Batched mini-epochs" },
];

const cnnLayers = [
  { icon: ImageIcon, label: "MRI Input", sub: "224 × 224 × 3" },
  { icon: Cpu, label: "MobileNetV2", sub: "Pre-trained backbone" },
  { icon: Layers, label: "Dense Layers", sub: "128 units · ReLU + Dropout" },
  { icon: Sparkles, label: "Classification", sub: "4-class Softmax" },
];

const vitLayers = [
  { icon: ImageIcon, label: "MRI Input", sub: "224 × 224" },
  { icon: Hash, label: "Patch Embeddings", sub: "16 × 16 token patches" },
  { icon: Boxes, label: "Transformer Encoder", sub: "Multi-head self-attention" },
  { icon: GitBranch, label: "Classification Head", sub: "[CLS] → MLP → Softmax" },
];

const journey = [
  { icon: Database, label: "Dataset Preparation", note: "Cleaning, balancing & splits" },
  { icon: Cpu, label: "CNN Training", note: "MobileNetV2 transfer learning" },
  { icon: Network, label: "ViT Fine-Tuning", note: "Attention-based experimentation" },
  { icon: FlaskConical, label: "GAN Experimentation", note: "Exploratory synthetic MRI" },
  { icon: Eye, label: "Grad-CAM Analysis", note: "Visualizing model focus" },
  { icon: Rocket, label: "Deployment", note: "Interactive web demo" },
];

const limitations = [
  { icon: Database, title: "Academic Dataset", desc: "Trained on a public Kaggle dataset — not representative of real clinical distributions." },
  { icon: Cog, title: "Limited Hardware", desc: "Experiments ran on consumer GPUs, constraining ViT depth and batch size." },
  { icon: Microscope, title: "No Clinical Validation", desc: "Predictions were never reviewed or validated by medical professionals." },
  { icon: ShieldAlert, title: "Research-Only Implementation", desc: "Intended purely for educational and engineering demonstration." },
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
      </div>
    </div>
  );
}

function Arrow() {
  return (
    <div className="flex justify-center py-1">
      <ArrowDown className="h-4 w-4 text-muted-foreground/60" />
    </div>
  );
}

function VerticalPipeline({ steps }: { steps: { icon: any; label: string; sub: string }[] }) {
  return (
    <div className="space-y-2">
      {steps.map((s, i) => (
        <div key={s.label}>
          <PipelineNode icon={s.icon} label={s.label} sub={s.sub} accent={i === 0 || i === steps.length - 1} />
          {i < steps.length - 1 && <Arrow />}
        </div>
      ))}
    </div>
  );
}

function MethodologyPage() {
  return (
    <div id="main-content" className="min-h-dvh bg-background text-foreground">
      <Navbar />

      <main className="pt-28">
        {/* Hero */}
        <section className="mx-auto max-w-6xl px-4 pb-16">
          <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-border/60 bg-secondary/40 px-3 py-1 text-xs text-muted-foreground">
            <span className="h-1.5 w-1.5 rounded-full bg-primary" /> Research Methodology
          </div>
          <h1 className="font-display text-4xl font-semibold tracking-tight md:text-6xl">
            The engineering workflow <span className="bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">behind Brain Tumor Detection</span>
          </h1>
          <p className="mt-4 max-w-3xl text-muted-foreground">
            A transparent look at the technical decisions, model architectures, and iterative experiments
            that shaped the project — from raw MRI data to a deployed interactive demo.
          </p>
        </section>

        {/* Dataset */}
        <section className="mx-auto max-w-6xl px-4 py-16">
          <SectionTitle
            eyebrow="Dataset"
            title="Kaggle Brain MRI Dataset"
            desc="A publicly available multi-class brain MRI dataset spanning four diagnostic categories."
          />
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            {classes.map((c) => (
              <div key={c.name} className={`relative overflow-hidden rounded-2xl border border-border/60 glass p-5 transition-all hover:border-primary/50 hover:shadow-glow`}>
                <div className={`pointer-events-none absolute inset-0 bg-gradient-to-br ${c.tint} opacity-60`} />
                <div className="relative">
                  <div className="flex items-center justify-between">
                    <div className="text-xs uppercase tracking-wider text-muted-foreground">Class</div>
                    <div className="font-mono text-xs text-muted-foreground">{c.count}</div>
                  </div>
                  <div className="mt-2 font-display text-xl font-semibold">{c.name}</div>
                  <p className="mt-2 text-sm text-muted-foreground">{c.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Data Preparation Pipeline */}
        <section className="mx-auto max-w-3xl px-4 py-16">
          <SectionTitle
            eyebrow="Data Preparation"
            title="From raw scans to training samples"
            desc="A reproducible preprocessing pipeline applied to every MRI before it reaches the model."
          />
          <VerticalPipeline steps={prepSteps} />
        </section>

        {/* CNN & ViT Pipelines */}
        <section className="mx-auto max-w-6xl px-4 py-16">
          <SectionTitle
            eyebrow="Architectures"
            title="CNN & Vision Transformer pipelines"
            desc="Two distinct inductive biases — convolutional locality vs. global self-attention."
          />
          <div className="grid gap-8 lg:grid-cols-2">
            <div>
              <div className="mb-4 flex items-center gap-2 text-sm text-muted-foreground">
                <Cpu className="h-4 w-4 text-primary" /> CNN Pipeline · MobileNetV2
              </div>
              <VerticalPipeline steps={cnnLayers} />
            </div>
            <div>
              <div className="mb-4 flex items-center gap-2 text-sm text-muted-foreground">
                <Network className="h-4 w-4 text-accent" /> Vision Transformer Pipeline
              </div>
              <VerticalPipeline steps={vitLayers} />
            </div>
          </div>
        </section>

        {/* Comparison Table */}
        <section className="mx-auto max-w-6xl px-4 py-16">
          <SectionTitle
            eyebrow="Model Comparison"
            title="CNN vs. Vision Transformer"
            desc="A side-by-side engineering comparison — not a benchmark race."
          />
          <div className="overflow-hidden rounded-2xl border border-border/60 glass">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-secondary/40 text-left text-xs uppercase tracking-wider text-muted-foreground">
                  <tr>
                    <th className="px-5 py-3">Aspect</th>
                    <th className="px-5 py-3">CNN (MobileNetV2)</th>
                    <th className="px-5 py-3">Vision Transformer</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/40">
                  {[
                    ["Architecture", "Depthwise separable convolutions", "Patch-based self-attention"],
                    ["Parameter Count", "~3.5M", "~86M"],
                    ["Inference Speed", "Fast on CPU & mobile", "Heavier — favors GPU"],
                    ["Explainability", "Strong with Grad-CAM", "Attention rollouts (less localized)"],
                    ["Strengths", "Lightweight · spatial priors", "Global context · scales with data"],
                    ["Limitations", "Limited long-range reasoning", "Data hungry · compute intensive"],
                  ].map(([a, b, c]) => (
                    <tr key={a} className="transition-colors hover:bg-secondary/20">
                      <td className="px-5 py-4 font-medium">{a}</td>
                      <td className="px-5 py-4 text-muted-foreground">{b}</td>
                      <td className="px-5 py-4 text-muted-foreground">{c}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </section>

        {/* Development Journey */}
        <section className="mx-auto max-w-6xl px-4 py-16">
          <SectionTitle
            eyebrow="Development Journey"
            title="An iterative engineering timeline"
            desc="Each milestone informed the next — from dataset prep to a live web demo."
          />
          <div className="relative">
            <div className="absolute left-5 top-0 hidden h-full w-px bg-gradient-to-b from-primary/40 via-border to-transparent md:block" />
            <ol className="space-y-5">
              {journey.map((j, i) => (
                <li key={j.label} className="relative md:pl-16">
                  <div className="absolute left-0 top-1 hidden h-10 w-10 place-items-center rounded-xl bg-gradient-to-br from-primary to-accent shadow-glow md:grid">
                    <j.icon className="h-5 w-5 text-primary-foreground" />
                  </div>
                  <div className="rounded-2xl border border-border/60 glass p-5 transition-all hover:border-primary/50">
                    <div className="flex items-center gap-2 text-xs text-muted-foreground">
                      <span className="font-mono">Step {String(i + 1).padStart(2, "0")}</span>
                      <span className="h-1 w-1 rounded-full bg-muted-foreground/40" />
                      <span>{j.note}</span>
                    </div>
                    <div className="mt-1 font-display text-lg font-semibold">{j.label}</div>
                  </div>
                </li>
              ))}
            </ol>
          </div>
        </section>

        {/* Limitations */}
        <section className="mx-auto max-w-6xl px-4 py-16">
          <SectionTitle
            eyebrow="Project Limitations"
            title="An honest scope statement"
            desc="Transparency about what this project is — and what it isn't."
          />
          <div className="grid gap-4 md:grid-cols-2">
            {limitations.map((l) => (
              <div key={l.title} className="rounded-2xl border border-border/60 glass p-5 transition-all hover:border-amber-400/40">
                <div className="flex items-start gap-4">
                  <div className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-amber-500/10 text-amber-300">
                    <l.icon className="h-5 w-5" />
                  </div>
                  <div>
                    <div className="font-medium">{l.title}</div>
                    <p className="mt-1 text-sm text-muted-foreground">{l.desc}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="mt-8 flex items-start gap-3 rounded-2xl border border-amber-500/30 bg-amber-500/5 p-5">
            <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-amber-400" />
            <p className="text-sm text-amber-100/90">
              This project is an academic and research implementation and is not intended for clinical diagnosis.
            </p>
          </div>
        </section>

        {/* Conclusion */}
        <section className="mx-auto max-w-4xl px-4 py-20">
          <SectionTitle eyebrow="Conclusion" title="What this project demonstrates" />
          <div className="rounded-2xl border border-border/60 glass p-6 md:p-8">
            <p className="text-muted-foreground">
              Brain Tumor Detection is an end-to-end exploration of modern deep learning applied to medical imaging —
              spanning classical CNN architectures, transformer-based vision models, generative experimentation,
              and explainability tooling. The emphasis throughout has been on{" "}
              <span className="text-foreground">engineering rigor, transparency, and iterative experimentation</span>{" "}
              rather than chasing a leaderboard score.
            </p>
            <ul className="mt-5 grid gap-3 md:grid-cols-2">
              {[
                "Practical understanding of preprocessing & augmentation",
                "Hands-on training of CNN and ViT models",
                "Use of Grad-CAM for model interpretability",
                "Exploration of GAN-based data synthesis",
              ].map((p) => (
                <li key={p} className="flex items-start gap-2 text-sm">
                  <CheckCircle2 className="mt-0.5 h-4 w-4 text-primary" />
                  <span className="text-muted-foreground">{p}</span>
                </li>
              ))}
            </ul>
          </div>
        </section>
      </main>

      <Footer />
    </div>
  );
}
