import { createFileRoute, Link } from "@tanstack/react-router";
import {
  ArrowLeft,
  Sparkles,
  Brain,
  Database,
  Layers,
  Zap,
  MonitorOff,
  Waves,
  AlertCircle,
  ImageOff,
  TrendingUp,
  Frown,
  Aperture,
  ShieldAlert,
} from "lucide-react";
import { Navbar } from "@/components/site/Navbar";
import { Footer } from "@/components/site/Footer";

export const Route = createFileRoute("/synthetic-mri")({
  head: () => ({
    meta: [
      { title: "Synthetic MRI Generation — Brain Tumor Detection" },
      {
        name: "description",
        content:
          "Exploratory research into GAN-based synthetic MRI generation for brain tumor classification — architecture, challenges, and research findings.",
      },
      { property: "og:title", content: "Synthetic MRI Generation — Brain Tumor Detection" },
      {
        property: "og:description",
        content:
          "An educational exploration of GAN-based synthetic MRI generation — architecture, training challenges, and research insights.",
      },
    ],
  }),
  component: SyntheticMriPage,
});

/* ────────────── Placeholder MRI Image ────────────── */

function PlaceholderMRI({ label }: { label?: string }) {
  return (
    <svg viewBox="0 0 200 200" className="h-full w-full">
      <defs>
        <radialGradient id="synthBrain" cx="50%" cy="45%" r="50%">
          <stop offset="0%" stopColor="#b8c5d6" stopOpacity="0.85" />
          <stop offset="60%" stopColor="#7a8fa8" stopOpacity="0.6" />
          <stop offset="100%" stopColor="#4a5a6e" stopOpacity="0.35" />
        </radialGradient>
        <filter id="synthBlur"><feGaussianBlur stdDeviation="2" /></filter>
        <filter id="synthBlurLight"><feGaussianBlur stdDeviation="1" /></filter>
      </defs>
      <rect width="200" height="200" fill="#0a0e17" />
      <ellipse cx="100" cy="95" rx="70" ry="65" fill="url(#synthBrain)" />
      <ellipse cx="130" cy="85" rx="20" ry="18" fill="#6b7d94" opacity="0.5" filter="url(#synthBlur)" />
      <path d="M60 80 Q90 70 120 78" fill="none" stroke="#9aacbf" strokeWidth="1.5" opacity="0.4" />
      <path d="M65 100 Q95 90 125 98" fill="none" stroke="#9aacbf" strokeWidth="1.5" opacity="0.3" />
      <path d="M70 120 Q100 110 130 118" fill="none" stroke="#9aacbf" strokeWidth="1.5" opacity="0.25" />
      {/* Synthetic artifacts */}
      <ellipse cx="45" cy="150" rx="15" ry="8" fill="#4a5a6e" opacity="0.3" filter="url(#synthBlur)" />
      <ellipse cx="155" cy="60" rx="12" ry="6" fill="#5a6a7e" opacity="0.25" filter="url(#synthBlurLight)" />
      {Array.from({ length: 15 }).map((_, i) => (
        <circle
          key={i}
          cx={30 + ((i * 37) % 140)}
          cy={30 + ((i * 23) % 140)}
          r={0.3 + (i % 3) * 0.4}
          fill="#c0cdd9"
          opacity={0.1 + (i % 5) * 0.03}
        />
      ))}
      {label && (
        <text x="100" y="190" textAnchor="middle" fill="#7a8fa8" fontSize="8" fontFamily="monospace" opacity="0.6">
          {label}
        </text>
      )}
    </svg>
  );
}

/* ────────────── Architecture Diagram ────────────── */

function ArchitectureDiagram() {
  const genSteps = [
    { label: "Noise Vector", sub: "z ~ N(0, I)", dim: "100-d" },
    { label: "Dense Layers", sub: "FC projection", dim: "4×4×512" },
    { label: "Upsampling", sub: "Transposed Conv", dim: "8×8 → 16×16" },
    { label: "Upsampling", sub: "Transposed Conv", dim: "32×32 → 64×64" },
    { label: "Upsampling", sub: "Transposed Conv", dim: "128×128" },
  ];

  const discSteps = [
    { label: "MRI Image", sub: "Real or Fake", dim: "128×128" },
    { label: "Conv Layers", sub: "Strided Conv", dim: "64×64 → 32×32" },
    { label: "Conv Layers", sub: "Strided Conv", dim: "16×16 → 8×8" },
    { label: "Dense Layer", sub: "FC classification", dim: "1-d" },
    { label: "Output", sub: "Real vs Fake", dim: "Probability" },
  ];

  return (
    <div className="relative rounded-2xl border border-border bg-black/40 p-6 sm:p-8">
      <div className="grid gap-8 lg:grid-cols-2">
        {/* Generator Column */}
        <div className="space-y-3">
          <div className="mb-4 inline-flex items-center gap-2 rounded-full glass px-3 py-1 text-[10px] font-mono uppercase tracking-widest text-primary">
            <Sparkles className="h-3 w-3" />
            Generator
          </div>
          <div className="space-y-0">
            {genSteps.map((step, i) => (
              <div key={i}>
                <div className="rounded-xl glass-strong px-4 py-3">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-xs font-medium text-foreground">{step.label}</div>
                      <div className="text-[10px] text-muted-foreground">{step.sub}</div>
                    </div>
                    <span className="rounded-md bg-primary/10 px-2 py-0.5 font-mono text-[10px] text-primary ring-1 ring-primary/20">
                      {step.dim}
                    </span>
                  </div>
                </div>
                {i < genSteps.length - 1 && (
                  <div className="flex justify-center py-1.5">
                    <div className="h-4 w-px bg-gradient-to-b from-primary/40 to-primary/10" />
                  </div>
                )}
              </div>
            ))}
          </div>
          <div className="rounded-xl border border-accent/20 bg-accent/5 p-3 text-center">
            <span className="text-xs font-mono text-accent">Synthetic MRI Output</span>
          </div>
        </div>

        {/* Discriminator Column */}
        <div className="space-y-3">
          <div className="mb-4 inline-flex items-center gap-2 rounded-full glass px-3 py-1 text-[10px] font-mono uppercase tracking-widest text-destructive">
            <ShieldAlert className="h-3 w-3" />
            Discriminator
          </div>
          <div className="space-y-0">
            {discSteps.map((step, i) => (
              <div key={i}>
                <div className="rounded-xl glass-strong px-4 py-3">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-xs font-medium text-foreground">{step.label}</div>
                      <div className="text-[10px] text-muted-foreground">{step.sub}</div>
                    </div>
                    <span className="rounded-md bg-destructive/10 px-2 py-0.5 font-mono text-[10px] text-destructive ring-1 ring-destructive/20">
                      {step.dim}
                    </span>
                  </div>
                </div>
                {i < discSteps.length - 1 && (
                  <div className="flex justify-center py-1.5">
                    <div className="h-4 w-px bg-gradient-to-b from-destructive/40 to-destructive/10" />
                  </div>
                )}
              </div>
            ))}
          </div>
          <div className="rounded-xl border border-brand-2/20 bg-brand-2/5 p-3 text-center">
            <span className="text-xs font-mono text-brand-2">Real / Fake Classification</span>
          </div>
        </div>
      </div>

      {/* Adversarial flow indicator */}
      <div className="mt-6 flex items-center justify-center gap-3">
        <div className="flex items-center gap-2 rounded-full glass px-3 py-1.5 text-[10px] font-mono text-muted-foreground">
          <Sparkles className="h-3 w-3 text-primary" />
          <span className="text-foreground/80">Generator</span>
          <span className="text-muted-foreground">← adversarial →</span>
          <span className="text-foreground/80">Discriminator</span>
          <ShieldAlert className="h-3 w-3 text-destructive" />
        </div>
      </div>
    </div>
  );
}

/* ────────────── Section Data ────────────── */

const introCards = [
  {
    icon: Database,
    title: "Data Scarcity",
    body: "Medical imaging datasets are notoriously difficult to acquire. Privacy regulations, institutional approvals, and expert annotation requirements limit dataset sizes to hundreds or low thousands of images.",
  },
  {
    icon: Brain,
    title: "Why Synthetic Images?",
    body: "Synthetic data could augment scarce datasets, improve class balance, and provide controlled variations for stress-testing models — all without compromising patient privacy.",
  },
  {
    icon: Layers,
    title: "Research Motivation",
    body: "This exploration investigated whether a Deep Convolutional GAN could learn the statistical distribution of brain MRI scans and generate anatomically plausible synthetic examples.",
  },
];

const trainingChallenges = [
  {
    icon: MonitorOff,
    title: "Limited Hardware",
    body: "Training a GAN requires significant GPU memory and compute. Experiments were constrained to a single consumer GPU, forcing smaller batch sizes and longer training times.",
  },
  {
    icon: Waves,
    title: "Training Instability",
    body: "The adversarial objective creates a delicate balance. Small hyperparameter changes caused oscillating losses, vanishing gradients, or sudden divergence mid-training.",
  },
  {
    icon: AlertCircle,
    title: "Mode Collapse",
    body: "The generator frequently converged to producing a limited set of similar outputs, especially for minority classes. Multiple restarts and label smoothing were required to maintain diversity.",
  },
  {
    icon: ImageOff,
    title: "Image Quality Limitations",
    body: "Generated images exhibited anatomical inconsistencies — blurred boundaries, implausible structures, and artifacts that distinguish them from real scans upon close inspection.",
  },
];

const galleryItems = [
  { label: "Glioma", tag: "synthetic" },
  { label: "Meningioma", tag: "synthetic" },
  { label: "Pituitary", tag: "synthetic" },
  { label: "No Tumor", tag: "synthetic" },
  { label: "Glioma", tag: "synthetic" },
  { label: "Meningioma", tag: "synthetic" },
];

const researchFindings = [
  {
    icon: Sparkles,
    title: "Promising Visual Structures",
    body: "At a glance, generated images resembled real brain MRI scans. The GAN successfully learned coarse anatomical layouts — skull shape, ventricle placement, and gray-white matter contrast.",
    tone: "positive" as const,
  },
  {
    icon: Frown,
    title: "Anatomical Inconsistencies",
    body: "Closer inspection revealed subtle but critical errors: asymmetric hemispheres, blurred sulcal patterns, and ventricles with implausible shapes. These artifacts risked confusing the classifier.",
    tone: "caution" as const,
  },
  {
    icon: ShieldAlert,
    title: "Not Included in Final Pipeline",
    body: "Due to anatomical fidelity concerns, synthetic images were not merged into the final classification pipeline. The real dataset alone provided more reliable training signals.",
    tone: "neutral" as const,
  },
  {
    icon: TrendingUp,
    title: "Valuable Research Insights",
    body: "Despite not being production-ready, the GAN experiments deepened understanding of dataset characteristics, augmentation strategies, and the unique challenges of medical generative modeling.",
    tone: "positive" as const,
  },
];

const futureDirections = [
  {
    title: "StyleGAN",
    body: "Style-based architectures with progressive growing could dramatically improve synthesis quality and control over generated features.",
  },
  {
    title: "Diffusion Models",
    body: "Diffusion-based approaches like Stable Diffusion have surpassed GANs in image quality. Conditional diffusion on MRI data is a promising next research step.",
  },
  {
    title: "Larger Datasets",
    body: "Training on multi-institutional datasets (e.g., BraTS) with tens of thousands of volumes would provide the statistical depth needed for high-fidelity generation.",
  },
];

/* ────────────── Sections ────────────── */

function IntroductionSection() {
  return (
    <section>
      <div className="mb-5">
        <div className="font-mono text-[10px] uppercase tracking-widest text-primary">Introduction</div>
        <h2 className="mt-1 font-display text-2xl font-semibold">Why Synthetic Medical Images?</h2>
        <p className="mt-2 text-sm text-muted-foreground">
          Medical imaging datasets are expensive, private, and difficult to scale. Generative models offer
          a compelling research direction for augmenting scarce data without breaching patient confidentiality.
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-3">
        {introCards.map((item) => (
          <div
            key={item.title}
            className="group rounded-2xl glass p-5 transition-all hover:-translate-y-0.5"
          >
            <div className="relative flex items-start gap-3">
              <span className="grid h-9 w-9 flex-none place-items-center rounded-xl bg-gradient-to-br from-primary/20 to-accent/20 ring-1 ring-primary/30">
                <item.icon className="h-4 w-4 text-primary" />
              </span>
              <div>
                <h4 className="font-display text-sm font-semibold">{item.title}</h4>
                <p className="mt-1 text-xs leading-relaxed text-muted-foreground">{item.body}</p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function ArchitectureSection() {
  return (
    <section>
      <div className="mb-5">
        <div className="font-mono text-[10px] uppercase tracking-widest text-primary">Architecture</div>
        <h2 className="mt-1 font-display text-2xl font-semibold">GAN Architecture for MRI Synthesis</h2>
        <p className="mt-2 text-sm text-muted-foreground">
          A Deep Convolutional GAN with transposed convolutions in the generator and strided convolutions
          in the discriminator. The two networks train adversarially to produce realistic synthetic scans.
        </p>
      </div>

      <ArchitectureDiagram />
    </section>
  );
}

function TrainingChallengesSection() {
  return (
    <section>
      <div className="mb-5">
        <div className="font-mono text-[10px] uppercase tracking-widest text-destructive">Challenges</div>
        <h2 className="mt-1 font-display text-2xl font-semibold">Training Obstacles</h2>
        <p className="mt-2 text-sm text-muted-foreground">
          GAN training on medical imagery is notoriously unstable. These were the key challenges encountered during the research phase.
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        {trainingChallenges.map((item) => (
          <div
            key={item.title}
            className="group rounded-2xl glass p-5 transition-all hover:-translate-y-0.5"
          >
            <div className="flex items-start gap-3">
              <span className="mt-0.5 grid h-8 w-8 flex-none place-items-center rounded-lg bg-destructive/10 ring-1 ring-destructive/20">
                <item.icon className="h-4 w-4 text-destructive" />
              </span>
              <div>
                <h4 className="font-display text-sm font-semibold">{item.title}</h4>
                <p className="mt-1 text-xs leading-relaxed text-muted-foreground">{item.body}</p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function GallerySection() {
  return (
    <section>
      <div className="mb-5">
        <div className="font-mono text-[10px] uppercase tracking-widest text-brand-2">Gallery</div>
        <h2 className="mt-1 font-display text-2xl font-semibold">Sample Generation Gallery</h2>
        <p className="mt-2 text-sm text-muted-foreground">
          Representative synthetic outputs from the trained conditional DCGAN. Each image is generated
          from a random latent vector conditioned on a target class label.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        {galleryItems.map((item, i) => (
          <div
            key={i}
            className="group overflow-hidden rounded-2xl glass-strong p-2 transition-all hover:-translate-y-1"
          >
            <div className="relative aspect-square overflow-hidden rounded-xl bg-black">
              <PlaceholderMRI label={`${item.label} · 128×128`} />
              <div className="absolute left-2 top-2 inline-flex items-center gap-1 rounded-full glass px-2 py-0.5 text-[10px] font-mono uppercase tracking-widest text-foreground/80">
                <Sparkles className="h-2.5 w-2.5 text-primary" />
                {item.label}
              </div>
              <div className="absolute inset-0 flex items-center justify-center opacity-0 transition-opacity group-hover:opacity-100">
                <div className="rounded-full glass px-3 py-1.5 text-[10px] font-mono text-foreground/80">
                  Synthetic
                </div>
              </div>
            </div>
            <div className="mt-2 flex items-center justify-between px-1">
              <span className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
                Sample 0{i + 1}
              </span>
              <span className="text-[10px] font-mono text-muted-foreground">128×128</span>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function ResearchFindingsSection() {
  const toneClasses = {
    positive: "bg-primary/10 ring-primary/20",
    caution: "bg-destructive/10 ring-destructive/20",
    neutral: "bg-brand-2/10 ring-brand-2/20",
  };

  const iconTone = {
    positive: "text-primary",
    caution: "text-destructive",
    neutral: "text-brand-2",
  };

  return (
    <section>
      <div className="mb-5">
        <div className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">Findings</div>
        <h2 className="mt-1 font-display text-2xl font-semibold">Research Findings</h2>
        <p className="mt-2 text-sm text-muted-foreground">
          An honest assessment of what the GAN experiments achieved — and why the generated images
          remained a research exploration rather than entering the production pipeline.
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        {researchFindings.map((item) => (
          <div
            key={item.title}
            className="group rounded-2xl glass-strong p-5 transition-all hover:-translate-y-0.5"
          >
            <div className="flex items-start gap-3">
              <span className={`grid h-9 w-9 flex-none place-items-center rounded-xl ring-1 ${toneClasses[item.tone]}`}>
                <item.icon className={`h-4 w-4 ${iconTone[item.tone]}`} />
              </span>
              <div>
                <h4 className="font-display text-sm font-semibold">{item.title}</h4>
                <p className="mt-1 text-xs leading-relaxed text-muted-foreground">{item.body}</p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function FutureDirectionsSection() {
  return (
    <section>
      <div className="mb-5">
        <div className="font-mono text-[10px] uppercase tracking-widest text-accent">Future</div>
        <h2 className="mt-1 font-display text-2xl font-semibold">Future Directions</h2>
        <p className="mt-2 text-sm text-muted-foreground">
          The GAN experiments opened several promising avenues for future research in medical image synthesis.
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-3">
        {futureDirections.map((item, i) => (
          <div
            key={item.title}
            className="group rounded-2xl glass p-5 transition-all hover:-translate-y-0.5"
          >
            <div className="mb-3 flex items-center gap-2">
              <span className="grid h-8 w-8 place-items-center rounded-lg bg-gradient-to-br from-accent/20 to-primary/20 ring-1 ring-accent/30">
                <Aperture className="h-4 w-4 text-accent" />
              </span>
              <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
                Direction 0{i + 1}
              </span>
            </div>
            <h4 className="font-display text-sm font-semibold">{item.title}</h4>
            <p className="mt-1 text-xs leading-relaxed text-muted-foreground">{item.body}</p>
          </div>
        ))}
      </div>
    </section>
  );
}

function Disclaimer() {
  return (
    <div className="rounded-2xl glass p-4">
      <div className="flex items-start gap-3">
        <ShieldAlert className="mt-0.5 h-4 w-4 flex-none text-primary" />
        <div>
          <p className="text-xs font-medium text-foreground">Research Exploration Only</p>
          <p className="mt-1 text-xs text-muted-foreground">
            The synthetic MRI generation work documented here is an academic research exploration.
            Generated images were not used in any final classification pipeline and are shown purely
            for educational and research purposes. This is not a medical diagnostic system.
          </p>
        </div>
      </div>
    </div>
  );
}

/* ────────────── Page ────────────── */

function SyntheticMriPage() {
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
              <Sparkles className="h-3.5 w-3.5 text-primary" />
              Generative AI
            </div>
            <h1 className="mt-5 font-display text-4xl font-semibold leading-[1.05] tracking-tight sm:text-5xl md:text-6xl">
              Exploratory{" "}
              <span className="text-gradient">Synthetic MRI Generation</span>
            </h1>
            <p className="mt-4 max-w-2xl text-sm text-muted-foreground sm:text-base">
              A research exploration into using Generative Adversarial Networks to synthesize
              brain MRI scans. This page documents the architecture, challenges, and insights
              gained from the GAN experiments — presented as an educational research branch.
            </p>
          </div>

          {/* Sections */}
          <div className="mt-12 space-y-16">
            <IntroductionSection />
            <ArchitectureSection />
            <TrainingChallengesSection />
            <GallerySection />
            <ResearchFindingsSection />
            <FutureDirectionsSection />
          </div>

          {/* Disclaimer */}
          <div className="mt-12">
            <Disclaimer />
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
}
