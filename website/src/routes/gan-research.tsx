import { createFileRoute, Link } from "@tanstack/react-router";
import {
  ArrowLeft,
  Brain,
  Network,
  GitBranch,
  AlertCircle,
  Layers,
  Zap,
  ShieldAlert,
  Sparkles,
  ImageOff,
  Lightbulb,
} from "lucide-react";
import { Navbar } from "@/components/site/Navbar";
import { Footer } from "@/components/site/Footer";

export const Route = createFileRoute("/gan-research")({
  head: () => ({
    meta: [
      { title: "GAN Research — Brain Tumor Detection" },
      {
        name: "description",
        content:
          "Exploratory synthetic MRI generation research using Generative Adversarial Networks for data augmentation and class-balance experiments.",
      },
      { property: "og:title", content: "GAN Research — Brain Tumor Detection" },
      {
        property: "og:description",
        content:
          "Exploring GAN-based synthetic MRI generation as a research direction for brain tumor classification data augmentation.",
      },
    ],
  }),
  component: GanResearchPage,
});

/* ────────────── Architecture Diagram ────────────── */

function ArchitectureDiagram() {
  return (
    <div className="relative rounded-2xl border border-border bg-black/40 p-6 sm:p-8">
      <div className="flex flex-col items-center gap-6 sm:flex-row sm:justify-center sm:gap-10">
        {/* Generator */}
        <div className="flex flex-col items-center gap-3">
          <div className="inline-flex items-center gap-2 rounded-full glass px-3 py-1 text-[10px] font-mono uppercase tracking-widest text-primary">
            <Sparkles className="h-3 w-3" />
            Generator
          </div>
          <div className="w-40 space-y-2 rounded-xl glass-strong p-4 text-center">
            <div className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
              Input
            </div>
            <div className="rounded-lg bg-secondary/50 py-2 text-xs font-mono text-foreground/80">
              Latent Vector z ~ N(0, I)
            </div>
            <div className="mx-auto h-5 w-px bg-border" />
            <div className="space-y-1.5">
              <div className="rounded-md bg-primary/10 px-2 py-1.5 text-[10px] font-mono text-primary ring-1 ring-primary/20">
                Transposed Conv 1
              </div>
              <div className="rounded-md bg-primary/10 px-2 py-1.5 text-[10px] font-mono text-primary ring-1 ring-primary/20">
                Transposed Conv 2
              </div>
              <div className="rounded-md bg-primary/10 px-2 py-1.5 text-[10px] font-mono text-primary ring-1 ring-primary/20">
                Transposed Conv 3
              </div>
            </div>
            <div className="mx-auto h-5 w-px bg-border" />
            <div className="rounded-lg bg-accent/10 py-2 text-xs font-mono text-accent ring-1 ring-accent/20">
              Synthetic MRI 128×128
            </div>
          </div>
        </div>

        {/* Discriminator */}
        <div className="flex flex-col items-center gap-3">
          <div className="inline-flex items-center gap-2 rounded-full glass px-3 py-1 text-[10px] font-mono uppercase tracking-widest text-destructive">
            <ShieldAlert className="h-3 w-3" />
            Discriminator
          </div>
          <div className="w-40 space-y-2 rounded-xl glass-strong p-4 text-center">
            <div className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
              Input
            </div>
            <div className="rounded-lg bg-secondary/50 py-2 text-xs font-mono text-foreground/80">
              Real or Fake MRI
            </div>
            <div className="mx-auto h-5 w-px bg-border" />
            <div className="space-y-1.5">
              <div className="rounded-md bg-destructive/10 px-2 py-1.5 text-[10px] font-mono text-destructive ring-1 ring-destructive/20">
                Conv Block 1
              </div>
              <div className="rounded-md bg-destructive/10 px-2 py-1.5 text-[10px] font-mono text-destructive ring-1 ring-destructive/20">
                Conv Block 2
              </div>
              <div className="rounded-md bg-destructive/10 px-2 py-1.5 text-[10px] font-mono text-destructive ring-1 ring-destructive/20">
                Conv Block 3
              </div>
            </div>
            <div className="mx-auto h-5 w-px bg-border" />
            <div className="rounded-lg bg-brand-2/10 py-2 text-xs font-mono text-brand-2 ring-1 ring-brand-2/20">
              Real / Fake
            </div>
          </div>
        </div>
      </div>

      {/* Adversarial arrows */}
      <div className="mt-6 flex items-center justify-center gap-4">
        <div className="flex items-center gap-2 rounded-full glass px-3 py-1.5 text-[10px] font-mono text-muted-foreground">
          <GitBranch className="h-3 w-3 text-primary" />
          <span className="text-foreground/80">Generator</span>
          <span className="text-muted-foreground">tries to fool</span>
          <span className="text-foreground/80">Discriminator</span>
        </div>
      </div>
    </div>
  );
}

/* ────────────── Placeholder Gallery Images ────────────── */

function PlaceholderSyntheticMRI({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 200 200" className={className}>
      <defs>
        <radialGradient id="ganBrain" cx="50%" cy="45%" r="50%">
          <stop offset="0%" stopColor="#b8c5d6" stopOpacity="0.8" />
          <stop offset="60%" stopColor="#7a8fa8" stopOpacity="0.6" />
          <stop offset="100%" stopColor="#4a5a6e" stopOpacity="0.4" />
        </radialGradient>
        <filter id="ganBlur">
          <feGaussianBlur stdDeviation="2" />
        </filter>
      </defs>
      <rect width="200" height="200" fill="#0a0e17" />
      <ellipse cx="100" cy="95" rx="70" ry="65" fill="url(#ganBrain)" />
      <ellipse cx="130" cy="85" rx="20" ry="18" fill="#6b7d94" opacity="0.5" filter="url(#ganBlur)" />
      <path d="M60 80 Q90 70 120 78" fill="none" stroke="#9aacbf" strokeWidth="1.5" opacity="0.4" />
      <path d="M65 100 Q95 90 125 98" fill="none" stroke="#9aacbf" strokeWidth="1.5" opacity="0.3" />
      <path d="M70 120 Q100 110 130 118" fill="none" stroke="#9aacbf" strokeWidth="1.5" opacity="0.25" />
      {/* Synthetic artifacts hint */}
      <ellipse cx="45" cy="150" rx="15" ry="8" fill="#4a5a6e" opacity="0.3" filter="url(#ganBlur)" />
      {Array.from({ length: 15 }).map((_, i) => (
        <circle
          key={i}
          cx={30 + Math.random() * 140}
          cy={30 + Math.random() * 140}
          r={0.3 + Math.random() * 1}
          fill="#c0cdd9"
          opacity={0.1 + Math.random() * 0.15}
        />
      ))}
    </svg>
  );
}

/* ────────────── Section Components ────────────── */

const architectureDetails = [
  {
    icon: Network,
    title: "DCGAN Architecture",
    body: "A Deep Convolutional GAN with transposed convolutions in the generator and strided convolutions in the discriminator. The generator maps a 100-dimensional latent vector through four upsampling layers to a 128×128 grayscale MRI.",
  },
  {
    icon: Layers,
    title: "Conditional Generation",
    body: "Class-conditional labels (Glioma, Meningioma, Pituitary, No Tumor) are embedded into both generator and discriminator to enable controlled generation of specific tumor types.",
  },
  {
    icon: Zap,
    title: "Spectral Normalization",
    body: "Applied to discriminator layers to stabilize training and prevent mode collapse. WGAN-GP loss was also experimented with to improve gradient quality.",
  },
];

const trainingChallenges = [
  {
    title: "Mode Collapse",
    body: "The generator frequently collapsed to producing a limited variety of outputs, especially for the minority Pituitary class. Multiple restarts and label smoothing were required to maintain diversity.",
  },
  {
    title: "Limited Training Data",
    body: "With only ~3,000 real MRI slices available, the discriminator overfitted quickly. Heavy augmentation and early stopping on a validation split were essential.",
  },
  {
    title: "Anatomical Fidelity",
    body: "Generated images often exhibited anatomically implausible structures — fused ventricles, asymmetric hemispheres, or blurred sulcal patterns that would never appear in real scans.",
  },
  {
    title: "Class Imbalance Propagation",
    body: "The underrepresented classes produced noticeably lower-quality generations, reinforcing rather than correcting the dataset imbalance the GAN was meant to address.",
  },
];

function ArchitectureSection() {
  return (
    <section>
      <div className="mb-5">
        <div className="font-mono text-[10px] uppercase tracking-widest text-primary">
          Architecture
        </div>
        <h2 className="mt-1 font-display text-2xl font-semibold">DCGAN for MRI Synthesis</h2>
        <p className="mt-2 text-sm text-muted-foreground">
          A class-conditional Deep Convolutional GAN designed to learn the distribution
          of brain MRI scans and generate plausible synthetic examples.
        </p>
      </div>

      <ArchitectureDiagram />

      <div className="mt-5 grid gap-4 sm:grid-cols-3">
        {architectureDetails.map((item, i) => (
          <div
            key={item.title}
            className="group relative rounded-2xl glass p-5 transition-all hover:-translate-y-0.5"
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

function TrainingChallengesSection() {
  return (
    <section>
      <div className="mb-5">
        <div className="font-mono text-[10px] uppercase tracking-widest text-destructive">
          Challenges
        </div>
        <h2 className="mt-1 font-display text-2xl font-semibold">Training Obstacles</h2>
        <p className="mt-2 text-sm text-muted-foreground">
          GAN training on medical images is notoriously difficult. These were the
          primary obstacles encountered during the research phase.
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        {trainingChallenges.map((item, i) => (
          <div
            key={item.title}
            className="group relative rounded-2xl glass p-5 transition-all hover:-translate-y-0.5"
          >
            <div className="flex items-start gap-3">
              <span className="mt-0.5 grid h-8 w-8 flex-none place-items-center rounded-lg bg-destructive/10 ring-1 ring-destructive/20">
                <AlertCircle className="h-4 w-4 text-destructive" />
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
  const labels = ["Glioma", "Meningioma", "Pituitary", "No Tumor", "Glioma", "Meningioma"];

  return (
    <section>
      <div className="mb-5">
        <div className="font-mono text-[10px] uppercase tracking-widest text-brand-2">
          Gallery
        </div>
        <h2 className="mt-1 font-display text-2xl font-semibold">Synthetic Generation Samples</h2>
        <p className="mt-2 text-sm text-muted-foreground">
          Representative outputs from the trained conditional DCGAN. While visually
          plausible at a glance, closer inspection reveals anatomical inconsistencies.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        {labels.map((label, i) => (
          <div key={i} className="group overflow-hidden rounded-2xl glass-strong p-2 transition-all hover:-translate-y-1">
            <div className="relative aspect-square overflow-hidden rounded-xl bg-black">
              <PlaceholderSyntheticMRI className="h-full w-full" />
              <div className="absolute left-2 top-2 inline-flex items-center gap-1 rounded-full glass px-2 py-0.5 text-[10px] font-mono uppercase tracking-widest text-foreground/80">
                <Sparkles className="h-2.5 w-2.5 text-primary" />
                {label}
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

function NotIncludedSection() {
  return (
    <section>
      <div className="mb-5">
        <div className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
          Outcome
        </div>
        <h2 className="mt-1 font-display text-2xl font-semibold">Why Synthetic Data Was Not Used</h2>
        <p className="mt-2 text-sm text-muted-foreground">
          A transparent account of why generated images remained an experimental
          branch rather than being merged into the final classification pipeline.
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <div className="rounded-2xl glass-strong p-5">
          <div className="flex items-start gap-3">
            <span className="grid h-9 w-9 flex-none place-items-center rounded-xl bg-destructive/10 ring-1 ring-destructive/20">
              <ImageOff className="h-4 w-4 text-destructive" />
            </span>
            <div>
              <h4 className="font-display text-sm font-semibold">Anatomical Inconsistencies</h4>
              <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
                Synthetic scans contained subtle but critical anatomical errors — blurred gray-white
                matter boundaries, implausible ventricle shapes, and missing sulcal patterns. Feeding
                these into the classifier risked teaching the model to rely on artifacts rather than
                real pathological features.
              </p>
            </div>
          </div>
        </div>

        <div className="rounded-2xl glass-strong p-5">
          <div className="flex items-start gap-3">
            <span className="grid h-9 w-9 flex-none place-items-center rounded-xl bg-destructive/10 ring-1 ring-destructive/20">
              <ShieldAlert className="h-4 w-4 text-destructive" />
            </span>
            <div>
              <h4 className="font-display text-sm font-semibold">Domain Shift Risk</h4>
              <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
                Adding synthetic data created a domain gap between real and generated images.
                Rather than improving generalization, preliminary experiments showed a slight
                degradation in validation accuracy when synthetic samples were mixed in.
              </p>
            </div>
          </div>
        </div>

        <div className="rounded-2xl glass-strong p-5">
          <div className="flex items-start gap-3">
            <span className="grid h-9 w-9 flex-none place-items-center rounded-xl bg-brand-2/10 ring-1 ring-brand-2/20">
              <AlertCircle className="h-4 w-4 text-brand-2" />
            </span>
            <div>
              <h4 className="font-display text-sm font-semibold">Mode Collapse Residue</h4>
              <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
                Despite mitigation techniques, the generator still exhibited mode collapse for
                minority classes. This meant synthetic Pituitary samples were nearly identical,
                offering no real diversity benefit and potentially reinforcing false correlations.
              </p>
            </div>
          </div>
        </div>

        <div className="rounded-2xl glass-strong p-5">
          <div className="flex items-start gap-3">
            <span className="grid h-9 w-9 flex-none place-items-center rounded-xl bg-primary/10 ring-1 ring-primary/20">
              <Lightbulb className="h-4 w-4 text-primary" />
            </span>
            <div>
              <h4 className="font-display text-sm font-semibold">Research Value Remains</h4>
              <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
                Even though synthetic data was not included in final predictions, the GAN experiments
                provided valuable insights into dataset characteristics, augmentation strategies, and
                the difficulty of generative modeling for medical imagery. This page documents
                that exploration.
              </p>
            </div>
          </div>
        </div>
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
            The GAN experiments documented here were conducted as part of an academic research
            project. Synthetic images were not used in any final classification pipeline and are
            shown purely for educational and exploratory purposes.
          </p>
        </div>
      </div>
    </div>
  );
}

/* ────────────── Page ────────────── */

function GanResearchPage() {
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
              An experimental research branch investigating GAN-based data augmentation for brain
              tumor classification. The goal was to study whether synthetic MRI scans could improve
              class balance and model robustness.
            </p>
          </div>

          {/* Sections */}
          <div className="mt-12 space-y-16">
            <ArchitectureSection />
            <TrainingChallengesSection />
            <GallerySection />
            <NotIncludedSection />
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
