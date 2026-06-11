import { ArrowRight, Play, Sparkles } from "lucide-react";
import { Link } from "@tanstack/react-router";

const techs = [
  { name: "MobileNetV2", desc: "Efficient CNN backbone" },
  { name: "Vision Transformer", desc: "Attention-based vision" },
  { name: "GAN", desc: "Synthetic MRI generation" },
  { name: "Grad-CAM", desc: "Visual explainability" },
];

export function Hero() {
  return (
    <section className="relative overflow-hidden pt-36 pb-24 md:pt-44 md:pb-32">
      <div className="absolute inset-0 grid-bg" aria-hidden />
      <div
        className="absolute inset-0 -z-10"
        style={{ background: "var(--gradient-hero)" }}
        aria-hidden
      />

      <div className="mx-auto max-w-6xl px-4">
        <div className="mx-auto max-w-4xl text-center animate-fade-up">
          <div className="inline-flex items-center gap-2 rounded-full glass px-3 py-1 text-xs text-muted-foreground">
            <Sparkles className="h-3.5 w-3.5 text-primary" />
            Research Project · Computer Vision · Explainable AI
          </div>

          <h1 className="mt-6 font-display text-4xl font-semibold leading-[1.05] tracking-tight sm:text-6xl md:text-7xl">
            <span className="text-gradient">Brain Tumor Detection</span>
            <br />
            <span className="text-foreground/90">using CNN and Vision Transformer</span>
          </h1>

          <p className="mx-auto mt-6 max-w-2xl text-base text-muted-foreground sm:text-lg">
            A deep learning–based MRI classification and explainable AI research project
            comparing convolutional and transformer architectures on multi-class brain tumor data.
          </p>

          <div className="mt-9 flex flex-wrap items-center justify-center gap-3">
            <a
              href="#overview"
              className="group inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-primary to-accent px-5 py-3 text-sm font-medium text-primary-foreground shadow-glow transition-transform hover:scale-[1.03]"
            >
              Explore Project
              <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
            </a>
            <Link
              to="/demo"
              className="inline-flex items-center gap-2 rounded-xl glass px-5 py-3 text-sm font-medium text-foreground hover:bg-secondary/60"
            >
              <Play className="h-4 w-4" />
              View Demo
            </Link>
          </div>
        </div>

        <div className="mt-20 grid grid-cols-2 gap-3 sm:gap-4 md:grid-cols-4">
          {techs.map((t, i) => (
            <div
              key={t.name}
              className="group relative animate-fade-up rounded-2xl glass p-5 transition-all hover:-translate-y-1 hover:border-primary/40"
              style={{ animationDelay: `${120 * i}ms` }}
            >
              <div className="absolute inset-x-6 top-0 h-px bg-gradient-to-r from-transparent via-primary/60 to-transparent opacity-0 transition-opacity group-hover:opacity-100" />
              <div className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
                0{i + 1}
              </div>
              <div className="mt-2 font-display text-lg font-semibold">{t.name}</div>
              <div className="mt-1 text-xs text-muted-foreground">{t.desc}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
