import { Layers3, Boxes, Wand2, Lightbulb, ArrowRight } from "lucide-react";
import { Link } from "@tanstack/react-router";

const items = [
  {
    icon: Layers3,
    title: "Transfer Learning",
    body: "Leveraging pretrained ImageNet representations to bootstrap performance on a comparatively small medical imaging dataset.",
  },
  {
    icon: Boxes,
    title: "Vision Transformers",
    body: "Exploring patch-based attention as an alternative inductive bias to convolutions for radiological imagery.",
  },
  {
    icon: Wand2,
    title: "Synthetic Data Generation",
    body: "Experimenting with GAN-generated MRI samples to study augmentation strategies and class-balance effects.",
    link: "/gan-research",
  },
  {
    icon: Lightbulb,
    title: "Explainability",
    body: "Grad-CAM visualizations and attention maps provide qualitative insight into where each model looks when classifying scans.",
  },
];

export function Research() {
  return (
    <section id="research" className="relative py-24 md:py-32">
      <div className="mx-auto max-w-6xl px-4">
        <div className="max-w-2xl">
          <div className="font-mono text-xs uppercase tracking-widest text-primary">
            03 — Research Highlights
          </div>
          <h2 className="mt-3 font-display text-3xl font-semibold tracking-tight sm:text-5xl">
            Themes explored in this <span className="text-gradient">project</span>
          </h2>
        </div>

        <div className="mt-12 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {items.map((it, i) => {
            if (it.link) {
              return (
                <Link
                  key={it.title}
                  to={it.link}
                  className="group relative rounded-2xl glass p-6 transition-all hover:-translate-y-1 hover:border-primary/40"
                >
                  <div className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
                    R.0{i + 1}
                  </div>
                  <div className="mt-3 inline-flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-primary/20 to-accent/20 ring-1 ring-primary/30">
                    <it.icon className="h-5 w-5 text-primary" />
                  </div>
                  <h3 className="mt-4 font-display text-lg font-semibold">{it.title}</h3>
                  <p className="mt-2 text-sm leading-relaxed text-muted-foreground">{it.body}</p>
                </Link>
              );
            }
            return (
              <div
                key={it.title}
                className="group relative rounded-2xl glass p-6 transition-all hover:-translate-y-1 hover:border-primary/40"
              >
                <div className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
                  R.0{i + 1}
                </div>
                <div className="mt-3 inline-flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-primary/20 to-accent/20 ring-1 ring-primary/30">
                  <it.icon className="h-5 w-5 text-primary" />
                </div>
                <h3 className="mt-4 font-display text-lg font-semibold">{it.title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-muted-foreground">{it.body}</p>
              </div>
            );
          })}
        </div>

        <div id="demo" className="mt-16 overflow-hidden rounded-3xl glass-strong p-8 sm:p-12">
          <div className="flex flex-col items-start gap-6 sm:flex-row sm:items-center sm:justify-between">
            <div className="max-w-xl">
              <h3 className="font-display text-2xl font-semibold tracking-tight sm:text-3xl">
                Try the interactive MRI demo
              </h3>
              <p className="mt-2 text-sm text-muted-foreground">
                Upload an MRI scan and watch the CNN and Vision Transformer pipelines classify it
                side by side, with animated confidence bars and a plain-language summary.
              </p>
            </div>
            <Link
              to="/demo"
              className="group inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-primary to-accent px-5 py-3 text-sm font-medium text-primary-foreground shadow-glow transition-transform hover:scale-[1.03]"
            >
              Launch demo
              <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
            </Link>
          </div>
        </div>
      </div>
    </section>
  );
}
