import { Scan, Layers, Network, Eye } from "lucide-react";

const items = [
  {
    icon: Scan,
    title: "MRI Classification",
    body: "Multi-class classification across Glioma, Meningioma, Pituitary, and No-Tumor MRI scans from the Kaggle Brain MRI dataset.",
  },
  {
    icon: Layers,
    title: "CNN Approach",
    body: "Transfer learning with MobileNetV2 — a lightweight convolutional backbone fine-tuned with augmentation and dropout regularization.",
  },
  {
    icon: Network,
    title: "Transformer Approach",
    body: "Fine-tuned ViT-Base/16 treating MRI scans as sequences of patches, capturing long-range spatial relationships via self-attention.",
  },
  {
    icon: Eye,
    title: "Explainable AI",
    body: "Grad-CAM heatmaps visualize the regions driving each CNN prediction, surfacing model focus for qualitative inspection.",
  },
];

export function Overview() {
  return (
    <section id="overview" className="relative py-24 md:py-32">
      <div className="mx-auto max-w-6xl px-4">
        <div className="max-w-2xl">
          <div className="font-mono text-xs uppercase tracking-widest text-primary">
            01 — Overview
          </div>
          <h2 className="mt-3 font-display text-3xl font-semibold tracking-tight sm:text-5xl">
            A study in <span className="text-gradient">vision architectures</span>
          </h2>
          <p className="mt-4 text-muted-foreground">
            An end-to-end research pipeline exploring how convolutional networks and vision
            transformers learn to distinguish tumor categories from MRI imagery — paired with
            visual explanations of model behavior.
          </p>
        </div>

        <div className="mt-12 grid gap-4 sm:grid-cols-2">
          {items.map((it) => (
            <div
              key={it.title}
              className="group relative overflow-hidden rounded-2xl glass p-6 transition-all hover:-translate-y-1"
            >
              <div className="absolute -right-16 -top-16 h-40 w-40 rounded-full bg-primary/10 blur-3xl transition-opacity group-hover:opacity-100" />
              <div className="relative">
                <div className="inline-flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br from-primary/20 to-accent/20 ring-1 ring-primary/30">
                  <it.icon className="h-5 w-5 text-primary" />
                </div>
                <h3 className="mt-5 font-display text-xl font-semibold">{it.title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-muted-foreground">{it.body}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
