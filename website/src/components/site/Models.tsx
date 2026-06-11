import { Cpu, Sparkles, Check } from "lucide-react";

const models = [
  {
    badge: "Convolutional",
    name: "MobileNetV2",
    tag: "CNN · Transfer Learning",
    desc: "A compact convolutional architecture leveraging depthwise-separable convolutions and inverted residual blocks for efficient feature extraction.",
    points: [
      "Pretrained ImageNet weights, fine-tuned head",
      "224×224 input, 4-class softmax output",
      "Heavy data augmentation pipeline",
      "Pairs naturally with Grad-CAM explanations",
    ],
    icon: Cpu,
    accent: "from-primary/30 to-primary/0",
  },
  {
    badge: "Transformer",
    name: "ViT-Base/16",
    tag: "Vision Transformer",
    desc: "Treats each MRI as a sequence of 16×16 patches and learns global spatial dependencies via multi-head self-attention.",
    points: [
      "google/vit-base-patch16-224 backbone",
      "Patch embeddings + attention layers",
      "Captures long-range spatial context",
      "Independent comparison head to CNN",
    ],
    icon: Sparkles,
    accent: "from-accent/30 to-accent/0",
  },
];

export function Models() {
  return (
    <section id="models" className="relative py-24 md:py-32">
      <div className="mx-auto max-w-6xl px-4">
        <div className="flex items-end justify-between gap-6 max-w-3xl">
          <div>
            <div className="font-mono text-xs uppercase tracking-widest text-primary">
              02 — Model Comparison
            </div>
            <h2 className="mt-3 font-display text-3xl font-semibold tracking-tight sm:text-5xl">
              Two architectures, <span className="text-gradient">one dataset</span>
            </h2>
            <p className="mt-4 text-muted-foreground">
              A side-by-side study of how a lightweight CNN and a transformer-based vision
              model approach the same MRI classification task.
            </p>
          </div>
        </div>

        <div className="mt-12 grid gap-5 lg:grid-cols-2">
          {models.map((m) => (
            <article
              key={m.name}
              className="group relative overflow-hidden rounded-3xl glass-strong p-7"
            >
              <div
                className={`pointer-events-none absolute -top-24 right-0 h-64 w-64 rounded-full bg-gradient-to-br ${m.accent} blur-3xl`}
              />
              <div className="relative">
                <div className="flex items-center gap-3">
                  <span className="inline-flex h-10 w-10 items-center justify-center rounded-xl bg-foreground/5 ring-1 ring-foreground/10">
                    <m.icon className="h-5 w-5 text-primary" />
                  </span>
                  <span className="rounded-full bg-foreground/5 px-2.5 py-1 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
                    {m.badge}
                  </span>
                </div>

                <h3 className="mt-5 font-display text-3xl font-semibold">{m.name}</h3>
                <div className="mt-1 text-sm text-primary/90">{m.tag}</div>

                <p className="mt-4 text-sm leading-relaxed text-muted-foreground">{m.desc}</p>

                <ul className="mt-6 space-y-2.5">
                  {m.points.map((p) => (
                    <li key={p} className="flex items-start gap-3 text-sm">
                      <span className="mt-0.5 inline-flex h-5 w-5 flex-none items-center justify-center rounded-md bg-primary/15 ring-1 ring-primary/30">
                        <Check className="h-3 w-3 text-primary" />
                      </span>
                      <span className="text-foreground/80">{p}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </article>
          ))}
        </div>
      </div>
    </section>
  );
}
