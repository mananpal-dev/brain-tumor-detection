import { Github, Linkedin, Globe, Brain } from "lucide-react";

const socials = [
  { icon: Github, label: "GitHub", href: "#" },
  { icon: Linkedin, label: "LinkedIn", href: "#" },
  { icon: Globe, label: "Portfolio", href: "#" },
];

export function Footer() {
  return (
    <footer className="relative border-t border-border/60 py-12">
      <div className="mx-auto max-w-6xl px-4">
        <div className="flex flex-col items-start justify-between gap-8 md:flex-row md:items-center">
          <div>
            <div className="flex items-center gap-2">
              <span className="grid h-8 w-8 place-items-center rounded-lg bg-gradient-to-br from-primary to-accent">
                <Brain className="h-4 w-4 text-primary-foreground" />
              </span>
              <span className="font-display text-sm font-semibold">Brain Tumor Detection Research</span>
            </div>
            <p className="mt-3 max-w-md text-xs text-muted-foreground">
              An academic research and portfolio project exploring CNN and Vision Transformer
              architectures for MRI classification. Not a medical product.
            </p>
          </div>

          <div className="flex items-center gap-2">
            {socials.map((s) => (
              <a
                key={s.label}
                href={s.href}
                aria-label={s.label}
                className="inline-flex h-10 w-10 items-center justify-center rounded-xl glass transition-all hover:-translate-y-0.5 hover:border-primary/40"
              >
                <s.icon className="h-4 w-4" />
              </a>
            ))}
          </div>
        </div>

        <div className="mt-10 flex flex-col items-start justify-between gap-2 border-t border-border/60 pt-6 text-xs text-muted-foreground md:flex-row md:items-center">
          <div>© {new Date().getFullYear()} Brain Tumor Detection Research · Built for academic purposes.</div>
          <div className="font-mono">Manan Pal · B.Tech CSE</div>
        </div>
      </div>
    </footer>
  );
}
