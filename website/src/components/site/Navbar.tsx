import { Brain, Menu, X } from "lucide-react";
import { Link } from "@tanstack/react-router";
import { useEffect, useState } from "react";

const links = [
  { href: "/", label: "Home" },
  { href: "/research", label: "Research" },
  { href: "/methodology", label: "Methodology" },
  { href: "/explainable-ai", label: "Explainable AI" },
  { href: "/explainability", label: "Grad-CAM" },
  { href: "/synthetic-mri", label: "Synthetic MRI" },
  { href: "/gan-research", label: "GAN Research" },
];

export function Navbar() {
  const [open, setOpen] = useState(false);

  // Lock background scroll when the mobile menu is open
  useEffect(() => {
    if (!open) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prev;
    };
  }, [open]);

  // Close on Escape
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  const linkBase =
    "rounded-lg px-3 py-1.5 text-sm text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background";

  return (
    <header className="fixed top-0 inset-x-0 z-50">
      <div className="mx-auto mt-3 max-w-6xl px-3 sm:mt-4 sm:px-4">
        <nav
          aria-label="Primary"
          className="glass-strong flex items-center justify-between gap-3 rounded-2xl px-3 py-2.5 sm:px-4 sm:py-3"
        >
          <Link
            to="/"
            aria-label="Brain Tumor Detection — home"
            className="flex items-center gap-2 rounded-lg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            <span className="grid h-8 w-8 place-items-center rounded-lg bg-gradient-to-br from-primary to-accent shadow-glow">
              <Brain className="h-4 w-4 text-primary-foreground" aria-hidden />
            </span>
            <span className="font-display text-sm font-semibold tracking-tight">
              Brain Tumor Detection
            </span>
          </Link>

          <ul className="hidden items-center gap-1 lg:flex">
            {links.map((l) => (
              <li key={l.label}>
                <Link
                  to={l.href}
                  activeOptions={{ exact: l.href === "/" }}
                  activeProps={{
                    className: "text-foreground bg-secondary",
                    "aria-current": "page",
                  }}
                  className={linkBase}
                >
                  {l.label}
                </Link>
              </li>
            ))}
          </ul>

          <div className="flex items-center gap-2">
            <Link
              to="/demo"
              className="rounded-lg bg-foreground/95 px-3.5 py-1.5 text-sm font-medium text-background transition-transform hover:scale-[1.03] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background sm:px-4"
            >
              View Demo
            </Link>
            <button
              type="button"
              onClick={() => setOpen((v) => !v)}
              aria-expanded={open}
              aria-controls="mobile-nav"
              aria-label={open ? "Close menu" : "Open menu"}
              className="grid h-10 w-10 place-items-center rounded-lg text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring lg:hidden"
            >
              {open ? <Menu className="h-5 w-5 rotate-90 opacity-0 transition" aria-hidden /> : null}
              {open ? <X className="h-5 w-5" aria-hidden /> : <Menu className="h-5 w-5" aria-hidden />}
            </button>
          </div>
        </nav>
      </div>

      {/* Mobile menu */}
      <div
        id="mobile-nav"
        className={`lg:hidden ${open ? "pointer-events-auto" : "pointer-events-none"}`}
      >
        <div
          onClick={() => setOpen(false)}
          aria-hidden
          className={`fixed inset-0 bg-background/60 backdrop-blur-sm transition-opacity ${
            open ? "opacity-100" : "opacity-0"
          }`}
        />
        <div
          className={`fixed inset-x-3 top-20 origin-top rounded-2xl glass-strong p-2 shadow-card transition-all sm:inset-x-4 ${
            open ? "scale-100 opacity-100" : "scale-95 opacity-0"
          }`}
        >
          <ul className="flex flex-col">
            {links.map((l) => (
              <li key={l.label}>
                <Link
                  to={l.href}
                  onClick={() => setOpen(false)}
                  activeOptions={{ exact: l.href === "/" }}
                  activeProps={{
                    className: "text-foreground bg-secondary",
                    "aria-current": "page",
                  }}
                  className="block rounded-xl px-4 py-3 text-base text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                >
                  {l.label}
                </Link>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </header>
  );
}
