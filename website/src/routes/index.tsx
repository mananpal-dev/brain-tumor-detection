import { createFileRoute } from "@tanstack/react-router";
import { Navbar } from "@/components/site/Navbar";
import { Hero } from "@/components/site/Hero";
import { Overview } from "@/components/site/Overview";
import { Models } from "@/components/site/Models";
import { Research } from "@/components/site/Research";
import { Footer } from "@/components/site/Footer";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Brain Tumor Detection — Brain Tumor Detection with CNN & Vision Transformer" },
      {
        name: "description",
        content:
          "Deep learning–based MRI classification and explainable AI research project comparing MobileNetV2 and Vision Transformer architectures.",
      },
      { property: "og:title", content: "Brain Tumor Detection — CNN & Vision Transformer MRI Research" },
      {
        property: "og:description",
        content:
          "Academic research project exploring CNN and ViT architectures for brain MRI classification with Grad-CAM explainability.",
      },
    ],
  }),
  component: Index,
});

function Index() {
  return (
    <div id="main-content" className="min-h-dvh bg-background text-foreground">
      <Navbar />
      <main>
        <Hero />
        <Overview />
        <Models />
        <Research />
      </main>
      <Footer />
    </div>
  );
}
