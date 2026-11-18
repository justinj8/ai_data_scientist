import Plotly from "plotly.js-dist-min";
import { useEffect, useRef } from "react";
import type { PlotlyFigure } from "../api/types";

const COLORWAY = ["#0F7C86", "#4C51BF", "#C98A1B", "#C2416B", "#5B6B7F", "#2E7D4F", "#8B5CF6", "#D97706"];

// Plotly Express hard-codes its default palette into each trace; map it onto ours.
const PLOTLY_DEFAULTS = ["#636efa", "#ef553b", "#00cc96", "#ab63fa", "#ffa15a", "#19d3f3", "#ff6692", "#b6e880"];

function recolor<T>(value: T): T {
  if (typeof value === "string") {
    const i = PLOTLY_DEFAULTS.indexOf(value.toLowerCase());
    return (i === -1 ? value : COLORWAY[i % COLORWAY.length]) as T;
  }
  if (Array.isArray(value)) return value.map(recolor) as T;
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, recolor(v)])) as T;
  }
  return value;
}

function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/** Renders a Plotly figure from the backend, restyled to match the app theme. */
export function PlotlyChart({ figure, height = 320 }: { figure: PlotlyFigure; height?: number }) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const draw = () => {
      const ink = cssVar("--ink-2");
      const rule = cssVar("--rule");
      const layout = {
        ...(figure.layout ?? {}),
        template: undefined,
        height,
        autosize: true,
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { family: "Schibsted Grotesk, system-ui, sans-serif", size: 12, color: ink },
        colorway: COLORWAY,
        margin: { l: 56, r: 16, t: 44, b: 48, ...((figure.layout?.margin as object) ?? {}) },
        xaxis: { gridcolor: rule, zerolinecolor: rule, ...((figure.layout?.xaxis as object) ?? {}) },
        yaxis: { gridcolor: rule, zerolinecolor: rule, ...((figure.layout?.yaxis as object) ?? {}) },
        legend: { orientation: "h", y: -0.2, ...((figure.layout?.legend as object) ?? {}) },
        title: figure.layout?.title
          ? { ...(figure.layout.title as object), font: { size: 13, color: cssVar("--ink") }, x: 0, xanchor: "left" }
          : undefined,
      };
      void Plotly.react(el, recolor(figure.data) as Plotly.Data[], layout as Partial<Plotly.Layout>, {
        responsive: true,
        displaylogo: false,
        modeBarButtonsToRemove: ["lasso2d", "select2d"],
      });
    };
    draw();
    const observer = new MutationObserver(draw); // re-theme on light/dark switch
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => {
      observer.disconnect();
      Plotly.purge(el);
    };
  }, [figure, height]);

  return <div ref={ref} className="w-full" style={{ minHeight: height }} />;
}

export async function downloadChartPng(el: HTMLElement, name: string) {
  const plot = el.querySelector(".js-plotly-plot") as HTMLElement | null;
  if (plot) await Plotly.downloadImage(plot, { format: "png", filename: name, width: 1200, height: 700 });
}
