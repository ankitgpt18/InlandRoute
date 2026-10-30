// ============================================================
// InlandRoute  -  Minimalist Non-AI Navigability Badge Component
// ============================================================

import React from "react";
import type { NavigabilityClass } from "@/types";

export interface NavigabilityBadgeProps {
  navigabilityClass: NavigabilityClass | string;
  size?: "sm" | "md" | "lg";
  className?: string;
}

export function NavigabilityBadge({
  navigabilityClass,
  size = "md",
  className = "",
}: NavigabilityBadgeProps) {
  const cls = String(navigabilityClass).toLowerCase();

  const labels: Record<string, string> = {
    navigable: "Navigable",
    conditional: "Conditional",
    non_navigable: "Non-Navigable",
  };

  const dotColors: Record<string, string> = {
    navigable: "bg-emerald-500",
    conditional: "bg-amber-400",
    non_navigable: "bg-rose-500",
  };

  const sizeStyles: Record<string, string> = {
    sm: "text-[11px] px-2 py-0.5",
    md: "text-xs px-2.5 py-1",
    lg: "text-xs px-3 py-1 font-medium",
  };

  const sizeClass = sizeStyles[size] ?? sizeStyles.md;

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-md border border-neutral-800 bg-neutral-900 text-neutral-200 font-medium ${sizeClass} ${className}`}
    >
      <span
        className={`h-1.5 w-1.5 rounded-full shrink-0 ${
          dotColors[cls] ?? "bg-neutral-500"
        }`}
      />
      <span>{labels[cls] ?? navigabilityClass}</span>
    </span>
  );
}

export function NavigabilityDot({
  navigabilityClass,
}: {
  navigabilityClass: NavigabilityClass | string;
}) {
  const cls = String(navigabilityClass).toLowerCase();
  const color =
    cls === "navigable"
      ? "bg-emerald-500"
      : cls === "conditional"
      ? "bg-amber-400"
      : "bg-rose-500";
  return <span className={`inline-block w-2 h-2 rounded-full ${color}`} />;
}

export function MapLegendCard() {
  return (
    <div className="p-3 text-xs space-y-2 bg-neutral-950/95 border border-neutral-800 shadow-xl rounded-lg">
      <div className="font-semibold text-white text-[11px] uppercase tracking-wider">
        Navigability Standard
      </div>
      <div className="space-y-1.5 text-[11px] text-neutral-300">
        <div className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-emerald-500" />
          <span>Navigable (LAD &gt; 2.5m)</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-amber-400" />
          <span>Conditional (2.0m - 2.5m)</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-rose-500" />
          <span>Restricted (LAD &lt; 2.0m)</span>
        </div>
      </div>
    </div>
  );
}
