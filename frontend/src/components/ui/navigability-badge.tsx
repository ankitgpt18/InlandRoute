// ============================================================
// InlandRoute — Minimalist Navigability Badge Component
// ============================================================

import React from 'react';
import type { NavigabilityClass } from '@/types';

export interface NavigabilityBadgeProps {
  navigabilityClass: NavigabilityClass | string;
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

export function NavigabilityBadge({
  navigabilityClass,
  size = 'md',
  className = '',
}: NavigabilityBadgeProps) {
  const cls = String(navigabilityClass).toLowerCase();

  const labels: Record<string, string> = {
    navigable: 'Navigable',
    conditional: 'Conditional',
    non_navigable: 'Non-Navigable',
  };

  const bgStyles: Record<string, string> = {
    navigable: 'badge-navigable',
    conditional: 'badge-conditional',
    non_navigable: 'badge-non-navigable',
  };

  const sizeStyles: Record<string, string> = {
    sm: 'text-[10px] px-2 py-0.5 font-bold',
    md: 'text-xs px-2.5 py-1 font-semibold',
    lg: 'text-xs px-3 py-1.5 font-bold tracking-wide',
  };

  const styleClass = bgStyles[cls] ?? 'bg-slate-500/15 border-slate-500/30 text-slate-400';
  const sizeClass = sizeStyles[size] ?? sizeStyles.md;

  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border ${styleClass} ${sizeClass} ${className}`}>
      <span className="w-1.5 h-1.5 rounded-full bg-current" />
      <span>{labels[cls] ?? navigabilityClass}</span>
    </span>
  );
}

export function NavigabilityDot({ navigabilityClass }: { navigabilityClass: NavigabilityClass | string }) {
  const cls = String(navigabilityClass).toLowerCase();
  const color =
    cls === 'navigable' ? 'bg-emerald-500' : cls === 'conditional' ? 'bg-amber-500' : 'bg-rose-500';
  return <span className={`inline-block w-2 h-2 rounded-full ${color}`} />;
}

export function MapLegendCard() {
  return (
    <div className="p-3.5 text-xs space-y-2.5 bg-zinc-900/90 border border-zinc-700 backdrop-blur-md shadow-2xl rounded-xl">
      <div className="font-bold text-white text-[11px] uppercase tracking-wider">
        IWAI Navigability Legend
      </div>
      <div className="space-y-1.5 text-zinc-300 font-medium">
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full bg-emerald-400" />
          <span>Navigable (depth ≥ 3.0m)</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full bg-amber-400" />
          <span>Conditional (2.0m – 3.0m)</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full bg-rose-400" />
          <span>Non-Navigable (&lt; 2.0m)</span>
        </div>
      </div>
    </div>
  );
}
