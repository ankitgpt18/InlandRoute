import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';
import { NavigabilityClass, AlertSeverity, Month, MONTH_LABELS } from '@/types';

// ─── Core className merger ────────────────────────────────────────────────────

/**
 * Merges Tailwind CSS class names, resolving conflicts intelligently.
 * Combines clsx (conditional classes) with tailwind-merge (conflict resolution).
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

// ─── Navigability helpers ─────────────────────────────────────────────────────

/**
 * Returns the hex color for a navigability class.
 */
export function getNavigabilityColor(cls: NavigabilityClass): string {
  switch (cls) {
    case 'navigable':     return '#22c55e';
    case 'conditional':   return '#f59e0b';
    case 'non_navigable': return '#ef4444';
    default:              return '#6b7280';
  }
}

/**
 * Returns Tailwind text-color class for a navigability class.
 */
export function getNavigabilityTextClass(cls: NavigabilityClass): string {
  switch (cls) {
    case 'navigable':     return 'text-slate-400';
    case 'conditional':   return 'text-slate-400';
    case 'non_navigable': return 'text-slate-400';
    default:              return 'text-slate-400';
  }
}

/**
 * Returns Tailwind background-color class for a navigability class.
 */
export function getNavigabilityBgClass(cls: NavigabilityClass): string {
  switch (cls) {
    case 'navigable':     return 'bg-green-500/20 border-green-500/30';
    case 'conditional':   return 'bg-amber-500/20 border-amber-500/30';
    case 'non_navigable': return 'bg-red-500/20   border-red-500/30';
    default:              return 'bg-slate-500/20 border-slate-500/30';
  }
}

/**
 * Returns a human-readable label for a navigability class.
 */
export function getNavigabilityLabel(cls: NavigabilityClass): string {
  switch (cls) {
    case 'navigable':     return 'Navigable';
    case 'conditional':   return 'Conditional';
    case 'non_navigable': return 'Non-Navigable';
    default:              return 'Unknown';
  }
}

/**
 * Returns the glow shadow class for a navigability class.
 */
export function getNavigabilityGlowClass(cls: NavigabilityClass): string {
  switch (cls) {
    case 'navigable':     return 'shadow-glow-green';
    case 'conditional':   return 'shadow-glow-amber';
    case 'non_navigable': return 'shadow-glow-red';
    default:              return '';
  }
}

/**
 * Derives a NavigabilityClass from a depth value against standard thresholds.
 */
export function classifyDepth(
  depth_m: number,
  navigable_threshold = 3.0,
  conditional_threshold = 2.0,
): NavigabilityClass {
  if (depth_m >= navigable_threshold)   return 'navigable';
  if (depth_m >= conditional_threshold) return 'conditional';
  return 'non_navigable';
}

// ─── Alert helpers ────────────────────────────────────────────────────────────

/**
 * Returns Tailwind color classes for an alert severity.
 */
export function getAlertSeverityClasses(severity: AlertSeverity): {
  text: string;
  bg: string;
  border: string;
  dot: string;
} {
  switch (severity) {
    case 'CRITICAL':
      return {
        text:   'text-slate-400',
        bg:     'bg-red-500/15',
        border: 'border-red-500/40',
        dot:    'bg-red-500',
      };
    case 'WARNING':
      return {
        text:   'text-slate-400',
        bg:     'bg-amber-500/15',
        border: 'border-amber-500/40',
        dot:    'bg-amber-500',
      };
    case 'INFO':
    default:
      return {
        text:   'text-slate-400',
        bg:     'bg-blue-500/15',
        border: 'border-blue-500/40',
        dot:    'bg-blue-500',
      };
  }
}

/**
 * Returns the hex color for an alert severity.
 */
export function getAlertSeverityColor(severity: AlertSeverity): string {
  switch (severity) {
    case 'CRITICAL': return '#ef4444';
    case 'WARNING':  return '#f59e0b';
    case 'INFO':     return '#3b82f6';
    default:         return '#6b7280';
  }
}

// ─── Number formatters ────────────────────────────────────────────────────────

/**
 * Formats a depth value in metres, e.g. "3.45 m".
 */
export function formatDepth(meters: number, decimals = 2): string {
  return `${meters.toFixed(decimals)} m`;
}

/**
 * Formats a distance in kilometres, e.g. "1,234.5 km".
 */
export function formatKm(km: number, decimals = 1): string {
  return `${km.toLocaleString('en-IN', { maximumFractionDigits: decimals })} km`;
}

/**
 * Formats a probability (0–1) as a percentage string, e.g. "87.3%".
 */
export function formatPct(value: number, decimals = 1): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

/**
 * Formats a plain 0–100 score as a percentage string, e.g. "73%".
 */
export function formatScore(score: number, decimals = 0): string {
  return `${score.toFixed(decimals)}%`;
}

/**
 * Formats a large number with Indian locale grouping, e.g. "12,34,567".
 */
export function formatNumber(
  value: number,
  options?: Intl.NumberFormatOptions,
): string {
  return value.toLocaleString('en-IN', options);
}

/**
 * Compact number formatter for stat cards (e.g. 1200 → "1.2K").
 */
export function formatCompact(value: number): string {
  if (Math.abs(value) >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (Math.abs(value) >= 1_000)     return `${(value / 1_000).toFixed(1)}K`;
  return value.toString();
}

/**
 * Formats a velocity in m/s with 2 decimal places.
 */
export function formatVelocity(ms: number): string {
  return `${ms.toFixed(2)} m/s`;
}

/**
 * Formats a width in metres.
 */
export function formatWidth(m: number): string {
  return `${Math.round(m)} m`;
}

// ─── Date / time helpers ──────────────────────────────────────────────────────

/**
 * Returns the abbreviated month label for a numeric month (1-indexed).
 */
export function getMonthLabel(month: Month | number): string {
  return MONTH_LABELS[month as Month] ?? 'N/A';
}

/**
 * Returns an ordered array of all month numbers [1…12].
 */
export const MONTHS: Month[] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

/**
 * Returns an array of years from startYear up to (and including) endYear.
 */
export function yearRange(startYear: number, endYear: number): number[] {
  return Array.from(
    { length: endYear - startYear + 1 },
    (_, i) => startYear + i,
  );
}

/**
 * Formats an ISO timestamp string as a human-readable date-time,
 * e.g. "14 Apr 2024, 09:32 IST".
 */
export function formatTimestamp(iso: string): string {
  const date = new Date(iso);
  return date.toLocaleString('en-IN', {
    day:      '2-digit',
    month:    'short',
    year:     'numeric',
    hour:     '2-digit',
    minute:   '2-digit',
    timeZone: 'Asia/Kolkata',
    timeZoneName: 'short',
  });
}

/**
 * Returns a relative time string, e.g. "2 hours ago" or "just now".
 */
export function relativeTime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const seconds = Math.floor(diff / 1000);
  if (seconds < 60)   return 'just now';
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60)   return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24)     return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

/**
 * Returns true if the given month is within the Indian monsoon season (Jun–Sep).
 */
export function isMonsoonMonth(month: Month | number): boolean {
  return month >= 6 && month <= 9;
}

/**
 * Returns the season name for a given month.
 */
export function getSeason(month: Month | number): string {
  if (month >= 6 && month <= 9)   return 'Monsoon';
  if (month >= 10 && month <= 11) return 'Post-Monsoon';
  if (month >= 12 || month <= 2)  return 'Winter';
  return 'Pre-Monsoon';
}

// ─── Map / GeoJSON helpers ────────────────────────────────────────────────────

/**
 * Linearly interpolates between two hex colors by a factor t ∈ [0, 1].
 */
export function lerpColor(hex1: string, hex2: string, t: number): string {
  const parse = (h: string) =>
    [
      parseInt(h.slice(1, 3), 16),
      parseInt(h.slice(3, 5), 16),
      parseInt(h.slice(5, 7), 16),
    ] as const;

  const [r1, g1, b1] = parse(hex1);
  const [r2, g2, b2] = parse(hex2);

  const r = Math.round(r1 + (r2 - r1) * t);
  const g = Math.round(g1 + (g2 - g1) * t);
  const b = Math.round(b1 + (b2 - b1) * t);

  return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
}

/**
 * Converts a hex color to an RGBA string.
 */
export function hexToRgba(hex: string, alpha = 1): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

/**
 * Returns a color based on a depth value mapped against known thresholds.
 * Useful for continuous depth-heat overlays.
 */
export function depthToColor(depth_m: number): string {
  // < 2 m → red, 2–3 m → amber, ≥ 3 m → green
  if (depth_m >= 3.0) return '#22c55e';
  if (depth_m >= 2.0) return '#f59e0b';
  return '#ef4444';
}

/**
 * Clamps a number between min and max.
 */
export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

/**
 * Maps a value from one range to another.
 */
export function mapRange(
  value: number,
  inMin: number,
  inMax: number,
  outMin: number,
  outMax: number,
): number {
  return outMin + ((value - inMin) / (inMax - inMin)) * (outMax - outMin);
}

// ─── Array / data helpers ─────────────────────────────────────────────────────

/**
 * Groups an array by a key-deriving function.
 */
export function groupBy<T>(
  array: T[],
  keyFn: (item: T) => string,
): Record<string, T[]> {
  return array.reduce<Record<string, T[]>>((acc, item) => {
    const key = keyFn(item);
    (acc[key] ??= []).push(item);
    return acc;
  }, {});
}

/**
 * Calculates the mean of an array of numbers.
 */
export function mean(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

/**
 * Sorts an array of objects by a numeric property, descending.
 */
export function sortByDesc<T>(array: T[], keyFn: (item: T) => number): T[] {
  return [...array].sort((a, b) => keyFn(b) - keyFn(a));
}

/**
 * Returns the unique values of an array.
 */
export function unique<T>(array: T[]): T[] {
  return [...new Set(array)];
}

// ─── Misc ─────────────────────────────────────────────────────────────────────

/**
 * Generates a deterministic pastel color from a string seed.
 * Useful for assigning consistent colors to year lines in trend charts.
 */
export function stringToColor(str: string): string {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = str.charCodeAt(i) + ((hash << 5) - hash);
    hash |= 0;
  }
  const hue = Math.abs(hash) % 360;
  return `hsl(${hue}, 70%, 60%)`;
}

/**
 * Year-indexed color palette for trend charts (2020–2025).
 */
export const YEAR_COLORS: Record<number, string> = {
  2020: '#818cf8', // indigo-400
  2021: '#34d399', // emerald-400
  2022: '#fb923c', // orange-400
  2023: '#60a5fa', // blue-400
  2024: '#f472b6', // pink-400
  2025: '#a78bfa', // violet-400
};

/**
 * Returns a color for a given year from the palette, falling back to a
 * deterministically generated color for years outside the map.
 */
export function getYearColor(year: number): string {
  return YEAR_COLORS[year] ?? stringToColor(String(year));
}

/**
 * Sleep helper for staggered animations in dev / demo mode.
 */
export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
