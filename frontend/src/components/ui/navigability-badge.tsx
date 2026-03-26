// ============================================================
// AIDSTL — Inland Waterway Navigability Prediction System
// Navigability Badge Component
// ============================================================

'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { CheckCircle2, AlertTriangle, XCircle, Circle } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { NavigabilityClass } from '@/types';

// ─── Types ────────────────────────────────────────────────────────────────────

export type BadgeSize   = 'xs' | 'sm' | 'md' | 'lg';
export type BadgeVariant = 'filled' | 'outlined' | 'subtle' | 'ghost' | 'glow';

export interface NavigabilityBadgeProps {
  /** The navigability classification to display */
  navigabilityClass: NavigabilityClass;
  /** Visual size of the badge */
  size?: BadgeSize;
  /** Visual style variant */
  variant?: BadgeVariant;
  /** Show the status icon alongside text */
  showIcon?: boolean;
  /** Show pulsing dot for CRITICAL / urgent states */
  pulse?: boolean;
  /** Animate entrance with framer-motion */
  animate?: boolean;
  /** Override display label */
  label?: string;
  /** Additional className */
  className?: string;
  /** Click handler */
  onClick?: () => void;
  /** Render as a pill (fully rounded) vs default slightly-rounded */
  pill?: boolean;
}

// ─── Config ───────────────────────────────────────────────────────────────────

interface ClassConfig {
  label:       string;
  shortLabel:  string;
  icon:        React.ElementType;
  colors: {
    filled:   string;
    outlined: string;
    subtle:   string;
    ghost:    string;
    glow:     string;
  };
  dotColor:  string;
  glowColor: string;
  hex:       string;
}

const CLASS_CONFIG: Record<NavigabilityClass, ClassConfig> = {
  navigable: {
    label:      'Navigable',
    shortLabel: 'NAV',
    icon:       CheckCircle2,
    colors: {
      filled:   'bg-emerald-500 text-white border-emerald-500',
      outlined: 'bg-transparent text-emerald-400 border-emerald-500/60',
      subtle:   'bg-emerald-500/12 text-emerald-400 border-emerald-500/25',
      ghost:    'bg-transparent text-emerald-400 border-transparent',
      glow:     'bg-emerald-500/15 text-emerald-300 border-emerald-500/40',
    },
    dotColor:  'bg-emerald-400',
    glowColor: 'shadow-[0_0_16px_rgba(34,197,94,0.45)]',
    hex:       '#22c55e',
  },

  conditional: {
    label:      'Conditional',
    shortLabel: 'COND',
    icon:       AlertTriangle,
    colors: {
      filled:   'bg-amber-500 text-white border-amber-500',
      outlined: 'bg-transparent text-amber-400 border-amber-500/60',
      subtle:   'bg-amber-500/12 text-amber-400 border-amber-500/25',
      ghost:    'bg-transparent text-amber-400 border-transparent',
      glow:     'bg-amber-500/15 text-amber-300 border-amber-500/40',
    },
    dotColor:  'bg-amber-400',
    glowColor: 'shadow-[0_0_16px_rgba(245,158,11,0.45)]',
    hex:       '#f59e0b',
  },

  non_navigable: {
    label:      'Non-Navigable',
    shortLabel: 'CLOSED',
    icon:       XCircle,
    colors: {
      filled:   'bg-red-500 text-white border-red-500',
      outlined: 'bg-transparent text-red-400 border-red-500/60',
      subtle:   'bg-red-500/12 text-red-400 border-red-500/25',
      ghost:    'bg-transparent text-red-400 border-transparent',
      glow:     'bg-red-500/15 text-red-300 border-red-500/40',
    },
    dotColor:  'bg-red-400',
    glowColor: 'shadow-[0_0_16px_rgba(239,68,68,0.45)]',
    hex:       '#ef4444',
  },
};

// ─── Size Config ──────────────────────────────────────────────────────────────

interface SizeConfig {
  container: string;
  text:      string;
  icon:      number;
  dot:       string;
  gap:       string;
}

const SIZE_CONFIG: Record<BadgeSize, SizeConfig> = {
  xs: {
    container: 'px-1.5 py-0.5',
    text:      'text-[9px] font-bold tracking-wider',
    icon:      9,
    dot:       'w-1.5 h-1.5',
    gap:       'gap-1',
  },
  sm: {
    container: 'px-2 py-0.5',
    text:      'text-[10px] font-bold tracking-wider',
    icon:      10,
    dot:       'w-1.5 h-1.5',
    gap:       'gap-1',
  },
  md: {
    container: 'px-2.5 py-1',
    text:      'text-[11px] font-semibold tracking-wide',
    icon:      12,
    dot:       'w-2 h-2',
    gap:       'gap-1.5',
  },
  lg: {
    container: 'px-3.5 py-1.5',
    text:      'text-xs font-semibold tracking-wide',
    icon:      14,
    dot:       'w-2 h-2',
    gap:       'gap-2',
  },
};

// ─── Pulsing Dot ──────────────────────────────────────────────────────────────

function PulsingDot({
  dotColor,
  dotSizeClass,
}: {
  dotColor: string;
  dotSizeClass: string;
}) {
  return (
    <span className={cn('relative inline-flex flex-shrink-0', dotSizeClass)}>
      {/* Ping ring */}
      <span
        className={cn(
          'animate-ping absolute inline-flex h-full w-full rounded-full opacity-60',
          dotColor,
        )}
      />
      {/* Solid dot */}
      <span
        className={cn(
          'relative inline-flex rounded-full h-full w-full',
          dotColor,
        )}
      />
    </span>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────

/**
 * NavigabilityBadge
 *
 * Displays the navigability classification of a river segment with
 * configurable size, variant, icon, and optional pulse animation.
 *
 * @example
 * <NavigabilityBadge navigabilityClass="navigable" size="md" variant="glow" pulse />
 * <NavigabilityBadge navigabilityClass="conditional" size="sm" variant="subtle" showIcon />
 * <NavigabilityBadge navigabilityClass="non_navigable" size="lg" variant="filled" />
 */
export function NavigabilityBadge({
  navigabilityClass,
  size      = 'md',
  variant   = 'subtle',
  showIcon  = false,
  pulse     = false,
  animate   = false,
  label,
  className,
  onClick,
  pill      = true,
}: NavigabilityBadgeProps) {
  const config     = CLASS_CONFIG[navigabilityClass];
  const sizeConf   = SIZE_CONFIG[size];
  const colorClass = config.colors[variant];
  const isGlow     = variant === 'glow';

  const displayLabel = label ?? (size === 'xs' ? config.shortLabel : config.label).toUpperCase();
  const Icon         = config.icon;

  const baseClass = cn(
    // Layout
    'inline-flex items-center border select-none whitespace-nowrap',
    // Gap between elements
    sizeConf.gap,
    // Padding
    sizeConf.container,
    // Border radius
    pill ? 'rounded-full' : 'rounded-md',
    // Colour
    colorClass,
    // Glow shadow (only for glow variant)
    isGlow && config.glowColor,
    // Clickable states
    onClick && 'cursor-pointer transition-transform active:scale-95',
    className,
  );

  const content = (
    <>
      {/* Pulsing dot OR icon */}
      {pulse ? (
        <PulsingDot dotColor={config.dotColor} dotSizeClass={sizeConf.dot} />
      ) : showIcon ? (
        <Icon size={sizeConf.icon} strokeWidth={2.5} className="flex-shrink-0" />
      ) : (
        /* Static dot */
        <span
          className={cn('rounded-full flex-shrink-0', sizeConf.dot, config.dotColor)}
        />
      )}

      {/* Label */}
      <span className={sizeConf.text}>{displayLabel}</span>
    </>
  );

  if (animate) {
    return (
      <motion.span
        className={baseClass}
        initial={{ opacity: 0, scale: 0.8, y: 4 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.8, y: -4 }}
        transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
        onClick={onClick}
        role={onClick ? 'button' : undefined}
        tabIndex={onClick ? 0 : undefined}
      >
        {content}
      </motion.span>
    );
  }

  return (
    <span
      className={baseClass}
      onClick={onClick}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
    >
      {content}
    </span>
  );
}

// ─── Compact Dot Variant ──────────────────────────────────────────────────────

/**
 * NavigabilityDot
 *
 * Ultra-compact colour dot with optional tooltip, for use inside tables,
 * calendars, or map legends where space is limited.
 */
export interface NavigabilityDotProps {
  navigabilityClass: NavigabilityClass;
  size?: number;
  pulse?: boolean;
  /** Show label on hover via title attribute */
  showTooltip?: boolean;
  className?: string;
}

export function NavigabilityDot({
  navigabilityClass,
  size     = 10,
  pulse    = false,
  showTooltip = true,
  className,
}: NavigabilityDotProps) {
  const config = CLASS_CONFIG[navigabilityClass];

  if (pulse) {
    return (
      <span
        className={cn('relative inline-flex flex-shrink-0', className)}
        style={{ width: size, height: size }}
        title={showTooltip ? config.label : undefined}
      >
        <span
          className={cn('animate-ping absolute inline-flex h-full w-full rounded-full opacity-60', config.dotColor)}
        />
        <span className={cn('relative inline-flex rounded-full h-full w-full', config.dotColor)} />
      </span>
    );
  }

  return (
    <span
      className={cn('inline-block rounded-full flex-shrink-0', config.dotColor, className)}
      style={{ width: size, height: size }}
      title={showTooltip ? config.label : undefined}
    />
  );
}

// ─── Color Legend Row ─────────────────────────────────────────────────────────

/**
 * NavigabilityLegend
 *
 * Renders a compact horizontal or vertical legend showing all three
 * navigability classes. Ideal for map overlays and report headers.
 */
export interface NavigabilityLegendProps {
  direction?: 'horizontal' | 'vertical';
  size?: BadgeSize;
  variant?: BadgeVariant;
  showIcon?: boolean;
  className?: string;
}

export function NavigabilityLegend({
  direction = 'horizontal',
  size      = 'sm',
  variant   = 'subtle',
  showIcon  = false,
  className,
}: NavigabilityLegendProps) {
  const classes: NavigabilityClass[] = ['navigable', 'conditional', 'non_navigable'];

  return (
    <div
      className={cn(
        'flex flex-wrap gap-2',
        direction === 'vertical' ? 'flex-col items-start' : 'flex-row items-center',
        className,
      )}
      role="list"
      aria-label="Navigability legend"
    >
      {classes.map((cls) => (
        <div key={cls} role="listitem">
          <NavigabilityBadge
            navigabilityClass={cls}
            size={size}
            variant={variant}
            showIcon={showIcon}
          />
        </div>
      ))}
    </div>
  );
}

// ─── Inline Status Text ───────────────────────────────────────────────────────

/**
 * NavigabilityText
 *
 * Renders the navigability label as inline coloured text (no background),
 * suitable for embedding inside sentences or table cells.
 */
export interface NavigabilityTextProps {
  navigabilityClass: NavigabilityClass;
  className?: string;
  bold?: boolean;
}

export function NavigabilityText({
  navigabilityClass,
  className,
  bold = true,
}: NavigabilityTextProps) {
  const config = CLASS_CONFIG[navigabilityClass];

  const colorMap: Record<NavigabilityClass, string> = {
    navigable:     'text-emerald-400',
    conditional:   'text-amber-400',
    non_navigable: 'text-red-400',
  };

  return (
    <span
      className={cn(
        colorMap[navigabilityClass],
        bold ? 'font-semibold' : 'font-normal',
        className,
      )}
    >
      {config.label}
    </span>
  );
}

// ─── Map Legend Component ─────────────────────────────────────────────────────

/**
 * MapLegendCard
 *
 * Floating card shown on the interactive river map to explain colour coding.
 */
export function MapLegendCard({ className }: { className?: string }) {
  const items: { cls: NavigabilityClass; depth: string; width: string }[] = [
    { cls: 'navigable',     depth: '≥ 3.0 m', width: '≥ 50 m' },
    { cls: 'conditional',   depth: '2.0–3.0 m', width: '30–50 m' },
    { cls: 'non_navigable', depth: '< 2.0 m', width: '< 30 m'  },
  ];

  return (
    <div
      className={cn(
        'glass-card px-3 py-3 rounded-xl min-w-[180px]',
        className,
      )}
    >
      {/* Header */}
      <div className="flex items-center gap-1.5 mb-2.5">
        <Circle size={10} className="text-slate-500" />
        <span className="text-[10px] font-bold tracking-widest text-slate-500 uppercase">
          Navigability
        </span>
      </div>

      {/* Rows */}
      <div className="flex flex-col gap-2">
        {items.map(({ cls, depth, width }) => {
          const cfg = CLASS_CONFIG[cls];
          return (
            <div key={cls} className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <span
                  className={cn('w-2.5 h-2.5 rounded-full flex-shrink-0', cfg.dotColor)}
                />
                <span className="text-[11px] font-medium text-slate-300">
                  {cfg.label}
                </span>
              </div>
              <div className="text-right">
                <div className="text-[10px] text-slate-500 tabular-nums">{depth}</div>
                <div className="text-[10px] text-slate-600 tabular-nums">{width}</div>
              </div>
            </div>
          );
        })}
      </div>

      {/* IWAI standard note */}
      <div className="mt-2.5 pt-2 border-t border-white/[0.06]">
        <p className="text-[9px] text-slate-600 leading-relaxed">
          IWAI LAD standard · 1,500 DWT barge
        </p>
      </div>
    </div>
  );
}
