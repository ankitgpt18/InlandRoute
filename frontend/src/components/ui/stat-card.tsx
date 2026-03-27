// ============================================================
// AIDSTL — Inland Waterway Navigability Prediction System
// StatCard Component — Animated glassmorphism metric card
// ============================================================

'use client';

import React, { useEffect, useRef, useState } from 'react';
import { motion, useInView, useSpring, useTransform } from 'framer-motion';
import {
  TrendingUp,
  TrendingDown,
  Minus,
  ArrowUpRight,
  ArrowDownRight,
  Info,
} from 'lucide-react';
import { cn } from '@/lib/utils';

// ─── Types ────────────────────────────────────────────────────────────────────

export type StatCardVariant =
  | 'default'
  | 'navigable'
  | 'conditional'
  | 'non_navigable'
  | 'accent'
  | 'info';

export type TrendDirection = 'up' | 'down' | 'neutral';

export interface StatCardProps {
  /** Primary label shown above the value */
  label: string;
  /** The numeric value to display (animated count-up) */
  value: number;
  /** Unit appended after value, e.g. "km", "%", "m" */
  unit?: string;
  /** Number of decimal places to display */
  decimals?: number;
  /** Secondary descriptive text below the value */
  subtitle?: string;
  /** Trend percentage change (positive or negative) */
  trend?: number;
  /** Explicit direction override for trend arrow */
  trendDirection?: TrendDirection;
  /** Label for trend context, e.g. "vs last month" */
  trendLabel?: string;
  /** Whether upward trend is good (green) or bad (red) — default: true */
  trendUpIsGood?: boolean;
  /** Lucide icon component to display */
  icon?: React.ElementType;
  /** Color variant of the card */
  variant?: StatCardVariant;
  /** Show a subtle loading skeleton overlay */
  loading?: boolean;
  /** Tooltip text shown on info icon hover */
  tooltip?: string;
  /** Click handler — makes the card interactive */
  onClick?: () => void;
  /** Whether card is in a selected/active state */
  isActive?: boolean;
  /** Optional prefix before value (e.g. "₹", "~") */
  prefix?: string;
  /** Animation delay in seconds */
  animationDelay?: number;
  /** Additional className */
  className?: string;
  /** Optional sparkline data (7 values) */
  sparklineData?: number[];
}

// ─── Variant Config ───────────────────────────────────────────────────────────

interface VariantConfig {
  cardBg:      string;
  cardBorder:  string;
  cardGlow:    string;
  iconBg:      string;
  iconColor:   string;
  valueColor:  string;
  accentLine:  string;
}

const VARIANT_CONFIG: Record<StatCardVariant, VariantConfig> = {
  default: {
    cardBg:     'bg-white',
    cardBorder: 'border-slate-200 shadow-sm',
    cardGlow:   '',
    iconBg:     'bg-slate-50',
    iconColor:  'text-slate-500',
    valueColor: 'text-slate-900',
    accentLine: 'bg-slate-100',
  },
  navigable: {
    cardBg:     'bg-white',
    cardBorder: 'border-emerald-100 shadow-sm',
    cardGlow:   'shadow-[0_0_24px_rgba(34,197,94,0.05)]',
    iconBg:     'bg-emerald-50 text-slate-600',
    iconColor:  'text-slate-500',
    valueColor: 'text-slate-600',
    accentLine: 'bg-emerald-100',
  },
  conditional: {
    cardBg:     'bg-white',
    cardBorder: 'border-amber-100 shadow-sm',
    cardGlow:   'shadow-[0_0_24px_rgba(245,158,11,0.05)]',
    iconBg:     'bg-amber-50 text-slate-600',
    iconColor:  'text-slate-500',
    valueColor: 'text-slate-600',
    accentLine: 'bg-amber-100',
  },
  non_navigable: {
    cardBg:     'bg-white',
    cardBorder: 'border-red-100 shadow-sm',
    cardGlow:   'shadow-[0_0_24px_rgba(239,68,68,0.05)]',
    iconBg:     'bg-red-50 text-slate-600',
    iconColor:  'text-slate-500',
    valueColor: 'text-slate-600',
    accentLine: 'bg-red-100',
  },
  accent: {
    cardBg:     'bg-white',
    cardBorder: 'border-blue-100 shadow-sm',
    cardGlow:   'shadow-[0_0_24px_rgba(59,130,246,0.05)]',
    iconBg:     'bg-blue-50 text-slate-600',
    iconColor:  'text-slate-500',
    valueColor: 'text-slate-600',
    accentLine: 'bg-blue-100',
  },
  info: {
    cardBg:     'bg-white',
    cardBorder: 'border-sky-100 shadow-sm',
    cardGlow:   'shadow-[0_0_24px_rgba(14,165,233,0.05)]',
    iconBg:     'bg-sky-50 text-slate-600',
    iconColor:  'text-slate-500',
    valueColor: 'text-slate-600',
    accentLine: 'bg-sky-100',
  },
};

// ─── Animated Counter Hook ────────────────────────────────────────────────────

function useCountUp(
  target: number,
  decimals: number,
  duration: number,
  delay: number,
  inView: boolean,
): string {
  const spring = useSpring(0, {
    stiffness: 60,
    damping: 20,
    restDelta: 0.001,
  });

  const [displayValue, setDisplayValue] = useState('0');

  useEffect(() => {
    if (!inView) return;
    const timer = setTimeout(() => {
      spring.set(target);
    }, delay * 1000);
    return () => clearTimeout(timer);
  }, [inView, target, spring, delay]);

  useEffect(() => {
    const unsubscribe = spring.on('change', (v) => {
      setDisplayValue(v.toFixed(decimals));
    });
    return unsubscribe;
  }, [spring, decimals]);

  return displayValue;
}

// ─── Mini Sparkline ───────────────────────────────────────────────────────────

function MiniSparkline({
  data,
  color,
}: {
  data: number[];
  color: string;
}) {
  if (!data || data.length < 2) return null;

  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const width = 64;
  const height = 24;
  const padX = 2;
  const padY = 2;

  const points = data.map((v, i) => {
    const x = padX + (i / (data.length - 1)) * (width - padX * 2);
    const y = padY + (1 - (v - min) / range) * (height - padY * 2);
    return `${x},${y}`;
  });

  const polylineStr = points.join(' ');

  // Build fill polygon (close path to bottom)
  const firstX = padX;
  const lastX  = padX + (width - padX * 2);
  const fillPoints = [
    `${firstX},${height}`,
    ...points,
    `${lastX},${height}`,
  ].join(' ');

  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      className="overflow-visible"
      aria-hidden="true"
    >
      <defs>
        <linearGradient id={`spark-fill-${color.replace('#', '')}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%"   stopColor={color} stopOpacity={0.25} />
          <stop offset="100%" stopColor={color} stopOpacity={0.0}  />
        </linearGradient>
      </defs>
      {/* Fill area */}
      <polygon
        points={fillPoints}
        fill={`url(#spark-fill-${color.replace('#', '')})`}
      />
      {/* Line */}
      <polyline
        points={polylineStr}
        fill="none"
        stroke={color}
        strokeWidth={1.5}
        strokeLinecap="round"
        strokeLinejoin="round"
        opacity={0.8}
      />
      {/* Last point dot */}
      {(() => {
        const lastPt = points[points.length - 1].split(',');
        return (
          <circle
            cx={parseFloat(lastPt[0])}
            cy={parseFloat(lastPt[1])}
            r={2}
            fill={color}
            opacity={0.9}
          />
        );
      })()}
    </svg>
  );
}

// ─── Trend Indicator ──────────────────────────────────────────────────────────

function TrendIndicator({
  trend,
  direction,
  label,
  upIsGood,
}: {
  trend: number;
  direction: TrendDirection;
  label?: string;
  upIsGood: boolean;
}) {
  const isGood =
    direction === 'neutral'
      ? false
      : upIsGood
        ? direction === 'up'
        : direction === 'down';

  const isNeutral = direction === 'neutral' || Math.abs(trend) < 0.05;

  const colorClass = isNeutral
    ? 'text-slate-500'
    : isGood
      ? 'text-slate-400'
      : 'text-slate-400';

  const bgClass = isNeutral
    ? 'bg-slate-500/10'
    : isGood
      ? 'bg-emerald-500/10'
      : 'bg-red-500/10';

  const TrendIcon = isNeutral ? Minus : isGood ? TrendingUp : TrendingDown;
  const ArrowIcon = isNeutral ? Minus : direction === 'up' ? ArrowUpRight : ArrowDownRight;

  return (
    <div className={cn('inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full', bgClass)}>
      <ArrowIcon size={10} className={cn('flex-shrink-0', colorClass)} strokeWidth={2.5} />
      <span className={cn('text-[10px] font-bold tabular-nums', colorClass)}>
        {isNeutral ? '—' : `${Math.abs(trend).toFixed(1)}%`}
      </span>
      {label && (
        <span className="text-[9px] text-slate-400 font-normal ml-0.5 hidden sm:inline">
          {label}
        </span>
      )}
    </div>
  );
}

// ─── Loading Skeleton ─────────────────────────────────────────────────────────

function StatCardSkeleton({ className }: { className?: string }) {
  return (
    <div
      className={cn(
        'relative overflow-hidden rounded-2xl border border-slate-200 bg-white p-5 shadow-sm',
        className,
      )}
    >
      <div className="flex items-start justify-between mb-4">
        <div className="skeleton h-4 w-24 rounded" />
        <div className="skeleton h-10 w-10 rounded-xl" />
      </div>
      <div className="skeleton h-8 w-32 rounded mb-2" />
      <div className="skeleton h-3 w-20 rounded mb-3" />
      <div className="skeleton h-3 w-16 rounded" />
    </div>
  );
}

// ─── Main StatCard ────────────────────────────────────────────────────────────

/**
 * StatCard
 *
 * An animated glassmorphism metric card featuring:
 * - Smooth spring-based count-up animation triggered when scrolled into view
 * - Trend indicator with directional arrow and colour coding
 * - Optional mini sparkline for 7-day trend
 * - Glassmorphism design with per-variant colour theming
 * - Hover lift + glow effects
 * - Accessible markup with aria labels
 *
 * @example
 * <StatCard
 *   label="Navigable Length"
 *   value={847}
 *   unit="km"
 *   variant="navigable"
 *   trend={4.2}
 *   trendDirection="up"
 *   icon={Navigation}
 *   sparklineData={[720, 740, 760, 800, 810, 830, 847]}
 * />
 */
export function StatCard({
  label,
  value,
  unit,
  decimals        = 0,
  subtitle,
  trend,
  trendDirection,
  trendLabel      = 'vs last month',
  trendUpIsGood   = true,
  icon: Icon,
  variant         = 'default',
  loading         = false,
  tooltip,
  onClick,
  isActive        = false,
  prefix,
  animationDelay  = 0,
  className,
  sparklineData,
}: StatCardProps) {
  const ref    = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: '-40px' });

  // Derive trend direction from value if not provided
  const resolvedDirection: TrendDirection =
    trendDirection ??
    (trend === undefined || Math.abs(trend) < 0.05
      ? 'neutral'
      : trend > 0
        ? 'up'
        : 'down');

  // Animated count-up value
  const animatedValue = useCountUp(value, decimals, 1.2, animationDelay, inView);

  const cfg = VARIANT_CONFIG[variant];

  // Sparkline accent colour
  const sparklineColorMap: Record<StatCardVariant, string> = {
    default:       '#64748b',
    navigable:     '#22c55e',
    conditional:   '#f59e0b',
    non_navigable: '#ef4444',
    accent:        '#3b82f6',
    info:          '#0ea5e9',
  };
  const sparkColor = sparklineColorMap[variant];

  if (loading) {
    return <StatCardSkeleton className={className} />;
  }

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 16 }}
      animate={inView ? { opacity: 1, y: 0 } : { opacity: 0, y: 16 }}
      transition={{
        duration: 0.45,
        delay: animationDelay,
        ease: [0.16, 1, 0.3, 1],
      }}
      whileHover={onClick ? { y: -2, scale: 1.005 } : { y: -1 }}
      whileTap={onClick ? { scale: 0.99 } : undefined}
      onClick={onClick}
      className={cn(
        // Layout
        'relative overflow-hidden rounded-2xl border p-5 flex flex-col gap-0',
        // Background & border
        cfg.cardBg,
        cfg.cardBorder,
        // Glow
        cfg.cardGlow,
        // Transition
        'transition-all duration-300',
        // Interactive
        onClick && 'cursor-pointer',
        // Active state ring
        isActive && [
          'ring-2',
          variant === 'navigable'     && 'ring-emerald-500/40',
          variant === 'conditional'   && 'ring-amber-500/40',
          variant === 'non_navigable' && 'ring-red-500/40',
          variant === 'accent'        && 'ring-blue-500/40',
          variant === 'info'          && 'ring-sky-500/40',
          variant === 'default'       && 'ring-slate-500/40',
        ],
        className,
      )}
      role={onClick ? 'button' : 'article'}
      aria-label={`${label}: ${prefix ?? ''}${animatedValue}${unit ? ` ${unit}` : ''}`}
      tabIndex={onClick ? 0 : undefined}
    >
      {/* ── Subtle top accent line ──────────────────────────── */}
      <div
        className={cn(
          'absolute top-0 left-6 right-6 h-px opacity-60',
          cfg.accentLine,
        )}
      />

      {/* ── Top row: label + icon ───────────────────────────── */}
      <div className="flex items-start justify-between gap-2 mb-3">
        {/* Label + optional tooltip */}
        <div className="flex items-center gap-1.5 min-w-0">
          <span className="text-[11px] font-semibold tracking-widest text-slate-400 uppercase leading-tight truncate">
            {label}
          </span>
          {tooltip && (
            <div className="group relative flex-shrink-0">
              <Info
                size={11}
                className="text-slate-400 hover:text-slate-400 cursor-help transition-colors"
              />
              {/* Tooltip popup */}
              <div
                className="
                  absolute bottom-full left-1/2 -translate-x-1/2 mb-2 z-50
                  w-48 px-2.5 py-2 rounded-lg
                  bg-slate-100 border border-slate-300
                  text-[11px] text-slate-700 leading-relaxed
                  shadow-xl pointer-events-none
                  opacity-0 group-hover:opacity-100
                  transition-opacity duration-150
                  whitespace-normal
                "
              >
                {tooltip}
                {/* Arrow */}
                <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-slate-800" />
              </div>
            </div>
          )}
        </div>

        {/* Icon */}
        {Icon && (
          <motion.div
            initial={{ scale: 0.8, opacity: 0 }}
            animate={inView ? { scale: 1, opacity: 1 } : {}}
            transition={{ delay: animationDelay + 0.1, duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
            className={cn(
              'flex-shrink-0 w-10 h-10 rounded-xl flex items-center justify-center',
              cfg.iconBg,
            )}
          >
            <Icon size={18} className={cfg.iconColor} strokeWidth={2} />
          </motion.div>
        )}
      </div>

      {/* ── Value row ───────────────────────────────────────── */}
      <div className="flex items-end gap-2 mb-1">
        <div className="flex items-baseline gap-1 min-w-0">
          {/* Prefix */}
          {prefix && (
            <span className={cn('text-lg font-semibold', cfg.valueColor, 'opacity-70')}>
              {prefix}
            </span>
          )}

          {/* Main value — animated */}
          <motion.span
            key={value} // Re-trigger animation when value changes
            className={cn(
              'font-extrabold leading-none tracking-tight tabular-nums',
              'text-3xl sm:text-4xl',
              cfg.valueColor,
            )}
            initial={{ opacity: 0 }}
            animate={inView ? { opacity: 1 } : {}}
            transition={{ delay: animationDelay + 0.05, duration: 0.2 }}
          >
            {animatedValue}
          </motion.span>

          {/* Unit */}
          {unit && (
            <span className={cn('text-base font-semibold mb-0.5', cfg.valueColor, 'opacity-60')}>
              {unit}
            </span>
          )}
        </div>

        {/* Sparkline (aligned to bottom of value) */}
        {sparklineData && sparklineData.length >= 2 && (
          <motion.div
            initial={{ opacity: 0, x: 8 }}
            animate={inView ? { opacity: 1, x: 0 } : {}}
            transition={{ delay: animationDelay + 0.3, duration: 0.4 }}
            className="mb-1 flex-shrink-0"
          >
            <MiniSparkline data={sparklineData} color={sparkColor} />
          </motion.div>
        )}
      </div>

      {/* ── Subtitle ────────────────────────────────────────── */}
      {subtitle && (
        <motion.p
          initial={{ opacity: 0 }}
          animate={inView ? { opacity: 1 } : {}}
          transition={{ delay: animationDelay + 0.2, duration: 0.3 }}
          className="text-[12px] text-slate-500 font-medium leading-snug mb-2 clamp-2"
        >
          {subtitle}
        </motion.p>
      )}

      {/* ── Trend indicator ─────────────────────────────────── */}
      {trend !== undefined && (
        <motion.div
          initial={{ opacity: 0, y: 4 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ delay: animationDelay + 0.25, duration: 0.3 }}
          className="mt-auto pt-2 flex items-center gap-2"
        >
          <TrendIndicator
            trend={trend}
            direction={resolvedDirection}
            label={trendLabel}
            upIsGood={trendUpIsGood}
          />
        </motion.div>
      )}

      {/* ── Hover shimmer overlay ───────────────────────────── */}
      {onClick && (
        <motion.div
          className="absolute inset-0 rounded-2xl pointer-events-none"
          initial={{ opacity: 0 }}
          whileHover={{ opacity: 1 }}
          style={{
            background:
              'linear-gradient(135deg, rgba(15,23,42,0.02) 0%, rgba(15,23,42,0.0) 60%)',
          }}
        />
      )}
    </motion.div>
  );
}

// ─── Stat Card Grid Helper ────────────────────────────────────────────────────

/**
 * StatCardGrid
 *
 * Responsive grid wrapper for a row of StatCards.
 * Renders 1 col on mobile, 2 on sm, and up to 4 on xl.
 */
export function StatCardGrid({
  children,
  cols = 4,
  className,
}: {
  children: React.ReactNode;
  cols?: 2 | 3 | 4;
  className?: string;
}) {
  const colMap = {
    2: 'sm:grid-cols-2',
    3: 'sm:grid-cols-2 lg:grid-cols-3',
    4: 'sm:grid-cols-2 xl:grid-cols-4',
  };

  return (
    <div
      className={cn(
        'grid grid-cols-1 gap-4',
        colMap[cols],
        className,
      )}
    >
      {children}
    </div>
  );
}
