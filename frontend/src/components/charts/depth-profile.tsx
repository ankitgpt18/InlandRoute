// ============================================================
// AIDSTL — Inland Waterway Navigability Prediction System
// DepthProfileChart — Longitudinal depth profile with CI bands
// ============================================================

'use client';

import React, { useMemo, useState } from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Legend,
  type TooltipProps,
} from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';
import {
  TrendingDown,
  AlertTriangle,
  Waves,
  ArrowDown,
  Info,
  ChevronLeft,
  ChevronRight,
  Anchor,
} from 'lucide-react';
import { useAppStore } from '@/store/app-store';
import { getMockDepthProfile } from '@/lib/mock-data';
import { NavigabilityBadge } from '@/components/ui/navigability-badge';
import { cn } from '@/lib/utils';
import type { WaterwayId, NavigabilityClass } from '@/types';

// ─── Constants ─────────────────────────────────────────────────────────────────

const NAVIGABLE_THRESHOLD   = 3.0; // metres
const CONDITIONAL_THRESHOLD = 2.0; // metres

const MONTH_NAMES = [
  'January','February','March','April',
  'May','June','July','August',
  'September','October','November','December',
] as const;

const NAV_COLORS: Record<NavigabilityClass, string> = {
  navigable:     '#22c55e',
  conditional:   '#f59e0b',
  non_navigable: '#ef4444',
};

// ─── Types ──────────────────────────────────────────────────────────────────────

interface DepthProfileChartProps {
  /** Override waterway (defaults to store selection) */
  waterwayId?: WaterwayId;
  /** Override month (defaults to store selection) */
  month?: number;
  /** Override year (defaults to store selection) */
  year?: number;
  /** Card height in px */
  height?: number;
  /** Show month navigation controls */
  showMonthNav?: boolean;
  /** Show summary stats bar */
  showStats?: boolean;
  /** Compact mode — hides labels and reduces padding */
  compact?: boolean;
  className?: string;
}

interface ChartDataPoint {
  km:                number;
  depth:             number;
  lowerCI:           number;
  upperCI:           number;
  width:             number;
  navigabilityClass: NavigabilityClass;
  segmentId:         string;
  landmark?:         string;
}

// ─── Custom Tooltip ─────────────────────────────────────────────────────────────

function DepthTooltip({ active, payload, label }: TooltipProps<number, string>) {
  if (!active || !payload || payload.length === 0) return null;

  const dataPoint = payload[0]?.payload as ChartDataPoint;
  if (!dataPoint) return null;

  const cls        = dataPoint.navigabilityClass;
  const depthColor = NAV_COLORS[cls];

  const clsLabels: Record<NavigabilityClass, string> = {
    navigable:     'Navigable',
    conditional:   'Conditional',
    non_navigable: 'Non-Navigable',
  };

  const marginToThreshold =
    cls === 'navigable'
      ? dataPoint.depth - NAVIGABLE_THRESHOLD
      : cls === 'conditional'
        ? dataPoint.depth - CONDITIONAL_THRESHOLD
        : CONDITIONAL_THRESHOLD - dataPoint.depth;

  return (
    <div
      className="
        bg-slate-900/98 backdrop-blur-xl
        border border-white/10 rounded-xl
        px-3 py-3 min-w-[200px]
        shadow-2xl shadow-black/60
        pointer-events-none
      "
      style={{ borderColor: `${depthColor}30` }}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-2 pb-2 border-b border-white/[0.07]">
        <div>
          <div className="text-[12px] font-bold text-slate-100 tabular-nums">
            km {(label as number).toFixed(1)}
          </div>
          {dataPoint.landmark && (
            <div className="text-[10px] text-slate-500 font-medium mt-0.5">
              📍 {dataPoint.landmark}
            </div>
          )}
        </div>
        <span
          className="text-[9px] font-bold tracking-wider uppercase px-2 py-0.5 rounded-full"
          style={{
            backgroundColor: `${depthColor}18`,
            color:            depthColor,
            border:           `1px solid ${depthColor}35`,
          }}
        >
          {clsLabels[cls]}
        </span>
      </div>

      {/* Depth value (large) */}
      <div className="flex items-baseline gap-1 mb-2">
        <span
          className="text-2xl font-extrabold tracking-tight tabular-nums leading-none"
          style={{ color: depthColor }}
        >
          {dataPoint.depth.toFixed(2)}
        </span>
        <span className="text-sm font-semibold text-slate-500">m</span>
        <span className="text-[10px] text-slate-600 ml-1">depth</span>
      </div>

      {/* CI band */}
      <div className="flex items-center gap-1.5 mb-2">
        <div
          className="h-1.5 rounded-full flex-1"
          style={{ background: `linear-gradient(90deg, ${depthColor}40, ${depthColor}20)` }}
        >
          <div
            className="h-full rounded-full"
            style={{
              width:      `${Math.max(20, Math.min(100, (dataPoint.depth / (dataPoint.upperCI || 1)) * 100))}%`,
              background: depthColor,
            }}
          />
        </div>
      </div>
      <div className="grid grid-cols-2 gap-x-3 gap-y-1.5 text-[11px]">
        <div>
          <span className="text-slate-600">CI Lower</span>
          <div className="font-semibold text-slate-300 tabular-nums">
            {dataPoint.lowerCI.toFixed(2)} m
          </div>
        </div>
        <div>
          <span className="text-slate-600">CI Upper</span>
          <div className="font-semibold text-slate-300 tabular-nums">
            {dataPoint.upperCI.toFixed(2)} m
          </div>
        </div>
        <div>
          <span className="text-slate-600">Width</span>
          <div className="font-semibold text-slate-300 tabular-nums">
            {dataPoint.width.toFixed(0)} m
          </div>
        </div>
        <div>
          <span className="text-slate-600">
            {cls === 'navigable' ? 'Margin' : 'Deficit'}
          </span>
          <div
            className="font-semibold tabular-nums"
            style={{ color: cls === 'non_navigable' ? '#ef4444' : '#22c55e' }}
          >
            {marginToThreshold >= 0 ? '+' : ''}{marginToThreshold.toFixed(2)} m
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Custom X-Axis Tick ──────────────────────────────────────────────────────────

function CustomXTick({
  x, y, payload,
}: {
  x?: number; y?: number; payload?: { value: number };
}) {
  if (!payload) return null;
  return (
    <g transform={`translate(${x},${y})`}>
      <text
        x={0} y={0} dy={14}
        textAnchor="middle"
        fill="#475569"
        fontSize={10}
        fontFamily="Inter, sans-serif"
        fontWeight={500}
      >
        {payload.value.toFixed(0)}
      </text>
    </g>
  );
}

// ─── Custom Y-Axis Tick ──────────────────────────────────────────────────────────

function CustomYTick({
  x, y, payload,
}: {
  x?: number; y?: number; payload?: { value: number };
}) {
  if (!payload) return null;
  return (
    <g transform={`translate(${x},${y})`}>
      <text
        x={-4} y={0} dy={4}
        textAnchor="end"
        fill="#475569"
        fontSize={10}
        fontFamily="Inter, sans-serif"
        fontWeight={500}
      >
        {payload.value.toFixed(0)}m
      </text>
    </g>
  );
}

// ─── Threshold Label ──────────────────────────────────────────────────────────────

function ThresholdLabel({
  viewBox,
  value,
  color,
  description,
}: {
  viewBox?: { x?: number; y?: number; width?: number };
  value:        number;
  color:        string;
  description:  string;
}) {
  const { x = 0, y = 0, width = 0 } = viewBox ?? {};
  return (
    <g>
      <foreignObject
        x={x + width - 130}
        y={y - 20}
        width={128}
        height={18}
      >
        <div
          style={{
            display:        'flex',
            alignItems:     'center',
            justifyContent: 'flex-end',
            gap:            4,
            padding:        '2px 6px',
            borderRadius:   4,
            background:     `${color}18`,
            border:         `1px solid ${color}35`,
          }}
        >
          <span style={{ fontSize: 9, fontWeight: 700, color, fontFamily: 'Inter, sans-serif', letterSpacing: '0.04em' }}>
            {description.toUpperCase()} · {value}m
          </span>
        </div>
      </foreignObject>
    </g>
  );
}

// ─── Bottleneck Marker ────────────────────────────────────────────────────────────

interface BottleneckMarkerProps {
  km:    number;
  depth: number;
}

function BottleneckBadge({ km, depth }: BottleneckMarkerProps) {
  return (
    <div className="
      flex items-center gap-2 px-3 py-2 rounded-xl
      bg-red-500/10 border border-red-500/25
      text-red-400
    ">
      <ArrowDown size={14} className="flex-shrink-0" />
      <div>
        <div className="text-[11px] font-bold">Bottleneck</div>
        <div className="text-[10px] font-medium text-red-500">
          km {km.toFixed(1)} · {depth.toFixed(2)} m
        </div>
      </div>
    </div>
  );
}

// ─── Summary Stats Bar ────────────────────────────────────────────────────────────

function SummaryStats({
  minDepth,
  maxDepth,
  meanDepth,
  bottleneckKm,
  bottleneckDepth,
  navigableThreshold,
  conditionalThreshold,
  navigablePct,
}: {
  minDepth:             number;
  maxDepth:             number;
  meanDepth:            number;
  bottleneckKm:         number;
  bottleneckDepth:      number;
  navigableThreshold:   number;
  conditionalThreshold: number;
  navigablePct:         number;
}) {
  const stats = [
    {
      label:  'Min Depth',
      value:  `${minDepth.toFixed(2)}m`,
      color:  minDepth >= navigableThreshold ? 'text-emerald-400' :
              minDepth >= conditionalThreshold ? 'text-amber-400' : 'text-red-400',
      sub:    minDepth >= navigableThreshold ? '✓ Above threshold' :
              minDepth >= conditionalThreshold ? '⚠ Conditional zone' : '✕ Below threshold',
    },
    {
      label:  'Mean Depth',
      value:  `${meanDepth.toFixed(2)}m`,
      color:  meanDepth >= navigableThreshold ? 'text-emerald-400' :
              meanDepth >= conditionalThreshold ? 'text-amber-400' : 'text-red-400',
      sub:    `Avg over full reach`,
    },
    {
      label:  'Max Depth',
      value:  `${maxDepth.toFixed(2)}m`,
      color:  'text-blue-400',
      sub:    'Peak channel depth',
    },
    {
      label:  'Navigable',
      value:  `${navigablePct.toFixed(0)}%`,
      color:  navigablePct >= 70 ? 'text-emerald-400' :
              navigablePct >= 40 ? 'text-amber-400' : 'text-red-400',
      sub:    `Of total length`,
    },
  ];

  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
      {stats.map((s) => (
        <div
          key={s.label}
          className="
            flex flex-col gap-0.5 px-3 py-2.5 rounded-xl
            bg-white/[0.03] border border-white/[0.06]
          "
        >
          <span className="text-[10px] font-semibold tracking-widest text-slate-500 uppercase">
            {s.label}
          </span>
          <span className={cn('text-xl font-extrabold tracking-tight tabular-nums leading-tight', s.color)}>
            {s.value}
          </span>
          <span className="text-[10px] text-slate-600 leading-tight">{s.sub}</span>
        </div>
      ))}
    </div>
  );
}

// ─── Main Component ──────────────────────────────────────────────────────────────

export function DepthProfileChart({
  waterwayId: waterwayIdProp,
  month:      monthProp,
  year:       yearProp,
  height      = 320,
  showMonthNav = true,
  showStats    = true,
  compact      = false,
  className,
}: DepthProfileChartProps) {
  const storeWaterway = useAppStore((s) => s.selectedWaterway);
  const storeMonth    = useAppStore((s) => s.selectedMonth);
  const storeYear     = useAppStore((s) => s.selectedYear);
  const goToPrev      = useAppStore((s) => s.goToPreviousMonth);
  const goToNext      = useAppStore((s) => s.goToNextMonth);

  const waterwayId = waterwayIdProp ?? storeWaterway;
  const month      = monthProp     ?? storeMonth;
  const year       = yearProp      ?? storeYear;

  // Local month override for independent month nav inside the chart
  const [localMonthOffset, setLocalMonthOffset] = useState(0);

  const effectiveMonth = useMemo(() => {
    let m = month + localMonthOffset;
    if (m > 12) m = m - 12;
    if (m < 1)  m = m + 12;
    return m as 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12;
  }, [month, localMonthOffset]);

  // ── Fetch data ───────────────────────────────────────────────────────────
  const profileData = useMemo(
    () => getMockDepthProfile(waterwayId, effectiveMonth),
    [waterwayId, effectiveMonth],
  );

  // ── Transform to chart-ready format ──────────────────────────────────────
  const chartData = useMemo<ChartDataPoint[]>(() => {
    if (!profileData?.points) return [];
    return profileData.points.map((pt) => ({
      km:                pt.km,
      depth:             pt.depth_m,
      lowerCI:           pt.depth_lower_ci,
      upperCI:           pt.depth_upper_ci,
      width:             pt.width_m,
      navigabilityClass: pt.navigability_class as NavigabilityClass,
      segmentId:         pt.segment_id,
      landmark:          pt.landmark,
    }));
  }, [profileData]);

  // ── Derived metrics ───────────────────────────────────────────────────────
  const { minDepth, maxDepth, meanDepth, navigablePct, yDomainMax } = useMemo(() => {
    if (chartData.length === 0) {
      return { minDepth: 0, maxDepth: 10, meanDepth: 0, navigablePct: 0, yDomainMax: 12 };
    }
    const depths    = chartData.map((d) => d.depth);
    const uppers    = chartData.map((d) => d.upperCI);
    const navCount  = chartData.filter((d) => d.depth >= NAVIGABLE_THRESHOLD).length;
    const min       = Math.min(...depths);
    const max       = Math.max(...depths);
    const upperMax  = Math.max(...uppers);
    const mean      = depths.reduce((a, b) => a + b, 0) / depths.length;
    const navPct    = (navCount / chartData.length) * 100;
    const domainMax = Math.ceil(Math.max(upperMax, NAVIGABLE_THRESHOLD + 2) * 1.1);
    return {
      minDepth:    min,
      maxDepth:    max,
      meanDepth:   mean,
      navigablePct: navPct,
      yDomainMax:  domainMax,
    };
  }, [chartData]);

  // ── Landmarks ─────────────────────────────────────────────────────────────
  const landmarks = useMemo(
    () => chartData.filter((d) => d.landmark).map((d) => ({ km: d.km, name: d.landmark! })),
    [chartData],
  );

  // ── Bottleneck ────────────────────────────────────────────────────────────
  const bottleneck = useMemo(() => {
    const worstPt = chartData.reduce<ChartDataPoint | null>(
      (worst, pt) => (!worst || pt.depth < worst.depth ? pt : worst),
      null,
    );
    return worstPt;
  }, [chartData]);

  // ── X domain ──────────────────────────────────────────────────────────────
  const xDomain = useMemo<[number, number]>(() => {
    if (chartData.length === 0) return [0, 100];
    return [chartData[0].km, chartData[chartData.length - 1].km];
  }, [chartData]);

  // ── Monsoon indicator ─────────────────────────────────────────────────────
  const isMonsoon = effectiveMonth >= 6 && effectiveMonth <= 9;

  // ── Loading / empty state ─────────────────────────────────────────────────
  if (!profileData || chartData.length === 0) {
    return (
      <div
        className={cn(
          'flex items-center justify-center rounded-2xl',
          'bg-white/[0.03] border border-white/[0.06]',
          className,
        )}
        style={{ height }}
      >
        <div className="text-center">
          <Waves size={32} className="text-slate-700 mx-auto mb-3" />
          <p className="text-sm text-slate-500 font-medium">No depth profile data</p>
          <p className="text-xs text-slate-700 mt-1">
            {MONTH_NAMES[effectiveMonth - 1]} {year}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className={cn('flex flex-col', className)}>
      {/* ── Header ──────────────────────────────────────────────────────── */}
      {!compact && (
        <div className="flex items-start justify-between gap-4 mb-4">
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-sm font-bold text-slate-100">
                Longitudinal Depth Profile
              </h3>
              {isMonsoon && (
                <span className="
                  text-[9px] font-bold tracking-wider uppercase
                  px-2 py-0.5 rounded-full
                  bg-blue-500/15 text-blue-400 border border-blue-500/25
                ">
                  Monsoon
                </span>
              )}
            </div>
            <p className="text-xs text-slate-500 mt-0.5">
              {waterwayId} · {MONTH_NAMES[effectiveMonth - 1]} {year} ·
              {' '}{xDomain[0].toFixed(0)}–{xDomain[1].toFixed(0)} km reach
            </p>
          </div>

          {/* Month navigation */}
          {showMonthNav && (
            <div className="flex items-center gap-1 flex-shrink-0">
              <button
                onClick={() => setLocalMonthOffset((v) => v - 1)}
                className="
                  w-7 h-7 flex items-center justify-center rounded-lg
                  text-slate-500 hover:text-slate-300 hover:bg-white/6
                  border border-white/6 transition-colors duration-150
                "
                aria-label="Previous month"
              >
                <ChevronLeft size={13} />
              </button>
              <span className="text-[11px] font-semibold text-slate-400 min-w-[90px] text-center">
                {MONTH_NAMES[effectiveMonth - 1]}
              </span>
              <button
                onClick={() => setLocalMonthOffset((v) => v + 1)}
                className="
                  w-7 h-7 flex items-center justify-center rounded-lg
                  text-slate-500 hover:text-slate-300 hover:bg-white/6
                  border border-white/6 transition-colors duration-150
                "
                aria-label="Next month"
              >
                <ChevronRight size={13} />
              </button>
            </div>
          )}
        </div>
      )}

      {/* ── Summary stats ───────────────────────────────────────────────── */}
      {showStats && !compact && (
        <SummaryStats
          minDepth={minDepth}
          maxDepth={maxDepth}
          meanDepth={meanDepth}
          bottleneckKm={bottleneck?.km ?? 0}
          bottleneckDepth={bottleneck?.depth ?? 0}
          navigableThreshold={NAVIGABLE_THRESHOLD}
          conditionalThreshold={CONDITIONAL_THRESHOLD}
          navigablePct={navigablePct}
        />
      )}

      {/* ── Chart ───────────────────────────────────────────────────────── */}
      <motion.div
        key={`${waterwayId}-${effectiveMonth}`}
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
        style={{ height }}
      >
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart
            data={chartData}
            margin={{ top: 16, right: 16, left: 4, bottom: 4 }}
          >
            <defs>
              {/* CI band gradient */}
              <linearGradient id="ciGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%"   stopColor="#38bdf8" stopOpacity={0.12} />
                <stop offset="100%" stopColor="#38bdf8" stopOpacity={0.02} />
              </linearGradient>

              {/* Depth area gradient */}
              <linearGradient id="depthGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%"   stopColor="#3b82f6" stopOpacity={0.35} />
                <stop offset="60%"  stopColor="#0369a1" stopOpacity={0.20} />
                <stop offset="100%" stopColor="#0c4a6e" stopOpacity={0.05} />
              </linearGradient>

              {/* Non-navigable zone gradient (fills below conditional threshold) */}
              <linearGradient id="dangerGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%"   stopColor="#ef4444" stopOpacity={0.0}  />
                <stop offset="100%" stopColor="#ef4444" stopOpacity={0.12} />
              </linearGradient>

              {/* Clip path to show only danger zone below threshold */}
              <clipPath id="belowConditional">
                <rect x="0" y="0" width="100%" height={`${(CONDITIONAL_THRESHOLD / yDomainMax) * 100}%`} />
              </clipPath>
            </defs>

            <CartesianGrid
              strokeDasharray="3 4"
              stroke="rgba(255,255,255,0.04)"
              vertical={false}
            />

            <XAxis
              dataKey="km"
              type="number"
              domain={xDomain}
              tickCount={8}
              tick={<CustomXTick />}
              axisLine={{ stroke: 'rgba(255,255,255,0.06)' }}
              tickLine={false}
              label={{
                value:    'Distance (km)',
                position: 'insideBottom',
                offset:   -2,
                fill:     '#334155',
                fontSize: 10,
                fontFamily: 'Inter, sans-serif',
                fontWeight: 500,
              }}
            />

            <YAxis
              domain={[0, yDomainMax]}
              tickCount={6}
              tick={<CustomYTick />}
              axisLine={{ stroke: 'rgba(255,255,255,0.06)' }}
              tickLine={false}
              width={40}
            />

            {/* ── Navigable threshold — 3.0 m ── */}
            <ReferenceLine
              y={NAVIGABLE_THRESHOLD}
              stroke="#22c55e"
              strokeWidth={1.5}
              strokeDasharray="6 4"
              opacity={0.7}
              label={
                <ThresholdLabel
                  value={NAVIGABLE_THRESHOLD}
                  color="#22c55e"
                  description="Navigable"
                />
              }
            />

            {/* ── Conditional threshold — 2.0 m ── */}
            <ReferenceLine
              y={CONDITIONAL_THRESHOLD}
              stroke="#f59e0b"
              strokeWidth={1.5}
              strokeDasharray="6 4"
              opacity={0.7}
              label={
                <ThresholdLabel
                  value={CONDITIONAL_THRESHOLD}
                  color="#f59e0b"
                  description="Conditional"
                />
              }
            />

            {/* ── Landmark reference lines ── */}
            {landmarks.slice(0, 6).map((lm) => (
              <ReferenceLine
                key={lm.km}
                x={lm.km}
                stroke="rgba(148,163,184,0.15)"
                strokeWidth={1}
                strokeDasharray="2 4"
                label={{
                  value:    lm.name,
                  position: 'insideTopRight',
                  fill:     '#334155',
                  fontSize: 9,
                  fontFamily: 'Inter, sans-serif',
                  angle:    -45,
                  offset:   4,
                }}
              />
            ))}

            {/* ── Bottleneck reference line ── */}
            {bottleneck && (
              <ReferenceLine
                x={bottleneck.km}
                stroke="#ef4444"
                strokeWidth={1.5}
                strokeDasharray="3 3"
                opacity={0.6}
              />
            )}

            {/* ── CI upper band (invisible stroke, only fill used below) ── */}
            <Area
              type="monotoneX"
              dataKey="upperCI"
              stroke="none"
              fill="url(#ciGradient)"
              fillOpacity={1}
              legendType="none"
              activeDot={false}
              isAnimationActive={true}
              animationDuration={800}
              animationEasing="ease-out"
            />

            {/* ── CI lower band (cancels out lower portion of CI fill) ── */}
            <Area
              type="monotoneX"
              dataKey="lowerCI"
              stroke="none"
              fill={`#020817`}
              fillOpacity={1}
              legendType="none"
              activeDot={false}
              isAnimationActive={true}
              animationDuration={800}
              animationEasing="ease-out"
            />

            {/* ── Main depth area ── */}
            <Area
              type="monotoneX"
              dataKey="depth"
              name="Predicted Depth"
              stroke="#3b82f6"
              strokeWidth={2.5}
              fill="url(#depthGradient)"
              fillOpacity={1}
              dot={false}
              activeDot={{
                r:           5,
                fill:        '#3b82f6',
                stroke:      '#020817',
                strokeWidth: 2,
              }}
              isAnimationActive={true}
              animationDuration={1000}
              animationEasing="ease-out"
            />

            <Tooltip
              content={<DepthTooltip />}
              cursor={{
                stroke:      'rgba(148,163,184,0.2)',
                strokeWidth: 1,
                strokeDasharray: '4 3',
              }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </motion.div>

      {/* ── Chart footnote: threshold legend ────────────────────────────── */}
      {!compact && (
        <div className="flex items-center flex-wrap gap-4 mt-3 pt-3 border-t border-white/[0.05]">
          {/* Threshold legend items */}
          <div className="flex items-center gap-1.5">
            <div className="w-8 border-t-2 border-dashed border-emerald-500/70" />
            <span className="text-[10px] text-slate-500">
              Navigable threshold (3.0 m)
            </span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-8 border-t-2 border-dashed border-amber-500/70" />
            <span className="text-[10px] text-slate-500">
              Conditional threshold (2.0 m)
            </span>
          </div>
          <div className="flex items-center gap-1.5">
            <div
              className="w-8 h-3 rounded-sm"
              style={{ background: 'linear-gradient(90deg, rgba(59,130,246,0.3), rgba(59,130,246,0.1))' }}
            />
            <span className="text-[10px] text-slate-500">
              Predicted depth
            </span>
          </div>
          <div className="flex items-center gap-1.5">
            <div
              className="w-8 h-3 rounded-sm"
              style={{ background: 'rgba(56,189,248,0.12)' }}
            />
            <span className="text-[10px] text-slate-500">
              90% confidence interval
            </span>
          </div>

          {/* IWAI standard badge */}
          <div className="ml-auto flex items-center gap-1.5">
            <Anchor size={10} className="text-slate-600" />
            <span className="text-[10px] text-slate-600 font-medium">
              IWAI LAD standard · 1,500 DWT
            </span>
          </div>
        </div>
      )}

      {/* ── Bottleneck warning card ──────────────────────────────────────── */}
      {!compact && bottleneck && bottleneck.depth < NAVIGABLE_THRESHOLD && (
        <motion.div
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4, duration: 0.3 }}
          className="mt-3"
        >
          <div className="
            flex items-center justify-between gap-3
            px-4 py-3 rounded-xl
            bg-amber-500/8 border border-amber-500/20
          ">
            <div className="flex items-center gap-2.5">
              <div className="
                w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0
                bg-amber-500/15
              ">
                <AlertTriangle size={16} className="text-amber-400" />
              </div>
              <div>
                <div className="text-[12px] font-semibold text-amber-300">
                  Shallowest Point Detected
                </div>
                <div className="text-[11px] text-amber-600 mt-0.5">
                  km {bottleneck.km.toFixed(1)} · {bottleneck.depth.toFixed(2)} m depth
                  {' '}({(NAVIGABLE_THRESHOLD - bottleneck.depth).toFixed(2)} m below navigable threshold)
                </div>
              </div>
            </div>
            <NavigabilityBadge
              navigabilityClass={bottleneck.navigabilityClass}
              size="sm"
              variant="subtle"
            />
          </div>
        </motion.div>
      )}
    </div>
  );
}

// ─── Compact Variant (for dashboard sidebar) ──────────────────────────────────────

export function DepthProfileCompact({
  waterwayId,
  month,
  height = 160,
  className,
}: {
  waterwayId?: WaterwayId;
  month?:      number;
  height?:     number;
  className?:  string;
}) {
  return (
    <DepthProfileChart
      waterwayId={waterwayId}
      month={month}
      height={height}
      showMonthNav={false}
      showStats={false}
      compact={true}
      className={className}
    />
  );
}
