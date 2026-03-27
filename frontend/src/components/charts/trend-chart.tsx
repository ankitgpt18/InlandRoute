// ============================================================
// AIDSTL — Inland Waterway Navigability Prediction System
// TrendChart — Multi-year navigability trend with Recharts
// ============================================================

'use client';

import React, { useMemo, useState } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Legend,
  Area,
  AreaChart,
  type TooltipProps,
} from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';
import {
  TrendingUp,
  TrendingDown,
  Minus,
  Calendar,
  BarChart3,
  Info,
  Eye,
  EyeOff,
  ChevronDown,
} from 'lucide-react';
import { useAppStore } from '@/store/app-store';
import { getMockTrends } from '@/lib/mock-data';
import { cn } from '@/lib/utils';
import type { WaterwayId } from '@/types';

// ─── Constants ──────────────────────────────────────────────────────────────────

const MONTH_SHORT = [
  'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
] as const;

const MONTH_FULL = [
  'January', 'February', 'March', 'April', 'May', 'June',
  'July', 'August', 'September', 'October', 'November', 'December',
] as const;

// ─── Types ──────────────────────────────────────────────────────────────────────

interface TrendYear {
  year:                     number;
  color:                    string;
  annual_mean_navigable_pct: number;
  data: {
    month:           number;
    label:           string;
    navigable_pct:   number;
    mean_depth_m:    number;
  }[];
}

interface ChartMode {
  id:    'navigability' | 'depth';
  label: string;
  unit:  string;
  yKey:  'navigable_pct' | 'mean_depth_m';
  yDomain: [number, number];
}

const CHART_MODES: ChartMode[] = [
  {
    id:      'navigability',
    label:   'Navigability %',
    unit:    '%',
    yKey:    'navigable_pct',
    yDomain: [0, 100],
  },
  {
    id:      'depth',
    label:   'Mean Depth',
    unit:    'm',
    yKey:    'mean_depth_m',
    yDomain: [0, 12],
  },
];

interface TrendChartProps {
  waterwayId?:   WaterwayId;
  /** Height of the chart area in px */
  height?:       number;
  /** Show chart type toggle (line vs area) */
  showTypeToggle?: boolean;
  /** Show mode toggle (navigability vs depth) */
  showModeToggle?: boolean;
  /** Show per-year summary stats below chart */
  showYearStats?: boolean;
  /** Compact mode */
  compact?:      boolean;
  className?:    string;
}

// ─── Custom Tooltip ─────────────────────────────────────────────────────────────

function TrendTooltip({
  active,
  payload,
  label,
  mode,
}: TooltipProps<number, string> & { mode: ChartMode }) {
  if (!active || !payload || payload.length === 0) return null;

  const monthIdx = (typeof label === 'number' ? label : parseInt(label as string, 10)) - 1;
  const monthName = MONTH_FULL[monthIdx] ?? String(label);

  // Sort by value descending
  const sorted = [...payload].sort((a, b) => (b.value ?? 0) - (a.value ?? 0));

  return (
    <div
      className="
        bg-white
        border border-slate-200 rounded-xl
        px-3.5 py-3 min-w-[200px]
        shadow-xl
        pointer-events-none
      "
    >
      {/* Month header */}
      <div className="flex items-center gap-2 mb-2.5 pb-2 border-b border-slate-900/[0.07]">
        <Calendar size={11} className="text-slate-500 flex-shrink-0" />
        <span className="text-[12px] font-bold text-slate-900">{monthName}</span>
        {/* Monsoon indicator */}
        {monthIdx >= 5 && monthIdx <= 8 && (
          <span className="ml-auto text-[9px] font-bold text-slate-400 tracking-wider uppercase">
            Monsoon
          </span>
        )}
      </div>

      {/* Year rows */}
      <div className="flex flex-col gap-1.5">
        {sorted.map((entry) => {
          const val = typeof entry.value === 'number' ? entry.value : 0;
          return (
            <div
              key={entry.dataKey}
              className="flex items-center justify-between gap-4"
            >
              <div className="flex items-center gap-1.5">
                <span
                  className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                  style={{ backgroundColor: entry.color ?? '#64748b' }}
                />
                <span className="text-[11px] font-semibold text-slate-700">
                  {entry.name}
                </span>
              </div>
              <span
                className="text-[12px] font-bold tabular-nums text-slate-800"
              >
                {val.toFixed(mode.id === 'depth' ? 2 : 1)}{mode.unit}
              </span>
            </div>
          );
        })}
      </div>

      {/* Range (highest - lowest) */}
      {sorted.length > 1 && (
        <div className="mt-2 pt-2 border-t border-slate-900/[0.06]">
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-slate-400">Year range</span>
            <span className="text-[10px] font-semibold text-slate-400 tabular-nums">
              {((sorted[0].value ?? 0) - (sorted[sorted.length - 1].value ?? 0)).toFixed(
                mode.id === 'depth' ? 2 : 1,
              )}{mode.unit}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Custom Legend ───────────────────────────────────────────────────────────────

function TrendLegend({
  years,
  hidden,
  onToggle,
}: {
  years:    TrendYear[];
  hidden:   Set<number>;
  onToggle: (year: number) => void;
}) {
  return (
    <div className="flex items-center flex-wrap gap-2">
      {years.map((y) => {
        const isHidden = hidden.has(y.year);
        return (
          <motion.button
            key={y.year}
            onClick={() => onToggle(y.year)}
            whileHover={{ scale: 1.04 }}
            whileTap={{ scale: 0.96 }}
            className={cn(
              'flex items-center gap-1.5 px-2.5 py-1 rounded-full border transition-all duration-150',
              isHidden
                ? 'border-slate-900/[0.06] bg-transparent opacity-40'
                : 'border-transparent',
            )}
            style={
              !isHidden
                ? {
                    backgroundColor: `${y.color}18`,
                    border:          `1px solid ${y.color}40`,
                  }
                : {}
            }
            aria-pressed={!isHidden}
            title={isHidden ? `Show ${y.year}` : `Hide ${y.year}`}
          >
            <span
              className="w-2.5 h-2.5 rounded-full flex-shrink-0 transition-opacity"
              style={{
                backgroundColor: y.color,
                opacity:          isHidden ? 0.3 : 1,
              }}
            />
            <span
              className="text-[11px] font-bold tabular-nums"
              style={{ color: isHidden ? '#64748b' : y.color }}
            >
              {y.year}
            </span>
            {!isHidden && (
              <span
                className="text-[10px] font-medium ml-0.5 hidden sm:inline"
                style={{ color: `${y.color}aa` }}
              >
                {y.annual_mean_navigable_pct.toFixed(0)}%
              </span>
            )}
          </motion.button>
        );
      })}
    </div>
  );
}

// ─── Year Stats Cards ─────────────────────────────────────────────────────────────

function YearStatsRow({
  years,
  hidden,
  mode,
}: {
  years:  TrendYear[];
  hidden: Set<number>;
  mode:   ChartMode;
}) {
  const visibleYears = years.filter((y) => !hidden.has(y.year));
  if (visibleYears.length === 0) return null;

  // Find best/worst year
  const vals = visibleYears.map((y) => ({
    year: y.year,
    val:
      mode.id === 'navigability'
        ? y.annual_mean_navigable_pct
        : y.data.reduce((s, d) => s + d.mean_depth_m, 0) / y.data.length,
    color: y.color,
  }));
  const best  = vals.reduce((a, b) => (b.val > a.val ? b : a));
  const worst = vals.reduce((a, b) => (b.val < a.val ? b : a));

  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4 pt-4 border-t border-slate-900/[0.05]">
      {vals.map(({ year, val, color }) => {
        const isBest  = year === best.year;
        const isWorst = year === worst.year;

        // Year-over-year change
        const prevYearVals = vals.find((v) => v.year === year - 1);
        const yoyChange = prevYearVals ? val - prevYearVals.val : null;

        return (
          <motion.div
            key={year}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
            className="
              flex flex-col gap-1 px-3 py-2.5 rounded-xl
              bg-white/[0.03] border border-slate-900/[0.06]
              relative overflow-hidden
            "
            style={{ borderColor: `${color}25` }}
          >
            {/* Top accent */}
            <div
              className="absolute top-0 left-4 right-4 h-px opacity-50"
              style={{ backgroundColor: color }}
            />

            {/* Year label + badge */}
            <div className="flex items-center justify-between">
              <span
                className="text-[11px] font-bold"
                style={{ color }}
              >
                {year}
              </span>
              {(isBest || isWorst) && (
                <span
                  className={cn(
                    'text-[9px] font-bold tracking-wider uppercase px-1.5 py-0.5 rounded-full',
                    isBest
                      ? 'bg-emerald-500/15 text-slate-400 border border-emerald-500/25'
                      : 'bg-red-500/15 text-slate-400 border border-red-500/25',
                  )}
                >
                  {isBest ? 'Best' : 'Worst'}
                </span>
              )}
            </div>

            {/* Value */}
            <div
              className="text-xl font-extrabold tracking-tight tabular-nums leading-tight"
              style={{ color }}
            >
              {val.toFixed(mode.id === 'depth' ? 2 : 1)}
              <span
                className="text-[11px] font-semibold ml-1 opacity-60"
                style={{ color }}
              >
                {mode.unit}
              </span>
            </div>

            {/* YoY change */}
            {yoyChange !== null && (
              <div
                className={cn(
                  'flex items-center gap-1 text-[10px] font-semibold',
                  Math.abs(yoyChange) < 0.5
                    ? 'text-slate-500'
                    : yoyChange > 0
                      ? 'text-slate-400'
                      : 'text-slate-400',
                )}
              >
                {Math.abs(yoyChange) < 0.5 ? (
                  <Minus size={10} />
                ) : yoyChange > 0 ? (
                  <TrendingUp size={10} />
                ) : (
                  <TrendingDown size={10} />
                )}
                {yoyChange >= 0 ? '+' : ''}
                {yoyChange.toFixed(1)}{mode.unit} vs {year - 1}
              </div>
            )}
          </motion.div>
        );
      })}
    </div>
  );
}

// ─── Chart Type Toggle ────────────────────────────────────────────────────────────

function ChartTypeToggle({
  isArea,
  onToggle,
}: {
  isArea:   boolean;
  onToggle: () => void;
}) {
  return (
    <div className="flex items-center rounded-lg border border-slate-900/[0.08] overflow-hidden">
      <button
        onClick={() => !isArea && onToggle()}
        className={cn(
          'flex items-center gap-1.5 px-2.5 py-1.5 text-[11px] font-semibold transition-all duration-150',
          !isArea
            ? 'bg-blue-500/20 text-slate-300'
            : 'text-slate-500 hover:text-slate-700 hover:bg-white/[0.04]',
        )}
        aria-pressed={!isArea}
        title="Line chart"
      >
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
          <polyline
            points="1,11 4,6 7,8 10,3 13,5"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        <span className="hidden sm:inline">Line</span>
      </button>
      <div className="w-px h-5 bg-white/[0.08]" />
      <button
        onClick={() => isArea && onToggle()}
        className={cn(
          'flex items-center gap-1.5 px-2.5 py-1.5 text-[11px] font-semibold transition-all duration-150',
          isArea
            ? 'bg-blue-500/20 text-slate-300'
            : 'text-slate-500 hover:text-slate-700 hover:bg-white/[0.04]',
        )}
        aria-pressed={isArea}
        title="Area chart"
      >
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
          <path
            d="M1 11 L4 6 L7 8 L10 3 L13 5 L13 12 L1 12 Z"
            fill="currentColor"
            opacity="0.4"
          />
          <polyline
            points="1,11 4,6 7,8 10,3 13,5"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        <span className="hidden sm:inline">Area</span>
      </button>
    </div>
  );
}

// ─── Mode Toggle ──────────────────────────────────────────────────────────────────

function ModeToggle({
  current,
  onChange,
}: {
  current:  ChartMode;
  onChange: (m: ChartMode) => void;
}) {
  return (
    <div className="flex items-center rounded-lg border border-slate-900/[0.08] overflow-hidden">
      {CHART_MODES.map((mode, idx) => (
        <React.Fragment key={mode.id}>
          {idx > 0 && <div className="w-px h-5 bg-white/[0.08]" />}
          <button
            onClick={() => onChange(mode)}
            className={cn(
              'px-2.5 py-1.5 text-[11px] font-semibold transition-all duration-150 whitespace-nowrap',
              current.id === mode.id
                ? 'bg-blue-500/20 text-slate-300'
                : 'text-slate-500 hover:text-slate-700 hover:bg-white/[0.04]',
            )}
            aria-pressed={current.id === mode.id}
          >
            {mode.label}
          </button>
        </React.Fragment>
      ))}
    </div>
  );
}

// ─── Custom X-Axis Tick ────────────────────────────────────────────────────────────

function XTick({
  x,
  y,
  payload,
}: {
  x?: number;
  y?: number;
  payload?: { value: number };
}) {
  if (!payload) return null;
  const mi = payload.value - 1;
  const isMonsoon = mi >= 5 && mi <= 8;
  return (
    <g transform={`translate(${x},${y})`}>
      <text
        x={0}
        y={0}
        dy={14}
        textAnchor="middle"
        fill={isMonsoon ? '#38bdf8' : '#475569'}
        fontSize={10}
        fontFamily="Inter, sans-serif"
        fontWeight={isMonsoon ? 700 : 500}
      >
        {MONTH_SHORT[mi]}
      </text>
      {isMonsoon && (
        <circle cx={0} cy={24} r={1.5} fill="#38bdf8" opacity={0.6} />
      )}
    </g>
  );
}

// ─── Custom Y-Axis Tick ────────────────────────────────────────────────────────────

function YTick({
  x,
  y,
  payload,
  unit,
}: {
  x?: number;
  y?: number;
  payload?: { value: number };
  unit: string;
}) {
  if (!payload) return null;
  return (
    <g transform={`translate(${x},${y})`}>
      <text
        x={-4}
        y={0}
        dy={4}
        textAnchor="end"
        fill="#475569"
        fontSize={10}
        fontFamily="Inter, sans-serif"
        fontWeight={500}
      >
        {payload.value}{unit}
      </text>
    </g>
  );
}

// ─── Monsoon Season Band ──────────────────────────────────────────────────────────

function MonsoonBand({
  yAxisMap,
}: {
  yAxisMap?: unknown;
}) {
  // This is rendered as a recharts customized background
  // We inject it as a reference area via SVG rects at chart level
  return null; // handled via ReferenceLine approach below
}

// ─── Change Summary Banner ────────────────────────────────────────────────────────

function ChangeSummaryBanner({
  years,
  mode,
}: {
  years:  TrendYear[];
  mode:   ChartMode;
}) {
  if (years.length < 2) return null;

  const sorted     = [...years].sort((a, b) => a.year - b.year);
  const first      = sorted[0];
  const last       = sorted[sorted.length - 1];

  const firstVal =
    mode.id === 'navigability'
      ? first.annual_mean_navigable_pct
      : first.data.reduce((s, d) => s + d.mean_depth_m, 0) / first.data.length;

  const lastVal =
    mode.id === 'navigability'
      ? last.annual_mean_navigable_pct
      : last.data.reduce((s, d) => s + d.mean_depth_m, 0) / last.data.length;

  const delta     = lastVal - firstVal;
  const pctChange = firstVal !== 0 ? (delta / firstVal) * 100 : 0;
  const isUp      = delta > 0;
  const isFlat    = Math.abs(delta) < 0.5;

  const TIcon = isFlat ? Minus : isUp ? TrendingUp : TrendingDown;
  const color = isFlat ? '#64748b' : isUp ? '#22c55e' : '#ef4444';
  const bg    = isFlat ? 'bg-slate-200/20 border-slate-300/30' : isUp ? 'bg-emerald-500/10 border-emerald-500/20' : 'bg-red-500/10 border-red-500/20';

  return (
    <motion.div
      initial={{ opacity: 0, y: -6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
      className={cn(
        'flex items-center gap-3 px-4 py-2.5 rounded-xl border mb-4',
        bg,
      )}
    >
      <div
        className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0"
        style={{ backgroundColor: `${color}20` }}
      >
        <TIcon size={16} style={{ color }} />
      </div>
      <div className="flex-1 min-w-0">
        <span className="text-[12px] font-semibold text-slate-800">
          {isFlat
            ? 'Stable trend'
            : isUp
              ? 'Improving trend'
              : 'Declining trend'}{' '}
          over {last.year - first.year} years
        </span>
        <span className="text-[11px] text-slate-500 ml-2">
          {first.year}→{last.year}
        </span>
      </div>
      <div className="text-right flex-shrink-0">
        <div
          className="text-[16px] font-extrabold tabular-nums leading-tight"
          style={{ color }}
        >
          {delta >= 0 ? '+' : ''}
          {delta.toFixed(mode.id === 'depth' ? 2 : 1)}{mode.unit}
        </div>
        <div className="text-[10px] text-slate-500 tabular-nums">
          {pctChange >= 0 ? '+' : ''}
          {pctChange.toFixed(1)}% total change
        </div>
      </div>
    </motion.div>
  );
}

// ─── Main Component ────────────────────────────────────────────────────────────────

export function TrendChart({
  waterwayId:   waterwayIdProp,
  height        = 300,
  showTypeToggle = true,
  showModeToggle = true,
  showYearStats  = true,
  compact        = false,
  className,
}: TrendChartProps) {
  const storeWaterway = useAppStore((s) => s.selectedWaterway);
  const waterwayId    = waterwayIdProp ?? storeWaterway;

  // ── Local state ──────────────────────────────────────────────────────────
  const [mode,       setMode]       = useState<ChartMode>(CHART_MODES[0]);
  const [isArea,     setIsArea]     = useState(false);
  const [hiddenYears, setHiddenYears] = useState<Set<number>>(new Set());
  const [showBanner, setShowBanner] = useState(true);

  // ── Fetch data ────────────────────────────────────────────────────────────
  const trendsData = useMemo(
    () => getMockTrends(waterwayId),
    [waterwayId],
  );

  // ── Parse years ───────────────────────────────────────────────────────────
  const years = useMemo<TrendYear[]>(() => {
    if (!trendsData?.trends) return [];
    return trendsData.trends as unknown as TrendYear[];
  }, [trendsData]);

  // ── Merge monthly data across years for recharts ──────────────────────────
  // Shape: [{ month: 1, 2020: 72.4, 2021: 68.1, ... }, ...]
  const chartData = useMemo(() => {
    return Array.from({ length: 12 }, (_, mi) => {
      const point: Record<string, number> = { month: mi + 1 };
      years.forEach((y) => {
        const monthData = y.data?.find((d: any) => d.month === mi + 1);
        if (monthData) {
          point[String(y.year)] = monthData[mode.yKey];
        }
      });
      return point;
    });
  }, [years, mode]);

  // ── Toggle year visibility ─────────────────────────────────────────────────
  const toggleYear = (year: number) => {
    setHiddenYears((prev) => {
      const next = new Set(prev);
      if (next.has(year)) {
        next.delete(year);
      } else {
        // Don't hide if it's the only visible year
        if (years.length - next.size <= 1) return prev;
        next.add(year);
      }
      return next;
    });
  };

  // ── Y-axis domain ──────────────────────────────────────────────────────────
  const yDomain = useMemo<[number, number]>(() => {
    if (mode.id === 'navigability') return [0, 100];
    // Auto-range for depth
    const allVals = years
      .filter((y) => !hiddenYears.has(y.year))
      .flatMap((y) => y.data.map((d) => d.mean_depth_m));
    if (allVals.length === 0) return [0, 12];
    const min = Math.max(0, Math.floor(Math.min(...allVals) - 1));
    const max = Math.ceil(Math.max(...allVals) + 1);
    return [min, max];
  }, [years, hiddenYears, mode]);

  // ── Reference lines for navigability thresholds ────────────────────────────
  const referenceLines = useMemo(() => {
    if (mode.id === 'navigability') {
      return [
        { y: 80, color: '#22c55e', label: '80% navigable', dash: '6 3' },
        { y: 50, color: '#f59e0b', label: '50% threshold', dash: '4 4' },
      ];
    }
    return [
      { y: 3.0, color: '#22c55e', label: 'Navigable (3m)', dash: '6 3' },
      { y: 2.0, color: '#f59e0b', label: 'Conditional (2m)', dash: '4 4' },
    ];
  }, [mode]);

  // ── Empty state ────────────────────────────────────────────────────────────
  if (!trendsData || years.length === 0) {
    return (
      <div
        className={cn(
          'flex flex-col items-center justify-center',
          'bg-white/[0.03] border border-slate-900/[0.06] rounded-2xl',
          'py-12 text-center',
          className,
        )}
        style={{ height }}
      >
        <BarChart3 size={32} className="text-slate-700 mb-3" />
        <p className="text-sm text-slate-500 font-medium">No trend data available</p>
        <p className="text-xs text-slate-700 mt-1">{waterwayId}</p>
      </div>
    );
  }

  // ── Recharts component (Line or Area) ──────────────────────────────────────
  const ChartComponent = isArea ? AreaChart : LineChart;

  return (
    <div className={cn('flex flex-col', className)}>

      {/* ── Header ───────────────────────────────────────────────────────── */}
      {!compact && (
        <div className="flex items-start justify-between gap-4 mb-4">
          <div>
            <h3 className="text-sm font-bold text-slate-900">
              Multi-Year Navigability Trends
            </h3>
            <p className="text-xs text-slate-500 mt-0.5">
              {waterwayId} · {years[0]?.year}–{years[years.length - 1]?.year} ·
              {' '}Monthly {mode.label.toLowerCase()} comparison
            </p>
          </div>

          {/* Controls */}
          <div className="flex items-center gap-2 flex-shrink-0 flex-wrap justify-end">
            {showModeToggle && (
              <ModeToggle current={mode} onChange={setMode} />
            )}
            {showTypeToggle && (
              <ChartTypeToggle
                isArea={isArea}
                onToggle={() => setIsArea((v) => !v)}
              />
            )}
          </div>
        </div>
      )}

      {/* ── Change summary banner ─────────────────────────────────────────── */}
      {!compact && showBanner && (
        <ChangeSummaryBanner years={years} mode={mode} />
      )}

      {/* ── Year legend / toggle ──────────────────────────────────────────── */}
      <div className="flex items-center justify-between gap-3 mb-3">
        <TrendLegend years={years} hidden={hiddenYears} onToggle={toggleYear} />

        {/* Info / banner toggle */}
        {!compact && (
          <button
            onClick={() => setShowBanner((v) => !v)}
            className="text-slate-400 hover:text-slate-400 transition-colors flex-shrink-0"
            title={showBanner ? 'Hide summary' : 'Show summary'}
          >
            <Info size={14} />
          </button>
        )}
      </div>

      {/* ── Chart ─────────────────────────────────────────────────────────── */}
      <motion.div
        key={`${waterwayId}-${mode.id}`}
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
        style={{ height }}
      >
        <ResponsiveContainer width="100%" height="100%">
          <ChartComponent
            data={chartData}
            margin={{ top: 8, right: 16, left: 4, bottom: 8 }}
          >
            <defs>
              {years.map((y) => (
                <linearGradient
                  key={y.year}
                  id={`area-gradient-${y.year}`}
                  x1="0"
                  y1="0"
                  x2="0"
                  y2="1"
                >
                  <stop offset="0%"   stopColor={y.color} stopOpacity={0.25} />
                  <stop offset="100%" stopColor={y.color} stopOpacity={0.02} />
                </linearGradient>
              ))}
            </defs>

            <CartesianGrid
              strokeDasharray="3 4"
              stroke="rgba(15,23,42,0.04)"
              vertical={false}
            />

            <XAxis
              dataKey="month"
              type="number"
              domain={[1, 12]}
              ticks={[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]}
              tick={<XTick />}
              axisLine={{ stroke: 'rgba(15,23,42,0.06)' }}
              tickLine={false}
              interval={0}
            />

            <YAxis
              domain={yDomain}
              tickCount={mode.id === 'navigability' ? 6 : 5}
              tick={<YTick unit={mode.unit} />}
              axisLine={{ stroke: 'rgba(15,23,42,0.06)' }}
              tickLine={false}
              width={38}
            />

            {/* ── Reference lines ── */}
            {referenceLines.map((ref) => (
              <ReferenceLine
                key={ref.y}
                y={ref.y}
                stroke={ref.color}
                strokeWidth={1.5}
                strokeDasharray={ref.dash}
                opacity={0.55}
                label={{
                  value:    ref.label,
                  position: 'insideTopRight',
                  fill:     ref.color,
                  fontSize: 9,
                  fontFamily: 'Inter, sans-serif',
                  fontWeight: 600,
                  opacity:  0.7,
                  dx: -4,
                  dy: 4,
                }}
              />
            ))}

            {/* ── Current month reference ── */}
            <ReferenceLine
              x={new Date().getMonth() + 1}
              stroke="rgba(59,130,246,0.4)"
              strokeWidth={1}
              strokeDasharray="3 3"
              label={{
                value:    'Now',
                position: 'insideTopLeft',
                fill:     '#3b82f6',
                fontSize: 9,
                fontFamily: 'Inter, sans-serif',
                fontWeight: 700,
                opacity:  0.6,
              }}
            />

            {/* ── Data series per year ── */}
            {years.map((y) => {
              if (hiddenYears.has(y.year)) return null;
              const yearKey = String(y.year);

              if (isArea) {
                return (
                  <Area
                    key={y.year}
                    type="monotone"
                    dataKey={yearKey}
                    name={yearKey}
                    stroke={y.color}
                    strokeWidth={2}
                    fill={`url(#area-gradient-${y.year})`}
                    fillOpacity={1}
                    dot={false}
                    activeDot={{
                      r:           4,
                      fill:        y.color,
                      stroke:      '#020817',
                      strokeWidth: 2,
                    }}
                    isAnimationActive={true}
                    animationDuration={900}
                    animationEasing="ease-out"
                    connectNulls={true}
                  />
                );
              }

              return (
                <Line
                  key={y.year}
                  type="monotone"
                  dataKey={yearKey}
                  name={yearKey}
                  stroke={y.color}
                  strokeWidth={2}
                  dot={false}
                  activeDot={{
                    r:           4,
                    fill:        y.color,
                    stroke:      '#020817',
                    strokeWidth: 2,
                  }}
                  isAnimationActive={true}
                  animationDuration={900}
                  animationEasing="ease-out"
                  connectNulls={true}
                />
              );
            })}

            <Tooltip
              content={<TrendTooltip mode={mode} />}
              cursor={{
                stroke:          'rgba(148,163,184,0.15)',
                strokeWidth:     1,
                strokeDasharray: '4 3',
              }}
            />
          </ChartComponent>
        </ResponsiveContainer>
      </motion.div>

      {/* ── Chart footnote ────────────────────────────────────────────────── */}
      {!compact && (
        <div className="flex items-center flex-wrap gap-4 mt-3 pt-3 border-t border-slate-900/[0.05]">
          {referenceLines.map((ref) => (
            <div key={ref.y} className="flex items-center gap-1.5">
              <div
                className="w-7 border-t-2 border-dashed"
                style={{ borderColor: `${ref.color}80` }}
              />
              <span className="text-[10px] text-slate-400">{ref.label}</span>
            </div>
          ))}
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-sky-400/60" />
            <span className="text-[10px] text-slate-400">Monsoon months (Jun–Sep)</span>
          </div>
          <div className="ml-auto text-[10px] text-slate-700">
            Source: HydroFormer v1.0 · Sentinel-2
          </div>
        </div>
      )}

      {/* ── Year stat cards ───────────────────────────────────────────────── */}
      {showYearStats && !compact && (
        <YearStatsRow
          years={years.filter((y) => !hiddenYears.has(y.year))}
          hidden={hiddenYears}
          mode={mode}
        />
      )}
    </div>
  );
}

// ─── Compact variant ────────────────────────────────────────────────────────────

export function TrendChartCompact({
  waterwayId,
  height = 180,
  className,
}: Pick<TrendChartProps, 'waterwayId' | 'height' | 'className'>) {
  return (
    <TrendChart
      waterwayId={waterwayId}
      height={height}
      showTypeToggle={false}
      showModeToggle={false}
      showYearStats={false}
      compact={true}
      className={className}
    />
  );
}
