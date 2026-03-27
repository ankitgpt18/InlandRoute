// ============================================================
// AIDSTL — Inland Waterway Navigability Prediction System
// SeasonalCalendar — 12-month navigability heatmap by segment
// ============================================================

"use client";

import React, { useMemo, useState, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Calendar,
  ChevronDown,
  Info,
  TrendingUp,
  AlertTriangle,
  CheckCircle2,
  Droplets,
  Anchor,
  Download,
  Eye,
  EyeOff,
} from "lucide-react";
import { useAppStore } from "@/store/app-store";
import { getMockSeasonalCalendar } from "@/lib/mock-data";
import { NavigabilityBadge } from "@/components/ui/navigability-badge";
import { cn } from "@/lib/utils";
import type { WaterwayId, NavigabilityClass } from "@/types";

// ─── Constants ─────────────────────────────────────────────────────────────────

const MONTH_SHORT = [
  "Jan",
  "Feb",
  "Mar",
  "Apr",
  "May",
  "Jun",
  "Jul",
  "Aug",
  "Sep",
  "Oct",
  "Nov",
  "Dec",
] as const;

const MONTH_FULL = [
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "December",
] as const;

// Meteorological seasons for NW-1 / NW-2 (India)
const SEASON_BANDS: { label: string; months: number[]; color: string }[] = [
  { label: "Pre-Monsoon", months: [3, 4, 5], color: "rgba(245,158,11,0.12)" },
  { label: "Monsoon", months: [6, 7, 8, 9], color: "rgba(59,130,246,0.10)" },
  { label: "Post-Monsoon", months: [10, 11], color: "rgba(34,197,94,0.08)" },
  { label: "Winter", months: [12, 1, 2], color: "rgba(148,163,184,0.06)" },
];

// ─── Types ─────────────────────────────────────────────────────────────────────

interface MonthCell {
  month: number;
  navigability_class: NavigabilityClass;
  predicted_depth_m: number;
  probability: number;
  risk_score: number;
  is_monsoon: boolean;
  label: string;
}

interface CalendarRow {
  segment_id: string;
  km_start: number;
  km_end: number;
  km_label: string;
  navigable_months_count: number;
  best_month: number;
  worst_month: number;
  months: MonthCell[];
}

interface HoveredCell {
  segmentIdx: number;
  monthIdx: number;
  cell: MonthCell;
  row: CalendarRow;
  rect: DOMRect;
}

interface SeasonalCalendarProps {
  waterwayId?: WaterwayId;
  year?: number;
  /** Maximum number of segment rows to display */
  maxRows?: number;
  /** Show column-level monthly summary row */
  showSummary?: boolean;
  /** Show season band backgrounds */
  showSeasonBands?: boolean;
  /** Show the best/worst period highlights */
  showHighlights?: boolean;
  className?: string;
}

// ─── Colour helpers ────────────────────────────────────────────────────────────

const NAV_HEX: Record<NavigabilityClass, string> = {
  navigable: "#22c55e",
  conditional: "#f59e0b",
  non_navigable: "#ef4444",
};

const NAV_LABEL: Record<NavigabilityClass, string> = {
  navigable: "Navigable",
  conditional: "Conditional",
  non_navigable: "Non-Navigable",
};

/** Returns Tailwind background class for a navigability class + opacity level */
function cellBgClass(cls: NavigabilityClass, probability: number): string {
  const opacity = Math.round(probability * 9) * 10; // 10–90 in steps of 10
  const clamp = Math.max(20, Math.min(90, opacity));

  const map: Record<NavigabilityClass, Record<number, string>> = {
    navigable: {
      20: "bg-emerald-500/20",
      30: "bg-emerald-500/30",
      40: "bg-emerald-500/40",
      50: "bg-emerald-500/50",
      60: "bg-emerald-500/60",
      70: "bg-emerald-500/70",
      80: "bg-emerald-500/80",
      90: "bg-emerald-500/90",
    },
    conditional: {
      20: "bg-amber-500/20",
      30: "bg-amber-500/30",
      40: "bg-amber-500/40",
      50: "bg-amber-500/50",
      60: "bg-amber-500/60",
      70: "bg-amber-500/70",
      80: "bg-amber-500/80",
      90: "bg-amber-500/90",
    },
    non_navigable: {
      20: "bg-red-500/20",
      30: "bg-red-500/30",
      40: "bg-red-500/40",
      50: "bg-red-500/50",
      60: "bg-red-500/60",
      70: "bg-red-500/70",
      80: "bg-red-500/80",
      90: "bg-red-500/90",
    },
  };

  const bucket = (Math.round(clamp / 10) *
    10) as keyof (typeof map)[NavigabilityClass];
  return map[cls][bucket] ?? "bg-slate-200/40";
}

/** Returns border class for selected/highlighted cells */
function cellBorderClass(cls: NavigabilityClass): string {
  switch (cls) {
    case "navigable":
      return "border-emerald-400/60";
    case "conditional":
      return "border-amber-400/60";
    case "non_navigable":
      return "border-red-400/60";
  }
}

// ─── Tooltip ──────────────────────────────────────────────────────────────────

interface CellTooltipProps {
  hoveredCell: HoveredCell;
  containerRef: React.RefObject<HTMLDivElement>;
}

function CellTooltip({ hoveredCell, containerRef }: CellTooltipProps) {
  const { cell, row } = hoveredCell;
  const cls = cell.navigability_class;
  const accentColor = NAV_HEX[cls];

  // Position: prefer right of cell, flip left if near right edge
  const containerRect = containerRef.current?.getBoundingClientRect();
  const cellRect = hoveredCell.rect;

  let left = cellRect.right - (containerRect?.left ?? 0) + 8;
  let top = cellRect.top - (containerRect?.top ?? 0) - 8;

  // Flip to left if too close to right edge
  const tooltipWidth = 220;
  if (containerRect && left + tooltipWidth > containerRect.width) {
    left = cellRect.left - (containerRect.left ?? 0) - tooltipWidth - 8;
  }

  // Keep within vertical bounds
  if (containerRect && top + 240 > containerRect.height) {
    top = containerRect.height - 248;
  }
  top = Math.max(0, top);

  const marginToNav = cell.predicted_depth_m - 3.0;
  const marginToCond = cell.predicted_depth_m - 2.0;
  const isOk = cls === "navigable";

  return (
    <motion.div
      key={`${hoveredCell.segmentIdx}-${hoveredCell.monthIdx}`}
      initial={{ opacity: 0, scale: 0.92, y: 4 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.88, y: -4 }}
      transition={{ duration: 0.13, ease: [0.16, 1, 0.3, 1] }}
      className="absolute z-50 pointer-events-none"
      style={{ left, top, width: tooltipWidth }}
    >
      <div
        className="
          bg-white
          border rounded-xl px-3.5 py-3
          shadow-xl
        "
        style={{ borderColor: `${accentColor}50` }}
      >
        {/* ── Header ── */}
        <div
          className="flex items-center justify-between mb-2.5 pb-2"
          style={{ borderBottom: `1px solid ${accentColor}20` }}
        >
          <div>
            <div className="text-[12px] font-bold text-slate-900">
              {MONTH_FULL[cell.month - 1]}
            </div>
            <div className="text-[10px] text-slate-500 mt-0.5 tabular-nums">
              {row.km_label}
            </div>
          </div>
          <span
            className="text-[9px] font-bold tracking-wider uppercase px-2 py-0.5 rounded-full"
            style={{
              backgroundColor: `${accentColor}18`,
              color: accentColor,
              border: `1px solid ${accentColor}35`,
            }}
          >
            {NAV_LABEL[cls]}
          </span>
        </div>

        {/* ── Depth (hero value) ── */}
        <div className="flex items-baseline gap-1 mb-2.5">
          <span
            className="text-2xl font-extrabold tracking-tight tabular-nums leading-none text-slate-800"
          >
            {cell.predicted_depth_m.toFixed(2)}
          </span>
          <span className="text-sm font-semibold text-slate-500">m</span>
          <span className="text-[10px] text-slate-400 ml-0.5">
            predicted depth
          </span>
        </div>

        {/* ── Metric rows ── */}
        <div className="grid grid-cols-2 gap-x-4 gap-y-1.5">
          <MetricRow
            label="Probability"
            value={`${(cell.probability * 100).toFixed(0)}%`}
          />
          <MetricRow
            label="Risk Score"
            value={`${(cell.risk_score * 100).toFixed(0)}`}
            highlight={
              cell.risk_score > 0.6
                ? "#ef4444"
                : cell.risk_score > 0.3
                  ? "#f59e0b"
                  : "#22c55e"
            }
          />
          <MetricRow
            label={isOk ? "Margin to NAV" : "Deficit to NAV"}
            value={`${marginToNav >= 0 ? "+" : ""}${marginToNav.toFixed(2)} m`}
            highlight={marginToNav >= 0 ? "#22c55e" : "#ef4444"}
          />
          <MetricRow
            label={
              cls === "non_navigable" ? "Deficit to COND" : "Margin to COND"
            }
            value={`${marginToCond >= 0 ? "+" : ""}${marginToCond.toFixed(2)} m`}
            highlight={marginToCond >= 0 ? "#22c55e" : "#f59e0b"}
          />
        </div>

        {/* ── Monsoon badge ── */}
        {cell.is_monsoon && (
          <div
            className="
            mt-2.5 flex items-center gap-1.5
            px-2 py-1 rounded-lg
            bg-blue-500/10 border border-blue-500/20
          "
          >
            <Droplets size={11} className="text-slate-400 flex-shrink-0" />
            <span className="text-[10px] font-semibold text-slate-400">
              Monsoon month — elevated discharge
            </span>
          </div>
        )}
      </div>
    </motion.div>
  );
}

function MetricRow({
  label,
  value,
  highlight,
}: {
  label: string;
  value: string;
  highlight?: string;
}) {
  return (
    <div>
      <div className="text-[10px] text-slate-400 font-medium leading-none mb-0.5">
        {label}
      </div>
      <div
        className="text-[12px] font-bold tabular-nums text-slate-800"
      >
        {value}
      </div>
    </div>
  );
}

// ─── Month Column Header ───────────────────────────────────────────────────────

function MonthHeader({
  monthIndex,
  isCurrentMonth,
  isMonsoon,
  navigablePct,
}: {
  monthIndex: number;
  isCurrentMonth: boolean;
  isMonsoon: boolean;
  navigablePct: number;
}) {
  const barColor =
    navigablePct >= 70 ? "#22c55e" : navigablePct >= 40 ? "#f59e0b" : "#ef4444";

  return (
    <div className="flex flex-col items-center gap-1 px-0.5">
      {/* Month abbreviation */}
      <span
        className={cn(
          "text-[10px] font-bold tracking-wide",
          isCurrentMonth
            ? "text-slate-400"
            : isMonsoon
              ? "text-slate-400"
              : "text-slate-500",
        )}
      >
        {MONTH_SHORT[monthIndex]}
      </span>

      {/* Monsoon dot */}
      {isMonsoon && (
        <span className="w-1.5 h-1.5 rounded-full bg-sky-500/60 flex-shrink-0" />
      )}
      {!isMonsoon && <span className="w-1.5 h-1.5 flex-shrink-0" />}

      {/* Navigability mini-bar (summary column) */}
      <div className="w-full h-1 rounded-full bg-white/[0.06] overflow-hidden">
        <motion.div
          className="h-full rounded-full"
          style={{ backgroundColor: barColor }}
          initial={{ width: 0 }}
          animate={{ width: `${navigablePct}%` }}
          transition={{
            duration: 0.6,
            delay: monthIndex * 0.03,
            ease: [0.16, 1, 0.3, 1],
          }}
        />
      </div>
    </div>
  );
}

// ─── Season Band Labels ────────────────────────────────────────────────────────

function SeasonLabels() {
  return (
    <div className="flex w-full mb-1 pl-[80px]" aria-hidden="true">
      {SEASON_BANDS.map((season) => (
        <div
          key={season.label}
          className="flex items-center justify-center text-[9px] font-semibold text-slate-400 tracking-wider uppercase"
          style={{ flex: season.months.length }}
        >
          {season.label}
        </div>
      ))}
    </div>
  );
}

// ─── Summary Stats for the calendar ───────────────────────────────────────────

function CalendarSummary({
  rows,
  waterwayId,
  year,
}: {
  rows: CalendarRow[];
  waterwayId: WaterwayId;
  year: number;
}) {
  const { peakMonth, worstMonth, avgNavPct, totalNavMonths } = useMemo(() => {
    // Per-month navigable percentage across all segments
    const monthNav = Array.from({ length: 12 }, (_, mi) => {
      const count = rows.filter(
        (r) => r.months[mi]?.navigability_class === "navigable",
      ).length;
      return (count / rows.length) * 100;
    });

    const peak = monthNav.indexOf(Math.max(...monthNav));
    const worst = monthNav.indexOf(Math.min(...monthNav));
    const avg = monthNav.reduce((a, b) => a + b, 0) / 12;

    // Count fully navigable months (>= 80% of segments navigable)
    const navMonths = monthNav.filter((v) => v >= 80).length;

    return {
      peakMonth: peak,
      worstMonth: worst,
      avgNavPct: avg,
      totalNavMonths: navMonths,
    };
  }, [rows]);

  const items = [
    {
      icon: CheckCircle2,
      color: "text-slate-400",
      bg: "bg-emerald-500/10 border-emerald-500/20",
      label: "Avg Navigability",
      value: `${avgNavPct.toFixed(0)}%`,
      sub: "across all months",
    },
    {
      icon: TrendingUp,
      color: "text-slate-400",
      bg: "bg-blue-500/10 border-blue-500/20",
      label: "Best Month",
      value: MONTH_SHORT[peakMonth],
      sub: "peak navigability",
    },
    {
      icon: AlertTriangle,
      color: "text-slate-400",
      bg: "bg-amber-500/10 border-amber-500/20",
      label: "Worst Month",
      value: MONTH_SHORT[worstMonth],
      sub: "min navigability",
    },
    {
      icon: Anchor,
      color: "text-slate-400",
      bg: "bg-sky-500/10 border-sky-500/20",
      label: "Reliable Months",
      value: `${totalNavMonths}/12`,
      sub: "≥ 80% navigable",
    },
  ];

  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4 pt-4 border-t border-slate-900/[0.05]">
      {items.map((item) => (
        <div
          key={item.label}
          className={cn(
            "flex items-center gap-2.5 px-3 py-2.5 rounded-xl border",
            item.bg,
          )}
        >
          <item.icon size={15} className={cn("flex-shrink-0", item.color)} />
          <div>
            <div className="text-[10px] font-semibold tracking-widest text-slate-500 uppercase">
              {item.label}
            </div>
            <div
              className={cn(
                "text-lg font-extrabold tracking-tight tabular-nums leading-tight",
                item.color,
              )}
            >
              {item.value}
            </div>
            <div className="text-[10px] text-slate-400">{item.sub}</div>
          </div>
        </div>
      ))}
    </div>
  );
}

// ─── Row Left Sidebar ──────────────────────────────────────────────────────────

function SegmentRowLabel({
  row,
  isSelected,
  onClick,
}: {
  row: CalendarRow;
  isSelected: boolean;
  onClick: () => void;
}) {
  const navPct = (row.navigable_months_count / 12) * 100;

  return (
    <button
      onClick={onClick}
      className={cn(
        "flex flex-col items-start justify-center w-full h-full px-2 py-1",
        "rounded-lg transition-all duration-150 text-left",
        isSelected
          ? "bg-blue-500/15 border border-blue-500/30"
          : "hover:bg-white/[0.04] border border-transparent",
      )}
    >
      <span
        className={cn(
          "text-[11px] font-bold tabular-nums leading-tight",
          isSelected ? "text-slate-300" : "text-slate-700",
        )}
      >
        {row.km_label}
      </span>
      {/* Navigability bar */}
      <div className="w-full h-1 mt-1 rounded-full bg-white/[0.06] overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{
            width: `${navPct}%`,
            backgroundColor:
              navPct >= 70 ? "#22c55e" : navPct >= 40 ? "#f59e0b" : "#ef4444",
          }}
        />
      </div>
    </button>
  );
}

// ─── Individual Cell ───────────────────────────────────────────────────────────

function CalendarCell({
  cell,
  isHovered,
  isSelected,
  isCurrentMonth,
  animDelay,
  onMouseEnter,
  onMouseLeave,
  onClick,
  cellRef,
}: {
  cell: MonthCell;
  isHovered: boolean;
  isSelected: boolean;
  isCurrentMonth: boolean;
  animDelay: number;
  onMouseEnter: (e: React.MouseEvent) => void;
  onMouseLeave: () => void;
  onClick: () => void;
  cellRef?: (el: HTMLButtonElement | null) => void;
}) {
  const cls = cell.navigability_class;
  const depth = cell.predicted_depth_m;

  return (
    <motion.button
      ref={cellRef}
      initial={{ opacity: 0, scale: 0.75 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{
        duration: 0.25,
        delay: animDelay,
        ease: [0.16, 1, 0.3, 1],
      }}
      whileHover={{ scale: 1.12, zIndex: 10 }}
      whileTap={{ scale: 0.95 }}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
      onClick={onClick}
      className={cn(
        // Base
        "relative w-full h-full rounded-md border transition-all duration-150",
        "flex flex-col items-center justify-center overflow-hidden cursor-pointer",
        // Background by class + probability
        cellBgClass(cls, cell.probability),
        // Border
        isHovered || isSelected
          ? cn("border-2", cellBorderClass(cls))
          : "border-transparent hover:border-slate-300",
        // Current month ring
        isCurrentMonth && "ring-1 ring-blue-500/40",
      )}
      style={{ minHeight: 28 }}
      aria-label={`${MONTH_FULL[cell.month - 1]}: ${NAV_LABEL[cls]}, ${depth.toFixed(1)}m`}
    >
      {/* Depth text (shown at larger sizes) */}
      <span
        className="text-[9px] font-bold tabular-nums leading-none text-slate-800"
      >
        {depth.toFixed(1)}
      </span>

      {/* Monsoon drop indicator */}
      {cell.is_monsoon && (
        <span
          className="absolute top-0.5 right-0.5 w-1 h-1 rounded-full bg-sky-400/80"
          aria-hidden="true"
        />
      )}

      {/* High-risk indicator */}
      {cell.risk_score > 0.65 && (
        <span
          className="absolute bottom-0.5 left-0.5 w-1 h-1 rounded-full bg-red-400/80"
          aria-hidden="true"
        />
      )}
    </motion.button>
  );
}

// ─── Monthly Summary Row ───────────────────────────────────────────────────────

function MonthlySummaryRow({
  rows,
  currentMonth,
}: {
  rows: CalendarRow[];
  currentMonth: number;
}) {
  const summaries = useMemo(
    () =>
      Array.from({ length: 12 }, (_, mi) => {
        const navCount = rows.filter(
          (r) => r.months[mi]?.navigability_class === "navigable",
        ).length;
        const condCount = rows.filter(
          (r) => r.months[mi]?.navigability_class === "conditional",
        ).length;
        const navPct = (navCount / rows.length) * 100;
        const cls: NavigabilityClass =
          navPct >= 60
            ? "navigable"
            : navPct >= 30
              ? "conditional"
              : "non_navigable";
        return { navPct, condCount, cls, month: mi + 1 };
      }),
    [rows],
  );

  return (
    <div
      className="flex items-center gap-1 mt-1"
      role="row"
      aria-label="Monthly summary"
    >
      {/* Spacer for row label column */}
      <div className="w-[80px] flex-shrink-0">
        <span className="text-[9px] font-semibold text-slate-400 uppercase tracking-wider px-2">
          Overall
        </span>
      </div>

      {/* Summary cells */}
      {summaries.map((s, mi) => {
        const isCurrentMonth = s.month === currentMonth;
        return (
          <div
            key={mi}
            className="flex-1 h-7 rounded-md flex items-center justify-center"
            style={{
              backgroundColor: `${NAV_HEX[s.cls]}22`,
              border: `1px solid ${NAV_HEX[s.cls]}30`,
              outline: isCurrentMonth
                ? `2px solid rgba(59,130,246,0.5)`
                : undefined,
              outlineOffset: isCurrentMonth ? "1px" : undefined,
            }}
            title={`${MONTH_FULL[mi]}: ${s.navPct.toFixed(0)}% navigable`}
          >
            <span
              className="text-[9px] font-bold tabular-nums"
              style={{ color: NAV_HEX[s.cls] }}
            >
              {s.navPct.toFixed(0)}%
            </span>
          </div>
        );
      })}
    </div>
  );
}

// ─── Main Component ────────────────────────────────────────────────────────────

export function SeasonalCalendar({
  waterwayId: waterwayIdProp,
  year: yearProp,
  maxRows = 18,
  showSummary = true,
  showSeasonBands = true,
  showHighlights = true,
  className,
}: SeasonalCalendarProps) {
  const storeWaterway = useAppStore((s) => s.selectedWaterway);
  const storeYear = useAppStore((s) => s.selectedYear);
  const storeMonth = useAppStore((s) => s.selectedMonth);

  const waterwayId = waterwayIdProp ?? storeWaterway;
  const year = yearProp ?? storeYear;
  const currentMonth = storeMonth;

  // ── State ──────────────────────────────────────────────────────────────────
  const [hoveredCell, setHoveredCell] = useState<HoveredCell | null>(null);
  const [selectedSeg, setSelectedSeg] = useState<string | null>(null);
  const [showAll, setShowAll] = useState(false);
  const [showSeasons, setShowSeasons] = useState(showSeasonBands);
  const [filter, setFilter] = useState<NavigabilityClass | "all">("all");

  const containerRef = useRef<HTMLDivElement>(null);

  // ── Fetch data ─────────────────────────────────────────────────────────────
  const calendarData = useMemo(
    () => getMockSeasonalCalendar(waterwayId, year),
    [waterwayId, year],
  );

  // ── Build rows from calendar data ──────────────────────────────────────────
  const rows = useMemo<CalendarRow[]>(() => {
    if (!calendarData?.rows) return [];

    return calendarData.rows.map((sr) => ({
      segment_id: sr.segment_id,
      km_start: sr.km_start,
      km_end: sr.km_end,
      km_label: sr.km_label,
      navigable_months_count: sr.navigable_months_count,
      best_month: sr.best_month,
      worst_month: sr.worst_month,
      months: sr.months.map((m) => ({
        month: m.month,
        navigability_class: m.navigability_class as NavigabilityClass,
        predicted_depth_m: m.predicted_depth_m,
        probability: m.probability,
        risk_score: m.risk_score,
        is_monsoon: m.is_monsoon,
        label: m.label,
      })),
    }));
  }, [calendarData]);

  // ── Filter rows ────────────────────────────────────────────────────────────
  const filteredRows = useMemo(() => {
    let filtered = rows;
    if (filter !== "all") {
      filtered = rows.filter((r) =>
        r.months.some((m) => m.navigability_class === filter),
      );
    }
    const visible = showAll ? filtered : filtered.slice(0, maxRows);
    return visible;
  }, [rows, filter, showAll, maxRows]);

  // ── Per-month navigable % (for column headers) ─────────────────────────────
  const monthNavPcts = useMemo(
    () =>
      Array.from({ length: 12 }, (_, mi) => {
        if (rows.length === 0) return 0;
        const count = rows.filter(
          (r) => r.months[mi]?.navigability_class === "navigable",
        ).length;
        return (count / rows.length) * 100;
      }),
    [rows],
  );

  // ── Monsoon month flags ────────────────────────────────────────────────────
  const isMonsoonMonth = (m: number) => m >= 6 && m <= 9;

  // ── Hover handlers ─────────────────────────────────────────────────────────
  const handleCellEnter = useCallback(
    (
      e: React.MouseEvent,
      segIdx: number,
      monthIdx: number,
      cell: MonthCell,
      row: CalendarRow,
    ) => {
      const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
      setHoveredCell({ segmentIdx: segIdx, monthIdx, cell, row, rect });
    },
    [],
  );

  const handleCellLeave = useCallback(() => {
    setHoveredCell(null);
  }, []);

  // ── Empty state ────────────────────────────────────────────────────────────
  if (!calendarData || rows.length === 0) {
    return (
      <div
        className={cn(
          "flex flex-col items-center justify-center",
          "bg-white/[0.03] border border-slate-900/[0.06] rounded-2xl",
          "py-16 px-8 text-center",
          className,
        )}
      >
        <Calendar size={36} className="text-slate-700 mb-4" />
        <p className="text-sm font-semibold text-slate-500">
          No calendar data available
        </p>
        <p className="text-xs text-slate-700 mt-1">
          {waterwayId} · {year}
        </p>
      </div>
    );
  }

  return (
    <div className={cn("flex flex-col", className)}>
      {/* ── Header ──────────────────────────────────────────────────────── */}
      <div className="flex items-start justify-between gap-4 mb-4">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-bold text-slate-900">
              Seasonal Navigability Calendar
            </h3>
            <span
              className="
              text-[9px] font-bold tracking-wider uppercase
              px-2 py-0.5 rounded-full
              bg-white/[0.06] text-slate-500 border border-slate-900/[0.08]
            "
            >
              {year}
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-0.5">
            {waterwayId} · {rows.length} segments · Cell colour = navigability
            class · Depth shown in metres
          </p>
        </div>

        {/* Controls */}
        <div className="flex items-center gap-2 flex-shrink-0">
          {/* Season bands toggle */}
          <button
            onClick={() => setShowSeasons((v) => !v)}
            className="
              flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg
              text-[11px] font-semibold
              bg-white/[0.04] border border-slate-900/[0.08]
              text-slate-400 hover:text-slate-800
              transition-colors duration-150
            "
            title="Toggle season bands"
          >
            {showSeasons ? <EyeOff size={12} /> : <Eye size={12} />}
            <span className="hidden sm:inline">Seasons</span>
          </button>

          {/* Filter dropdown */}
          <FilterDropdown current={filter} onChange={setFilter} />
        </div>
      </div>

      {/* ── Season band background container ────────────────────────────── */}
      <div ref={containerRef} className="relative">
        {/* Season labels row */}
        {showSeasons && <SeasonLabels />}

        {/* ── Month column headers ─────────────────────────────────────── */}
        <div className="flex items-end gap-1 mb-2 pl-[80px]">
          {Array.from({ length: 12 }, (_, mi) => (
            <div key={mi} className="flex-1 min-w-0">
              <MonthHeader
                monthIndex={mi}
                isCurrentMonth={mi + 1 === currentMonth}
                isMonsoon={isMonsoonMonth(mi + 1)}
                navigablePct={monthNavPcts[mi]}
              />
            </div>
          ))}
        </div>

        {/* ── Season band backgrounds ──────────────────────────────────── */}
        {showSeasons && (
          <div className="absolute top-6 bottom-0 left-[80px] right-0 flex pointer-events-none z-0">
            {Array.from({ length: 12 }, (_, mi) => {
              const band = SEASON_BANDS.find((b) => b.months.includes(mi + 1));
              return (
                <div
                  key={mi}
                  className="flex-1"
                  style={{ backgroundColor: band?.color ?? "transparent" }}
                />
              );
            })}
          </div>
        )}

        {/* ── Grid rows ────────────────────────────────────────────────── */}
        <div className="relative z-10 flex flex-col gap-1 max-h-[480px] overflow-y-auto thin-scrollbar pr-0.5">
          <AnimatePresence mode="popLayout">
            {filteredRows.map((row, si) => (
              <motion.div
                key={row.segment_id}
                layout
                initial={{ opacity: 0, x: -8 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 8 }}
                transition={{
                  duration: 0.2,
                  delay: si * 0.012,
                  ease: [0.16, 1, 0.3, 1],
                }}
                className="flex items-stretch gap-1 h-[34px]"
                role="row"
              >
                {/* Row label */}
                <div className="w-[80px] flex-shrink-0">
                  <SegmentRowLabel
                    row={row}
                    isSelected={selectedSeg === row.segment_id}
                    onClick={() =>
                      setSelectedSeg((v) =>
                        v === row.segment_id ? null : row.segment_id,
                      )
                    }
                  />
                </div>

                {/* 12 month cells */}
                {row.months.map((cell, mi) => (
                  <div key={mi} className="flex-1 min-w-0" role="gridcell">
                    <CalendarCell
                      cell={cell}
                      isHovered={
                        hoveredCell?.segmentIdx === si &&
                        hoveredCell?.monthIdx === mi
                      }
                      isSelected={selectedSeg === row.segment_id}
                      isCurrentMonth={mi + 1 === currentMonth}
                      animDelay={si * 0.008 + mi * 0.005}
                      onMouseEnter={(e) =>
                        handleCellEnter(e, si, mi, cell, row)
                      }
                      onMouseLeave={handleCellLeave}
                      onClick={() => {
                        setSelectedSeg((v) =>
                          v === row.segment_id ? null : row.segment_id,
                        );
                      }}
                    />
                  </div>
                ))}
              </motion.div>
            ))}
          </AnimatePresence>
        </div>

        {/* ── Monthly summary row ────────────────────────────────────────── */}
        {showSummary && rows.length > 0 && (
          <div className="mt-2">
            <MonthlySummaryRow
              rows={filteredRows}
              currentMonth={currentMonth}
            />
          </div>
        )}

        {/* ── Hover tooltip ─────────────────────────────────────────────── */}
        <AnimatePresence>
          {hoveredCell && (
            <CellTooltip
              hoveredCell={hoveredCell}
              containerRef={containerRef as React.RefObject<HTMLDivElement>}
            />
          )}
        </AnimatePresence>
      </div>

      {/* ── Show more / collapse ────────────────────────────────────────── */}
      {rows.length > maxRows && (
        <div className="mt-3 flex justify-center">
          <button
            onClick={() => setShowAll((v) => !v)}
            className="
              flex items-center gap-1.5 px-4 py-2 rounded-xl
              text-[12px] font-semibold
              bg-white/[0.04] border border-slate-900/[0.08]
              text-slate-400 hover:text-slate-800 hover:bg-white/[0.07]
              transition-all duration-150
            "
          >
            <motion.span
              animate={{ rotate: showAll ? 180 : 0 }}
              transition={{ duration: 0.2 }}
            >
              <ChevronDown size={14} />
            </motion.span>
            {showAll
              ? `Show fewer segments`
              : `Show all ${rows.length} segments`}
          </button>
        </div>
      )}

      {/* ── Legend ──────────────────────────────────────────────────────── */}
      <div className="flex items-center flex-wrap gap-4 mt-4 pt-3 border-t border-slate-900/[0.05]">
        {/* Navigability class legend */}
        <div className="flex items-center gap-3">
          {(
            ["navigable", "conditional", "non_navigable"] as NavigabilityClass[]
          ).map((cls) => (
            <div key={cls} className="flex items-center gap-1.5">
              <span
                className="w-4 h-4 rounded-md flex-shrink-0"
                style={{
                  backgroundColor: `${NAV_HEX[cls]}55`,
                  border: `1px solid ${NAV_HEX[cls]}70`,
                }}
              />
              <span className="text-[10px] text-slate-500 font-medium">
                {NAV_LABEL[cls]}
              </span>
            </div>
          ))}
        </div>

        {/* Indicator legend */}
        <div className="flex items-center gap-3 ml-auto">
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-sky-400/80" />
            <span className="text-[10px] text-slate-400">Monsoon</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-red-400/80" />
            <span className="text-[10px] text-slate-400">High risk</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span
              className="w-4 h-4 rounded-md"
              style={{
                outline: "2px solid rgba(59,130,246,0.5)",
                outlineOffset: 1,
                backgroundColor: "transparent",
              }}
            />
            <span className="text-[10px] text-slate-400">Current month</span>
          </div>
        </div>
      </div>

      {/* ── Summary stats ───────────────────────────────────────────────── */}
      {showHighlights && rows.length > 0 && (
        <CalendarSummary rows={rows} waterwayId={waterwayId} year={year} />
      )}
    </div>
  );
}

// ─── Filter Dropdown ───────────────────────────────────────────────────────────

function FilterDropdown({
  current,
  onChange,
}: {
  current: NavigabilityClass | "all";
  onChange: (v: NavigabilityClass | "all") => void;
}) {
  const [open, setOpen] = useState(false);

  const OPTIONS: {
    value: NavigabilityClass | "all";
    label: string;
    color?: string;
  }[] = [
    { value: "all", label: "All Classes" },
    { value: "navigable", label: "Navigable", color: "#22c55e" },
    { value: "conditional", label: "Conditional", color: "#f59e0b" },
    { value: "non_navigable", label: "Non-Navigable", color: "#ef4444" },
  ];

  const currentOpt = OPTIONS.find((o) => o.value === current) ?? OPTIONS[0];

  return (
    <div className="relative">
      <button
        onClick={() => setOpen((v) => !v)}
        className="
          flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg
          text-[11px] font-semibold
          bg-white/[0.04] border border-slate-900/[0.08]
          text-slate-400 hover:text-slate-800
          transition-colors duration-150
        "
      >
        {currentOpt.color && (
          <span
            className="w-2 h-2 rounded-full flex-shrink-0"
            style={{ backgroundColor: currentOpt.color }}
          />
        )}
        <span className="hidden sm:inline">{currentOpt.label}</span>
        <ChevronDown
          size={11}
          className={`transition-transform duration-200 ${open ? "rotate-180" : ""}`}
        />
      </button>

      <AnimatePresence>
        {open && (
          <>
            <div
              className="fixed inset-0 z-40"
              onClick={() => setOpen(false)}
            />
            <motion.div
              initial={{ opacity: 0, y: -6, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -6, scale: 0.95 }}
              transition={{ duration: 0.12, ease: [0.16, 1, 0.3, 1] }}
              className="
                absolute right-0 top-full mt-1.5 w-44 z-50
                bg-white/95 backdrop-blur-xl
                border border-slate-900/[0.1] rounded-xl
                shadow-2xl overflow-hidden p-1
              "
            >
              {OPTIONS.map((opt) => (
                <button
                  key={opt.value}
                  onClick={() => {
                    onChange(opt.value);
                    setOpen(false);
                  }}
                  className={cn(
                    "w-full flex items-center gap-2.5 px-3 py-2.5 rounded-lg",
                    "text-[12px] font-medium transition-colors duration-100",
                    current === opt.value
                      ? "bg-blue-500/15 text-slate-300"
                      : "text-slate-400 hover:text-slate-900 hover:bg-white/[0.06]",
                  )}
                >
                  {opt.color ? (
                    <span
                      className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                      style={{ backgroundColor: opt.color }}
                    />
                  ) : (
                    <span className="w-2.5 h-2.5 rounded-full bg-slate-600 flex-shrink-0" />
                  )}
                  {opt.label}
                  {current === opt.value && (
                    <span className="ml-auto text-[10px] text-slate-400 font-bold">
                      ✓
                    </span>
                  )}
                </button>
              ))}
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  );
}

// ─── Compact Variant ──────────────────────────────────────────────────────────

/**
 * SeasonalCalendarCompact
 *
 * Slimmer version for the dashboard right panel —
 * fewer rows, no summary stats, no highlights.
 */
export function SeasonalCalendarCompact({
  waterwayId,
  year,
  maxRows = 8,
  className,
}: Pick<
  SeasonalCalendarProps,
  "waterwayId" | "year" | "maxRows" | "className"
>) {
  return (
    <SeasonalCalendar
      waterwayId={waterwayId}
      year={year}
      maxRows={maxRows}
      showSummary={true}
      showSeasonBands={false}
      showHighlights={false}
      className={className}
    />
  );
}
