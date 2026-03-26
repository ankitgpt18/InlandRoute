// ============================================================
// AIDSTL — Inland Waterway Navigability Prediction System
// Maps Page — Full-screen interactive river navigability map
// ============================================================

'use client';

import React, { useMemo, useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  MapPin,
  Layers,
  ChevronDown,
  ChevronUp,
  ChevronRight,
  ChevronLeft,
  X,
  Waves,
  Ruler,
  Gauge,
  TrendingUp,
  TrendingDown,
  Activity,
  Navigation,
  AlertTriangle,
  CheckCircle2,
  Info,
  Maximize2,
  BarChart2,
  Eye,
  EyeOff,
  SlidersHorizontal,
  RefreshCw,
  Download,
  Anchor,
  Droplets,
  Wind,
  Satellite,
  Clock,
  ArrowUpRight,
  ArrowDownRight,
  Minus,
  Search,
  Filter,
} from 'lucide-react';
import { useAppStore } from '@/store/app-store';
import {
  getMockNavigabilityMap,
  getMockDepthProfile,
  getMockAlerts,
  buildNW1GeoJSON,
  buildNW2GeoJSON,
} from '@/lib/mock-data';
import { RiverMap } from '@/components/maps/river-map';
import { DepthProfileChart } from '@/components/charts/depth-profile';
import { NavigabilityBadge } from '@/components/ui/navigability-badge';
import { cn } from '@/lib/utils';
import type { NavigabilityClass, WaterwayId } from '@/types';

// ─── Constants ─────────────────────────────────────────────────────────────────

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

const NAV_LABELS: Record<NavigabilityClass, string> = {
  navigable:     'Navigable',
  conditional:   'Conditional',
  non_navigable: 'Non-Navigable',
};

// ─── Types ──────────────────────────────────────────────────────────────────────

interface SegmentInfo {
  segment_id:        string;
  waterway_id:       WaterwayId;
  km_start:          number;
  km_end:            number;
  navigability_class: NavigabilityClass;
  depth_m:           number;
  width_m:           number;
  confidence:        number;
  velocity_ms?:      number;
  state?:            string;
}

type PanelMode = 'segments' | 'profile' | 'alerts' | 'settings';

// ─── Helpers ────────────────────────────────────────────────────────────────────

function formatKm(start: number, end: number): string {
  return `${start.toFixed(0)}–${end.toFixed(0)} km`;
}

function formatRelativeTime(dateStr: string): string {
  const date    = new Date(dateStr);
  const diffMs  = Date.now() - date.getTime();
  const diffMin = Math.floor(diffMs / 60_000);
  if (diffMin < 1)  return 'Just now';
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24)  return `${diffHr}h ago`;
  return `${Math.floor(diffHr / 24)}d ago`;
}

// ─── Metric Pill ────────────────────────────────────────────────────────────────

function MetricPill({
  label,
  value,
  color,
  icon: Icon,
}: {
  label:  string;
  value:  string;
  color?: string;
  icon?:  React.ElementType;
}) {
  return (
    <div className="flex flex-col gap-0.5 px-3 py-2 rounded-xl bg-white/[0.04] border border-white/[0.07]">
      <div className="flex items-center gap-1.5">
        {Icon && <Icon size={10} className="text-slate-500 flex-shrink-0" />}
        <span className="text-[9px] font-semibold text-slate-500 uppercase tracking-wider">{label}</span>
      </div>
      <span
        className="text-[15px] font-extrabold tabular-nums tracking-tight leading-none"
        style={{ color: color ?? '#f8fafc' }}
      >
        {value}
      </span>
    </div>
  );
}

// ─── Segment List Item ──────────────────────────────────────────────────────────

function SegmentListItem({
  segment,
  isSelected,
  isHovered,
  onClick,
  onHover,
}: {
  segment:    SegmentInfo;
  isSelected: boolean;
  isHovered:  boolean;
  onClick:    () => void;
  onHover:    (id: string | null) => void;
}) {
  const cls        = segment.navigability_class;
  const accentColor = NAV_COLORS[cls];
  const isCritical  = cls === 'non_navigable';

  return (
    <motion.button
      layout
      whileHover={{ x: 2 }}
      whileTap={{ scale: 0.98 }}
      onClick={onClick}
      onMouseEnter={() => onHover(segment.segment_id)}
      onMouseLeave={() => onHover(null)}
      className={cn(
        'relative w-full text-left px-3 py-2.5 rounded-xl border transition-all duration-150 overflow-hidden',
        isSelected
          ? 'bg-blue-500/15 border-blue-500/35'
          : isHovered
            ? 'bg-white/[0.06] border-white/[0.12]'
            : 'bg-white/[0.02] border-white/[0.06] hover:bg-white/[0.05]',
      )}
    >
      {/* Left accent bar */}
      <div
        className="absolute left-0 top-2 bottom-2 w-0.5 rounded-r-full"
        style={{ backgroundColor: accentColor, opacity: isSelected ? 1 : 0.4 }}
      />

      <div className="flex items-center justify-between gap-2 pl-2">
        {/* Left: km + state */}
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span
              className={cn(
                'text-[12px] font-bold tabular-nums',
                isSelected ? 'text-blue-300' : 'text-slate-200',
              )}
            >
              {formatKm(segment.km_start, segment.km_end)}
            </span>
            {segment.state && (
              <span className="text-[9px] text-slate-600 font-medium truncate hidden sm:block">
                {segment.state}
              </span>
            )}
          </div>
          <div className="flex items-center gap-2 mt-0.5">
            <span className="text-[10px] text-slate-500 tabular-nums">
              {segment.depth_m.toFixed(2)} m
            </span>
            <span className="text-slate-700 text-[10px]">·</span>
            <span className="text-[10px] text-slate-600 tabular-nums">
              {segment.width_m.toFixed(0)} m wide
            </span>
          </div>
        </div>

        {/* Right: badge + confidence */}
        <div className="flex flex-col items-end gap-1 flex-shrink-0">
          <NavigabilityBadge
            navigabilityClass={cls}
            size="xs"
            variant="subtle"
            pulse={isCritical && isSelected}
          />
          <span className="text-[9px] text-slate-600 tabular-nums">
            {(segment.confidence * 100).toFixed(0)}% conf
          </span>
        </div>
      </div>

      {/* Critical pulsing overlay */}
      {isCritical && isSelected && (
        <motion.div
          className="absolute inset-0 rounded-xl pointer-events-none"
          animate={{ opacity: [0, 0.05, 0] }}
          transition={{ duration: 2, repeat: Infinity }}
          style={{ background: 'rgba(239,68,68,0.3)' }}
        />
      )}
    </motion.button>
  );
}

// ─── Segment Detail Card ──────────────────────────────────────────────────────

function SegmentDetailCard({
  segment,
  onClose,
  month,
  year,
}: {
  segment: SegmentInfo;
  onClose: () => void;
  month:   number;
  year:    number;
}) {
  const cls         = segment.navigability_class;
  const accentColor = NAV_COLORS[cls];

  const thresholdDiff =
    cls === 'navigable'
      ? segment.depth_m - 3.0
      : cls === 'conditional'
        ? segment.depth_m - 2.0
        : 2.0 - segment.depth_m;

  const isAbove = cls !== 'non_navigable';

  const metrics = [
    {
      label: 'Depth',
      value: `${segment.depth_m.toFixed(2)} m`,
      color: accentColor,
      icon:  Waves,
    },
    {
      label: 'Width',
      value: `${segment.width_m.toFixed(0)} m`,
      color: '#38bdf8',
      icon:  Ruler,
    },
    {
      label: 'Confidence',
      value: `${(segment.confidence * 100).toFixed(1)}%`,
      color: '#8b5cf6',
      icon:  Gauge,
    },
    ...(segment.velocity_ms !== undefined
      ? [{
          label: 'Velocity',
          value: `${segment.velocity_ms.toFixed(2)} m/s`,
          color: '#f59e0b',
          icon:  Activity,
        }]
      : []),
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 16, scale: 0.97 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: 8, scale: 0.97 }}
      transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
      className="rounded-2xl border overflow-hidden"
      style={{
        background:  `linear-gradient(135deg, ${accentColor}0a 0%, rgba(15,23,42,0.97) 60%)`,
        borderColor: `${accentColor}30`,
      }}
    >
      {/* Top accent */}
      <div
        className="h-0.5 w-full"
        style={{ background: `linear-gradient(90deg, transparent, ${accentColor}, transparent)` }}
      />

      <div className="p-4">
        {/* Header */}
        <div className="flex items-start justify-between gap-2 mb-3">
          <div>
            <div className="flex items-center gap-2 flex-wrap">
              <h3 className="text-[14px] font-bold text-slate-100">
                Segment {segment.segment_id.split('-').pop()}
              </h3>
              <NavigabilityBadge
                navigabilityClass={cls}
                size="sm"
                variant="glow"
                animate
              />
            </div>
            <div className="flex items-center gap-2 mt-1">
              <MapPin size={10} className="text-slate-500 flex-shrink-0" />
              <span className="text-[11px] font-semibold text-slate-400">
                {segment.waterway_id}
              </span>
              <span className="text-slate-700 text-[10px]">·</span>
              <span className="text-[11px] text-slate-500 tabular-nums font-medium">
                {formatKm(segment.km_start, segment.km_end)}
              </span>
              {segment.state && (
                <>
                  <span className="text-slate-700 text-[10px]">·</span>
                  <span className="text-[11px] text-slate-600">{segment.state}</span>
                </>
              )}
            </div>
          </div>
          <button
            onClick={onClose}
            className="flex-shrink-0 w-7 h-7 rounded-lg flex items-center justify-center text-slate-500 hover:text-slate-300 hover:bg-white/[0.08] transition-colors duration-150"
            aria-label="Close segment detail"
          >
            <X size={14} />
          </button>
        </div>

        {/* Metrics grid */}
        <div className="grid grid-cols-2 gap-2 mb-3">
          {metrics.map((m) => {
            const Icon = m.icon;
            return (
              <MetricPill
                key={m.label}
                label={m.label}
                value={m.value}
                color={m.color}
                icon={Icon}
              />
            );
          })}
        </div>

        {/* Depth vs threshold */}
        <div
          className="px-3 py-2.5 rounded-xl border mb-3"
          style={{
            background:   `${isAbove ? '#22c55e' : '#ef4444'}0c`,
            borderColor:  `${isAbove ? '#22c55e' : '#ef4444'}25`,
          }}
        >
          <div className="flex items-center justify-between mb-1.5">
            <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider">
              {isAbove ? 'Depth Margin' : 'Depth Deficit'}
            </span>
            {isAbove ? (
              <ArrowUpRight size={12} className="text-emerald-400" />
            ) : (
              <ArrowDownRight size={12} className="text-red-400" />
            )}
          </div>
          <div className="flex items-baseline gap-1">
            <span
              className="text-[20px] font-extrabold tabular-nums leading-none"
              style={{ color: isAbove ? '#22c55e' : '#ef4444' }}
            >
              {isAbove ? '+' : '-'}{Math.abs(thresholdDiff).toFixed(2)}
            </span>
            <span className="text-[12px] font-semibold text-slate-500">m</span>
            <span className="text-[10px] text-slate-600 ml-1">
              {isAbove ? 'above' : 'below'}{' '}
              {cls === 'navigable' ? 'navigable' : 'conditional'} threshold
            </span>
          </div>

          {/* Depth bar */}
          <div className="mt-2 h-1.5 rounded-full bg-white/[0.06] overflow-hidden relative">
            {/* Threshold marker */}
            <div
              className="absolute top-0 bottom-0 w-0.5 bg-white/30 z-10"
              style={{
                left: `${Math.min(100, (cls === 'navigable' ? 3.0 : 2.0) / Math.max(segment.depth_m + 1, 5) * 100)}%`,
              }}
            />
            {/* Fill */}
            <motion.div
              className="h-full rounded-full"
              style={{ backgroundColor: accentColor }}
              initial={{ width: 0 }}
              animate={{
                width: `${Math.min(100, (segment.depth_m / Math.max(segment.depth_m + 1, 5)) * 100)}%`,
              }}
              transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
            />
          </div>
        </div>

        {/* Month/Year tag */}
        <div className="flex items-center justify-between text-[10px] text-slate-600">
          <div className="flex items-center gap-1.5">
            <Clock size={9} />
            <span>{MONTH_NAMES[month - 1]} {year} composite</span>
          </div>
          <div className="flex items-center gap-1.5">
            <Satellite size={9} />
            <span>Sentinel-2 L2A · 10m</span>
          </div>
        </div>
      </div>
    </motion.div>
  );
}

// ─── Segment List Panel ────────────────────────────────────────────────────────

function SegmentListPanel({
  segments,
  selectedId,
  onSelect,
  filterClass,
  onFilterChange,
  searchQuery,
  onSearchChange,
}: {
  segments:       SegmentInfo[];
  selectedId:     string | null;
  onSelect:       (id: string) => void;
  filterClass:    NavigabilityClass | 'all';
  onFilterChange: (v: NavigabilityClass | 'all') => void;
  searchQuery:    string;
  onSearchChange: (v: string) => void;
}) {
  const hoveredSegmentId    = useAppStore((s) => s.hoveredSegmentId);
  const setHoveredSegmentId = useAppStore((s) => s.setHoveredSegmentId);

  // Filter segments
  const filtered = useMemo(() => {
    let list = segments;
    if (filterClass !== 'all') {
      list = list.filter((s) => s.navigability_class === filterClass);
    }
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      list = list.filter(
        (s) =>
          s.segment_id.toLowerCase().includes(q) ||
          s.km_start.toString().includes(q) ||
          s.km_end.toString().includes(q) ||
          (s.state ?? '').toLowerCase().includes(q),
      );
    }
    return list;
  }, [segments, filterClass, searchQuery]);

  const filterOptions: { value: NavigabilityClass | 'all'; label: string; color: string }[] = [
    { value: 'all',          label: 'All',     color: '#94a3b8' },
    { value: 'navigable',    label: 'Nav',     color: '#22c55e' },
    { value: 'conditional',  label: 'Cond',    color: '#f59e0b' },
    { value: 'non_navigable',label: 'Closed',  color: '#ef4444' },
  ];

  // Counts
  const counts: Record<NavigabilityClass | 'all', number> = useMemo(() => ({
    all:           segments.length,
    navigable:     segments.filter((s) => s.navigability_class === 'navigable').length,
    conditional:   segments.filter((s) => s.navigability_class === 'conditional').length,
    non_navigable: segments.filter((s) => s.navigability_class === 'non_navigable').length,
  }), [segments]);

  return (
    <div className="flex flex-col h-full">
      {/* Search */}
      <div className="relative mb-2">
        <Search
          size={12}
          className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500 pointer-events-none"
        />
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => onSearchChange(e.target.value)}
          placeholder="Search km, state…"
          className="
            w-full pl-8 pr-3 py-2 rounded-xl
            bg-white/[0.04] border border-white/[0.08]
            text-[12px] text-slate-300 placeholder:text-slate-600
            focus:outline-none focus:border-blue-500/50 focus:bg-white/[0.06]
            transition-all duration-150
          "
        />
        {searchQuery && (
          <button
            onClick={() => onSearchChange('')}
            className="absolute right-2.5 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300"
          >
            <X size={11} />
          </button>
        )}
      </div>

      {/* Filter tabs */}
      <div className="flex items-center gap-1 mb-3 p-0.5 bg-white/[0.03] rounded-xl border border-white/[0.06]">
        {filterOptions.map((opt) => {
          const isActive = filterClass === opt.value;
          return (
            <button
              key={opt.value}
              onClick={() => onFilterChange(opt.value)}
              className={cn(
                'flex-1 flex items-center justify-center gap-1 py-1.5 rounded-lg text-[10px] font-bold uppercase tracking-wider transition-all duration-150',
                isActive ? 'text-white' : 'text-slate-500 hover:text-slate-300',
              )}
              style={
                isActive
                  ? { background: `${opt.color}22`, color: opt.color, border: `1px solid ${opt.color}40` }
                  : {}
              }
            >
              <span>{opt.label}</span>
              <span
                className="text-[9px] tabular-nums"
                style={{ color: isActive ? `${opt.color}cc` : '#475569' }}
              >
                {counts[opt.value]}
              </span>
            </button>
          );
        })}
      </div>

      {/* Segment list */}
      <div className="flex-1 overflow-y-auto thin-scrollbar space-y-1 pr-0.5">
        <AnimatePresence mode="popLayout">
          {filtered.length === 0 ? (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex flex-col items-center justify-center py-10 text-center"
            >
              <MapPin size={24} className="text-slate-700 mb-2" />
              <p className="text-[12px] text-slate-500 font-medium">No segments match</p>
              <p className="text-[10px] text-slate-700 mt-1">
                {searchQuery ? 'Clear search to see all' : 'Adjust filter above'}
              </p>
            </motion.div>
          ) : (
            filtered.map((seg, i) => (
              <motion.div
                key={seg.segment_id}
                layout
                initial={{ opacity: 0, x: -8 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 8 }}
                transition={{ duration: 0.2, delay: i * 0.01, ease: [0.16, 1, 0.3, 1] }}
              >
                <SegmentListItem
                  segment={seg}
                  isSelected={selectedId === seg.segment_id}
                  isHovered={hoveredSegmentId === seg.segment_id}
                  onClick={() => onSelect(seg.segment_id)}
                  onHover={(id) => setHoveredSegmentId(id)}
                />
              </motion.div>
            ))
          )}
        </AnimatePresence>
      </div>

      {/* Footer count */}
      <div className="mt-2 text-center">
        <span className="text-[10px] text-slate-600">
          {filtered.length} of {segments.length} segments
        </span>
      </div>
    </div>
  );
}

// ─── Map Stats Overlay ─────────────────────────────────────────────────────────

function MapStatsFloater({
  navMap,
  month,
  year,
  waterway,
}: {
  navMap:    ReturnType<typeof getMockNavigabilityMap>;
  month:     number;
  year:      number;
  waterway:  WaterwayId;
}) {
  const isMonsoon = month >= 6 && month <= 9;

  return (
    <motion.div
      initial={{ opacity: 0, y: -12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.3, ease: [0.16, 1, 0.3, 1] }}
      className="
        absolute top-4 left-1/2 -translate-x-1/2 z-20
        flex items-center gap-0 overflow-hidden
        bg-slate-900/90 backdrop-blur-xl
        border border-white/[0.1] rounded-full
        shadow-2xl shadow-black/50
        pointer-events-none
      "
    >
      {/* Waterway pill */}
      <div className="flex items-center gap-2 px-3.5 py-2">
        <span
          className="w-2 h-2 rounded-full flex-shrink-0"
          style={{
            backgroundColor: waterway === 'NW-1' ? '#3b82f6' : '#8b5cf6',
            boxShadow:        `0 0 6px ${waterway === 'NW-1' ? '#3b82f6' : '#8b5cf6'}`,
          }}
        />
        <span className="text-[11px] font-bold text-slate-200">
          {waterway === 'NW-1' ? 'NW-1 · Ganga' : 'NW-2 · Brahmaputra'}
        </span>
      </div>

      <div className="w-px h-5 bg-white/[0.1]" />

      {/* Month */}
      <div className="flex items-center gap-1.5 px-3 py-2">
        <span className="text-[11px] font-semibold text-slate-400">
          {MONTH_NAMES[month - 1].slice(0, 3)} {year}
        </span>
        {isMonsoon && (
          <span className="text-[9px] font-bold text-sky-400">🌧</span>
        )}
      </div>

      <div className="w-px h-5 bg-white/[0.1]" />

      {/* Navigable % */}
      <div className="flex items-center gap-1.5 px-3 py-2">
        <span className="w-2 h-2 rounded-full bg-emerald-500 flex-shrink-0" />
        <span className="text-[11px] font-bold text-emerald-400 tabular-nums">
          {(navMap?.navigable_pct ?? 0).toFixed(0)}%
        </span>
        <span className="text-[10px] text-slate-500">navigable</span>
      </div>

      <div className="w-px h-5 bg-white/[0.1]" />

      {/* Total km */}
      <div className="flex items-center gap-1.5 px-3 py-2">
        <span className="text-[11px] font-bold text-blue-400 tabular-nums">
          {(navMap?.navigable_km ?? 0).toFixed(0)} km
        </span>
        <span className="text-[10px] text-slate-500">open</span>
      </div>
    </motion.div>
  );
}

// ─── Layer Settings Panel ──────────────────────────────────────────────────────

function LayerSettingsPanel() {
  const showDepthOverlay    = useAppStore((s) => s.showDepthOverlay);
  const showWidthOverlay    = useAppStore((s) => s.showWidthOverlay);
  const showAlerts          = useAppStore((s) => s.showAlerts);
  const showConfidenceLayer = useAppStore((s) => s.showConfidenceLayer);
  const mapStyle            = useAppStore((s) => s.mapStyle);
  const setShowDepthOverlay    = useAppStore((s) => s.setShowDepthOverlay);
  const setShowWidthOverlay    = useAppStore((s) => s.setShowWidthOverlay);
  const setShowAlerts          = useAppStore((s) => s.setShowAlerts);
  const setShowConfidenceLayer = useAppStore((s) => s.setShowConfidenceLayer);
  const setMapStyle            = useAppStore((s) => s.setMapStyle);

  const layers = [
    {
      id:      'depth',
      label:   'Depth Labels',
      sub:     'Show predicted depth on segments',
      active:  showDepthOverlay,
      toggle:  () => setShowDepthOverlay(!showDepthOverlay),
      color:   '#3b82f6',
    },
    {
      id:      'width',
      label:   'Width Indicators',
      sub:     'Visualise channel width',
      active:  showWidthOverlay,
      toggle:  () => setShowWidthOverlay(!showWidthOverlay),
      color:   '#0ea5e9',
    },
    {
      id:      'alerts',
      label:   'Risk Alerts',
      sub:     'Highlight alert segments',
      active:  showAlerts,
      toggle:  () => setShowAlerts(!showAlerts),
      color:   '#ef4444',
    },
    {
      id:      'confidence',
      label:   'Confidence Layer',
      sub:     'Show prediction uncertainty',
      active:  showConfidenceLayer,
      toggle:  () => setShowConfidenceLayer(!showConfidenceLayer),
      color:   '#8b5cf6',
    },
  ];

  const styles: { value: 'dark' | 'satellite' | 'light'; label: string }[] = [
    { value: 'dark',      label: 'Dark'      },
    { value: 'satellite', label: 'Satellite' },
    { value: 'light',     label: 'Light'     },
  ];

  return (
    <div className="flex flex-col gap-5">
      {/* Map style */}
      <div>
        <div className="text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-2.5">
          Map Style
        </div>
        <div className="grid grid-cols-3 gap-1.5">
          {styles.map((s) => (
            <button
              key={s.value}
              onClick={() => setMapStyle(s.value)}
              className={cn(
                'py-2 px-2 rounded-xl text-[11px] font-semibold border transition-all duration-150',
                mapStyle === s.value
                  ? 'bg-blue-500/20 border-blue-500/40 text-blue-300'
                  : 'bg-white/[0.03] border-white/[0.07] text-slate-500 hover:text-slate-300 hover:bg-white/[0.06]',
              )}
            >
              {s.label}
            </button>
          ))}
        </div>
      </div>

      {/* Layers */}
      <div>
        <div className="text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-2.5">
          Overlay Layers
        </div>
        <div className="space-y-2">
          {layers.map((layer) => (
            <button
              key={layer.id}
              onClick={layer.toggle}
              className={cn(
                'w-full flex items-center gap-3 px-3 py-2.5 rounded-xl border text-left transition-all duration-150',
                layer.active
                  ? 'bg-white/[0.05] border-white/[0.10]'
                  : 'bg-white/[0.02] border-white/[0.06] opacity-60 hover:opacity-80',
              )}
            >
              {/* Toggle */}
              <div
                className={cn(
                  'relative flex-shrink-0 w-8 h-4.5 rounded-full transition-all duration-300',
                  layer.active ? '' : 'bg-slate-700',
                )}
                style={layer.active ? { backgroundColor: `${layer.color}50`, border: `1px solid ${layer.color}60` } : {}}
              >
                <motion.div
                  className="absolute top-0.5 w-3 h-3 rounded-full"
                  animate={{ left: layer.active ? 'calc(100% - 14px)' : '2px' }}
                  transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                  style={{ backgroundColor: layer.active ? layer.color : '#475569' }}
                />
              </div>

              <div className="flex-1 min-w-0">
                <div className="text-[12px] font-semibold text-slate-200">{layer.label}</div>
                <div className="text-[10px] text-slate-500">{layer.sub}</div>
              </div>

              {layer.active ? (
                <Eye size={13} className="text-slate-400 flex-shrink-0" />
              ) : (
                <EyeOff size={13} className="text-slate-600 flex-shrink-0" />
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Navigability threshold reference */}
      <div>
        <div className="text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-2.5">
          Colour Reference
        </div>
        <div className="space-y-1.5">
          {(
            [
              { cls: 'navigable',     depth: '≥ 3.0 m', width: '≥ 50 m',  vessels: '1,500 DWT barge' },
              { cls: 'conditional',   depth: '2.0–3.0 m', width: '30–50 m', vessels: 'Shallow-draft only' },
              { cls: 'non_navigable', depth: '< 2.0 m',  width: '< 30 m',  vessels: 'Navigation unsafe' },
            ] as { cls: NavigabilityClass; depth: string; width: string; vessels: string }[]
          ).map((row) => (
            <div
              key={row.cls}
              className="flex items-center gap-3 px-3 py-2 rounded-xl border"
              style={{
                background:   `${NAV_COLORS[row.cls]}0c`,
                borderColor:  `${NAV_COLORS[row.cls]}25`,
              }}
            >
              <span
                className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                style={{ backgroundColor: NAV_COLORS[row.cls] }}
              />
              <div className="flex-1 min-w-0">
                <div className="text-[11px] font-semibold" style={{ color: NAV_COLORS[row.cls] }}>
                  {NAV_LABELS[row.cls]}
                </div>
                <div className="text-[9px] text-slate-600 tabular-nums">
                  {row.depth} · {row.width}
                </div>
              </div>
              <div className="text-[9px] text-slate-600 text-right hidden sm:block">
                {row.vessels}
              </div>
            </div>
          ))}
        </div>
        <div className="mt-2 text-[9px] text-slate-700 text-center">
          IWAI LAD standard · Gati Shakti Vishwavidyalaya
        </div>
      </div>
    </div>
  );
}

// ─── Alert Quick Panel ──────────────────────────────────────────────────────────

function AlertQuickPanel({ waterway }: { waterway: WaterwayId }) {
  const setSelectedSegment = useAppStore((s) => s.setSelectedSegmentId);
  const alerts = useMemo(() => getMockAlerts(waterway), [waterway]);
  const active = alerts.filter((a) => a.is_active);
  const critical = active.filter((a) => a.severity === 'CRITICAL');

  if (active.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-center">
        <div className="relative mb-4">
          {[0, 1].map((i) => (
            <motion.div
              key={i}
              className="absolute rounded-full border border-emerald-500/20"
              style={{ width: 36 + i * 20, height: 36 + i * 20, top: -(i * 10), left: -(i * 10) }}
              animate={{ opacity: [0.5, 0, 0.5] }}
              transition={{ duration: 2.5, repeat: Infinity, delay: i * 0.6 }}
            />
          ))}
          <div className="relative w-9 h-9 rounded-xl bg-emerald-500/15 border border-emerald-500/25 flex items-center justify-center">
            <CheckCircle2 size={18} className="text-emerald-400" />
          </div>
        </div>
        <p className="text-[13px] font-bold text-slate-300">All Clear</p>
        <p className="text-[11px] text-slate-600 mt-1">No active alerts for {waterway}</p>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {/* Summary */}
      <div className="flex items-center gap-2 mb-3">
        {critical.length > 0 && (
          <motion.span
            animate={{ scale: [1, 1.05, 1] }}
            transition={{ duration: 2, repeat: Infinity }}
            className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-red-500/12 border border-red-500/30 text-[10px] font-bold text-red-400"
          >
            <span className="relative flex h-1.5 w-1.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-red-500" />
            </span>
            {critical.length} Critical
          </motion.span>
        )}
        <span className="text-[11px] text-slate-500">
          {active.length} total active
        </span>
      </div>

      {/* Alert cards */}
      {active.slice(0, 6).map((alert, i) => {
        const isCritical = alert.severity === 'CRITICAL';
        const accentColor = isCritical ? '#ef4444' : alert.severity === 'WARNING' ? '#f59e0b' : '#0ea5e9';

        return (
          <motion.div
            key={alert.alert_id}
            initial={{ opacity: 0, x: -8 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.05, duration: 0.2 }}
            className={cn(
              'relative px-3 py-2.5 rounded-xl border overflow-hidden cursor-pointer',
              'transition-all duration-150 hover:border-white/[0.15] hover:bg-white/[0.05]',
            )}
            style={{
              background:  `${accentColor}08`,
              borderColor: `${accentColor}28`,
            }}
            onClick={() => setSelectedSegment(alert.segment_id)}
          >
            {isCritical && (
              <motion.div
                className="absolute left-0 top-0 bottom-0 w-0.5"
                style={{ background: accentColor }}
                animate={{ opacity: [0.5, 1, 0.5] }}
                transition={{ duration: 2, repeat: Infinity }}
              />
            )}

            <div className="flex items-start gap-2.5 pl-1.5">
              <div>
                <div className="text-[11px] font-bold leading-tight" style={{ color: accentColor }}>
                  {alert.title}
                </div>
                <div className="flex items-center gap-1.5 mt-0.5">
                  <span className="text-[9px] text-slate-500 tabular-nums">
                    km {alert.km_start.toFixed(0)}–{alert.km_end.toFixed(0)}
                  </span>
                  <span className="text-slate-700 text-[9px]">·</span>
                  <span className="text-[9px] text-slate-600 tabular-nums">
                    {(alert.predicted_value).toFixed(2)} {alert.unit}
                  </span>
                  <span className="text-slate-700 text-[9px]">·</span>
                  <span className="text-[9px] text-slate-600">
                    {formatRelativeTime(alert.created_at)}
                  </span>
                </div>
              </div>
              <div className="ml-auto flex-shrink-0">
                <span
                  className="text-[10px] font-bold tabular-nums"
                  style={{ color: accentColor }}
                >
                  {(alert.risk_score * 100).toFixed(0)}
                </span>
              </div>
            </div>
          </motion.div>
        );
      })}

      {active.length > 6 && (
        <p className="text-center text-[10px] text-slate-600 pt-1">
          +{active.length - 6} more alerts — visit Alerts page
        </p>
      )}
    </div>
  );
}

// ─── Side Panel Tab Bar ────────────────────────────────────────────────────────

const PANEL_TABS: {
  id:    PanelMode;
  label: string;
  icon:  React.ElementType;
}[] = [
  { id: 'segments', label: 'Segments', icon: Layers          },
  { id: 'profile',  label: 'Profile',  icon: BarChart2       },
  { id: 'alerts',   label: 'Alerts',   icon: AlertTriangle   },
  { id: 'settings', label: 'Layers',   icon: SlidersHorizontal },
];

function PanelTabBar({
  active,
  onChange,
  alertCount,
}: {
  active:     PanelMode;
  onChange:   (m: PanelMode) => void;
  alertCount: number;
}) {
  return (
    <div className="flex items-center gap-0.5 p-0.5 bg-white/[0.04] rounded-xl border border-white/[0.07] flex-shrink-0">
      {PANEL_TABS.map((tab) => {
        const Icon     = tab.icon;
        const isActive = active === tab.id;
        return (
          <motion.button
            key={tab.id}
            onClick={() => onChange(tab.id)}
            whileTap={{ scale: 0.95 }}
            className={cn(
              'relative flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg flex-1 justify-center',
              'text-[10px] font-bold uppercase tracking-wider transition-all duration-200',
              isActive
                ? 'bg-blue-500/20 text-blue-300'
                : 'text-slate-500 hover:text-slate-300 hover:bg-white/[0.05]',
            )}
          >
            <Icon size={11} strokeWidth={isActive ? 2.5 : 2} />
            <span className="hidden sm:inline">{tab.label}</span>

            {tab.id === 'alerts' && alertCount > 0 && (
              <span className="absolute -top-1 -right-1 min-w-[14px] h-3.5 px-1 bg-red-500 text-white text-[8px] font-bold rounded-full flex items-center justify-center leading-none">
                {alertCount > 9 ? '9+' : alertCount}
              </span>
            )}
          </motion.button>
        );
      })}
    </div>
  );
}

// ─── Main Maps Page ────────────────────────────────────────────────────────────

export default function MapsPage() {
  // ── Store ──────────────────────────────────────────────────────────────────
  const selectedWaterway    = useAppStore((s) => s.selectedWaterway);
  const selectedMonth       = useAppStore((s) => s.selectedMonth);
  const selectedYear        = useAppStore((s) => s.selectedYear);
  const selectedSegmentId   = useAppStore((s) => s.selectedSegmentId);
  const setSelectedSegment  = useAppStore((s) => s.setSelectedSegmentId);

  // ── Local state ────────────────────────────────────────────────────────────
  const [panelMode,    setPanelMode]    = useState<PanelMode>('segments');
  const [panelOpen,    setPanelOpen]    = useState(true);
  const [filterClass,  setFilterClass]  = useState<NavigabilityClass | 'all'>('all');
  const [searchQuery,  setSearchQuery]  = useState('');

  // ── Data ────────────────────────────────────────────────────────────────────
  const navMap = useMemo(
    () => getMockNavigabilityMap(selectedWaterway, selectedMonth),
    [selectedWaterway, selectedMonth],
  );

  const geojsonData = useMemo(
    () => selectedWaterway === 'NW-1'
      ? buildNW1GeoJSON(selectedMonth)
      : buildNW2GeoJSON(selectedMonth),
    [selectedWaterway, selectedMonth],
  );

  // Build segment list from GeoJSON features
  const segments = useMemo<SegmentInfo[]>(() => {
    if (!geojsonData?.features) return [];
    return geojsonData.features.map((f) => {
      const p = f.properties;
      return {
        segment_id:         p.segment_id as string,
        waterway_id:        p.waterway_id as WaterwayId,
        km_start:           p.km_start as number,
        km_end:             p.km_end as number,
        navigability_class: p.navigability_class as NavigabilityClass,
        depth_m:            p.depth_m as number,
        width_m:            p.width_m as number,
        confidence:         p.confidence as number,
        velocity_ms:        p.velocity_ms as number | undefined,
        state:              (p as any).state as string | undefined,
      };
    });
  }, [geojsonData]);

  const selectedSegment = useMemo(
    () => segments.find((s) => s.segment_id === selectedSegmentId) ?? null,
    [segments, selectedSegmentId],
  );

  const alerts    = useMemo(() => getMockAlerts(selectedWaterway), [selectedWaterway]);
  const alertCount = alerts.filter((a) => a.is_active && a.severity === 'CRITICAL').length;

  // ── Handlers ──────────────────────────────────────────────────────────────
  const handleSegmentSelect = useCallback((segId: string) => {
    setSelectedSegment(segId);
    setPanelMode('segments');
  }, [setSelectedSegment]);

  // ── Derived ───────────────────────────────────────────────────────────────
  const isMonsoon = selectedMonth >= 6 && selectedMonth <= 9;

  return (
    <div className="relative flex h-full overflow-hidden">

      {/* ── Full-screen Map ───────────────────────────────────────────────── */}
      <div className="flex-1 relative">
        {/* Floating stats bar */}
        <MapStatsFloater
          navMap={navMap}
          month={selectedMonth}
          year={selectedYear}
          waterway={selectedWaterway}
        />

        {/* The Map */}
        <RiverMap
          className="w-full h-full"
          onSegmentClick={(segId) => {
            handleSegmentSelect(segId);
          }}
        />

        {/* Panel toggle button (when panel is closed) */}
        <AnimatePresence>
          {!panelOpen && (
            <motion.button
              initial={{ opacity: 0, x: 16 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 16 }}
              onClick={() => setPanelOpen(true)}
              className="
                absolute right-4 top-1/2 -translate-y-1/2 z-20
                flex items-center gap-2 px-3 py-2.5 rounded-xl
                bg-slate-900/90 backdrop-blur-sm
                border border-white/[0.12]
                text-[11px] font-semibold text-slate-300 hover:text-white
                shadow-lg transition-all duration-150
              "
              aria-label="Open analysis panel"
            >
              <ChevronLeft size={14} />
              <span>Panel</span>
              {alertCount > 0 && (
                <span className="min-w-[18px] h-4.5 px-1.5 bg-red-500 text-white text-[9px] font-bold rounded-full flex items-center justify-center">
                  {alertCount}
                </span>
              )}
            </motion.button>
          )}
        </AnimatePresence>
      </div>

      {/* ── Right Analysis Panel ──────────────────────────────────────────── */}
      <AnimatePresence mode="wait">
        {panelOpen && (
          <motion.aside
            key="side-panel"
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: 360, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
            className="
              relative flex-shrink-0 flex flex-col
              bg-slate-900/95 backdrop-blur-xl
              border-l border-white/[0.07]
              overflow-hidden z-30
            "
          >
            <div className="flex flex-col h-full w-[360px]">

              {/* Panel header */}
              <div className="flex-shrink-0 flex items-center justify-between px-4 pt-3 pb-2">
                <div className="flex items-center gap-2">
                  <Navigation size={14} className="text-blue-400" />
                  <span className="text-[13px] font-bold text-slate-100">Analysis Panel</span>
                </div>
                <button
                  onClick={() => setPanelOpen(false)}
                  className="w-7 h-7 rounded-lg flex items-center justify-center text-slate-500 hover:text-slate-300 hover:bg-white/[0.07] transition-colors duration-150"
                  aria-label="Close panel"
                >
                  <ChevronRight size={14} />
                </button>
              </div>

              {/* Tab bar */}
              <div className="flex-shrink-0 px-4 pb-2">
                <PanelTabBar
                  active={panelMode}
                  onChange={setPanelMode}
                  alertCount={alertCount}
                />
              </div>

              {/* Divider */}
              <div className="flex-shrink-0 h-px bg-white/[0.06] mx-4" />

              {/* Selected segment detail (shown above content when applicable) */}
              <AnimatePresence>
                {selectedSegment && panelMode === 'segments' && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
                    className="flex-shrink-0 px-4 pt-3 overflow-hidden"
                  >
                    <SegmentDetailCard
                      segment={selectedSegment}
                      onClose={() => setSelectedSegment(null)}
                      month={selectedMonth}
                      year={selectedYear}
                    />
                    <div className="h-3" />
                    <div className="h-px bg-white/[0.05]" />
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Scrollable panel content */}
              <div className="flex-1 min-h-0 overflow-y-auto thin-scrollbar px-4 py-3">

                {/* ── Segments Tab ──────────────────────────────────────── */}
                {panelMode === 'segments' && (
                  <motion.div
                    key="segments"
                    initial={{ opacity: 0, x: 12 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -12 }}
                    transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
                    className="h-full flex flex-col"
                    style={{ minHeight: selectedSegment ? 200 : 400 }}
                  >
                    <SegmentListPanel
                      segments={segments}
                      selectedId={selectedSegmentId}
                      onSelect={(id) => setSelectedSegment(id)}
                      filterClass={filterClass}
                      onFilterChange={setFilterClass}
                      searchQuery={searchQuery}
                      onSearchChange={setSearchQuery}
                    />
                  </motion.div>
                )}

                {/* ── Profile Tab ───────────────────────────────────────── */}
                {panelMode === 'profile' && (
                  <motion.div
                    key="profile"
                    initial={{ opacity: 0, x: 12 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -12 }}
                    transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
                  >
                    {/* Heading */}
                    <div className="mb-3">
                      <h3 className="text-[13px] font-bold text-slate-100">Depth Profile</h3>
                      <p className="text-[11px] text-slate-500 mt-0.5">
                        {selectedWaterway} ·{' '}
                        {MONTH_NAMES[selectedMonth - 1]} {selectedYear}
                        {isMonsoon && (
                          <span className="ml-1.5 text-sky-400 font-semibold">🌧 Monsoon</span>
                        )}
                      </p>
                    </div>

                    {/* Quick stats */}
                    {(() => {
                      const profile = getMockDepthProfile(selectedWaterway, selectedMonth);
                      if (!profile) return null;
                      const depths  = profile.points.map((p) => p.depth_m);
                      const minD    = Math.min(...depths);
                      const maxD    = Math.max(...depths);
                      const meanD   = depths.reduce((a, b) => a + b, 0) / depths.length;

                      return (
                        <div className="grid grid-cols-3 gap-2 mb-4">
                          {[
                            { label: 'Min', value: `${minD.toFixed(2)}m`, color: minD >= 3 ? '#22c55e' : minD >= 2 ? '#f59e0b' : '#ef4444' },
                            { label: 'Mean', value: `${meanD.toFixed(2)}m`, color: '#3b82f6' },
                            { label: 'Max', value: `${maxD.toFixed(2)}m`, color: '#38bdf8' },
                          ].map((m) => (
                            <div
                              key={m.label}
                              className="flex flex-col gap-0.5 px-2.5 py-2 rounded-xl bg-white/[0.03] border border-white/[0.06] text-center"
                            >
                              <span className="text-[9px] font-semibold uppercase tracking-wider text-slate-500">{m.label}</span>
                              <span className="text-[14px] font-extrabold tabular-nums leading-tight" style={{ color: m.color }}>
                                {m.value}
                              </span>
                            </div>
                          ))}
                        </div>
                      );
                    })()}

                    <DepthProfileChart
                      waterwayId={selectedWaterway}
                      month={selectedMonth}
                      year={selectedYear}
                      height={260}
                      showMonthNav={true}
                      showStats={false}
                      compact={false}
                    />
                  </motion.div>
                )}

                {/* ── Alerts Tab ────────────────────────────────────────── */}
                {panelMode === 'alerts' && (
                  <motion.div
                    key="alerts"
                    initial={{ opacity: 0, x: 12 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -12 }}
                    transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
                  >
                    <div className="mb-3">
                      <h3 className="text-[13px] font-bold text-slate-100">Risk Alerts</h3>
                      <p className="text-[11px] text-slate-500 mt-0.5">
                        {selectedWaterway} · Click alert to highlight on map
                      </p>
                    </div>
                    <AlertQuickPanel waterway={selectedWaterway} />
                  </motion.div>
                )}

                {/* ── Settings Tab ──────────────────────────────────────── */}
                {panelMode === 'settings' && (
                  <motion.div
                    key="settings"
                    initial={{ opacity: 0, x: 12 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -12 }}
                    transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
                  >
                    <div className="mb-4">
                      <h3 className="text-[13px] font-bold text-slate-100">Map Layers & Style</h3>
                      <p className="text-[11px] text-slate-500 mt-0.5">
                        Customise what's shown on the river map
                      </p>
                    </div>
                    <LayerSettingsPanel />
                  </motion.div>
                )}
              </div>

              {/* Panel footer */}
              <div className="flex-shrink-0 border-t border-white/[0.06] px-4 py-2.5">
                <div className="flex items-center justify-between text-[10px] text-slate-600">
                  <div className="flex items-center gap-1.5">
                    <Satellite size={10} />
                    <span>Sentinel-2 · 10m · 5-day</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className="relative flex h-1.5 w-1.5">
                      <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                      <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-emerald-500" />
                    </span>
                    <span>HydroFormer live</span>
                  </div>
                </div>
              </div>
            </div>
          </motion.aside>
        )}
      </AnimatePresence>
    </div>
  );
}
