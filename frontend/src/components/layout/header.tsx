// ============================================================
// AIDSTL — Inland Waterway Navigability Prediction System
// Top Header Component
// ============================================================

'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Bell,
  Download,
  RefreshCw,
  Satellite,
  Signal,
  ChevronDown,
  AlertTriangle,
  CheckCircle2,
  Clock,
  FileText,
  Share2,
  Zap,
} from 'lucide-react';
import { useAppStore } from '@/store/app-store';
import { getMockAlerts } from '@/lib/mock-data';

// ─── Constants ────────────────────────────────────────────────────────────────

const MONTH_NAMES = [
  'January', 'February', 'March', 'April',
  'May', 'June', 'July', 'August',
  'September', 'October', 'November', 'December',
] as const;

const WATERWAY_META: Record<
  string,
  { fullName: string; river: string; stretch: string; color: string; segments: number }
> = {
  'NW-1': {
    fullName: 'National Waterway 1',
    river: 'Ganga',
    stretch: 'Varanasi → Haldia',
    color: '#3b82f6',
    segments: 278,
  },
  'NW-2': {
    fullName: 'National Waterway 2',
    river: 'Brahmaputra',
    stretch: 'Dhubri → Sadiya',
    color: '#8b5cf6',
    segments: 178,
  },
};

// ─── Helper: format relative time ─────────────────────────────────────────────

function formatRelativeTime(date: Date): string {
  const diffMs = Date.now() - date.getTime();
  const diffMin = Math.floor(diffMs / 60_000);
  if (diffMin < 1) return 'Just now';
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return `${diffHr}h ago`;
  return date.toLocaleDateString('en-IN', { day: 'numeric', month: 'short' });
}

// ─── Status Indicator ─────────────────────────────────────────────────────────

function StatusDot({ status }: { status: 'online' | 'degraded' | 'offline' }) {
  const colors = {
    online:   { dot: 'bg-emerald-400', ping: 'bg-emerald-400', label: 'Live Data',  text: 'text-emerald-400' },
    degraded: { dot: 'bg-amber-400',   ping: 'bg-amber-400',   label: 'Degraded',   text: 'text-amber-400'   },
    offline:  { dot: 'bg-red-400',     ping: 'bg-red-400',     label: 'Offline',    text: 'text-red-400'     },
  };
  const c = colors[status];

  return (
    <div className="flex items-center gap-1.5">
      <span className="relative flex h-2 w-2 flex-shrink-0">
        <span className={`animate-ping absolute inline-flex h-full w-full rounded-full ${c.ping} opacity-60`} />
        <span className={`relative inline-flex rounded-full h-2 w-2 ${c.dot}`} />
      </span>
      <span className={`text-[11px] font-semibold ${c.text}`}>{c.label}</span>
    </div>
  );
}

// ─── Alert Bell ───────────────────────────────────────────────────────────────

function AlertBell() {
  const selectedWaterway = useAppStore((s) => s.selectedWaterway);
  const setAlertsPanelOpen = useAppStore((s) => s.setAlertsPanelOpen);
  const alertsPanelOpen = useAppStore((s) => s.alertsPanelOpen);
  const [isAnimating, setIsAnimating] = useState(false);

  const alerts = getMockAlerts(selectedWaterway);
  const criticalCount = alerts.filter((a) => a.severity === 'CRITICAL').length;
  const totalCount = alerts.filter((a) => a.is_active).length;

  // Trigger bell ring animation on mount and when waterway changes
  useEffect(() => {
    setIsAnimating(true);
    const t = setTimeout(() => setIsAnimating(false), 800);
    return () => clearTimeout(t);
  }, [selectedWaterway]);

  return (
    <div className="relative">
      <motion.button
        onClick={() => setAlertsPanelOpen(!alertsPanelOpen)}
        className={`
          relative flex items-center justify-center w-9 h-9 rounded-xl
          border transition-all duration-200
          ${alertsPanelOpen
            ? 'bg-blue-500/15 border-blue-500/40 text-blue-400'
            : 'bg-white/[0.04] border-white/[0.08] text-slate-400 hover:text-slate-200 hover:bg-white/[0.07] hover:border-white/[0.12]'
          }
        `}
        whileTap={{ scale: 0.93 }}
        animate={
          isAnimating
            ? { rotate: [0, -18, 18, -12, 12, -6, 6, 0] }
            : { rotate: 0 }
        }
        transition={{ duration: 0.6, ease: 'easeInOut' }}
        aria-label={`${totalCount} active alerts`}
      >
        <Bell size={16} strokeWidth={alertsPanelOpen ? 2.5 : 2} />

        {/* Count badge */}
        {totalCount > 0 && (
          <motion.span
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            className={`
              absolute -top-1.5 -right-1.5 min-w-[18px] h-[18px] px-1
              flex items-center justify-center rounded-full
              text-[9px] font-bold text-white leading-none
              ${criticalCount > 0 ? 'bg-red-500' : 'bg-amber-500'}
            `}
          >
            {totalCount > 9 ? '9+' : totalCount}
          </motion.span>
        )}
      </motion.button>
    </div>
  );
}

// ─── Export / Report Dropdown ─────────────────────────────────────────────────

function ExportButton() {
  const [open, setOpen] = useState(false);
  const [exporting, setExporting] = useState<string | null>(null);

  const handleExport = async (format: string) => {
    setExporting(format);
    setOpen(false);
    // Simulate export delay
    await new Promise((r) => setTimeout(r, 1500));
    setExporting(null);
  };

  const OPTIONS = [
    { id: 'pdf',     icon: FileText, label: 'Export as PDF',     sub: 'Full navigability report' },
    { id: 'geojson', icon: Share2,   label: 'Download GeoJSON',  sub: 'River segment geometries' },
    { id: 'csv',     icon: Download, label: 'Export CSV',        sub: 'Predictions & features'   },
  ];

  return (
    <div className="relative">
      <motion.button
        onClick={() => setOpen((v) => !v)}
        whileTap={{ scale: 0.95 }}
        className={`
          flex items-center gap-2 px-3.5 py-2 rounded-xl text-sm font-semibold
          border transition-all duration-200
          ${exporting
            ? 'bg-blue-500/10 border-blue-500/30 text-blue-400 cursor-wait'
            : 'bg-blue-600 hover:bg-blue-500 border-blue-500 text-white shadow-lg shadow-blue-500/20'
          }
        `}
        disabled={!!exporting}
      >
        {exporting ? (
          <>
            <RefreshCw size={14} className="animate-spin" />
            <span className="hidden sm:inline">Exporting…</span>
          </>
        ) : (
          <>
            <Download size={14} />
            <span className="hidden sm:inline">Export</span>
            <ChevronDown
              size={12}
              className={`transition-transform duration-200 ${open ? 'rotate-180' : ''}`}
            />
          </>
        )}
      </motion.button>

      <AnimatePresence>
        {open && (
          <>
            {/* Backdrop */}
            <div
              className="fixed inset-0 z-50"
              onClick={() => setOpen(false)}
            />

            {/* Dropdown */}
            <motion.div
              initial={{ opacity: 0, y: -8, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -8, scale: 0.95 }}
              transition={{ duration: 0.15, ease: [0.16, 1, 0.3, 1] }}
              className="
                absolute right-0 top-full mt-2 w-56 z-50
                bg-slate-900/95 backdrop-blur-xl
                border border-white/10 rounded-xl
                shadow-2xl shadow-black/50
                overflow-hidden
              "
            >
              <div className="p-1">
                {OPTIONS.map((opt) => (
                  <button
                    key={opt.id}
                    onClick={() => handleExport(opt.id)}
                    className="
                      w-full flex items-center gap-3 px-3 py-2.5 rounded-lg
                      text-slate-300 hover:text-white hover:bg-white/6
                      transition-colors duration-150 text-left
                    "
                  >
                    <opt.icon size={15} className="text-slate-500 flex-shrink-0" />
                    <div>
                      <div className="text-[13px] font-medium">{opt.label}</div>
                      <div className="text-[11px] text-slate-500">{opt.sub}</div>
                    </div>
                  </button>
                ))}
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  );
}

// ─── Satellite Data Pill ──────────────────────────────────────────────────────

function SatellitePill({ month, year }: { month: number; year: number }) {
  return (
    <div className="hidden md:flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-white/[0.04] border border-white/[0.06]">
      <Satellite size={12} className="text-slate-500" />
      <span className="text-[11px] text-slate-500 font-medium">
        Sentinel-2
      </span>
      <span className="text-[11px] text-slate-600">·</span>
      <span className="text-[11px] text-slate-400 font-semibold tabular-nums">
        {MONTH_NAMES[month - 1].slice(0, 3)} {year}
      </span>
    </div>
  );
}

// ─── Model Confidence Badge ───────────────────────────────────────────────────

function ModelConfidenceBadge() {
  const r2 = 0.918;

  return (
    <div className="hidden lg:flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-white/[0.04] border border-white/[0.06]">
      <Zap size={12} className="text-blue-400" />
      <span className="text-[11px] text-slate-500 font-medium">HydroFormer</span>
      <span className="text-[11px] text-slate-600">·</span>
      <span className="text-[11px] text-blue-400 font-bold tabular-nums">
        R² {r2.toFixed(3)}
      </span>
    </div>
  );
}

// ─── Breadcrumb / Waterway Title ──────────────────────────────────────────────

function WaterwayTitle() {
  const selectedWaterway = useAppStore((s) => s.selectedWaterway);
  const selectedMonth = useAppStore((s) => s.selectedMonth);
  const selectedYear = useAppStore((s) => s.selectedYear);
  const meta = WATERWAY_META[selectedWaterway];

  return (
    <div className="flex flex-col justify-center min-w-0">
      <div className="flex items-center gap-2 flex-wrap">
        <AnimatePresence mode="wait">
          <motion.h1
            key={selectedWaterway}
            initial={{ opacity: 0, y: -6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 6 }}
            transition={{ duration: 0.2 }}
            className="text-base font-bold text-slate-100 leading-tight tracking-tight"
          >
            {meta.fullName}
          </motion.h1>
        </AnimatePresence>

        {/* Status dot */}
        <StatusDot status="online" />
      </div>

      <div className="flex items-center gap-2 mt-0.5 flex-wrap">
        {/* River & stretch */}
        <AnimatePresence mode="wait">
          <motion.span
            key={selectedWaterway + '-sub'}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="text-[12px] text-slate-500 font-medium"
          >
            {meta.river} · {meta.stretch}
          </motion.span>
        </AnimatePresence>

        <span className="text-slate-700 hidden sm:inline">·</span>

        {/* Segment count */}
        <span className="hidden sm:inline text-[12px] text-slate-600">
          {meta.segments.toLocaleString('en-IN')} segments
        </span>

        {/* Last updated */}
        <span className="text-slate-700 hidden md:inline">·</span>
        <div className="hidden md:flex items-center gap-1 text-[11px] text-slate-600">
          <Clock size={10} />
          <span>
            Updated {formatRelativeTime(new Date(Date.now() - 23 * 60_000))}
          </span>
        </div>
      </div>
    </div>
  );
}

// ─── Data Quality Indicators ──────────────────────────────────────────────────

function DataQualityBar() {
  const selectedMonth = useAppStore((s) => s.selectedMonth);

  // Simulate cloud cover — monsoon months (6-9) have higher cloud cover
  const isMonsoon = selectedMonth >= 6 && selectedMonth <= 9;
  const cloudCover = isMonsoon ? 42 : 12;
  const coverage = 100 - cloudCover;

  return (
    <div className="hidden xl:flex items-center gap-3 px-3 py-1.5 rounded-lg bg-white/[0.03] border border-white/[0.05]">
      <div className="flex items-center gap-1.5">
        <Signal size={11} className="text-slate-500" />
        <span className="text-[10px] text-slate-500 font-medium whitespace-nowrap">
          Cloud-free coverage
        </span>
      </div>

      {/* Mini progress bar */}
      <div className="w-20 h-1.5 rounded-full bg-white/[0.06] overflow-hidden">
        <motion.div
          className={`h-full rounded-full ${
            coverage >= 80 ? 'bg-emerald-500' :
            coverage >= 60 ? 'bg-amber-500' : 'bg-red-500'
          }`}
          initial={{ width: 0 }}
          animate={{ width: `${coverage}%` }}
          transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
        />
      </div>

      <span
        className={`text-[11px] font-bold tabular-nums ${
          coverage >= 80 ? 'text-emerald-400' :
          coverage >= 60 ? 'text-amber-400' : 'text-red-400'
        }`}
      >
        {coverage}%
      </span>
    </div>
  );
}

// ─── Refresh Button ───────────────────────────────────────────────────────────

function RefreshButton() {
  const [refreshing, setRefreshing] = useState(false);
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date());

  const handleRefresh = async () => {
    if (refreshing) return;
    setRefreshing(true);
    await new Promise((r) => setTimeout(r, 1200));
    setLastRefresh(new Date());
    setRefreshing(false);
  };

  return (
    <motion.button
      onClick={handleRefresh}
      whileTap={{ scale: 0.93 }}
      className="
        flex items-center justify-center w-9 h-9 rounded-xl
        bg-white/[0.04] border border-white/[0.08]
        text-slate-400 hover:text-slate-200 hover:bg-white/[0.07] hover:border-white/[0.12]
        transition-all duration-200
        disabled:opacity-50 disabled:cursor-not-allowed
      "
      disabled={refreshing}
      aria-label="Refresh data"
      title={`Last refreshed: ${formatRelativeTime(lastRefresh)}`}
    >
      <motion.div
        animate={refreshing ? { rotate: 360 } : { rotate: 0 }}
        transition={
          refreshing
            ? { duration: 0.8, repeat: Infinity, ease: 'linear' }
            : { duration: 0.3 }
        }
      >
        <RefreshCw size={15} />
      </motion.div>
    </motion.button>
  );
}

// ─── Alerts Quick Summary ─────────────────────────────────────────────────────

function AlertsQuickSummary() {
  const selectedWaterway = useAppStore((s) => s.selectedWaterway);
  const setAlertsPanelOpen = useAppStore((s) => s.setAlertsPanelOpen);

  const alerts = getMockAlerts(selectedWaterway);
  const criticalAlerts = alerts.filter((a) => a.severity === 'CRITICAL' && a.is_active);

  if (criticalAlerts.length === 0) {
    return (
      <div className="hidden lg:flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-emerald-500/8 border border-emerald-500/15">
        <CheckCircle2 size={12} className="text-emerald-400" />
        <span className="text-[11px] font-semibold text-emerald-400">All Clear</span>
      </div>
    );
  }

  return (
    <motion.button
      onClick={() => setAlertsPanelOpen(true)}
      className="
        hidden lg:flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg
        bg-red-500/10 border border-red-500/25
        hover:bg-red-500/15 hover:border-red-500/40
        transition-all duration-200 cursor-pointer
      "
      animate={{ scale: [1, 1.02, 1] }}
      transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
    >
      <AlertTriangle size={12} className="text-red-400" />
      <span className="text-[11px] font-bold text-red-400">
        {criticalAlerts.length} Critical
      </span>
    </motion.button>
  );
}

// ─── Main Header ──────────────────────────────────────────────────────────────

export function Header() {
  const selectedMonth    = useAppStore((s) => s.selectedMonth);
  const selectedYear     = useAppStore((s) => s.selectedYear);
  const selectedWaterway = useAppStore((s) => s.selectedWaterway);

  return (
    <header
      className="
        relative flex items-center h-16 flex-shrink-0 px-5
        bg-slate-900/60 backdrop-blur-xl
        border-b border-white/[0.06]
        z-50
      "
    >
      {/* Left — Waterway info */}
      <div className="flex items-center gap-4 min-w-0 flex-1">
        <WaterwayTitle />
      </div>

      {/* Centre — Data quality & model info pills */}
      <div className="hidden md:flex items-center gap-2 mx-4">
        <SatellitePill month={selectedMonth} year={selectedYear} />
        <ModelConfidenceBadge />
        <DataQualityBar />
      </div>

      {/* Right — Actions */}
      <div className="flex items-center gap-2 flex-shrink-0">
        {/* Critical alerts quick summary */}
        <AlertsQuickSummary />

        {/* Refresh */}
        <RefreshButton />

        {/* Alert bell */}
        <AlertBell />

        {/* Divider */}
        <div className="hidden sm:block w-px h-6 bg-white/[0.08] mx-1" />

        {/* Export */}
        <ExportButton />
      </div>

      {/* Bottom highlight line */}
      <div
        className="absolute bottom-0 left-0 right-0 h-px pointer-events-none"
        style={{
          background: `linear-gradient(90deg,
            transparent 0%,
            ${WATERWAY_META[selectedWaterway]?.color ?? '#3b82f6'}30 30%,
            ${WATERWAY_META[selectedWaterway]?.color ?? '#3b82f6'}50 50%,
            ${WATERWAY_META[selectedWaterway]?.color ?? '#3b82f6'}30 70%,
            transparent 100%
          )`,
        }}
      />
    </header>
  );
}
