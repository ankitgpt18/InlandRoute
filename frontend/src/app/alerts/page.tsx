// ============================================================
// InlandRoute - Inland Waterway Navigability Prediction System
// Alerts Page — Full risk alerts dashboard
// ============================================================

'use client';

import React, { useMemo, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  AlertTriangle,
  AlertCircle,
  Info,
  Bell,
  BellOff,
  Filter,
  RefreshCw,
  MapPin,
  Clock,
  TrendingDown,
  Shield,
  Zap,
  ChevronDown,
  ChevronUp,
  ArrowRight,
  Eye,
  X,
  Download,
  Activity,
  Droplets,
  Wind,
  Anchor,
  BarChart2,
  ShieldAlert,
  CheckCircle2,
  SlidersHorizontal,
  Waves,
  ExternalLink,
} from 'lucide-react';
import { useAppStore } from '@/store/app-store';
import { getMockAlerts } from '@/lib/mock-data';
import { NavigabilityBadge } from '@/components/ui/navigability-badge';
import { StatCard, StatCardGrid } from '@/components/ui/stat-card';
import { cn } from '@/lib/utils';
import type { WaterwayId, AlertSeverity, AlertType } from '@/types';

// ─── Types ────────────────────────────────────────────────────────────────────

interface Alert {
  alert_id:            string;
  waterway_id:         WaterwayId;
  segment_id:          string;
  km_start:            number;
  km_end:              number;
  severity:            AlertSeverity;
  alert_type:          AlertType;
  title:               string;
  description:         string;
  predicted_value:     number;
  threshold_value:     number;
  unit:                string;
  risk_score:          number;
  valid_from:          string;
  valid_until:         string;
  created_at:          string;
  is_active:           boolean;
  recommended_actions: string[];
  affected_vessels?:   string[];
}

// ─── Config Maps ──────────────────────────────────────────────────────────────

const SEVERITY_CONFIG: Record<AlertSeverity, {
  label:    string;
  color:    string;
  bg:       string;
  border:   string;
  glow:     string;
  icon:     React.ElementType;
  dotColor: string;
  pulse:    boolean;
  order:    number;
}> = {
  CRITICAL: {
    label:    'Critical',
    color:    'text-slate-400',
    bg:       'bg-red-500/10',
    border:   'border-red-500/30',
    glow:     'shadow-[0_0_24px_rgba(239,68,68,0.15)]',
    icon:     AlertCircle,
    dotColor: 'bg-red-400',
    pulse:    true,
    order:    0,
  },
  WARNING: {
    label:    'Warning',
    color:    'text-slate-400',
    bg:       'bg-amber-500/10',
    border:   'border-amber-500/25',
    glow:     '',
    icon:     AlertTriangle,
    dotColor: 'bg-amber-400',
    pulse:    false,
    order:    1,
  },
  INFO: {
    label:    'Info',
    color:    'text-slate-400',
    bg:       'bg-sky-500/8',
    border:   'border-sky-500/20',
    glow:     '',
    icon:     Info,
    dotColor: 'bg-sky-400',
    pulse:    false,
    order:    2,
  },
};

const ALERT_TYPE_CONFIG: Partial<Record<AlertType, { label: string; icon: React.ElementType; color: string }>> = {
  DEPTH_CRITICAL:    { label: 'Depth Critical',    icon: TrendingDown,  color: '#ef4444' },
  DEPTH_WARNING:     { label: 'Depth Warning',     icon: BarChart2,     color: '#f59e0b' },
  WIDTH_RESTRICTION: { label: 'Width Restriction', icon: Activity,      color: '#f59e0b' },
  VELOCITY_HIGH:     { label: 'High Velocity',     icon: Wind,          color: '#f59e0b' },
  SEASONAL_CLOSURE:  { label: 'Seasonal Closure',  icon: Anchor,        color: '#94a3b8' },
  OBSTACLE_DETECTED: { label: 'Obstacle',          icon: ShieldAlert,   color: '#ef4444' },
  FLOOD_RISK:        { label: 'Flood Risk',        icon: Droplets,      color: '#3b82f6' },
  DROUGHT_RISK:      { label: 'Drought Risk',      icon: Waves,         color: '#f59e0b' },
};

// ─── Helpers ──────────────────────────────────────────────────────────────────

function formatRelativeTime(dateStr: string): string {
  const date    = new Date(dateStr);
  const diffMs  = Date.now() - date.getTime();
  const diffMin = Math.floor(diffMs / 60_000);
  if (diffMin < 1)   return 'Just now';
  if (diffMin < 60)  return `${diffMin}m ago`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24)   return `${diffHr}h ago`;
  const diffDay = Math.floor(diffHr / 24);
  if (diffDay < 7)   return `${diffDay}d ago`;
  return date.toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' });
}

function formatDate(dateStr: string): string {
  return new Date(dateStr).toLocaleDateString('en-IN', {
    day: 'numeric', month: 'short', year: 'numeric',
  });
}

// ─── Pulsing Dot ──────────────────────────────────────────────────────────────

function PulsingDot({ severity, size = 8 }: { severity: AlertSeverity; size?: number }) {
  const cfg = SEVERITY_CONFIG[severity];
  return (
    <span className="relative inline-flex flex-shrink-0" style={{ width: size, height: size }}>
      {cfg.pulse && (
        <span
          className={cn('animate-ping absolute inline-flex h-full w-full rounded-full opacity-60', cfg.dotColor)}
        />
      )}
      <span className={cn('relative inline-flex rounded-full h-full w-full', cfg.dotColor)} />
    </span>
  );
}

// ─── Risk Score Bar ───────────────────────────────────────────────────────────

function RiskBar({ score, showLabel = true }: { score: number; showLabel?: boolean }) {
  const pct   = Math.round(score * 100);
  const color = pct >= 70 ? '#ef4444' : pct >= 40 ? '#f59e0b' : '#22c55e';
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 rounded-full bg-white/[0.06] overflow-hidden">
        <motion.div
          className="h-full rounded-full"
          style={{ backgroundColor: color }}
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
        />
      </div>
      {showLabel && (
        <span className="text-[10px] font-bold tabular-nums w-7 text-right" style={{ color }}>
          {pct}
        </span>
      )}
    </div>
  );
}

// ─── Alert Card (full expanded version) ──────────────────────────────────────

function AlertCard({
  alert,
  index,
  onViewOnMap,
}: {
  alert:       Alert;
  index:       number;
  onViewOnMap: (segId: string) => void;
}) {
  const [expanded, setExpanded] = useState(false);

  const sevCfg  = SEVERITY_CONFIG[alert.severity];
  const typeCfg = ALERT_TYPE_CONFIG[alert.alert_type];
  const SevIcon  = sevCfg.icon;
  const TypeIcon = typeCfg?.icon ?? AlertTriangle;

  const deficit     = alert.threshold_value - alert.predicted_value;
  const isCritical  = alert.severity === 'CRITICAL';
  const accentColor = isCritical ? '#ef4444' : alert.severity === 'WARNING' ? '#f59e0b' : '#0ea5e9';

  return (
    <motion.article
      layout
      initial={{ opacity: 0, y: 16, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: -8, scale: 0.98 }}
      transition={{ duration: 0.35, delay: index * 0.05, ease: [0.16, 1, 0.3, 1] }}
      className={cn(
        'relative overflow-hidden rounded-2xl border transition-all duration-300',
        sevCfg.bg,
        sevCfg.border,
        isCritical && sevCfg.glow,
      )}
    >
      {/* Critical pulsing left accent */}
      {isCritical && (
        <motion.div
          className="absolute left-0 top-0 bottom-0 w-1 rounded-r-full"
          style={{ background: `linear-gradient(180deg, #ef4444 0%, #dc2626 100%)` }}
          animate={{ opacity: [0.6, 1, 0.6] }}
          transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
        />
      )}

      {/* Top accent line */}
      <div
        className="absolute top-0 left-8 right-8 h-px opacity-60"
        style={{ backgroundColor: accentColor }}
      />

      <div className="px-5 pt-4 pb-3">
        {/* ── Row 1: Severity + Type + Time ──────────────────────────────── */}
        <div className="flex items-center justify-between gap-3 mb-3 flex-wrap">
          <div className="flex items-center gap-2 flex-wrap">
            {/* Severity badge */}
            <span
              className={cn(
                'inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-bold uppercase tracking-wider border',
                sevCfg.bg,
                sevCfg.border,
                sevCfg.color,
              )}
            >
              <PulsingDot severity={alert.severity} size={6} />
              {sevCfg.label}
            </span>

            {/* Alert type pill */}
            {typeCfg && (
              <span
                className="inline-flex items-center gap-1 text-[9px] font-semibold uppercase tracking-wider text-slate-500"
              >
                <TypeIcon size={9} />
                {typeCfg.label}
              </span>
            )}

            {/* Active indicator */}
            {alert.is_active && (
              <span className="inline-flex items-center gap-1 text-[9px] font-bold text-slate-400 uppercase tracking-wider">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                Active
              </span>
            )}
          </div>

          {/* Timestamp */}
          <div className="flex items-center gap-1.5 text-[10px] text-slate-500 font-medium">
            <Clock size={9} />
            <span>{formatRelativeTime(alert.created_at)}</span>
          </div>
        </div>

        {/* ── Row 2: Icon + Title + Location ─────────────────────────────── */}
        <div className="flex items-start gap-3 mb-4">
          {/* Icon */}
          <div
            className="flex-shrink-0 w-10 h-10 rounded-xl flex items-center justify-center mt-0.5"
            style={{ background: `${accentColor}18`, border: `1px solid ${accentColor}35` }}
          >
            <TypeIcon size={18} style={{ color: accentColor }} />
          </div>

          <div className="flex-1 min-w-0">
            <h3 className={cn('text-[14px] font-bold leading-tight mb-1', sevCfg.color)}>
              {alert.title}
            </h3>
            <div className="flex items-center gap-2 flex-wrap text-[11px]">
              <div className="flex items-center gap-1 text-slate-400 font-semibold">
                <MapPin size={10} className="text-slate-500" />
                {alert.waterway_id}
              </div>
              <span className="text-slate-700">·</span>
              <span className="text-slate-500 tabular-nums font-medium">
                km {alert.km_start.toFixed(0)}–{alert.km_end.toFixed(0)}
              </span>
              <span className="text-slate-700">·</span>
              <span className="text-slate-400 font-medium">
                Segment {alert.segment_id.split('-').pop()}
              </span>
            </div>
          </div>
        </div>

        {/* ── Row 3: Predicted vs Threshold metrics ──────────────────────── */}
        <div className="grid grid-cols-3 gap-2 mb-4">
          {/* Predicted value */}
          <div
            className="flex flex-col gap-0.5 px-3 py-2.5 rounded-xl border"
            style={{
              background:   `${accentColor}10`,
              borderColor:  `${accentColor}30`,
            }}
          >
            <span className="text-[9px] font-semibold uppercase tracking-wider text-slate-500">
              Predicted
            </span>
            <span
              className="text-xl font-extrabold tabular-nums leading-tight text-slate-800"
            >
              {alert.predicted_value.toFixed(2)}
              <span className="text-[11px] font-semibold ml-0.5 opacity-70">
                {alert.unit}
              </span>
            </span>
          </div>

          {/* Threshold */}
          <div className="flex flex-col gap-0.5 px-3 py-2.5 rounded-xl border border-emerald-500/20 bg-emerald-500/8">
            <span className="text-[9px] font-semibold uppercase tracking-wider text-slate-500">
              Threshold
            </span>
            <span className="text-xl font-extrabold tabular-nums leading-tight text-slate-400">
              {alert.threshold_value.toFixed(2)}
              <span className="text-[11px] font-semibold ml-0.5 opacity-70">
                {alert.unit}
              </span>
            </span>
          </div>

          {/* Deficit */}
          <div className="flex flex-col gap-0.5 px-3 py-2.5 rounded-xl border border-slate-900/[0.07] bg-white/[0.03]">
            <span className="text-[9px] font-semibold uppercase tracking-wider text-slate-500">
              Deficit
            </span>
            <span className="text-xl font-extrabold tabular-nums leading-tight text-slate-400">
              {Math.abs(deficit).toFixed(2)}
              <span className="text-[11px] font-semibold ml-0.5 opacity-70">
                {alert.unit}
              </span>
            </span>
          </div>
        </div>

        {/* ── Risk Score ─────────────────────────────────────────────────── */}
        <div className="mb-4">
          <div className="flex items-center justify-between mb-1.5">
            <span className="text-[9px] font-semibold uppercase tracking-wider text-slate-500">
              Risk Score
            </span>
            <span
              className="text-[11px] font-bold tabular-nums text-slate-800"
            >
              {(alert.risk_score * 100).toFixed(0)} / 100
            </span>
          </div>
          <RiskBar score={alert.risk_score} showLabel={false} />
        </div>

        {/* ── Description ────────────────────────────────────────────────── */}
        <p className={cn('text-[12px] text-slate-400 leading-relaxed', !expanded && 'line-clamp-2')}>
          {alert.description}
        </p>

        {/* ── Expanded section ────────────────────────────────────────────── */}
        <AnimatePresence>
          {expanded && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
              className="overflow-hidden"
            >
              <div className="mt-4 pt-4 border-t border-slate-900/[0.06] space-y-4">

                {/* Recommended actions */}
                {alert.recommended_actions?.length > 0 && (
                  <div>
                    <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-2.5">
                      Recommended Actions
                    </div>
                    <div className="space-y-2">
                      {alert.recommended_actions.map((action, i) => (
                        <motion.div
                          key={i}
                          initial={{ opacity: 0, x: -8 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: i * 0.06, duration: 0.2 }}
                          className="flex items-start gap-2.5"
                        >
                          <span
                            className="flex-shrink-0 w-5 h-5 rounded-full flex items-center justify-center text-[9px] font-bold mt-0.5"
                            style={{
                              background: `${accentColor}18`,
                              border:     `1px solid ${accentColor}35`,
                              color:       accentColor,
                            }}
                          >
                            {i + 1}
                          </span>
                          <span className="text-[12px] text-slate-400 leading-snug">{action}</span>
                        </motion.div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Affected vessels */}
                {alert.affected_vessels && alert.affected_vessels.length > 0 && (
                  <div>
                    <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-2">
                      Affected Vessel Types
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {alert.affected_vessels.map((v, i) => (
                        <span
                          key={i}
                          className="text-[11px] font-medium px-2.5 py-1 rounded-full bg-white/[0.05] border border-slate-900/[0.08] text-slate-400"
                        >
                          {v}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Alert metadata */}
                <div className="grid grid-cols-2 gap-3 pt-2 border-t border-slate-900/[0.05]">
                  <div>
                    <div className="text-[9px] font-semibold uppercase tracking-wider text-slate-400 mb-0.5">
                      Valid From
                    </div>
                    <div className="text-[11px] font-medium text-slate-400">{formatDate(alert.valid_from)}</div>
                  </div>
                  <div>
                    <div className="text-[9px] font-semibold uppercase tracking-wider text-slate-400 mb-0.5">
                      Valid Until
                    </div>
                    <div className="text-[11px] font-medium text-slate-400">{formatDate(alert.valid_until)}</div>
                  </div>
                  <div>
                    <div className="text-[9px] font-semibold uppercase tracking-wider text-slate-400 mb-0.5">
                      Alert ID
                    </div>
                    <div className="text-[11px] font-mono text-slate-500">{alert.alert_id.slice(-12)}</div>
                  </div>
                  <div>
                    <div className="text-[9px] font-semibold uppercase tracking-wider text-slate-400 mb-0.5">
                      Created
                    </div>
                    <div className="text-[11px] font-medium text-slate-400">{formatDate(alert.created_at)}</div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── Footer actions ──────────────────────────────────────────────── */}
        <div className="flex items-center justify-between mt-3 pt-3 border-t border-slate-900/[0.05]">
          {/* View on map */}
          <motion.button
            whileHover={{ x: 2 }}
            whileTap={{ scale: 0.96 }}
            onClick={() => onViewOnMap(alert.segment_id)}
            className="flex items-center gap-1.5 text-[12px] font-semibold text-slate-400 hover:text-slate-300 transition-colors duration-150"
          >
            <Eye size={13} />
            View on Map
            <ArrowRight size={11} />
          </motion.button>

          {/* Expand / collapse */}
          <button
            onClick={() => setExpanded((v) => !v)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[11px] font-medium text-slate-500 hover:text-slate-700 hover:bg-white/[0.05] transition-all duration-150"
            aria-expanded={expanded}
          >
            <motion.span
              animate={{ rotate: expanded ? 180 : 0 }}
              transition={{ duration: 0.2 }}
              className="flex items-center"
            >
              <ChevronDown size={13} />
            </motion.span>
            {expanded ? 'Collapse' : 'Expand'}
          </button>
        </div>
      </div>
    </motion.article>
  );
}

// ─── Severity Filter Tab ──────────────────────────────────────────────────────

function SeverityTabs({
  current,
  onChange,
  counts,
}: {
  current:  AlertSeverity | 'ALL';
  onChange: (v: AlertSeverity | 'ALL') => void;
  counts:   Record<AlertSeverity | 'ALL', number>;
}) {
  const tabs: (AlertSeverity | 'ALL')[] = ['ALL', 'CRITICAL', 'WARNING', 'INFO'];

  const colors: Record<AlertSeverity | 'ALL', string> = {
    ALL:      '#94a3b8',
    CRITICAL: '#ef4444',
    WARNING:  '#f59e0b',
    INFO:     '#0ea5e9',
  };

  return (
    <div className="flex items-center gap-1.5 p-1 bg-white/[0.03] rounded-xl border border-slate-900/[0.06]">
      {tabs.map((tab) => {
        const isActive = current === tab;
        const color    = colors[tab];
        const count    = counts[tab];

        return (
          <motion.button
            key={tab}
            onClick={() => onChange(tab)}
            whileTap={{ scale: 0.96 }}
            className={cn(
              'relative flex items-center gap-2 px-3.5 py-2 rounded-lg flex-1 justify-center',
              'text-[11px] font-bold uppercase tracking-wider transition-all duration-200',
              isActive
                ? 'text-slate-900'
                : 'text-slate-500 hover:text-slate-700 hover:bg-white/[0.04]',
            )}
            style={
              isActive
                ? {
                    background:  `${color}20`,
                    border:      `1px solid ${color}40`,
                    color,
                  }
                : {}
            }
          >
            {tab !== 'ALL' && <PulsingDot severity={tab as AlertSeverity} size={6} />}
            <span className="hidden sm:inline">{tab === 'ALL' ? 'All' : SEVERITY_CONFIG[tab as AlertSeverity].label}</span>
            {count > 0 && (
              <span
                className="min-w-[20px] h-5 px-1.5 rounded-full flex items-center justify-center text-[10px] font-bold"
                style={{
                  background: isActive ? `${color}30` : 'rgba(15,23,42,0.06)',
                  color:       isActive ? color : '#64748b',
                }}
              >
                {count}
              </span>
            )}
          </motion.button>
        );
      })}
    </div>
  );
}

// ─── Alert Type Filter ────────────────────────────────────────────────────────

function AlertTypeFilter({
  selected,
  onChange,
  available,
}: {
  selected:  AlertType | 'ALL';
  onChange:  (v: AlertType | 'ALL') => void;
  available: AlertType[];
}) {
  const [open, setOpen] = useState(false);

  const currentLabel = selected === 'ALL'
    ? 'All Types'
    : ALERT_TYPE_CONFIG[selected]?.label ?? selected;

  return (
    <div className="relative">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-2 px-3.5 py-2 rounded-xl border border-slate-900/[0.08] bg-white/[0.04] text-[12px] font-semibold text-slate-700 hover:text-slate-900 hover:bg-white/[0.07] transition-all duration-150"
      >
        <SlidersHorizontal size={13} className="text-slate-500" />
        {currentLabel}
        <ChevronDown
          size={11}
          className={cn('text-slate-500 transition-transform duration-200', open && 'rotate-180')}
        />
      </button>

      <AnimatePresence>
        {open && (
          <>
            <div className="fixed inset-0 z-40" onClick={() => setOpen(false)} />
            <motion.div
              initial={{ opacity: 0, y: -8, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -8, scale: 0.95 }}
              transition={{ duration: 0.15, ease: [0.16, 1, 0.3, 1] }}
              className="absolute left-0 top-full mt-1.5 w-52 z-50 bg-white/98 backdrop-blur-xl border border-slate-900/[0.1] rounded-xl shadow-2xl overflow-hidden p-1"
            >
              <button
                onClick={() => { onChange('ALL'); setOpen(false); }}
                className={cn(
                  'w-full flex items-center gap-2.5 px-3 py-2.5 rounded-lg text-[12px] font-medium transition-colors duration-100',
                  selected === 'ALL' ? 'bg-blue-500/15 text-slate-300' : 'text-slate-400 hover:text-slate-900 hover:bg-white/[0.06]',
                )}
              >
                <Filter size={12} className="flex-shrink-0" />
                All Types
                {selected === 'ALL' && <span className="ml-auto text-[10px] text-slate-400 font-bold">✓</span>}
              </button>
              {available.map((type) => {
                const cfg  = ALERT_TYPE_CONFIG[type];
                if (!cfg) return null;
                const Icon = cfg.icon;
                return (
                  <button
                    key={type}
                    onClick={() => { onChange(type); setOpen(false); }}
                    className={cn(
                      'w-full flex items-center gap-2.5 px-3 py-2.5 rounded-lg text-[12px] font-medium transition-colors duration-100',
                      selected === type ? 'bg-blue-500/15 text-slate-300' : 'text-slate-400 hover:text-slate-900 hover:bg-white/[0.06]',
                    )}
                  >
                    <Icon size={12} className="flex-shrink-0" style={{ color: cfg.color }} />
                    {cfg.label}
                    {selected === type && <span className="ml-auto text-[10px] text-slate-400 font-bold">✓</span>}
                  </button>
                );
              })}
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  );
}

// ─── Waterway Filter ──────────────────────────────────────────────────────────

function WaterwayToggle({
  selected,
  onChange,
}: {
  selected: WaterwayId | 'BOTH';
  onChange: (v: WaterwayId | 'BOTH') => void;
}) {
  const opts: (WaterwayId | 'BOTH')[] = ['BOTH', 'NW-1', 'NW-2'];
  const colors: Record<WaterwayId | 'BOTH', string> = {
    BOTH:   '#94a3b8',
    'NW-1': '#3b82f6',
    'NW-2': '#8b5cf6',
  };

  return (
    <div className="flex items-center gap-1 p-0.5 bg-white/[0.03] rounded-xl border border-slate-900/[0.06]">
      {opts.map((opt) => {
        const isActive = selected === opt;
        const color    = colors[opt];
        return (
          <button
            key={opt}
            onClick={() => onChange(opt)}
            className={cn(
              'px-3 py-1.5 rounded-lg text-[11px] font-bold transition-all duration-150',
              isActive ? 'text-slate-900' : 'text-slate-500 hover:text-slate-700',
            )}
            style={isActive ? { background: `${color}22`, color, border: `1px solid ${color}40` } : {}}
          >
            {opt}
          </button>
        );
      })}
    </div>
  );
}

// ─── Sort Control ─────────────────────────────────────────────────────────────

type SortKey = 'risk_score' | 'created_at' | 'km_start' | 'severity';

function SortControl({
  current,
  onChange,
}: {
  current:  SortKey;
  onChange: (v: SortKey) => void;
}) {
  const opts: { value: SortKey; label: string }[] = [
    { value: 'risk_score', label: 'Risk Score' },
    { value: 'severity',   label: 'Severity'   },
    { value: 'created_at', label: 'Newest'     },
    { value: 'km_start',   label: 'Location'   },
  ];

  return (
    <div className="flex items-center gap-1.5">
      <span className="text-[10px] text-slate-400 font-medium">Sort:</span>
      {opts.map((opt) => (
        <button
          key={opt.value}
          onClick={() => onChange(opt.value)}
          className={cn(
            'px-2.5 py-1 rounded-lg text-[10px] font-semibold border transition-all duration-150',
            current === opt.value
              ? 'bg-blue-500/15 text-slate-300 border-blue-500/30'
              : 'text-slate-500 hover:text-slate-700 border-slate-900/[0.06] bg-white/[0.03]',
          )}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}

// ─── Timeline Visualization ───────────────────────────────────────────────────

function AlertTimeline({ alerts }: { alerts: Alert[] }) {
  const active = alerts.filter((a) => a.is_active);
  if (active.length === 0) return null;

  // Group by day of week / time for a simple bar chart
  const hourBuckets = Array.from({ length: 24 }, (_, h) => ({
    hour:  h,
    count: active.filter((a) => new Date(a.created_at).getHours() === h).length,
  }));

  const maxCount = Math.max(...hourBuckets.map((b) => b.count), 1);

  return (
    <div className="p-4 rounded-2xl border border-slate-900/[0.06] bg-white/[0.02]">
      <div className="flex items-center justify-between mb-3">
        <div className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">
          Alert Activity (by hour of day)
        </div>
        <span className="text-[10px] text-slate-400">Last 24h</span>
      </div>
      <div className="flex items-end gap-0.5 h-12">
        {hourBuckets.map((bucket) => (
          <motion.div
            key={bucket.hour}
            className="flex-1 rounded-sm"
            style={{
              background: bucket.count > 0
                ? `rgba(59,130,246,${0.3 + (bucket.count / maxCount) * 0.7})`
                : 'rgba(15,23,42,0.04)',
              minWidth: 2,
            }}
            initial={{ height: 0 }}
            animate={{ height: `${Math.max(10, (bucket.count / maxCount) * 100)}%` }}
            transition={{ duration: 0.5, delay: bucket.hour * 0.01, ease: [0.16, 1, 0.3, 1] }}
            title={`${bucket.hour}:00 — ${bucket.count} alert${bucket.count !== 1 ? 's' : ''}`}
          />
        ))}
      </div>
      <div className="flex justify-between text-[9px] text-slate-700 mt-1">
        <span>00:00</span>
        <span>06:00</span>
        <span>12:00</span>
        <span>18:00</span>
        <span>24:00</span>
      </div>
    </div>
  );
}

// ─── Empty State ──────────────────────────────────────────────────────────────

function EmptyState({ filtered }: { filtered: boolean }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
      className="flex flex-col items-center justify-center py-20 px-8 text-center"
    >
      <div className="relative mb-5">
        {[0, 1, 2].map((i) => (
          <motion.div
            key={i}
            className="absolute rounded-full border border-emerald-500/20"
            style={{ width: 48 + i * 28, height: 48 + i * 28, top: -(i * 14), left: -(i * 14) }}
            animate={{ opacity: [0.5, 0, 0.5], scale: [1, 1.15, 1] }}
            transition={{ duration: 3, repeat: Infinity, delay: i * 0.7, ease: 'easeOut' }}
          />
        ))}
        <div className="relative w-12 h-12 rounded-2xl bg-emerald-500/15 border border-emerald-500/25 flex items-center justify-center">
          {filtered ? (
            <Filter size={22} className="text-slate-400" />
          ) : (
            <BellOff size={22} className="text-slate-400" />
          )}
        </div>
      </div>
      <h3 className="text-base font-bold text-slate-800 mb-2">
        {filtered ? 'No matching alerts' : 'All Clear — No Active Alerts'}
      </h3>
      <p className="text-[13px] text-slate-500 max-w-xs leading-relaxed">
        {filtered
          ? 'Try adjusting the severity filter, alert type, or waterway selection to see more results.'
          : 'All monitored segments are within safe navigability parameters. Well done!'}
      </p>
      {!filtered && (
        <div className="mt-4 flex items-center gap-2 px-4 py-2 rounded-xl bg-emerald-500/8 border border-emerald-500/20">
          <CheckCircle2 size={14} className="text-slate-400" />
          <span className="text-[12px] font-semibold text-slate-400">
            System operating normally
          </span>
        </div>
      )}
    </motion.div>
  );
}

// ─── Main Alerts Page ─────────────────────────────────────────────────────────

export default function AlertsPage() {
  // ── Store ──────────────────────────────────────────────────────────────────
  const selectedWaterway   = useAppStore((s) => s.selectedWaterway);
  const setSelectedSegment = useAppStore((s) => s.setSelectedSegmentId);
  const severityFilter     = useAppStore((s) => s.alertSeverityFilter);
  const setSeverityFilter  = useAppStore((s) => s.setAlertSeverityFilter);

  // ── Local state ────────────────────────────────────────────────────────────
  const [waterwayFilter, setWaterwayFilter] = useState<WaterwayId | 'BOTH'>('BOTH');
  const [typeFilter,     setTypeFilter]     = useState<AlertType | 'ALL'>('ALL');
  const [sortKey,        setSortKey]        = useState<SortKey>('risk_score');
  const [showAll,        setShowAll]        = useState(false);
  const [refreshing,     setRefreshing]     = useState(false);
  const [showTimeline,   setShowTimeline]   = useState(true);

  const PAGE_SIZE = 8;

  // ── Fetch data from both waterways ─────────────────────────────────────────
  const nw1Alerts = useMemo(() => getMockAlerts('NW-1') as Alert[], []);
  const nw2Alerts = useMemo(() => getMockAlerts('NW-2') as Alert[], []);
  const allAlerts  = useMemo(() => [...nw1Alerts, ...nw2Alerts], [nw1Alerts, nw2Alerts]);

  // ── Available alert types in current dataset ───────────────────────────────
  const availableTypes = useMemo(
    () => Array.from(new Set(allAlerts.map((a) => a.alert_type))),
    [allAlerts],
  );

  // ── Alert counts for filter badges ────────────────────────────────────────
  const activeAlerts = useMemo(() => allAlerts.filter((a) => a.is_active), [allAlerts]);

  const alertCounts: Record<AlertSeverity | 'ALL', number> = useMemo(() => ({
    ALL:      activeAlerts.length,
    CRITICAL: activeAlerts.filter((a) => a.severity === 'CRITICAL').length,
    WARNING:  activeAlerts.filter((a) => a.severity === 'WARNING').length,
    INFO:     activeAlerts.filter((a) => a.severity === 'INFO').length,
  }), [activeAlerts]);

  // ── Filtered + sorted alerts ───────────────────────────────────────────────
  const filteredAlerts = useMemo(() => {
    let list = activeAlerts;

    // Waterway filter
    if (waterwayFilter !== 'BOTH') {
      list = list.filter((a) => a.waterway_id === waterwayFilter);
    }

    // Severity filter
    if (severityFilter !== 'ALL') {
      list = list.filter((a) => a.severity === severityFilter);
    }

    // Type filter
    if (typeFilter !== 'ALL') {
      list = list.filter((a) => a.alert_type === typeFilter);
    }

    // Sort
    list = [...list].sort((a, b) => {
      switch (sortKey) {
        case 'risk_score':
          return b.risk_score - a.risk_score;
        case 'created_at':
          return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
        case 'km_start':
          return a.km_start - b.km_start;
        case 'severity': {
          const sevOrder = SEVERITY_CONFIG[a.severity].order - SEVERITY_CONFIG[b.severity].order;
          return sevOrder !== 0 ? sevOrder : b.risk_score - a.risk_score;
        }
        default:
          return 0;
      }
    });

    return list;
  }, [activeAlerts, waterwayFilter, severityFilter, typeFilter, sortKey]);

  const visibleAlerts = showAll
    ? filteredAlerts
    : filteredAlerts.slice(0, PAGE_SIZE);

  const hasMore = filteredAlerts.length > PAGE_SIZE && !showAll;

  // ── Derived stats ──────────────────────────────────────────────────────────
  const avgRiskScore   = activeAlerts.length
    ? activeAlerts.reduce((s, a) => s + a.risk_score, 0) / activeAlerts.length
    : 0;

  const criticalCount  = alertCounts.CRITICAL;
  const warningCount   = alertCounts.WARNING;
  const mostAffectedWW = nw1Alerts.filter(a => a.is_active).length >= nw2Alerts.filter(a => a.is_active).length
    ? 'NW-1' : 'NW-2';

  // ── Handlers ──────────────────────────────────────────────────────────────
  const handleViewOnMap = (segmentId: string) => {
    setSelectedSegment(segmentId);
  };

  const handleRefresh = async () => {
    if (refreshing) return;
    setRefreshing(true);
    await new Promise((r) => setTimeout(r, 1200));
    setRefreshing(false);
  };

  const handleResetFilters = () => {
    setSeverityFilter('ALL');
    setTypeFilter('ALL');
    setWaterwayFilter('BOTH');
    setSortKey('risk_score');
  };

  const isFiltered = severityFilter !== 'ALL' || typeFilter !== 'ALL' || waterwayFilter !== 'BOTH';

  // ── Animation ──────────────────────────────────────────────────────────────
  const containerVariants = {
    hidden:   { opacity: 0 },
    visible:  { opacity: 1, transition: { staggerChildren: 0.06, delayChildren: 0.05 } },
  };
  const itemVariants = {
    hidden:   { opacity: 0, y: 18 },
    visible:  { opacity: 1, y: 0, transition: { duration: 0.4, ease: [0.16, 1, 0.3, 1] } },
  };

  return (
    <motion.div
      className="p-5 space-y-5 max-w-[1400px] mx-auto"
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      {/* ── Page heading ──────────────────────────────────────────────────── */}
      <motion.div variants={itemVariants}>
        <div className="flex items-start justify-between gap-4 flex-wrap">
          <div>
            <div className="flex items-center gap-3">
              <h1 className="text-2xl font-extrabold text-slate-900 tracking-tight">
                Risk Alerts
              </h1>
              {criticalCount > 0 && (
                <motion.span
                  animate={{ scale: [1, 1.06, 1] }}
                  transition={{ duration: 2, repeat: Infinity }}
                  className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-red-500/12 border border-red-500/30 text-[10px] font-bold text-slate-400 uppercase tracking-wider"
                >
                  <PulsingDot severity="CRITICAL" size={6} />
                  {criticalCount} Critical
                </motion.span>
              )}
              {criticalCount === 0 && activeAlerts.length > 0 && (
                <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-amber-500/10 border border-amber-500/25 text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                  <AlertTriangle size={10} />
                  {activeAlerts.length} Active
                </span>
              )}
              {activeAlerts.length === 0 && (
                <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/25 text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                  <CheckCircle2 size={10} />
                  All Clear
                </span>
              )}
            </div>
            <p className="text-[13px] text-slate-500 mt-1">
              Monitoring {allAlerts.length} segments across NW-1 and NW-2 ·
              {' '}HydroFormer risk engine · Updated every 6 hours
            </p>
          </div>

          {/* Header actions */}
          <div className="flex items-center gap-2">
            {/* Refresh */}
            <motion.button
              onClick={handleRefresh}
              whileTap={{ scale: 0.93 }}
              disabled={refreshing}
              className="flex items-center gap-2 px-3.5 py-2 rounded-xl border border-slate-900/[0.08] bg-white/[0.04] text-[12px] font-semibold text-slate-400 hover:text-slate-800 hover:bg-white/[0.07] transition-all duration-150 disabled:opacity-50"
              aria-label="Refresh alerts"
            >
              <motion.div
                animate={refreshing ? { rotate: 360 } : { rotate: 0 }}
                transition={refreshing ? { duration: 0.8, repeat: Infinity, ease: 'linear' } : { duration: 0.3 }}
              >
                <RefreshCw size={13} />
              </motion.div>
              <span className="hidden sm:inline">{refreshing ? 'Refreshing…' : 'Refresh'}</span>
            </motion.button>

            {/* Export */}
            <button className="flex items-center gap-2 px-3.5 py-2 rounded-xl bg-blue-600 hover:bg-blue-500 border border-blue-500 text-[12px] font-semibold text-slate-900 transition-all duration-150 shadow-lg shadow-blue-500/20">
              <Download size={13} />
              <span className="hidden sm:inline">Export</span>
            </button>
          </div>
        </div>
      </motion.div>

      {/* ── Stats row ─────────────────────────────────────────────────────── */}
      <motion.div variants={itemVariants}>
        <StatCardGrid cols={4}>
          <StatCard
            label="Critical Alerts"
            value={criticalCount}
            decimals={0}
            subtitle="Immediate action required"
            icon={AlertCircle}
            variant={criticalCount > 0 ? 'non_navigable' : 'navigable'}
            animationDelay={0}
            trend={criticalCount > 0 ? -8.5 : 0}
            trendDirection={criticalCount > 0 ? 'down' : 'neutral'}
            trendLabel="vs last week"
            trendUpIsGood={false}
            tooltip="Segments where predicted depth has dropped below the non-navigable threshold (< 2.0m)"
          />
          <StatCard
            label="Warnings"
            value={warningCount}
            decimals={0}
            subtitle="Monitor closely"
            icon={AlertTriangle}
            variant={warningCount > 0 ? 'conditional' : 'navigable'}
            animationDelay={0.07}
            trend={warningCount > 0 ? -5.2 : 0}
            trendDirection={warningCount > 0 ? 'down' : 'neutral'}
            trendLabel="vs last week"
            trendUpIsGood={false}
          />
          <StatCard
            label="Avg Risk Score"
            value={avgRiskScore * 100}
            unit="%"
            decimals={1}
            subtitle={`Across ${activeAlerts.length} active alerts`}
            icon={Shield}
            variant={avgRiskScore > 0.6 ? 'non_navigable' : avgRiskScore > 0.3 ? 'conditional' : 'navigable'}
            animationDelay={0.14}
            tooltip="Mean risk score (0–100) across all active alert segments"
          />
          <StatCard
            label="Most Affected"
            value={
              waterwayFilter === 'BOTH'
                ? (mostAffectedWW === 'NW-1' ? nw1Alerts.filter(a => a.is_active).length : nw2Alerts.filter(a => a.is_active).length)
                : (waterwayFilter === 'NW-1' ? nw1Alerts.filter(a => a.is_active).length : nw2Alerts.filter(a => a.is_active).length)
            }
            decimals={0}
            subtitle={`${mostAffectedWW} — ${mostAffectedWW === 'NW-1' ? 'Ganga' : 'Brahmaputra'}`}
            icon={Waves}
            variant="accent"
            animationDelay={0.21}
          />
        </StatCardGrid>
      </motion.div>

      {/* ── Two-column layout: filters + list (left) and sidebar (right) ─── */}
      <motion.div variants={itemVariants} className="grid grid-cols-1 xl:grid-cols-4 gap-5">

        {/* ── Left: Filters + Alert list ───────────────────────────────── */}
        <div className="xl:col-span-3 space-y-4">

          {/* Filter controls */}
          <div className="p-4 rounded-2xl border border-slate-900/[0.06] bg-white/[0.02] space-y-3">
            {/* Row 1: Severity tabs */}
            <SeverityTabs
              current={severityFilter}
              onChange={setSeverityFilter}
              counts={alertCounts}
            />

            {/* Row 2: Waterway + Type + Sort */}
            <div className="flex items-center gap-3 flex-wrap">
              <WaterwayToggle
                selected={waterwayFilter}
                onChange={setWaterwayFilter}
              />

              <AlertTypeFilter
                selected={typeFilter}
                onChange={setTypeFilter}
                available={availableTypes}
              />

              <div className="hidden md:block h-5 w-px bg-white/[0.08]" />

              <SortControl current={sortKey} onChange={setSortKey} />

              {/* Reset filters */}
              {isFiltered && (
                <motion.button
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.9 }}
                  onClick={handleResetFilters}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[11px] font-semibold text-slate-400 hover:text-slate-300 bg-red-500/8 border border-red-500/20 hover:bg-red-500/12 transition-all duration-150"
                >
                  <X size={11} />
                  Reset Filters
                </motion.button>
              )}

              {/* Result count */}
              <div className="ml-auto text-[11px] text-slate-500 font-medium">
                {filteredAlerts.length} alert{filteredAlerts.length !== 1 ? 's' : ''}
                {isFiltered && (
                  <span className="ml-1 text-slate-400">
                    (filtered from {activeAlerts.length})
                  </span>
                )}
              </div>
            </div>
          </div>

          {/* Alert list */}
          <div className="space-y-3">
            <AnimatePresence mode="popLayout">
              {visibleAlerts.length === 0 ? (
                <EmptyState filtered={isFiltered} />
              ) : (
                visibleAlerts.map((alert, i) => (
                  <AlertCard
                    key={alert.alert_id}
                    alert={alert}
                    index={i}
                    onViewOnMap={handleViewOnMap}
                  />
                ))
              )}
            </AnimatePresence>
          </div>

          {/* Show more / less */}
          {filteredAlerts.length > PAGE_SIZE && (
            <motion.button
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              onClick={() => setShowAll((v) => !v)}
              className="w-full flex items-center justify-center gap-2 py-3 rounded-2xl text-[13px] font-semibold text-slate-400 hover:text-slate-800 bg-white/[0.03] border border-slate-900/[0.06] hover:bg-white/[0.06] hover:border-slate-900/[0.10] transition-all duration-150"
            >
              <motion.span
                animate={{ rotate: showAll ? 180 : 0 }}
                transition={{ duration: 0.2 }}
                className="flex items-center"
              >
                <ChevronDown size={16} />
              </motion.span>
              {showAll
                ? `Show fewer (${PAGE_SIZE})`
                : `Show all ${filteredAlerts.length} alerts`}
            </motion.button>
          )}
        </div>

        {/* ── Right sidebar ─────────────────────────────────────────────── */}
        <div className="xl:col-span-1 space-y-4">

          {/* Timeline */}
          <AlertTimeline alerts={allAlerts} />

          {/* Quick summary by waterway */}
          <div className="p-4 rounded-2xl border border-slate-900/[0.06] bg-white/[0.02]">
            <div className="text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-3">
              By Waterway
            </div>
            {[
              {
                id:     'NW-1' as WaterwayId,
                label:  'Ganga',
                count:  nw1Alerts.filter((a) => a.is_active).length,
                color:  '#3b82f6',
                total:  nw1Alerts.length,
              },
              {
                id:     'NW-2' as WaterwayId,
                label:  'Brahmaputra',
                count:  nw2Alerts.filter((a) => a.is_active).length,
                color:  '#8b5cf6',
                total:  nw2Alerts.length,
              },
            ].map((ww) => (
              <div key={ww.id} className="mb-3 last:mb-0">
                <div className="flex items-center justify-between mb-1.5">
                  <div className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full" style={{ backgroundColor: ww.color }} />
                    <span className="text-[11px] font-semibold text-slate-700">{ww.id}</span>
                    <span className="text-[10px] text-slate-500">· {ww.label}</span>
                  </div>
                  <span className="text-[11px] font-bold tabular-nums" style={{ color: ww.color }}>
                    {ww.count}
                  </span>
                </div>
                <div className="h-1.5 rounded-full bg-white/[0.06] overflow-hidden">
                  <motion.div
                    className="h-full rounded-full"
                    style={{ backgroundColor: ww.color }}
                    initial={{ width: 0 }}
                    animate={{ width: `${(ww.count / Math.max(nw1Alerts.filter(a=>a.is_active).length, nw2Alerts.filter(a=>a.is_active).length, 1)) * 100}%` }}
                    transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
                  />
                </div>
              </div>
            ))}
          </div>

          {/* Quick breakdown by type */}
          <div className="p-4 rounded-2xl border border-slate-900/[0.06] bg-white/[0.02]">
            <div className="text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-3">
              By Alert Type
            </div>
            <div className="space-y-2">
              {availableTypes.map((type) => {
                const cfg   = ALERT_TYPE_CONFIG[type];
                if (!cfg) return null;
                const Icon  = cfg.icon;
                const count = activeAlerts.filter((a) => a.alert_type === type).length;
                if (count === 0) return null;

                return (
                  <motion.button
                    key={type}
                    onClick={() => {
                      setTypeFilter(typeFilter === type ? 'ALL' : type);
                    }}
                    whileHover={{ x: 2 }}
                    whileTap={{ scale: 0.98 }}
                    className={cn(
                      'w-full flex items-center gap-2.5 px-3 py-2.5 rounded-xl border text-left transition-all duration-150',
                      typeFilter === type
                        ? 'bg-blue-500/15 border-blue-500/30'
                        : 'bg-white/[0.02] border-slate-900/[0.06] hover:bg-white/[0.05]',
                    )}
                  >
                    <div
                      className="w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0"
                      style={{ background: `${cfg.color}18`, border: `1px solid ${cfg.color}30` }}
                    >
                      <Icon size={13} style={{ color: cfg.color }} />
                    </div>
                    <span className="flex-1 text-[11px] font-medium text-slate-700 min-w-0 truncate">
                      {cfg.label}
                    </span>
                    <span
                      className="flex-shrink-0 min-w-[20px] h-5 px-1.5 rounded-full flex items-center justify-center text-[10px] font-bold"
                      style={{
                        background: `${cfg.color}18`,
                        color:       cfg.color,
                      }}
                    >
                      {count}
                    </span>
                  </motion.button>
                );
              })}
            </div>
          </div>

          {/* Webhook / subscribe CTA */}
          <div
            className="p-4 rounded-2xl border"
            style={{
              background:   'linear-gradient(135deg, rgba(59,130,246,0.08) 0%, rgba(139,92,246,0.06) 100%)',
              borderColor:  'rgba(59,130,246,0.2)',
            }}
          >
            <div className="flex items-center gap-2 mb-2">
              <Zap size={14} className="text-slate-400 flex-shrink-0" />
              <span className="text-[12px] font-bold text-slate-800">Alert Webhooks</span>
            </div>
            <p className="text-[11px] text-slate-500 leading-relaxed mb-3">
              Receive instant notifications when new critical alerts are generated for your monitored segments.
            </p>
            <button className="w-full flex items-center justify-center gap-1.5 py-2 rounded-xl bg-blue-600/80 hover:bg-blue-600 text-slate-900 text-[11px] font-bold transition-all duration-150 border border-blue-500/50">
              <Bell size={12} />
              Subscribe to Webhooks
              <ExternalLink size={10} className="opacity-60" />
            </button>
          </div>

          {/* System note */}
          <div className="text-center">
            <p className="text-[10px] text-slate-700 leading-relaxed">
              Alerts generated by HydroFormer v1.0<br />
              Sentinel-2 + CWC gauge fusion<br />
              Updated every 6 hours
            </p>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
}
