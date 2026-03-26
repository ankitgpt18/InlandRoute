// ============================================================
// AIDSTL — Inland Waterway Navigability Prediction System
// AlertList — Risk alerts panel with severity badges
// ============================================================

'use client';

import React, { useMemo, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  AlertTriangle,
  AlertCircle,
  Info,
  MapPin,
  Clock,
  ChevronDown,
  ChevronRight,
  ArrowRight,
  Waves,
  Anchor,
  Droplets,
  Wind,
  Eye,
  X,
  Filter,
  RefreshCw,
  BellOff,
  ExternalLink,
  TrendingDown,
  Activity,
  ShieldAlert,
  BarChart2,
} from 'lucide-react';
import { useAppStore } from '@/store/app-store';
import { getMockAlerts } from '@/lib/mock-data';
import { NavigabilityBadge } from '@/components/ui/navigability-badge';
import { cn } from '@/lib/utils';
import type { WaterwayId, AlertSeverity, AlertType } from '@/types';

// ─── Types ────────────────────────────────────────────────────────────────────

export interface Alert {
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

export interface AlertListProps {
  /** Override waterway (defaults to store selection) */
  waterwayId?:    WaterwayId;
  /** Maximum number of alerts to show before "Show more" */
  maxVisible?:    number;
  /** Show filter controls */
  showFilters?:   boolean;
  /** Show empty state illustration */
  showEmptyState?: boolean;
  /** Compact mode — hides descriptions and recommendations */
  compact?:       boolean;
  /** Called when user clicks "View on Map" */
  onViewOnMap?:   (segmentId: string) => void;
  className?:     string;
}

// ─── Severity Config ─────────────────────────────────────────────────────────

interface SeverityConfig {
  label:    string;
  color:    string;
  bg:       string;
  border:   string;
  icon:     React.ElementType;
  dotColor: string;
  pulse:    boolean;
  order:    number;
}

const SEVERITY_CONFIG: Record<AlertSeverity, SeverityConfig> = {
  CRITICAL: {
    label:    'Critical',
    color:    'text-red-400',
    bg:       'bg-red-500/12',
    border:   'border-red-500/30',
    icon:     AlertCircle,
    dotColor: 'bg-red-400',
    pulse:    true,
    order:    0,
  },
  WARNING: {
    label:    'Warning',
    color:    'text-amber-400',
    bg:       'bg-amber-500/12',
    border:   'border-amber-500/30',
    icon:     AlertTriangle,
    dotColor: 'bg-amber-400',
    pulse:    false,
    order:    1,
  },
  INFO: {
    label:    'Info',
    color:    'text-sky-400',
    bg:       'bg-sky-500/10',
    border:   'border-sky-500/25',
    icon:     Info,
    dotColor: 'bg-sky-400',
    pulse:    false,
    order:    2,
  },
};

// ─── Alert Type Config ────────────────────────────────────────────────────────

interface AlertTypeConfig {
  label: string;
  icon:  React.ElementType;
  color: string;
}

const ALERT_TYPE_CONFIG: Partial<Record<AlertType, AlertTypeConfig>> = {
  DEPTH_CRITICAL:    { label: 'Depth Critical',    icon: TrendingDown,  color: '#ef4444' },
  DEPTH_WARNING:     { label: 'Depth Warning',     icon: BarChart2,     color: '#f59e0b' },
  WIDTH_RESTRICTION: { label: 'Width Restriction', icon: Activity,      color: '#f59e0b' },
  VELOCITY_HIGH:     { label: 'High Velocity',     icon: Wind,          color: '#f59e0b' },
  SEASONAL_CLOSURE:  { label: 'Seasonal Closure',  icon: Anchor,        color: '#94a3b8' },
  OBSTACLE_DETECTED: { label: 'Obstacle Detected', icon: ShieldAlert,   color: '#ef4444' },
  FLOOD_RISK:        { label: 'Flood Risk',        icon: Droplets,      color: '#3b82f6' },
  DROUGHT_RISK:      { label: 'Drought Risk',      icon: Waves,         color: '#f59e0b' },
};

// ─── Helpers ──────────────────────────────────────────────────────────────────

function formatRelativeTime(dateStr: string): string {
  const date   = new Date(dateStr);
  const diffMs = Date.now() - date.getTime();
  const diffMin = Math.floor(diffMs / 60_000);
  if (diffMin < 1)  return 'Just now';
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24)  return `${diffHr}h ago`;
  const diffDay = Math.floor(diffHr / 24);
  if (diffDay < 7)  return `${diffDay}d ago`;
  return date.toLocaleDateString('en-IN', { day: 'numeric', month: 'short' });
}

function formatKm(start: number, end: number): string {
  return `${start.toFixed(0)}–${end.toFixed(0)} km`;
}

// ─── Pulsing Severity Dot ─────────────────────────────────────────────────────

function SeverityDot({
  severity,
  size = 8,
}: {
  severity: AlertSeverity;
  size?:    number;
}) {
  const cfg = SEVERITY_CONFIG[severity];

  return (
    <span
      className="relative inline-flex flex-shrink-0"
      style={{ width: size, height: size }}
    >
      {cfg.pulse && (
        <span
          className={cn(
            'animate-ping absolute inline-flex h-full w-full rounded-full opacity-60',
            cfg.dotColor,
          )}
        />
      )}
      <span
        className={cn(
          'relative inline-flex rounded-full h-full w-full',
          cfg.dotColor,
        )}
      />
    </span>
  );
}

// ─── Severity Badge ───────────────────────────────────────────────────────────

function SeverityBadge({ severity }: { severity: AlertSeverity }) {
  const cfg  = SEVERITY_CONFIG[severity];
  const Icon = cfg.icon;

  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 px-2 py-0.5 rounded-full',
        'text-[9px] font-bold tracking-wider uppercase border',
        cfg.bg,
        cfg.border,
        cfg.color,
      )}
    >
      <Icon size={9} strokeWidth={2.5} />
      {cfg.label}
    </span>
  );
}

// ─── Risk Score Bar ───────────────────────────────────────────────────────────

function RiskScoreBar({ score }: { score: number }) {
  const pct   = Math.round(score * 100);
  const color =
    pct >= 70 ? '#ef4444' :
    pct >= 40 ? '#f59e0b' : '#22c55e';

  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 rounded-full bg-white/[0.06] overflow-hidden">
        <motion.div
          className="h-full rounded-full"
          style={{ backgroundColor: color }}
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
        />
      </div>
      <span
        className="text-[10px] font-bold tabular-nums w-[28px] text-right"
        style={{ color }}
      >
        {pct}
      </span>
    </div>
  );
}

// ─── Alert Card ───────────────────────────────────────────────────────────────

interface AlertCardProps {
  alert:       Alert;
  index:       number;
  compact:     boolean;
  onViewOnMap: (segmentId: string) => void;
}

function AlertCard({ alert, index, compact, onViewOnMap }: AlertCardProps) {
  const [expanded, setExpanded] = useState(false);

  const sevCfg = SEVERITY_CONFIG[alert.severity];
  const typeCfg = ALERT_TYPE_CONFIG[alert.alert_type];
  const TypeIcon = typeCfg?.icon ?? AlertTriangle;

  const margin = alert.threshold_value - alert.predicted_value;
  const isCritical = alert.severity === 'CRITICAL';

  return (
    <motion.div
      layout
      initial={{ opacity: 0, x: -16, scale: 0.97 }}
      animate={{ opacity: 1, x: 0, scale: 1 }}
      exit={{ opacity: 0, x: 16, scale: 0.97 }}
      transition={{
        duration: 0.3,
        delay:    index * 0.06,
        ease:     [0.16, 1, 0.3, 1],
      }}
      className={cn(
        'relative overflow-hidden rounded-xl border',
        'transition-all duration-200',
        sevCfg.bg,
        sevCfg.border,
        isCritical && 'shadow-[0_0_20px_rgba(239,68,68,0.12)]',
        expanded && 'ring-1 ring-white/10',
      )}
    >
      {/* ── Critical pulsing left border ─── */}
      {isCritical && (
        <motion.div
          className="absolute left-0 top-0 bottom-0 w-0.5 rounded-r-full bg-red-500"
          animate={{ opacity: [0.5, 1, 0.5] }}
          transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
        />
      )}

      {/* ── Card header ─────────────────────────────────────── */}
      <div className="px-3.5 pt-3 pb-2.5">

        {/* Top row: severity + alert type + time */}
        <div className="flex items-center justify-between gap-2 mb-2">
          <div className="flex items-center gap-2 flex-wrap">
            <SeverityBadge severity={alert.severity} />
            {typeCfg && (
              <span
                className="inline-flex items-center gap-1 text-[9px] font-semibold uppercase tracking-wider"
                style={{ color: `${typeCfg.color}aa` }}
              >
                <TypeIcon size={9} />
                {typeCfg.label}
              </span>
            )}
          </div>
          <div className="flex items-center gap-1.5 flex-shrink-0">
            <Clock size={9} className="text-slate-600" />
            <span className="text-[10px] text-slate-600 font-medium tabular-nums">
              {formatRelativeTime(alert.created_at)}
            </span>
          </div>
        </div>

        {/* Alert title */}
        <div className="flex items-start gap-2.5">
          {/* Icon */}
          <div
            className="flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center mt-0.5"
            style={{
              backgroundColor: `${sevCfg.dotColor.replace('bg-', '')}20`,
              background:       `${isCritical ? 'rgba(239,68,68,0.15)' : alert.severity === 'WARNING' ? 'rgba(245,158,11,0.15)' : 'rgba(14,165,233,0.15)'}`,
            }}
          >
            <TypeIcon
              size={16}
              style={{
                color: isCritical ? '#ef4444' : alert.severity === 'WARNING' ? '#f59e0b' : '#0ea5e9',
              }}
            />
          </div>

          {/* Title + location */}
          <div className="flex-1 min-w-0">
            <h4 className={cn('text-[13px] font-bold leading-tight', sevCfg.color)}>
              {alert.title}
            </h4>
            <div className="flex items-center gap-1.5 mt-1 flex-wrap">
              <div className="flex items-center gap-1">
                <MapPin size={10} className="text-slate-600 flex-shrink-0" />
                <span className="text-[10px] font-semibold text-slate-500">
                  {alert.waterway_id}
                </span>
              </div>
              <span className="text-slate-700 text-[10px]">·</span>
              <span className="text-[10px] text-slate-500 font-medium tabular-nums">
                {formatKm(alert.km_start, alert.km_end)}
              </span>
              <span className="text-slate-700 text-[10px]">·</span>
              <span className="text-[10px] text-slate-600 font-medium tabular-nums">
                Seg {alert.segment_id.split('-').pop()}
              </span>
            </div>
          </div>
        </div>

        {/* ── Metric row: predicted vs threshold ─── */}
        <div className="flex items-stretch gap-2 mt-3">
          {/* Predicted value */}
          <div
            className="flex-1 px-2.5 py-2 rounded-lg"
            style={{
              background: `${isCritical ? 'rgba(239,68,68,0.08)' : alert.severity === 'WARNING' ? 'rgba(245,158,11,0.08)' : 'rgba(14,165,233,0.08)'}`,
              border:     `1px solid ${isCritical ? 'rgba(239,68,68,0.2)' : alert.severity === 'WARNING' ? 'rgba(245,158,11,0.2)' : 'rgba(14,165,233,0.2)'}`,
            }}
          >
            <div className="text-[9px] font-semibold text-slate-600 uppercase tracking-wider mb-0.5">
              Predicted
            </div>
            <div
              className="text-base font-extrabold tabular-nums leading-none"
              style={{
                color: isCritical ? '#ef4444' : alert.severity === 'WARNING' ? '#f59e0b' : '#0ea5e9',
              }}
            >
              {alert.predicted_value.toFixed(2)}
              <span className="text-[11px] font-semibold ml-0.5 opacity-70">
                {alert.unit}
              </span>
            </div>
          </div>

          {/* Arrow */}
          <div className="flex items-center flex-shrink-0">
            <ChevronRight size={14} className="text-slate-600" />
          </div>

          {/* Threshold value */}
          <div className="flex-1 px-2.5 py-2 rounded-lg bg-emerald-500/8 border border-emerald-500/20">
            <div className="text-[9px] font-semibold text-slate-600 uppercase tracking-wider mb-0.5">
              Threshold
            </div>
            <div className="text-base font-extrabold tabular-nums leading-none text-emerald-400">
              {alert.threshold_value.toFixed(2)}
              <span className="text-[11px] font-semibold ml-0.5 opacity-70">
                {alert.unit}
              </span>
            </div>
          </div>

          {/* Deficit */}
          <div className="flex-1 px-2.5 py-2 rounded-lg bg-white/[0.03] border border-white/[0.06]">
            <div className="text-[9px] font-semibold text-slate-600 uppercase tracking-wider mb-0.5">
              Deficit
            </div>
            <div className="text-base font-extrabold tabular-nums leading-none text-red-400">
              {Math.abs(margin).toFixed(2)}
              <span className="text-[11px] font-semibold ml-0.5 opacity-70">
                {alert.unit}
              </span>
            </div>
          </div>
        </div>

        {/* ── Risk score bar ─── */}
        <div className="mt-2.5">
          <div className="flex items-center justify-between mb-1">
            <span className="text-[9px] font-semibold text-slate-600 uppercase tracking-wider">
              Risk Score
            </span>
          </div>
          <RiskScoreBar score={alert.risk_score} />
        </div>

        {/* ── Description (compact: truncated) ─── */}
        {!compact && (
          <p
            className={cn(
              'text-[11px] text-slate-500 leading-relaxed mt-2.5',
              !expanded && 'line-clamp-2',
            )}
          >
            {alert.description}
          </p>
        )}

        {/* ── Expanded content ─── */}
        <AnimatePresence>
          {expanded && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
              className="overflow-hidden"
            >
              <div className="mt-3 pt-3 border-t border-white/[0.06]">
                {/* Recommended actions */}
                {alert.recommended_actions && alert.recommended_actions.length > 0 && (
                  <div className="mb-3">
                    <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-2">
                      Recommended Actions
                    </div>
                    <ul className="flex flex-col gap-1.5">
                      {alert.recommended_actions.map((action, i) => (
                        <motion.li
                          key={i}
                          initial={{ opacity: 0, x: -8 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: i * 0.05, duration: 0.2 }}
                          className="flex items-start gap-2"
                        >
                          <span
                            className="flex-shrink-0 w-4 h-4 rounded-full flex items-center justify-center mt-0.5 text-[8px] font-bold"
                            style={{
                              backgroundColor: `${isCritical ? 'rgba(239,68,68,0.15)' : 'rgba(245,158,11,0.15)'}`,
                              color:            isCritical ? '#ef4444' : '#f59e0b',
                            }}
                          >
                            {i + 1}
                          </span>
                          <span className="text-[11px] text-slate-400 leading-snug">
                            {action}
                          </span>
                        </motion.li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Affected vessels */}
                {alert.affected_vessels && alert.affected_vessels.length > 0 && (
                  <div className="mb-3">
                    <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-2">
                      Affected Vessel Types
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {alert.affected_vessels.map((v, i) => (
                        <span
                          key={i}
                          className="
                            text-[10px] font-medium px-2 py-0.5 rounded-full
                            bg-white/[0.06] border border-white/[0.08] text-slate-400
                          "
                        >
                          {v}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Valid period */}
                <div className="flex items-center gap-3 text-[10px] text-slate-600">
                  <div className="flex items-center gap-1">
                    <Clock size={10} />
                    <span>Valid: {new Date(alert.valid_from).toLocaleDateString('en-IN', { day: 'numeric', month: 'short' })}</span>
                    <span>→</span>
                    <span>{new Date(alert.valid_until).toLocaleDateString('en-IN', { day: 'numeric', month: 'short' })}</span>
                  </div>
                  <span>·</span>
                  <span>ID: {alert.alert_id.slice(-8)}</span>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── Footer actions ─── */}
        <div className="flex items-center justify-between mt-2.5 pt-2 border-t border-white/[0.05]">
          {/* View on map */}
          <motion.button
            whileHover={{ x: 2 }}
            whileTap={{ scale: 0.96 }}
            onClick={() => onViewOnMap(alert.segment_id)}
            className="
              flex items-center gap-1.5
              text-[11px] font-semibold text-blue-400 hover:text-blue-300
              transition-colors duration-150
            "
          >
            <Eye size={12} />
            View on Map
            <ArrowRight size={10} />
          </motion.button>

          {/* Expand / collapse */}
          <button
            onClick={() => setExpanded((v) => !v)}
            className="
              flex items-center gap-1 px-2 py-1 rounded-lg
              text-[10px] font-medium text-slate-500 hover:text-slate-300
              hover:bg-white/[0.05] transition-all duration-150
            "
            aria-expanded={expanded}
          >
            <motion.span
              animate={{ rotate: expanded ? 180 : 0 }}
              transition={{ duration: 0.2 }}
              className="flex items-center"
            >
              <ChevronDown size={12} />
            </motion.span>
            {expanded ? 'Less' : 'Details'}
          </button>
        </div>
      </div>
    </motion.div>
  );
}

// ─── Filter Bar ───────────────────────────────────────────────────────────────

function FilterBar({
  currentFilter,
  onFilterChange,
  alertCounts,
}: {
  currentFilter: AlertSeverity | 'ALL';
  onFilterChange: (f: AlertSeverity | 'ALL') => void;
  alertCounts: Record<AlertSeverity | 'ALL', number>;
}) {
  const options: (AlertSeverity | 'ALL')[] = ['ALL', 'CRITICAL', 'WARNING', 'INFO'];

  const colorMap: Record<AlertSeverity | 'ALL', string> = {
    ALL:      '#94a3b8',
    CRITICAL: '#ef4444',
    WARNING:  '#f59e0b',
    INFO:     '#0ea5e9',
  };

  const bgMap: Record<AlertSeverity | 'ALL', string> = {
    ALL:      'bg-slate-500/15 border-slate-500/30',
    CRITICAL: 'bg-red-500/15 border-red-500/30',
    WARNING:  'bg-amber-500/15 border-amber-500/30',
    INFO:     'bg-sky-500/15 border-sky-500/30',
  };

  return (
    <div className="flex items-center gap-1.5 flex-wrap">
      {options.map((opt) => {
        const count    = alertCounts[opt];
        const isActive = currentFilter === opt;
        const color    = colorMap[opt];

        return (
          <motion.button
            key={opt}
            onClick={() => onFilterChange(opt)}
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.97 }}
            className={cn(
              'flex items-center gap-1.5 px-2.5 py-1 rounded-full border',
              'text-[10px] font-bold uppercase tracking-wider',
              'transition-all duration-150',
              isActive
                ? bgMap[opt]
                : 'border-white/[0.07] bg-transparent hover:bg-white/[0.05]',
            )}
            style={{
              color: isActive ? color : '#64748b',
            }}
          >
            {opt !== 'ALL' && (
              <SeverityDot
                severity={opt as AlertSeverity}
                size={6}
              />
            )}
            {opt}
            {count > 0 && (
              <span
                className="
                  ml-0.5 min-w-[16px] h-4 px-1 rounded-full
                  flex items-center justify-center
                  text-[9px] font-bold
                "
                style={{
                  backgroundColor: isActive ? `${color}25` : 'rgba(255,255,255,0.06)',
                  color:            isActive ? color : '#64748b',
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

// ─── Empty State ──────────────────────────────────────────────────────────────

function EmptyState({ filtered }: { filtered: boolean }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="
        flex flex-col items-center justify-center
        py-12 px-6 text-center
      "
    >
      {/* Animated icon */}
      <div className="relative mb-4">
        {[0, 1, 2].map((i) => (
          <motion.div
            key={i}
            className="absolute rounded-full border border-emerald-500/20"
            style={{ width: 32 + i * 24, height: 32 + i * 24, top: -(i * 12), left: -(i * 12) }}
            animate={{ opacity: [0.4, 0, 0.4], scale: [1, 1.1, 1] }}
            transition={{
              duration: 3,
              repeat: Infinity,
              delay: i * 0.6,
            }}
          />
        ))}
        <div className="relative w-8 h-8 rounded-full bg-emerald-500/20 flex items-center justify-center">
          <BellOff size={16} className="text-emerald-400" />
        </div>
      </div>

      <h4 className="text-sm font-bold text-slate-300 mb-1">
        {filtered ? 'No matching alerts' : 'All Clear'}
      </h4>
      <p className="text-[12px] text-slate-600 max-w-[200px] leading-relaxed">
        {filtered
          ? 'Try adjusting the severity filter to see more alerts.'
          : 'No active risk alerts for this waterway at the selected time.'}
      </p>
    </motion.div>
  );
}

// ─── Alert Statistics Row ─────────────────────────────────────────────────────

function AlertStats({ alerts }: { alerts: Alert[] }) {
  const activeAlerts   = alerts.filter((a) => a.is_active);
  const criticalCount  = activeAlerts.filter((a) => a.severity === 'CRITICAL').length;
  const warningCount   = activeAlerts.filter((a) => a.severity === 'WARNING').length;
  const avgRisk        = activeAlerts.length > 0
    ? activeAlerts.reduce((s, a) => s + a.risk_score, 0) / activeAlerts.length
    : 0;

  const stats = [
    {
      label:  'Critical',
      value:  criticalCount,
      color:  criticalCount > 0 ? '#ef4444' : '#64748b',
      bg:     criticalCount > 0 ? 'bg-red-500/10 border-red-500/20' : 'bg-white/[0.03] border-white/[0.06]',
    },
    {
      label:  'Warning',
      value:  warningCount,
      color:  warningCount > 0 ? '#f59e0b' : '#64748b',
      bg:     warningCount > 0 ? 'bg-amber-500/10 border-amber-500/20' : 'bg-white/[0.03] border-white/[0.06]',
    },
    {
      label:  'Avg Risk',
      value:  `${(avgRisk * 100).toFixed(0)}%`,
      color:  avgRisk > 0.6 ? '#ef4444' : avgRisk > 0.3 ? '#f59e0b' : '#22c55e',
      bg:     'bg-white/[0.03] border-white/[0.06]',
    },
    {
      label:  'Active',
      value:  activeAlerts.length,
      color:  '#94a3b8',
      bg:     'bg-white/[0.03] border-white/[0.06]',
    },
  ];

  return (
    <div className="grid grid-cols-4 gap-2 mb-4">
      {stats.map((s) => (
        <motion.div
          key={s.label}
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
          className={cn(
            'flex flex-col gap-0.5 px-2.5 py-2 rounded-xl border text-center',
            s.bg,
          )}
        >
          <span
            className="text-lg font-extrabold tracking-tight tabular-nums leading-none"
            style={{ color: s.color }}
          >
            {s.value}
          </span>
          <span className="text-[9px] font-semibold uppercase tracking-wider text-slate-600">
            {s.label}
          </span>
        </motion.div>
      ))}
    </div>
  );
}

// ─── Main AlertList Component ─────────────────────────────────────────────────

export function AlertList({
  waterwayId:   waterwayIdProp,
  maxVisible    = 5,
  showFilters   = true,
  showEmptyState = true,
  compact       = false,
  onViewOnMap,
  className,
}: AlertListProps) {
  const storeWaterway       = useAppStore((s) => s.selectedWaterway);
  const setSelectedSegment  = useAppStore((s) => s.setSelectedSegmentId);
  const setDetailPanelOpen  = useAppStore((s) => s.setDetailPanelOpen);
  const severityFilter      = useAppStore((s) => s.alertSeverityFilter);
  const setSeverityFilter   = useAppStore((s) => s.setAlertSeverityFilter);
  const flyToSegment        = useAppStore((s) => s.flyToSegment);

  const waterwayId = waterwayIdProp ?? storeWaterway;

  const [showAll,    setShowAll]    = useState(false);
  const [refreshing, setRefreshing] = useState(false);

  // ── Fetch data ─────────────────────────────────────────────────────────────
  const allAlerts = useMemo(
    () => getMockAlerts(waterwayId),
    [waterwayId],
  );

  // ── Count per severity (for filter badges) ─────────────────────────────────
  const alertCounts = useMemo(() => {
    const active = allAlerts.filter((a) => a.is_active);
    return {
      ALL:      active.length,
      CRITICAL: active.filter((a) => a.severity === 'CRITICAL').length,
      WARNING:  active.filter((a) => a.severity === 'WARNING').length,
      INFO:     active.filter((a) => a.severity === 'INFO').length,
    };
  }, [allAlerts]);

  // ── Filter + sort ──────────────────────────────────────────────────────────
  const filteredAlerts = useMemo(() => {
    let filtered = allAlerts.filter((a) => a.is_active);

    if (severityFilter !== 'ALL') {
      filtered = filtered.filter((a) => a.severity === severityFilter);
    }

    // Sort: CRITICAL first, then WARNING, then INFO; within each group by risk_score desc
    return filtered.sort((a, b) => {
      const orderA = SEVERITY_CONFIG[a.severity].order;
      const orderB = SEVERITY_CONFIG[b.severity].order;
      if (orderA !== orderB) return orderA - orderB;
      return b.risk_score - a.risk_score;
    });
  }, [allAlerts, severityFilter]);

  // ── Visible slice ──────────────────────────────────────────────────────────
  const visibleAlerts = showAll
    ? filteredAlerts
    : filteredAlerts.slice(0, maxVisible);

  const hasMore = filteredAlerts.length > maxVisible && !showAll;

  // ── Handlers ──────────────────────────────────────────────────────────────
  const handleViewOnMap = (segmentId: string) => {
    setSelectedSegment(segmentId);
    setDetailPanelOpen(true);
    onViewOnMap?.(segmentId);
  };

  const handleRefresh = async () => {
    if (refreshing) return;
    setRefreshing(true);
    await new Promise((r) => setTimeout(r, 1000));
    setRefreshing(false);
  };

  return (
    <div className={cn('flex flex-col', className)}>

      {/* ── Header ──────────────────────────────────────────────────────── */}
      {!compact && (
        <div className="flex items-center justify-between gap-3 mb-3">
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-sm font-bold text-slate-100">Risk Alerts</h3>
              {alertCounts.CRITICAL > 0 && (
                <motion.span
                  animate={{ scale: [1, 1.08, 1] }}
                  transition={{ duration: 2, repeat: Infinity }}
                  className="
                    inline-flex items-center gap-1 px-2 py-0.5 rounded-full
                    bg-red-500/15 border border-red-500/30
                    text-[9px] font-bold text-red-400 uppercase tracking-wider
                  "
                >
                  <span className="w-1.5 h-1.5 rounded-full bg-red-400 animate-ping" />
                  {alertCounts.CRITICAL} Critical
                </motion.span>
              )}
            </div>
            <p className="text-[11px] text-slate-500 mt-0.5">
              {waterwayId} · {alertCounts.ALL} active alert{alertCounts.ALL !== 1 ? 's' : ''}
            </p>
          </div>

          {/* Refresh button */}
          <motion.button
            onClick={handleRefresh}
            whileTap={{ scale: 0.93 }}
            className="
              flex items-center justify-center w-8 h-8 rounded-lg
              text-slate-500 hover:text-slate-300
              bg-white/[0.04] border border-white/[0.08]
              hover:bg-white/[0.07]
              transition-all duration-150
            "
            disabled={refreshing}
            aria-label="Refresh alerts"
          >
            <motion.div
              animate={refreshing ? { rotate: 360 } : { rotate: 0 }}
              transition={
                refreshing
                  ? { duration: 0.8, repeat: Infinity, ease: 'linear' }
                  : { duration: 0.3 }
              }
            >
              <RefreshCw size={13} />
            </motion.div>
          </motion.button>
        </div>
      )}

      {/* ── Summary stats ────────────────────────────────────────────────── */}
      {!compact && allAlerts.length > 0 && (
        <AlertStats alerts={allAlerts} />
      )}

      {/* ── Filters ──────────────────────────────────────────────────────── */}
      {showFilters && !compact && (
        <div className="flex items-center gap-2 mb-3">
          <Filter size={11} className="text-slate-600 flex-shrink-0" />
          <FilterBar
            currentFilter={severityFilter}
            onFilterChange={setSeverityFilter}
            alertCounts={alertCounts}
          />
        </div>
      )}

      {/* ── Alert cards ──────────────────────────────────────────────────── */}
      <div className="flex flex-col gap-2">
        <AnimatePresence mode="popLayout">
          {visibleAlerts.length === 0 ? (
            showEmptyState ? (
              <EmptyState filtered={severityFilter !== 'ALL'} />
            ) : null
          ) : (
            visibleAlerts.map((alert, i) => (
              <AlertCard
                key={alert.alert_id}
                alert={alert as Alert}
                index={i}
                compact={compact}
                onViewOnMap={handleViewOnMap}
              />
            ))
          )}
        </AnimatePresence>
      </div>

      {/* ── Show more / less ──────────────────────────────────────────────── */}
      {(hasMore || showAll) && filteredAlerts.length > maxVisible && (
        <motion.button
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          onClick={() => setShowAll((v) => !v)}
          className="
            flex items-center justify-center gap-2 mt-3 py-2.5 rounded-xl
            text-[12px] font-semibold text-slate-400 hover:text-slate-200
            bg-white/[0.03] border border-white/[0.06]
            hover:bg-white/[0.06] hover:border-white/[0.10]
            transition-all duration-150
          "
        >
          <motion.span
            animate={{ rotate: showAll ? 180 : 0 }}
            transition={{ duration: 0.2 }}
            className="flex items-center"
          >
            <ChevronDown size={14} />
          </motion.span>
          {showAll
            ? 'Show fewer alerts'
            : `Show ${filteredAlerts.length - maxVisible} more alert${filteredAlerts.length - maxVisible !== 1 ? 's' : ''}`
          }
        </motion.button>
      )}

      {/* ── Footer note ──────────────────────────────────────────────────── */}
      {!compact && filteredAlerts.length > 0 && (
        <p className="text-[10px] text-slate-700 mt-3 text-center">
          Alerts generated by HydroFormer v1.0 · Updated every 6 hours
        </p>
      )}
    </div>
  );
}

// ─── Compact variant (for dashboard sidebar) ──────────────────────────────────

export function AlertListCompact({
  waterwayId,
  maxVisible = 3,
  onViewOnMap,
  className,
}: Pick<AlertListProps, 'waterwayId' | 'maxVisible' | 'onViewOnMap' | 'className'>) {
  return (
    <AlertList
      waterwayId={waterwayId}
      maxVisible={maxVisible}
      showFilters={false}
      showEmptyState={true}
      compact={true}
      onViewOnMap={onViewOnMap}
      className={className}
    />
  );
}
