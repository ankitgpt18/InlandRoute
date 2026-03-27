// ============================================================
// AIDSTL — Inland Waterway Navigability Prediction System
// Dashboard Page — Main overview with map, stats, alerts
// ============================================================

'use client';

import React, { useMemo } from 'react';
import { motion } from 'framer-motion';
import {
  Navigation,
  AlertTriangle,
  Gauge,
  Ruler,
  Waves,
  TrendingUp,
  Activity,
  Satellite,
} from 'lucide-react';
import { useAppStore } from '@/store/app-store';
import {
  getMockNavigabilityMap,
  getMockWaterwayStats,
  getMockAlerts,
} from '@/lib/mock-data';
import { StatCard, StatCardGrid } from '@/components/ui/stat-card';
import { RiverMap } from '@/components/maps/river-map';
import { AlertListCompact } from '@/components/alerts/alert-list';
import { SeasonalCalendarCompact } from '@/components/charts/seasonal-calendar';
import { DepthProfileCompact } from '@/components/charts/depth-profile';
import { NavigabilityBadge } from '@/components/ui/navigability-badge';
import { cn } from '@/lib/utils';

// ─── Month name helper ────────────────────────────────────────────────────────

const MONTH_NAMES = [
  'January', 'February', 'March', 'April',
  'May', 'June', 'July', 'August',
  'September', 'October', 'November', 'December',
] as const;

// ─── Section heading component ────────────────────────────────────────────────

function SectionHeading({
  title,
  subtitle,
  icon: Icon,
  action,
  className,
}: {
  title:     string;
  subtitle?: string;
  icon?:     React.ElementType;
  action?:   React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn('flex items-center justify-between gap-3', className)}>
      <div className="flex items-center gap-2.5 min-w-0">
        {Icon && (
          <div className="w-7 h-7 rounded-lg bg-blue-500/15 border border-blue-500/25 flex items-center justify-center flex-shrink-0">
            <Icon size={14} className="text-slate-400" />
          </div>
        )}
        <div className="min-w-0">
          <h2 className="text-sm font-bold text-slate-900 leading-tight">{title}</h2>
          {subtitle && (
            <p className="text-[11px] text-slate-500 mt-0.5 truncate">{subtitle}</p>
          )}
        </div>
      </div>
      {action && <div className="flex-shrink-0">{action}</div>}
    </div>
  );
}

// ─── Model Metrics Strip ──────────────────────────────────────────────────────

function ModelMetricsStrip() {
  const metrics = [
    { label: 'R²',   value: '0.918', color: '#22c55e', desc: 'Depth regression' },
    { label: 'RMSE', value: '1.24m', color: '#3b82f6', desc: 'Root mean sq. error' },
    { label: 'MAE',  value: '0.87m', color: '#8b5cf6', desc: 'Mean absolute error' },
    { label: 'F1',   value: '93.4%', color: '#f59e0b', desc: 'Classification F1' },
    { label: 'CI',   value: '90%',   color: '#0ea5e9', desc: 'Confidence interval' },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.1 }}
      className="
        flex items-center gap-0 overflow-hidden
        rounded-xl border border-slate-900/[0.06]
        bg-white/[0.02]
      "
    >
      {metrics.map((m, i) => (
        <React.Fragment key={m.label}>
          <div className="flex items-center gap-2.5 px-4 py-2.5 flex-1 min-w-0">
            {/* Colour dot */}
            <span
              className="w-2 h-2 rounded-full flex-shrink-0"
              style={{ backgroundColor: m.color }}
            />
            <div className="min-w-0">
              <div className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider leading-none">
                {m.label}
              </div>
              <div
                className="text-sm font-extrabold tabular-nums tracking-tight leading-tight mt-0.5"
                style={{ color: m.color }}
              >
                {m.value}
              </div>
              <div className="text-[9px] text-slate-400 leading-none mt-0.5 hidden sm:block">
                {m.desc}
              </div>
            </div>
          </div>
          {i < metrics.length - 1 && (
            <div className="w-px h-8 bg-white/[0.06] flex-shrink-0" />
          )}
        </React.Fragment>
      ))}

      {/* Model badge at end */}
      <div className="w-px h-8 bg-white/[0.06] flex-shrink-0" />
      <div className="flex items-center gap-2 px-4 py-2.5 flex-shrink-0">
        <div className="flex items-center gap-1.5">
          <span className="relative flex h-2 w-2 flex-shrink-0">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
          </span>
          <span className="text-[11px] font-bold text-slate-400">HydroFormer v1.0</span>
        </div>
        <span className="hidden md:block text-[10px] text-slate-400">
          TFT + Swin-T Ensemble
        </span>
      </div>
    </motion.div>
  );
}

// ─── Right Panel Tabs ─────────────────────────────────────────────────────────

type RightPanelTab = 'alerts' | 'calendar' | 'depth';

const RIGHT_PANEL_TABS: { id: RightPanelTab; label: string; icon: React.ElementType }[] = [
  { id: 'alerts',   label: 'Alerts',    icon: AlertTriangle },
  { id: 'calendar', label: 'Calendar',  icon: Activity      },
  { id: 'depth',    label: 'Depth',     icon: Waves         },
];

function RightPanelTabs({
  active,
  onChange,
  alertCount,
}: {
  active:     RightPanelTab;
  onChange:   (t: RightPanelTab) => void;
  alertCount: number;
}) {
  return (
    <div className="flex items-center gap-0.5 p-0.5 bg-white/[0.04] rounded-xl border border-slate-900/[0.06]">
      {RIGHT_PANEL_TABS.map((tab) => {
        const Icon     = tab.icon;
        const isActive = active === tab.id;
        return (
          <motion.button
            key={tab.id}
            onClick={() => onChange(tab.id)}
            whileTap={{ scale: 0.96 }}
            className={cn(
              'relative flex items-center gap-1.5 px-3 py-1.5 rounded-lg flex-1 justify-center',
              'text-[11px] font-semibold transition-all duration-200',
              isActive
                ? 'bg-blue-500/20 text-slate-300'
                : 'text-slate-500 hover:text-slate-700 hover:bg-white/[0.05]',
            )}
          >
            <Icon size={12} strokeWidth={isActive ? 2.5 : 2} />
            <span className="hidden sm:inline">{tab.label}</span>

            {/* Alert count badge */}
            {tab.id === 'alerts' && alertCount > 0 && (
              <span className="
                absolute -top-1 -right-1 min-w-[16px] h-4 px-1
                bg-red-500 text-slate-900 text-[9px] font-bold rounded-full
                flex items-center justify-center leading-none
              ">
                {alertCount > 9 ? '9+' : alertCount}
              </span>
            )}
          </motion.button>
        );
      })}
    </div>
  );
}

// ─── Main Dashboard Page ──────────────────────────────────────────────────────

export default function DashboardPage() {
  // ── Store ──────────────────────────────────────────────────────────────────
  const selectedWaterway    = useAppStore((s) => s.selectedWaterway);
  const selectedMonth       = useAppStore((s) => s.selectedMonth);
  const selectedYear        = useAppStore((s) => s.selectedYear);
  const setSelectedSegment  = useAppStore((s) => s.setSelectedSegmentId);

  // ── Local state ────────────────────────────────────────────────────────────
  const [rightTab, setRightTab] = React.useState<RightPanelTab>('alerts');

  // ── Data ───────────────────────────────────────────────────────────────────
  const navMap = useMemo(
    () => getMockNavigabilityMap(selectedWaterway, selectedMonth),
    [selectedWaterway, selectedMonth],
  );

  const stats = useMemo(
    () => getMockWaterwayStats(selectedWaterway, selectedYear),
    [selectedWaterway, selectedYear],
  );

  const alerts = useMemo(
    () => getMockAlerts(selectedWaterway),
    [selectedWaterway],
  );

  const activeAlerts   = alerts.filter((a) => a.is_active);
  const criticalAlerts = activeAlerts.filter((a) => a.severity === 'CRITICAL');

  // ── Derived stats ──────────────────────────────────────────────────────────
  const navigablePct     = navMap?.navigable_pct     ?? 0;
  const navigableKm      = navMap?.navigable_km       ?? 0;
  const totalKm          = navMap?.total_km           ?? 0;
  const conditionalKm    = navMap?.conditional_km     ?? 0;
  const nonNavigableKm   = navMap?.non_navigable_km   ?? 0;
  const meanConfidence   = navMap?.mean_confidence    ?? 0;
  const totalSegments    = navMap?.total_segments     ?? 0;

  // YoY change from stats
  const yoyNavChange     = stats?.yoy_navigability_change_pct ?? 2.4;
  const yoyDepthChange   = stats?.yoy_depth_change_m ?? 0.18;
  const annualMeanDepth  = stats?.annual_mean_depth_m ?? 4.2;

  // Month name for subtitle
  const monthName = MONTH_NAMES[selectedMonth - 1];
  const isMonsoon = selectedMonth >= 6 && selectedMonth <= 9;

  // ── Animation variants ─────────────────────────────────────────────────────
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.07, delayChildren: 0.05 },
    },
  };

  const itemVariants = {
    hidden:   { opacity: 0, y: 16 },
    visible:  { opacity: 1, y: 0, transition: { duration: 0.4, ease: [0.16, 1, 0.3, 1] } },
  };

  return (
    <motion.div
      className="flex flex-col h-full p-4 gap-4 overflow-hidden"
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      {/* ── Row 1: Page title + model metrics strip ───────────────────────── */}
      <motion.div variants={itemVariants} className="flex flex-col gap-3">
        {/* Page heading */}
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center gap-3">
              <h1 className="text-xl font-extrabold text-slate-900 tracking-tight">
                Overview
              </h1>
              {isMonsoon && (
                <span className="
                  text-[10px] font-bold tracking-widest uppercase
                  px-2.5 py-1 rounded-full
                  bg-sky-500/12 border border-sky-500/25 text-slate-400
                ">
                  🌧 Monsoon Season
                </span>
              )}
            </div>
            <p className="text-[12px] text-slate-500 mt-0.5">
              {selectedWaterway} ·{' '}
              {monthName} {selectedYear} ·{' '}
              {totalSegments} segments analysed
            </p>
          </div>

          {/* Live timestamp */}
          <div className="hidden md:flex items-center gap-2 text-[11px] text-slate-400">
            <Satellite size={11} className="text-slate-400" />
            <span>Sentinel-2 composite · 10 m resolution</span>
          </div>
        </div>

        {/* Model metrics strip */}
        <ModelMetricsStrip />
      </motion.div>

      {/* ── Row 2: Stat cards ─────────────────────────────────────────────── */}
      <motion.div variants={itemVariants}>
        <StatCardGrid cols={4}>
          {/* Navigable % */}
          <StatCard
            label="Navigable"
            value={navigablePct}
            unit="%"
            decimals={1}
            subtitle={`${navigableKm.toFixed(0)} km of ${totalKm.toFixed(0)} km passable`}
            trend={yoyNavChange}
            trendDirection={yoyNavChange >= 0 ? 'up' : 'down'}
            trendLabel="vs last year"
            trendUpIsGood={true}
            icon={Navigation}
            variant="navigable"
            animationDelay={0}
            sparklineData={
              stats?.monthly_stats
                ? stats.monthly_stats.slice(-7).map((m: { navigable_pct: number }) => m.navigable_pct)
                : [65, 72, 68, 75, 80, navigablePct - 3, navigablePct]
            }
            tooltip="Percentage of river segments meeting the IWAI navigability standard (depth ≥ 3.0m, width ≥ 50m)"
          />

          {/* Navigable km */}
          <StatCard
            label="Navigable Length"
            value={navigableKm}
            unit="km"
            decimals={0}
            subtitle={`${conditionalKm.toFixed(0)} km conditional · ${nonNavigableKm.toFixed(0)} km closed`}
            trend={yoyDepthChange * 8}
            trendDirection={yoyNavChange >= 0 ? 'up' : 'down'}
            trendLabel="vs last year"
            trendUpIsGood={true}
            icon={Ruler}
            variant="accent"
            animationDelay={0.07}
            sparklineData={[820, 840, 835, 855, 870, navigableKm - 20, navigableKm]}
          />

          {/* Critical alerts */}
          <StatCard
            label="Critical Alerts"
            value={criticalAlerts.length}
            decimals={0}
            subtitle={`${activeAlerts.length} total active · ${activeAlerts.filter(a => a.severity === 'WARNING').length} warnings`}
            trend={criticalAlerts.length > 0 ? -15.3 : 0}
            trendDirection={criticalAlerts.length > 0 ? 'down' : 'neutral'}
            trendLabel="vs last month"
            trendUpIsGood={false}
            icon={AlertTriangle}
            variant={criticalAlerts.length > 0 ? 'non_navigable' : 'navigable'}
            animationDelay={0.14}
            tooltip="Active critical risk alerts where predicted depth is below the non-navigable threshold"
          />

          {/* Model confidence */}
          <StatCard
            label="Model Confidence"
            value={meanConfidence * 100}
            unit="%"
            decimals={1}
            subtitle={`R² = 0.918 · RMSE = 1.24m · HydroFormer v1.0`}
            trend={2.1}
            trendDirection="up"
            trendLabel="vs baseline"
            trendUpIsGood={true}
            icon={Gauge}
            variant="info"
            animationDelay={0.21}
            sparklineData={[84, 86, 87, 88, 88.5, 89, meanConfidence * 100]}
            tooltip="Mean prediction confidence score across all active segment predictions, derived from the ensemble uncertainty quantification"
          />
        </StatCardGrid>
      </motion.div>

      {/* ── Row 3: Main content — Map (left) + Right panel ──────────────── */}
      <motion.div
        variants={itemVariants}
        className="flex gap-4 flex-1 min-h-0 overflow-hidden"
      >
        {/* ── Left: River Map ───────────────────────────────────────────── */}
        <div className="flex flex-col flex-1 min-w-0 gap-3">
          {/* Map section heading */}
          <SectionHeading
            title="Navigability Map"
            subtitle={`${selectedWaterway} · ${monthName} ${selectedYear} · Click segment for details`}
            icon={Navigation}
            action={
              <div className="flex items-center gap-2">
                {/* Quick navigability summary */}
                <div className="hidden lg:flex items-center gap-2">
                  <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-emerald-500/10 border border-emerald-500/20">
                    <span className="w-2 h-2 rounded-full bg-emerald-500 flex-shrink-0" />
                    <span className="text-[10px] font-bold text-slate-400 tabular-nums">
                      {navMap?.navigable_count ?? 0} nav
                    </span>
                  </div>
                  <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-amber-500/10 border border-amber-500/20">
                    <span className="w-2 h-2 rounded-full bg-amber-500 flex-shrink-0" />
                    <span className="text-[10px] font-bold text-slate-400 tabular-nums">
                      {navMap?.conditional_count ?? 0} cond
                    </span>
                  </div>
                  <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-red-500/10 border border-red-500/20">
                    <span className="w-2 h-2 rounded-full bg-red-500 flex-shrink-0" />
                    <span className="text-[10px] font-bold text-slate-400 tabular-nums">
                      {navMap?.non_navigable_count ?? 0} closed
                    </span>
                  </div>
                </div>
              </div>
            }
          />

          {/* The Map */}
          <div
            className="
              flex-1 min-h-0 rounded-2xl overflow-hidden
              border border-slate-900/[0.06]
              shadow-[0_4px_32px_rgba(0,0,0,0.5)]
            "
          >
            <RiverMap
              className="w-full h-full"
              onSegmentClick={(segId) => {
                setSelectedSegment(segId);
              }}
            />
          </div>
        </div>

        {/* ── Right panel ──────────────────────────────────────────────── */}
        <div
          className="
            flex flex-col w-[340px] xl:w-[380px] flex-shrink-0 gap-3 min-h-0 overflow-hidden
          "
        >
          {/* Tab switcher */}
          <RightPanelTabs
            active={rightTab}
            onChange={setRightTab}
            alertCount={criticalAlerts.length}
          />

          {/* Panel content */}
          <div
            className="
              flex-1 min-h-0 overflow-y-auto thin-scrollbar
              rounded-2xl border border-slate-900/[0.06]
              bg-white/[0.02] backdrop-blur-sm
              p-4
            "
          >
            {/* ── Alerts tab ─────────────────────────────────────────── */}
            {rightTab === 'alerts' && (
              <motion.div
                key="alerts"
                initial={{ opacity: 0, x: 12 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -12 }}
                transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
              >
                <AlertListCompact
                  waterwayId={selectedWaterway}
                  maxVisible={4}
                  onViewOnMap={(segId) => setSelectedSegment(segId)}
                />
              </motion.div>
            )}

            {/* ── Calendar tab ───────────────────────────────────────── */}
            {rightTab === 'calendar' && (
              <motion.div
                key="calendar"
                initial={{ opacity: 0, x: 12 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -12 }}
                transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
              >
                <SeasonalCalendarCompact
                  waterwayId={selectedWaterway}
                  year={selectedYear}
                  maxRows={6}
                />
              </motion.div>
            )}

            {/* ── Depth tab ──────────────────────────────────────────── */}
            {rightTab === 'depth' && (
              <motion.div
                key="depth"
                initial={{ opacity: 0, x: 12 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -12 }}
                transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
              >
                <div className="mb-3">
                  <h3 className="text-sm font-bold text-slate-900">Depth Profile</h3>
                  <p className="text-[11px] text-slate-500 mt-0.5">
                    {selectedWaterway} · {monthName} {selectedYear}
                  </p>
                </div>
                <DepthProfileCompact
                  waterwayId={selectedWaterway}
                  month={selectedMonth}
                  height={200}
                />

                {/* Quick depth stats */}
                <div className="mt-4 grid grid-cols-2 gap-2">
                  {[
                    {
                      label: 'Annual Mean',
                      value: `${annualMeanDepth.toFixed(2)}m`,
                      color: 'text-slate-400',
                      bg:    'bg-blue-500/10 border-blue-500/20',
                    },
                    {
                      label: 'YoY Change',
                      value: `${yoyDepthChange >= 0 ? '+' : ''}${yoyDepthChange.toFixed(2)}m`,
                      color: yoyDepthChange >= 0 ? 'text-slate-400' : 'text-slate-400',
                      bg:    yoyDepthChange >= 0
                        ? 'bg-emerald-500/10 border-emerald-500/20'
                        : 'bg-red-500/10 border-red-500/20',
                    },
                  ].map((s) => (
                    <div
                      key={s.label}
                      className={cn(
                        'flex flex-col gap-0.5 px-3 py-2.5 rounded-xl border',
                        s.bg,
                      )}
                    >
                      <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">
                        {s.label}
                      </span>
                      <span className={cn('text-xl font-extrabold tabular-nums tracking-tight leading-tight', s.color)}>
                        {s.value}
                      </span>
                    </div>
                  ))}
                </div>
              </motion.div>
            )}
          </div>

          {/* ── Quick waterway info footer ────────────────────────────── */}
          <div className="
            flex-shrink-0 px-3.5 py-2.5 rounded-xl
            bg-white/[0.03] border border-slate-900/[0.06]
          ">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider">
                  {selectedWaterway === 'NW-1' ? 'Ganga · Varanasi → Haldia' : 'Brahmaputra · Dhubri → Sadiya'}
                </div>
                <div className="flex items-center gap-2 mt-1">
                  <span className="text-xs font-bold text-slate-700">
                    {totalKm.toFixed(0)} km
                  </span>
                  <span className="text-slate-700 text-xs">·</span>
                  <span className="text-xs text-slate-500">
                    {totalSegments} × 5km segments
                  </span>
                </div>
              </div>
              <NavigabilityBadge
                navigabilityClass={
                  navigablePct >= 60
                    ? 'navigable'
                    : navigablePct >= 35
                      ? 'conditional'
                      : 'non_navigable'
                }
                size="sm"
                variant="glow"
                pulse={navigablePct < 35}
                animate
              />
            </div>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
}
