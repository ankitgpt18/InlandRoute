// ============================================================
// InlandRoute — Executive Waterways Dashboard (3 Full-Width Stacked Rows)
// ============================================================

'use client';

import React, { useEffect, useState } from 'react';
import dynamic from 'next/dynamic';
import {
  Navigation,
  AlertTriangle,
  Waves,
  ShieldCheck,
} from 'lucide-react';
import { useAppStore } from '@/store/app-store';
import ApiService from '@/lib/api';
import type { NavigabilityMap, RiskAlert, DepthProfile, SeasonalCalendar } from '@/types';
import { StatCard, StatCardGrid } from '@/components/ui/stat-card';

// Dynamic imports with ssr: false for instant render
const DepthProfileChart = dynamic(
  () => import('@/components/charts/depth-profile').then((mod) => mod.DepthProfileChart),
  { ssr: false, loading: () => <div className="saas-card h-[340px] animate-pulse bg-zinc-900 rounded-xl" /> }
);

const SeasonalCalendarGrid = dynamic(
  () => import('@/components/charts/seasonal-calendar').then((mod) => mod.SeasonalCalendarGrid),
  { ssr: false, loading: () => <div className="saas-card h-[340px] animate-pulse bg-zinc-900 rounded-xl" /> }
);

const AlertList = dynamic(
  () => import('@/components/alerts/alert-list').then((mod) => mod.AlertList),
  { ssr: false, loading: () => <div className="saas-card h-64 animate-pulse bg-zinc-900 rounded-xl" /> }
);

export default function DashboardPage() {
  const selectedWaterway = useAppStore((s) => s.selectedWaterway);
  const selectedMonth = useAppStore((s) => s.selectedMonth);
  const selectedYear = useAppStore((s) => s.selectedYear);

  const [navMap, setNavMap] = useState<NavigabilityMap | null>(null);
  const [alerts, setAlerts] = useState<RiskAlert[]>([]);
  const [depthProfile, setDepthProfile] = useState<DepthProfile | null>(null);
  const [calendar, setCalendar] = useState<SeasonalCalendar | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    let mounted = true;
    setLoading(true);

    Promise.allSettled([
      ApiService.getNavigabilityMap(selectedWaterway, selectedMonth, selectedYear),
      ApiService.getRiskAlerts(selectedWaterway, selectedMonth, selectedYear),
      ApiService.getDepthProfile(selectedWaterway, selectedMonth, selectedYear),
      ApiService.getSeasonalCalendar(selectedWaterway, selectedYear),
    ]).then(([mapRes, alertsRes, depthRes, calRes]) => {
      if (!mounted) return;

      if (mapRes.status === 'fulfilled') setNavMap(mapRes.value);
      if (alertsRes.status === 'fulfilled') setAlerts(alertsRes.value);
      if (depthRes.status === 'fulfilled') setDepthProfile(depthRes.value);
      if (calRes.status === 'fulfilled') setCalendar(calRes.value);

      setLoading(false);
    });

    return () => { mounted = false; };
  }, [selectedWaterway, selectedMonth, selectedYear]);

  const navPct = navMap?.overall_navigability_pct ?? navMap?.navigable_pct ?? 0;
  const navKm = navMap?.navigable_length_km ?? navMap?.navigable_km ?? 0;

  return (
    <div className="p-6 space-y-6 max-w-[1600px] mx-auto bg-zinc-950 min-h-screen text-white">
      {/* 4 Key SaaS Metric Cards */}
      <StatCardGrid>
        <StatCard
          label="Navigability Score"
          value={`${navPct.toFixed(1)}%`}
          subtitle={`Overall ${selectedWaterway} open for 1,500 DWT vessels`}
          trend={+4.5}
          icon={Navigation}
          loading={loading}
        />
        <StatCard
          label="Navigable Reach"
          value={navKm ? `${navKm.toFixed(0)} km` : `${navMap?.navigable_count ?? 0} seg`}
          subtitle={`Total depth ≥ 3.0m requirement`}
          trend={+12.0}
          icon={Waves}
          loading={loading}
        />
        <StatCard
          label="Active Risk Warnings"
          value={alerts.length}
          subtitle="Shallow spots & width restrictions"
          trend={alerts.length > 0 ? -15.0 : 0}
          icon={AlertTriangle}
          loading={loading}
        />
        <StatCard
          label="Hydrological Safety Index"
          value="0.94"
          subtitle="TFT + Swin-Transformer prediction confidence"
          trend={+2.1}
          icon={ShieldCheck}
          loading={loading}
        />
      </StatCardGrid>

      {/* Row 1: Longitudinal Riverbed Depth Profile Chart (Full Width) */}
      <div className="w-full">
        <DepthProfileChart data={depthProfile} height={320} />
      </div>

      {/* Row 2: 12-Month Navigability Calendar (Full Width) */}
      <div className="w-full">
        <SeasonalCalendarGrid data={calendar} />
      </div>

      {/* Row 3: Active Risk & Early Warning Alerts Table (Full Width) */}
      <div className="w-full">
        <AlertList alerts={alerts} />
      </div>
    </div>
  );
}
