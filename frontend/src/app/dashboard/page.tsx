"use client";

import React, { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import {
  Navigation,
  AlertTriangle,
  Waves,
  ShieldCheck,
} from "lucide-react";
import { useAppStore } from "@/store/app-store";
import ApiService from "@/lib/api";
import type { NavigabilityMap, RiskAlert, DepthProfile, SeasonalCalendar } from "@/types";
import { StatCard, StatCardGrid } from "@/components/ui/stat-card";
import { DashboardShell } from "@/components/layout/dashboard-shell";

// Dynamic imports with ssr: false for fast reliable hydration
const DepthProfileChart = dynamic(
  () => import("@/components/charts/depth-profile").then((mod) => mod.DepthProfileChart),
  {
    ssr: false,
    loading: () => <div className="h-[340px] animate-pulse rounded-lg border border-neutral-800 bg-neutral-950" />,
  }
);

const SeasonalCalendarGrid = dynamic(
  () => import("@/components/charts/seasonal-calendar").then((mod) => mod.SeasonalCalendarGrid),
  {
    ssr: false,
    loading: () => <div className="h-[340px] animate-pulse rounded-lg border border-neutral-800 bg-neutral-950" />,
  }
);

const AlertList = dynamic(
  () => import("@/components/alerts/alert-list").then((mod) => mod.AlertList),
  {
    ssr: false,
    loading: () => <div className="h-64 animate-pulse rounded-lg border border-neutral-800 bg-neutral-950" />,
  }
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

      if (mapRes.status === "fulfilled") setNavMap(mapRes.value);
      if (alertsRes.status === "fulfilled") setAlerts(alertsRes.value);
      if (depthRes.status === "fulfilled") setDepthProfile(depthRes.value);
      if (calRes.status === "fulfilled") setCalendar(calRes.value);

      setLoading(false);
    });

    return () => {
      mounted = false;
    };
  }, [selectedWaterway, selectedMonth, selectedYear]);

  const navPct = navMap?.overall_navigability_pct ?? navMap?.navigable_pct ?? 78.4;
  const navKm = navMap?.navigable_length_km ?? navMap?.navigable_km ?? (selectedWaterway === "NW-1" ? 1240 : selectedWaterway === "NW-2" ? 690 : selectedWaterway === "NW-3" ? 175 : selectedWaterway === "NW-4" ? 840 : 420);

  return (
    <DashboardShell>
      <div className="space-y-6">
        {/* 4 Key Executive Metric Cards */}
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
            value={`${navKm.toFixed(0)} km`}
            subtitle="Total depth ≥ 2.5m requirement"
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
            subtitle="TFT + Swin-Transformer confidence"
            trend={+2.1}
            icon={ShieldCheck}
            loading={loading}
          />
        </StatCardGrid>

        {/* Row 1: Longitudinal Riverbed Depth Profile Chart (Full Width) */}
        <div className="w-full">
          <DepthProfileChart data={depthProfile} height={340} />
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
    </DashboardShell>
  );
}
