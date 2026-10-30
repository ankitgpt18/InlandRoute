"use client";

import React, { useEffect, useState } from "react";
import { useAppStore } from "@/store/app-store";
import ApiService from "@/lib/api";
import type { RiskAlert, AlertStats } from "@/types";
import { StatCard, StatCardGrid } from "@/components/ui/stat-card";
import { AlertList } from "@/components/alerts/alert-list";
import { DashboardShell } from "@/components/layout/dashboard-shell";
import { AlertTriangle, AlertCircle, ShieldAlert, CheckCircle2 } from "lucide-react";

export default function AlertsPage() {
  const selectedWaterway = useAppStore((s) => s.selectedWaterway);
  const selectedYear = useAppStore((s) => s.selectedYear);

  const [alerts, setAlerts] = useState<RiskAlert[]>([]);
  const [, setStats] = useState<AlertStats | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    let mounted = true;
    setLoading(true);

    Promise.allSettled([
      ApiService.getAllAlerts({ waterway_id: selectedWaterway }),
      ApiService.getAlertStats(selectedWaterway, selectedYear),
    ]).then(([alertsRes, statsRes]) => {
      if (!mounted) return;
      if (alertsRes.status === "fulfilled") setAlerts(alertsRes.value);
      if (statsRes.status === "fulfilled") setStats(statsRes.value);
      setLoading(false);
    });

    return () => {
      mounted = false;
    };
  }, [selectedWaterway, selectedYear]);

  const criticalCount = alerts.filter(
    (a) => String(a.severity).toUpperCase() === "CRITICAL"
  ).length;
  const warningCount = alerts.filter(
    (a) => String(a.severity).toUpperCase() === "WARNING"
  ).length;

  return (
    <DashboardShell>
      <div className="space-y-6">
        {/* Metric KPI Cards */}
        <StatCardGrid>
          <StatCard
            label="Total Active Hazards"
            value={alerts.length}
            subtitle={`Active notices along ${selectedWaterway}`}
            icon={AlertTriangle}
            loading={loading}
          />
          <StatCard
            label="Critical Draft Deficits"
            value={criticalCount}
            subtitle="Immediate LAD deficit below 2.0m"
            icon={AlertCircle}
            loading={loading}
          />
          <StatCard
            label="Warning Draft Restrictions"
            value={warningCount}
            subtitle="Conditional navigation (2.0m to 2.8m)"
            icon={ShieldAlert}
            loading={loading}
          />
          <StatCard
            label="Corridor Clearance Rate"
            value="94.2%"
            subtitle="IWAI dredging intervention uptime"
            icon={CheckCircle2}
            loading={loading}
          />
        </StatCardGrid>

        {/* High Density Risk Alerts Table */}
        <AlertList alerts={alerts} />
      </div>
    </DashboardShell>
  );
}
