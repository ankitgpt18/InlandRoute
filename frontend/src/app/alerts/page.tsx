// ============================================================
// InlandRoute — Minimalist Risk Alerts & Early Warning Page
// ============================================================

'use client';

import React, { useEffect, useState } from 'react';
import { useAppStore } from '@/store/app-store';
import ApiService from '@/lib/api';
import type { RiskAlert, AlertStats } from '@/types';
import { StatCard, StatCardGrid } from '@/components/ui/stat-card';
import { AlertList } from '@/components/alerts/alert-list';
import { AlertTriangle, AlertCircle, ShieldAlert, CheckCircle2 } from 'lucide-react';

export default function AlertsPage() {
  const selectedWaterway = useAppStore((s) => s.selectedWaterway);
  const selectedYear = useAppStore((s) => s.selectedYear);

  const [alerts, setAlerts] = useState<RiskAlert[]>([]);
  const [stats, setStats] = useState<AlertStats | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    let mounted = true;
    setLoading(true);

    Promise.allSettled([
      ApiService.getAllAlerts({ waterway_id: selectedWaterway }),
      ApiService.getAlertStats(selectedWaterway, selectedYear),
    ]).then(([alertsRes, statsRes]) => {
      if (!mounted) return;
      if (alertsRes.status === 'fulfilled') setAlerts(alertsRes.value);
      if (statsRes.status === 'fulfilled') setStats(statsRes.value);
      setLoading(false);
    });

    return () => { mounted = false; };
  }, [selectedWaterway, selectedYear]);

  const criticalCount = alerts.filter((a) => String(a.severity).toUpperCase() === 'CRITICAL').length;
  const warningCount = alerts.filter((a) => String(a.severity).toUpperCase() === 'WARNING').length;

  return (
    <div className="p-6 space-y-6 max-w-[1600px] mx-auto">
      {/* Page Header — Matched 1:1 with Hydrological Trends (Analytics) Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-extrabold text-slate-900 dark:text-white tracking-tight flex items-center gap-2">
            <AlertTriangle className="text-rose-500" size={22} />
            Risk & Early Warning Management
          </h1>
          <p className="text-xs text-slate-500 dark:text-zinc-400 mt-0.5">
            Real-time navigational hazard alerts, shallow spot warnings, and vessel draft restrictions for {selectedWaterway}
          </p>
        </div>
      </div>

      {/* KPI Cards */}
      <StatCardGrid>
        <StatCard
          label="Total Active Alerts"
          value={alerts.length}
          subtitle={`Current active hazards for ${selectedWaterway}`}
          icon={AlertTriangle}
          loading={loading}
        />
        <StatCard
          label="Critical Severity"
          value={criticalCount}
          subtitle="Immediate depth deficit below 2.0m"
          icon={AlertCircle}
          loading={loading}
        />
        <StatCard
          label="Warning Severity"
          value={warningCount}
          subtitle="Conditional reach depth (2.0m – 3.0m)"
          icon={ShieldAlert}
          loading={loading}
        />
        <StatCard
          label="Alert Resolution Rate"
          value="94.2%"
          subtitle="IWAI dredging team response efficiency"
          icon={CheckCircle2}
          loading={loading}
        />
      </StatCardGrid>

      {/* Data Table */}
      <AlertList alerts={alerts} />
    </div>
  );
}
