// ============================================================
// InlandRoute — Minimalist Light/Dark Risk Alert Data Table Component
// ============================================================

'use client';

import React, { useState } from 'react';
import { Search, CheckCircle, ExternalLink } from 'lucide-react';
import type { RiskAlert } from '@/types';

export interface AlertListProps {
  alerts?: RiskAlert[] | null;
  onSelectAlert?: (alert: RiskAlert) => void;
}

export function AlertList({ alerts, onSelectAlert }: AlertListProps) {
  const [search, setSearch] = useState('');
  const [severityFilter, setSeverityFilter] = useState<string>('ALL');

  if (!alerts || alerts.length === 0) {
    return (
      <div className="saas-card p-6 text-center text-slate-500 dark:text-zinc-400 text-xs font-medium space-y-2">
        <CheckCircle size={24} className="mx-auto text-emerald-500" />
        <p className="font-bold text-slate-900 dark:text-white">All Clear — No Active Risk Alerts</p>
        <p>No critical shallow spots or width restriction alerts detected.</p>
      </div>
    );
  }

  const filteredAlerts = alerts.filter((a) => {
    const matchesSearch =
      (a.title ?? '').toLowerCase().includes(search.toLowerCase()) ||
      (a.segment_id ?? '').toLowerCase().includes(search.toLowerCase());

    const matchesSeverity =
      severityFilter === 'ALL' ||
      String(a.severity).toUpperCase() === severityFilter.toUpperCase();

    return matchesSearch && matchesSeverity;
  });

  return (
    <div className="saas-card p-5 space-y-4">
      {/* Top Filter Bar */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
        <div>
          <h3 className="text-sm font-bold text-slate-900 dark:text-white tracking-tight">
            Active Risk & Early Warning Alerts
          </h3>
          <p className="text-xs text-slate-500 dark:text-zinc-400">
            Real-time depth and width restriction alerts for fleet operators
          </p>
        </div>

        {/* Filters & Search */}
        <div className="flex items-center gap-2 w-full sm:w-auto">
          {/* Search */}
          <div className="relative flex-1 sm:w-48">
            <Search size={13} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-slate-400 dark:text-zinc-500" />
            <input
              type="text"
              placeholder="Filter segment or title..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full pl-8 pr-3 py-1.5 bg-slate-50 dark:bg-zinc-800 border border-slate-200 dark:border-zinc-700 rounded-lg text-xs text-slate-900 dark:text-white focus:outline-none focus:border-slate-400 dark:focus:border-zinc-500"
            />
          </div>

          {/* Severity Tabs */}
          <select
            value={severityFilter}
            onChange={(e) => setSeverityFilter(e.target.value)}
            className="px-2.5 py-1.5 bg-slate-50 dark:bg-zinc-800 border border-slate-200 dark:border-zinc-700 rounded-lg text-xs font-semibold text-slate-700 dark:text-zinc-300 cursor-pointer focus:outline-none"
          >
            <option value="ALL">All Severities</option>
            <option value="CRITICAL">Critical Only</option>
            <option value="HIGH">High</option>
            <option value="MEDIUM">Medium</option>
            <option value="LOW">Low</option>
          </select>
        </div>
      </div>

      {/* High Density SaaS Data Table */}
      <div className="overflow-x-auto thin-scrollbar">
        <table className="saas-table">
          <thead>
            <tr>
              <th className="w-24">Severity</th>
              <th className="w-28">Segment</th>
              <th>Alert Title & Recommendation</th>
              <th className="text-right w-28">Current / Target</th>
              <th className="text-right w-24">Risk Score</th>
              <th className="text-center w-20">Action</th>
            </tr>
          </thead>
          <tbody>
            {filteredAlerts.length === 0 ? (
              <tr>
                <td colSpan={6} className="text-center py-6 text-slate-500 dark:text-zinc-400 text-xs">
                  No alerts matching the selected filters.
                </td>
              </tr>
            ) : (
              filteredAlerts.map((alert) => {
                const sev = String(alert.severity).toUpperCase();
                let sevBadge = 'badge-conditional';
                if (sev === 'CRITICAL' || sev === 'HIGH') sevBadge = 'badge-non-navigable';
                if (sev === 'LOW' || sev === 'INFO') sevBadge = 'badge-navigable';

                return (
                  <tr key={alert.alert_id ?? alert.segment_id} className="hover:bg-slate-50 dark:hover:bg-zinc-800/80 transition-colors">
                    <td>
                      <span className={`pill ${sevBadge} text-[10px]`}>
                        {sev}
                      </span>
                    </td>
                    <td className="font-bold text-slate-900 dark:text-white">{alert.segment_id}</td>
                    <td>
                      <div className="font-semibold text-slate-900 dark:text-white">{alert.title}</div>
                      <div className="text-xs text-slate-500 dark:text-zinc-400 line-clamp-1">
                        {alert.description || alert.recommended_action || 'Operational precaution recommended.'}
                      </div>
                    </td>
                    <td className="text-right font-mono text-xs tabular-nums">
                      <div className="font-bold text-slate-900 dark:text-white">
                        {alert.current_depth_m ? `${alert.current_depth_m.toFixed(2)}m` : 'N/A'}
                      </div>
                      <div className="text-[10px] text-slate-400 dark:text-zinc-500">
                        min {alert.threshold_depth_m ?? 3.0}m
                      </div>
                    </td>
                    <td className="text-right font-mono font-bold text-slate-900 dark:text-white tabular-nums">
                      {((alert.risk_score ?? 0) * 100).toFixed(0)}%
                    </td>
                    <td className="text-center">
                      <button
                        onClick={() => onSelectAlert?.(alert)}
                        className="p-1.5 rounded-lg border border-slate-200 dark:border-zinc-700 text-slate-600 dark:text-zinc-400 hover:text-slate-900 dark:hover:text-white hover:bg-slate-100 dark:hover:bg-zinc-700 shadow-sm transition-all"
                        title="View Details"
                      >
                        <ExternalLink size={13} />
                      </button>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
