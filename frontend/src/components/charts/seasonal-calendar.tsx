// ============================================================
// InlandRoute — Minimalist Light/Dark Seasonal Calendar Component
// ============================================================

'use client';

import React from 'react';
import type { SeasonalCalendar } from '@/types';

export interface SeasonalCalendarProps {
  data?: SeasonalCalendar | null;
}

const MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

export function SeasonalCalendarGrid({ data }: SeasonalCalendarProps) {
  const rows = data?.segment_outlooks || data?.rows || [];

  if (!data || rows.length === 0) {
    return (
      <div className="saas-card p-6 text-slate-500 dark:text-zinc-400 text-xs font-medium text-center">
        No seasonal calendar outlook available for {data?.waterway_id ?? 'this waterway'}.
      </div>
    );
  }

  return (
    <div className="saas-card p-5 space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-bold text-slate-900 dark:text-white tracking-tight">
            12-Month Navigability Calendar
          </h3>
          <p className="text-xs text-slate-500 dark:text-zinc-400">
            Waterway reach segment outlook across months for {data.waterway_id} ({data.year})
          </p>
        </div>
        <div className="flex items-center gap-3 text-xs font-semibold">
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
            <span className="text-slate-700 dark:text-zinc-300">Navigable</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-amber-500" />
            <span className="text-slate-700 dark:text-zinc-300">Conditional</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-rose-500" />
            <span className="text-slate-700 dark:text-zinc-300">Non-Navigable</span>
          </div>
        </div>
      </div>

      <div className="overflow-x-auto thin-scrollbar">
        <table className="saas-table">
          <thead>
            <tr>
              <th className="w-32">Segment</th>
              {MONTH_NAMES.map((m) => (
                <th key={m} className="text-center">{m}</th>
              ))}
              <th className="text-right">Annual %</th>
            </tr>
          </thead>
          <tbody>
            {rows.slice(0, 15).map((row) => (
              <tr key={row.segment_id} className="hover:bg-slate-50 dark:hover:bg-zinc-800/80 transition-colors">
                <td className="font-semibold text-slate-900 dark:text-white">{row.segment_id}</td>
                {(row.monthly_outlooks ?? row.months ?? []).map((m) => {
                  const cls = String(m.navigability_class).toLowerCase();
                  const bg =
                    cls === 'navigable'
                      ? 'bg-emerald-500/15 border-emerald-500/30 text-emerald-700 dark:text-emerald-400'
                      : cls === 'conditional'
                      ? 'bg-amber-500/15 border-amber-500/30 text-amber-700 dark:text-amber-400'
                      : 'bg-rose-500/15 border-rose-500/30 text-rose-700 dark:text-rose-400';

                  return (
                    <td key={m.month} className="p-1 text-center">
                      <div
                        className={`py-1 rounded text-[11px] font-bold border ${bg}`}
                        title={`Segment ${row.segment_id} (${MONTH_NAMES[m.month - 1]}): ${m.predicted_depth_m.toFixed(2)}m depth`}
                      >
                        {m.predicted_depth_m.toFixed(1)}m
                      </div>
                    </td>
                  );
                })}
                <td className="text-right font-bold text-slate-900 dark:text-white">
                  {(row.annual_navigability_pct ?? 0).toFixed(0)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
