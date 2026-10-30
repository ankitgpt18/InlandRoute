// ============================================================
// InlandRoute  -  12-Month Navigability Calendar (Statsio Clean Dark UI)
// ============================================================

"use client";

import React from "react";
import type { SeasonalCalendar } from "@/types";

export interface SeasonalCalendarProps {
  data?: SeasonalCalendar | null;
}

const MONTH_NAMES = [
  "Jan", "Feb", "Mar", "Apr", "May", "Jun",
  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

export function SeasonalCalendarGrid({ data }: SeasonalCalendarProps) {
  const rows = data?.segment_outlooks || data?.rows || [];

  if (!data || rows.length === 0) {
    return (
      <div className="flex items-center justify-center rounded-lg border border-neutral-800 bg-black p-6 text-neutral-400 text-xs font-medium text-center">
        No seasonal calendar outlook available for {data?.waterway_id ?? "this waterway"}.
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-neutral-800 bg-black p-4 space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-neutral-800 pb-3">
        <div>
          <h3 className="text-sm font-semibold text-white tracking-tight">
            12-Month Navigability Calendar
          </h3>
          <p className="text-xs text-neutral-400 mt-0.5">
            Waterway reach segment outlook across months for {data.waterway_id} ({data.year})
          </p>
        </div>
        <div className="flex items-center gap-3 text-xs text-neutral-400">
          <div className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-emerald-500" />
            <span>&ge; 2.5m (Navigable)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-amber-400" />
            <span>2.0 - 2.5m</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-neutral-600" />
            <span>&lt; 2.0m</span>
          </div>
        </div>
      </div>

      <div className="overflow-x-auto thin-scrollbar">
        <table className="w-full text-left text-xs">
          <thead className="border-b border-neutral-800 bg-neutral-950 text-neutral-400">
            <tr>
              <th className="px-3 py-2.5 font-semibold uppercase tracking-wider text-[11px] w-32">Segment</th>
              {MONTH_NAMES.map((m) => (
                <th key={m} className="px-1 py-2.5 text-center font-semibold uppercase tracking-wider text-[11px]">
                  {m}
                </th>
              ))}
              <th className="px-3 py-2.5 text-right font-semibold uppercase tracking-wider text-[11px] w-24">Annual %</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-neutral-900 text-neutral-300">
            {rows.slice(0, 15).map((row) => (
              <tr key={row.segment_id} className="transition-colors hover:bg-neutral-900/40">
                <td className="px-3 py-2 font-mono text-xs font-semibold text-white whitespace-nowrap">
                  {row.segment_id}
                </td>
                {(row.monthly_outlooks ?? row.months ?? []).map((m) => {
                  const cls = String(m.navigability_class).toLowerCase();
                  const isNavigable = cls === "navigable";
                  const isConditional = cls === "conditional";

                  return (
                    <td key={m.month} className="p-1 text-center">
                      <div
                        className={`py-1 rounded font-mono text-[11px] transition-colors ${
                          isNavigable
                            ? "bg-neutral-900 text-neutral-200 border border-neutral-800"
                            : isConditional
                            ? "bg-neutral-900/60 text-amber-300/90 border border-neutral-800/80"
                            : "bg-neutral-950 text-neutral-500 border border-neutral-900"
                        }`}
                        title={`Segment ${row.segment_id} (${MONTH_NAMES[m.month - 1]}): ${m.predicted_depth_m.toFixed(2)}m depth`}
                      >
                        {m.predicted_depth_m.toFixed(1)}m
                      </div>
                    </td>
                  );
                })}
                <td className="px-3 py-2 text-right font-mono font-semibold text-white tabular-nums">
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
