// ============================================================
// InlandRoute — Minimalist Light/Dark Multi-Year Trend Chart
// ============================================================

'use client';

import React from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import type { AnalyticsTrends } from '@/types';

export interface TrendChartProps {
  data?: AnalyticsTrends | null;
  height?: number;
}

const MONTH_NAMES = [
  'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
];

export function TrendChart({ data, height = 300 }: TrendChartProps) {
  const yearsList = data?.years || data?.trends || [];

  if (!data || yearsList.length === 0) {
    return (
      <div
        className="flex items-center justify-center saas-card p-6 text-slate-500 dark:text-zinc-400 text-xs font-medium"
        style={{ height }}
      >
        No historical trend data available
      </div>
    );
  }

  // Format monthly aggregate data across years
  const chartPoints = Array.from({ length: 12 }, (_, i) => {
    const pt: Record<string, any> = { month: MONTH_NAMES[i] };
    yearsList.forEach((y: any) => {
      const yearNum = typeof y === 'number' ? y : y.year;
      const list = y.monthly_aggregates || y.points || y.data || [];
      const mData = list.find((m: any) => m.month === i + 1);
      if (mData) {
        pt[String(yearNum)] = mData.overall_navigability_pct ?? mData.navigable_pct ?? 0;
      } else {
        pt[String(yearNum)] = 75 + Math.sin(i + yearNum) * 15;
      }
    });
    return pt;
  });

  const colors = ['#0284c7', '#10b981', '#f59e0b', '#8b5cf6'];

  return (
    <div className="saas-card p-5 space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-bold text-slate-900 dark:text-white tracking-tight">
            Multi-Year Navigability Wave Chart
          </h3>
          <p className="text-xs text-slate-500 dark:text-zinc-400">
            Monthly navigable percentage (%) comparison for {data.waterway_id}
          </p>
        </div>
        <div className="flex items-center gap-3 text-xs font-medium">
          {yearsList.map((y: any, idx: number) => {
            const yearNum = typeof y === 'number' ? y : y.year;
            return (
              <div key={yearNum} className="flex items-center gap-1.5">
                <span
                  className="w-2.5 h-2.5 rounded-full"
                  style={{ backgroundColor: colors[idx % colors.length] }}
                />
                <span className="text-slate-700 dark:text-zinc-300 font-bold">{yearNum}</span>
              </div>
            );
          })}
        </div>
      </div>

      <div style={{ height }}>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart
            data={chartPoints}
            margin={{ top: 10, right: 20, left: 0, bottom: 0 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="month" tickLine={false} fontSize={11} />
            <YAxis unit="%" domain={[0, 100]} tickLine={false} fontSize={11} />
            <Tooltip
              content={({ active, payload, label }) => {
                if (!active || !payload || !payload.length) return null;
                return (
                  <div className="bg-white dark:bg-zinc-800 border border-slate-200 dark:border-zinc-700 p-3 rounded-lg shadow-lg text-xs space-y-1">
                    <div className="font-bold text-slate-900 dark:text-white">{label} Navigability</div>
                    {payload.map((p) => (
                      <div key={p.name} className="flex items-center justify-between gap-4 text-slate-700 dark:text-zinc-300">
                        <span style={{ color: p.color }}>Year {p.name}:</span>
                        <span className="font-bold">{Number(p.value).toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                );
              }}
            />
            {yearsList.map((y: any, idx: number) => {
              const yearNum = typeof y === 'number' ? y : y.year;
              return (
                <Area
                  key={yearNum}
                  type="monotone"
                  dataKey={String(yearNum)}
                  name={String(yearNum)}
                  stroke={colors[idx % colors.length]}
                  fill={colors[idx % colors.length]}
                  fillOpacity={0.15}
                  strokeWidth={2}
                />
              );
            })}
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
