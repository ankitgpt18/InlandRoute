// ============================================================
// InlandRoute — Minimalist Light/Dark Depth Profile Chart
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
  ReferenceLine,
  ResponsiveContainer,
} from 'recharts';
import type { DepthProfile } from '@/types';

export interface DepthProfileChartProps {
  data?: DepthProfile | null;
  height?: number;
}

export function DepthProfileChart({ data, height = 300 }: DepthProfileChartProps) {
  if (!data || !(data.profile_points || data.points) || (data.profile_points || data.points)?.length === 0) {
    return (
      <div
        className="flex items-center justify-center saas-card p-6 text-slate-500 dark:text-zinc-400 text-xs font-medium"
        style={{ height }}
      >
        No longitudinal depth profile data available
      </div>
    );
  }

  const pointsList = data.profile_points || data.points || [];

  const chartPoints = pointsList.map((pt) => ({
    km: pt.chainage_km ?? pt.km,
    depth: pt.predicted_depth_m ?? pt.depth_m,
    lowerCI: pt.depth_lower_ci,
    upperCI: pt.depth_upper_ci,
    segmentId: pt.segment_id,
  }));

  return (
    <div className="saas-card p-5 space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-bold text-slate-900 dark:text-white tracking-tight">
            Longitudinal Riverbed Depth Profile
          </h3>
          <p className="text-xs text-slate-500 dark:text-zinc-400">
            Predicted water depth along {data.waterway_id} reach ({data.total_length_km ?? 1540} km)
          </p>
        </div>
        <div className="flex items-center gap-4 text-xs font-medium text-slate-600 dark:text-zinc-400">
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-sky-500" />
            <span>Predicted Depth (m)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
            <span>IWAI 3.0m Target</span>
          </div>
        </div>
      </div>

      <div style={{ height }}>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart
            data={chartPoints}
            margin={{ top: 10, right: 20, left: 0, bottom: 0 }}
          >
            <defs>
              <linearGradient id="depthFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#0284c7" stopOpacity={0.4} />
                <stop offset="100%" stopColor="#0284c7" stopOpacity={0.05} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="km"
              unit=" km"
              tickLine={false}
              fontSize={11}
            />
            <YAxis
              unit=" m"
              tickLine={false}
              fontSize={11}
              domain={[0, 'auto']}
            />
            <Tooltip
              content={({ active, payload }) => {
                if (!active || !payload || !payload.length) return null;
                const point = payload[0].payload;
                return (
                  <div className="bg-white dark:bg-zinc-800 border border-slate-200 dark:border-zinc-700 p-3 rounded-lg shadow-lg text-xs space-y-1">
                    <div className="font-bold text-slate-900 dark:text-white">
                      Segment {point.segmentId} (km {point.km})
                    </div>
                    <div className="text-slate-700 dark:text-zinc-300">
                      Depth: <span className="font-bold">{point.depth?.toFixed(2)} m</span>
                    </div>
                    {point.lowerCI !== undefined && (
                      <div className="text-slate-500 dark:text-zinc-400 text-[11px]">
                        90% CI: {point.lowerCI?.toFixed(2)}m – {point.upperCI?.toFixed(2)}m
                      </div>
                    )}
                  </div>
                );
              }}
            />
            <ReferenceLine
              y={3.0}
              stroke="#10b981"
              strokeDasharray="4 4"
              label={{ value: '3.0m Threshold', fill: '#10b981', fontSize: 10, position: 'insideTopRight' }}
            />
            <Area
              type="monotone"
              dataKey="depth"
              stroke="#0284c7"
              strokeWidth={2}
              fill="url(#depthFill)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
