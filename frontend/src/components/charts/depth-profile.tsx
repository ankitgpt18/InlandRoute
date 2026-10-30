// ============================================================
// InlandRoute  -  Depth Profile Chart (Statsio Clean Dark UI)
// ============================================================

"use client";

import React from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";
import type { DepthProfile } from "@/types";

export interface DepthProfileChartProps {
  data?: DepthProfile | null;
  height?: number;
}

export function DepthProfileChart({
  data,
  height = 320,
}: DepthProfileChartProps) {
  if (
    !data ||
    !(data.profile_points || data.points) ||
    (data.profile_points || data.points)?.length === 0
  ) {
    return (
      <div
        className="flex items-center justify-center rounded-lg border border-neutral-800 bg-black p-6 text-neutral-400 text-xs font-medium"
        style={{ height }}
      >
        No longitudinal depth profile telemetry available
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
    <div className="rounded-lg border border-neutral-800 bg-black p-4 space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-neutral-800 pb-3">
        <div>
          <h3 className="text-sm font-semibold text-white tracking-tight">
            Longitudinal Riverbed Depth Profile
          </h3>
          <p className="text-xs text-neutral-400 mt-0.5">
            Predicted water fairway depth along {data.waterway_id} reach ({data.total_length_km ?? 1540} km)
          </p>
        </div>
        <div className="flex items-center gap-4 text-xs text-neutral-400">
          <div className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-cyan-400" />
            <span>Predicted Depth (m)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-neutral-500" />
            <span>IWAI 2.5m Standard</span>
          </div>
        </div>
      </div>

      <div style={{ height }}>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart
            data={chartPoints}
            margin={{ top: 10, right: 15, left: -15, bottom: 0 }}
          >
            <defs>
              <linearGradient id="depthFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#38bdf8" stopOpacity={0.25} />
                <stop offset="100%" stopColor="#38bdf8" stopOpacity={0.0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="2 2" stroke="#1f1f1f" vertical={false} />
            <XAxis
              dataKey="km"
              unit=" km"
              tickLine={false}
              axisLine={false}
              fontSize={11}
              tick={{ fill: "#a1a1aa" }}
            />
            <YAxis
              unit=" m"
              tickLine={false}
              axisLine={false}
              fontSize={11}
              domain={[0, "auto"]}
              tick={{ fill: "#a1a1aa" }}
            />
            <ReferenceLine
              y={2.5}
              stroke="#52525b"
              strokeDasharray="4 4"
              label={{
                value: "IWAI 2.50m LAD Standard",
                fill: "#71717a",
                fontSize: 10,
                position: "insideTopRight",
              }}
            />
            <Tooltip
              content={({ active, payload }) => {
                if (!active || !payload || !payload.length) return null;
                const point = payload[0].payload;
                return (
                  <div className="rounded-md border border-neutral-800 bg-neutral-950 p-2.5 shadow-xl text-xs space-y-1">
                    <div className="font-semibold text-white">Chainage {point.km} km</div>
                    <div className="text-[11px] text-neutral-400">Reach: {point.segmentId ?? "Main Channel"}</div>
                    <div className="font-mono text-cyan-400 font-semibold mt-1">
                      Depth: {point.depth ? point.depth.toFixed(2) : "N/A"} m
                    </div>
                  </div>
                );
              }}
            />
            <Area
              type="monotone"
              dataKey="depth"
              stroke="#38bdf8"
              strokeWidth={1.8}
              fill="url(#depthFill)"
              activeDot={{ r: 4, fill: "#fff", stroke: "#38bdf8", strokeWidth: 2 }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
