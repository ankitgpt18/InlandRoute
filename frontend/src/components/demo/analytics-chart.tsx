"use client";

import React from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  ResponsiveContainer,
  CartesianGrid,
  Area,
  ReferenceLine,
  ReferenceDot,
} from "recharts";
import { ArrowUp, Waves, Navigation } from "lucide-react";
import { MdOutlineKeyboardArrowDown } from "react-icons/md";

const chartData = [
  { station: "Prayagraj", depth: 2.1 },
  { station: "Varanasi", depth: 2.4 },
  { station: "Buxar", depth: 2.6 },
  { station: "Patna", depth: 2.9 },
  { station: "Munger", depth: 3.2 },
  { station: "Bhagalpur", depth: 3.1 },
  { station: "Farakka", depth: 3.8 },
  { station: "Haldia", depth: 4.5 },
];

export default function StaticAnalyticsChart() {
  const patnaPoint = chartData.find((d) => d.station === "Patna");

  return (
    <div className="pointer-events-none w-full overflow-hidden rounded-xl border border-neutral-800 bg-black">
      <div className="grid grid-cols-1 divide-y divide-zinc-800">
        <div className="flex flex-row items-start justify-between">
          <div className="flex">
            <div className="border-r border-zinc-800 pt-5">
              <div className="flex w-52 flex-col items-start border-b-2 border-transparent">
                <div className="ml-4 pb-3">
                  <p className="flex items-center gap-1.5 text-xs font-medium uppercase tracking-wider text-zinc-400">
                    <Waves className="h-3.5 w-3.5 text-cyan-400" />
                    Average Fairway LAD
                  </p>
                  <div className="mt-1 flex items-center space-x-3">
                    <p className="text-3xl font-semibold text-white">2.85 m</p>
                    <span className="flex items-center rounded bg-emerald-950 px-2 py-0.5 text-xs font-semibold text-emerald-400">
                      <ArrowUp className="h-3 w-3 mr-0.5" />
                      Safe
                    </span>
                  </div>
                </div>
              </div>
            </div>

            <div className="border-r border-zinc-800 pt-5">
              <div className="flex w-52 flex-col items-start border-b-2 border-zinc-300">
                <div className="ml-4 pb-3">
                  <p className="flex items-center gap-1.5 text-xs font-medium uppercase tracking-wider text-zinc-400">
                    <Navigation className="h-3.5 w-3.5 text-blue-400" />
                    Navigable Corridor
                  </p>
                  <div className="mt-1 flex items-center space-x-3">
                    <p className="text-3xl font-semibold text-white">96.4%</p>
                    <span className="flex items-center rounded bg-emerald-950 px-2 py-0.5 text-xs font-semibold text-emerald-400">
                      <ArrowUp className="h-3 w-3 mr-0.5" />
                      Active
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="p-3">
            <div className="flex w-[160px] items-center justify-between rounded-md border border-neutral-800 bg-neutral-950 px-3 py-1.5 text-[12px] text-neutral-300">
              <span>Sentinel-2 Pass (Oct)</span>
              <MdOutlineKeyboardArrowDown className="h-4 w-4" />
            </div>
          </div>
        </div>

        <div className="relative h-[380px] w-full p-4">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={chartData}
              margin={{ top: 30, right: 30, left: -10, bottom: 20 }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                vertical={false}
                stroke="#222222"
              />

              <defs>
                <linearGradient id="colorDepth" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#38bdf8" stopOpacity={0.4} />
                  <stop offset="50%" stopColor="#38bdf8" stopOpacity={0.1} />
                  <stop offset="100%" stopColor="#38bdf8" stopOpacity={0} />
                </linearGradient>
              </defs>

              <XAxis
                dataKey="station"
                tick={{ fill: "#9ca3af", fontSize: 11 }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                domain={[1.5, 5]}
                tick={{ fill: "#9ca3af", fontSize: 11 }}
                axisLine={false}
                tickLine={false}
                unit="m"
              />

              {/* Safe Navigation Threshold Line (2.5m standard draft) */}
              <ReferenceLine
                y={2.5}
                stroke="#f59e0b"
                strokeDasharray="4 4"
                label={{
                  value: "Draft Threshold (2.5m)",
                  fill: "#f59e0b",
                  fontSize: 10,
                  position: "insideTopRight",
                }}
              />

              <Area
                type="monotone"
                dataKey="depth"
                stroke="none"
                fill="url(#colorDepth)"
                fillOpacity={0.8}
                isAnimationActive={false}
              />
              <Line
                type="monotone"
                dataKey="depth"
                stroke="#38bdf8"
                strokeWidth={2.5}
                dot={{ r: 3, fill: "#38bdf8" }}
                activeDot={{
                  r: 6,
                  fill: "#38bdf8",
                  stroke: "#fff",
                  strokeWidth: 2,
                }}
                isAnimationActive={false}
              />
              {patnaPoint && (
                <>
                  <ReferenceLine x={patnaPoint.station} stroke="#52525b" />
                  <ReferenceDot
                    x={patnaPoint.station}
                    y={patnaPoint.depth}
                    r={6}
                    fill="#38bdf8"
                    stroke="#fff"
                    strokeWidth={2}
                  />
                </>
              )}
            </LineChart>
          </ResponsiveContainer>

          <div className="absolute left-[44%] top-20 rounded-md border border-neutral-800 bg-neutral-950/95 p-2.5 shadow-xl backdrop-blur-sm">
            <div className="flex items-center gap-2 font-medium">
              <span className="h-2 w-2 rounded-full bg-cyan-400"></span>
              <p className="text-xs text-zinc-300">Patna Fairway LAD</p>
              <p className="text-xs font-semibold text-white">2.90 m</p>
            </div>
            <p className="mt-1 text-[11px] text-zinc-500">Class III Vessel Navigable</p>
          </div>
        </div>
      </div>
    </div>
  );
}
