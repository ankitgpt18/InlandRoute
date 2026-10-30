"use client";

import React, { useState } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  ReferenceLine,
} from "recharts";
import { Waves, ShieldCheck, AlertTriangle } from "lucide-react";

interface AnalyticsChartProps {
  waterwayCode: string;
}

const RIVER_DATA: Record<
  string,
  { station: string; depth: number; flow: string }[]
> = {
  "NW-1": [
    { station: "Prayagraj", depth: 2.1, flow: "9,800 cumecs" },
    { station: "Mirzapur", depth: 2.35, flow: "11,200 cumecs" },
    { station: "Varanasi", depth: 2.45, flow: "14,200 cumecs" },
    { station: "Ghazipur", depth: 2.65, flow: "15,800 cumecs" },
    { station: "Buxar", depth: 2.7, flow: "16,400 cumecs" },
    { station: "Patna", depth: 2.9, flow: "18,400 cumecs" },
    { station: "Barh", depth: 2.8, flow: "18,900 cumecs" },
    { station: "Munger", depth: 3.2, flow: "20,100 cumecs" },
    { station: "Bhagalpur", depth: 3.1, flow: "19,100 cumecs" },
    { station: "Kahalgaon", depth: 2.95, flow: "21,500 cumecs" },
    { station: "Farakka", depth: 3.8, flow: "24,600 cumecs" },
    { station: "Haldia", depth: 4.5, flow: "Tidal Corridor" },
  ],
  "NW-2": [
    { station: "Dhubri", depth: 2.85, flow: "21,800 cumecs" },
    { station: "Jogighopa", depth: 2.95, flow: "22,400 cumecs" },
    { station: "Guwahati", depth: 3.4, flow: "25,100 cumecs" },
    { station: "Tezpur", depth: 3.15, flow: "24,000 cumecs" },
    { station: "Silghat", depth: 3.25, flow: "23,600 cumecs" },
    { station: "Neamati", depth: 2.75, flow: "20,200 cumecs" },
    { station: "Dibrugarh", depth: 2.6, flow: "18,900 cumecs" },
    { station: "Sadiya", depth: 2.3, flow: "15,400 cumecs" },
  ],
  "NW-3": [
    { station: "Kottapuram", depth: 2.4, flow: "Canal Reach" },
    { station: "Aluva", depth: 2.65, flow: "Canal Reach" },
    { station: "Kochi", depth: 3.5, flow: "Port Corridor" },
    { station: "Alappuzha", depth: 2.8, flow: "Canal Reach" },
    { station: "Kollam", depth: 2.45, flow: "Canal Reach" },
  ],
  "NW-4": [
    { station: "Kakinada", depth: 3.2, flow: "Deepwater Lock" },
    { station: "Rajahmundry", depth: 2.8, flow: "12,400 cumecs" },
    { station: "Eluru Canal", depth: 2.3, flow: "Canal Reach" },
    { station: "Vijayawada", depth: 2.9, flow: "14,800 cumecs" },
    { station: "Muktyala", depth: 2.6, flow: "10,200 cumecs" },
    { station: "Puducherry", depth: 3.1, flow: "Southern Lock" },
  ],
  "NW-5": [
    { station: "Talcher", depth: 2.2, flow: "Brahmani Upper" },
    { station: "Pankapal", depth: 2.45, flow: "8,900 cumecs" },
    { station: "Mangalgadi", depth: 2.9, flow: "11,500 cumecs" },
    { station: "Dhamra", depth: 4.2, flow: "Tidal Reach" },
    { station: "Paradip Link", depth: 4.8, flow: "Deepwater Port" },
  ],
};

export default function AnalyticsChart({ waterwayCode }: AnalyticsChartProps) {
  const chartData = RIVER_DATA[waterwayCode] ?? RIVER_DATA["NW-1"];
  const [activeTab, setActiveTab] = useState<"depth" | "ratio">("depth");

  const avgDepth = (
    chartData.reduce((acc, curr) => acc + curr.depth, 0) / chartData.length
  ).toFixed(2);

  const navigableCount = chartData.filter((d) => d.depth >= 2.5).length;
  const navigablePct = ((navigableCount / chartData.length) * 100).toFixed(1);

  return (
    <div className="w-full overflow-hidden rounded-lg border border-neutral-800 bg-black">
      {/* Metrics Row */}
      <div className="grid grid-cols-1 divide-y divide-neutral-800 sm:grid-cols-3 sm:divide-x sm:divide-y-0 border-b border-neutral-800">
        <button
          type="button"
          onClick={() => setActiveTab("depth")}
          className={`flex flex-col p-4 text-left transition-colors ${
            activeTab === "depth"
              ? "bg-neutral-900/60 border-b-2 border-b-white"
              : "hover:bg-neutral-950"
          }`}
        >
          <div className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-neutral-400">
            <Waves className="h-3.5 w-3.5 text-neutral-400" />
            <span>Average Fairway LAD</span>
          </div>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="text-2xl font-bold tracking-tight text-white font-mono">
              {avgDepth} m
            </span>
            <span className="inline-flex items-center gap-1 text-[11px] font-medium text-emerald-400">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
              Above Threshold
            </span>
          </div>
        </button>

        <button
          type="button"
          onClick={() => setActiveTab("ratio")}
          className={`flex flex-col p-4 text-left transition-colors ${
            activeTab === "ratio"
              ? "bg-neutral-900/60 border-b-2 border-b-white"
              : "hover:bg-neutral-950"
          }`}
        >
          <div className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-neutral-400">
            <ShieldCheck className="h-3.5 w-3.5 text-neutral-400" />
            <span>Navigable Ratio</span>
          </div>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="text-2xl font-bold tracking-tight text-white font-mono">
              {navigablePct}%
            </span>
            <span className="inline-flex items-center gap-1 text-[11px] font-medium text-neutral-300">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
              Operational
            </span>
          </div>
        </button>

        <div className="flex flex-col p-4 text-left">
          <div className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-neutral-400">
            <AlertTriangle className="h-3.5 w-3.5 text-neutral-400" />
            <span>Active Shoals Under Care</span>
          </div>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="text-2xl font-bold tracking-tight text-white font-mono">
              {waterwayCode === "NW-1" ? 4 : waterwayCode === "NW-2" ? 3 : 1}
            </span>
            <span className="inline-flex items-center gap-1 text-[11px] font-medium text-neutral-400">
              <span className="h-1.5 w-1.5 rounded-full bg-amber-400" />
              Dredging Active
            </span>
          </div>
        </div>
      </div>

      {/* Main Chart Area */}
      <div className="relative h-[320px] w-full p-4">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart
            data={chartData}
            margin={{ top: 15, right: 15, left: -15, bottom: 5 }}
          >
            <defs>
              <linearGradient id="fairwayGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#38bdf8" stopOpacity={0.25} />
                <stop offset="95%" stopColor="#38bdf8" stopOpacity={0.0} />
              </linearGradient>
            </defs>

            <CartesianGrid
              strokeDasharray="2 2"
              vertical={false}
              stroke="#1f1f1f"
            />

            <XAxis
              dataKey="station"
              tick={{ fill: "#a1a1aa", fontSize: 11 }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              domain={[1.5, 5]}
              tick={{ fill: "#a1a1aa", fontSize: 11 }}
              axisLine={false}
              tickLine={false}
              unit="m"
            />

            <ReferenceLine
              y={2.5}
              stroke="#52525b"
              strokeDasharray="4 4"
              label={{
                value: "IWAI Class-III Standard (2.50m)",
                fill: "#71717a",
                fontSize: 10,
                position: "insideTopRight",
              }}
            />

            <Tooltip
              content={({ active, payload, label }) => {
                if (active && payload && payload.length) {
                  const data = payload[0].payload;
                  return (
                    <div className="rounded-md border border-neutral-800 bg-neutral-950 p-2.5 shadow-xl text-xs">
                      <div className="font-semibold text-white">{label} Reach</div>
                      <div className="mt-1 font-mono text-cyan-400 font-semibold">
                        LAD: {data.depth.toFixed(2)} m
                      </div>
                      <div className="text-[11px] text-neutral-400 mt-0.5">
                        Discharge: {data.flow}
                      </div>
                    </div>
                  );
                }
                return null;
              }}
            />

            <Area
              type="monotone"
              dataKey="depth"
              stroke="#38bdf8"
              strokeWidth={2}
              fill="url(#fairwayGradient)"
              activeDot={{
                r: 4,
                fill: "#fff",
                stroke: "#38bdf8",
                strokeWidth: 2,
              }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
