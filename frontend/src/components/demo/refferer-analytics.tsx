"use client";

import React from "react";

const gauges = [
  { name: "CWC Patna Gandhighat", level: "+46.2 m MSL", status: "Normal", pct: 94 },
  { name: "CWC Varanasi Rajghat", level: "+68.4 m MSL", status: "Moderate", pct: 82 },
  { name: "CWC Farakka Barrage", level: "+19.1 m MSL", status: "Steady", pct: 76 },
  { name: "CWC Dhubri (NW-2)", level: "+28.3 m MSL", status: "Navigable", pct: 60 },
];

export default function ReffererAnalytics() {
  return (
    <section>
      <div className="overflow-hidden rounded-lg border border-neutral-800 bg-black shadow">
        <div className="flex items-center justify-between border-b border-neutral-800 px-4 py-4">
          <h2 className="text-sm font-semibold text-white">Hydrological Stations</h2>
          <span className="text-xs font-semibold text-neutral-400">
            WATER LEVEL
          </span>
        </div>
        <div className="relative p-2">
          <ul className="space-y-1">
            {gauges.map((gauge) => (
              <li
                key={gauge.name}
                className="relative flex h-8 items-center rounded-md"
              >
                <div
                  className="absolute left-0 top-0 h-full rounded-md bg-neutral-900"
                  style={{ width: `${gauge.pct}%` }}
                />
                <div className="relative z-10 flex w-full items-center justify-between px-3">
                  <div className="truncate text-xs font-medium text-white">
                    {gauge.name}
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-mono font-semibold text-emerald-400">
                      {gauge.level}
                    </span>
                    <span className="text-[10px] text-neutral-400">
                      {gauge.status}
                    </span>
                  </div>
                </div>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
}
