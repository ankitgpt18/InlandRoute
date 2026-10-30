"use client";

import React from "react";
import { Navigation } from "lucide-react";

const gauges = [
  { name: "CWC Patna Gandhighat", level: "+46.20 m MSL", discharge: "18,400 m³/s", pct: 95 },
  { name: "CWC Varanasi Rajghat", level: "+68.40 m MSL", discharge: "14,200 m³/s", pct: 88 },
  { name: "CWC Farakka Barrage", level: "+19.10 m MSL", discharge: "24,600 m³/s", pct: 82 },
  { name: "CWC Dhubri (NW-2)", level: "+28.30 m MSL", discharge: "21,800 m³/s", pct: 75 },
  { name: "CWC Prayagraj Sangam", level: "+79.80 m MSL", discharge: "9,800 m³/s", pct: 60 },
  { name: "CWC Bhagalpur Reach", level: "+31.50 m MSL", discharge: "19,100 m³/s", pct: 52 },
];

export default function GaugeAnalytics() {
  const maxPct = Math.max(...gauges.map((g) => g.pct));

  return (
    <section className="w-full">
      <div className="relative flex h-[365px] flex-col justify-between overflow-hidden rounded-lg border border-neutral-800 bg-black shadow">
        <div>
          <div className="flex items-center justify-between border-b border-neutral-800 px-4 py-4">
            <div className="flex items-center gap-2">
              <Navigation className="h-4 w-4 text-cyan-400" />
              <h2 className="text-sm font-semibold text-white">CWC Hydrological Gauges</h2>
            </div>
            <span className="text-xs font-semibold text-neutral-400 uppercase tracking-wider font-mono">
              STAGE & DISCHARGE
            </span>
          </div>

          <div className="p-2">
            <ul className="space-y-1">
              {gauges.map((gauge) => (
                <li
                  key={gauge.name}
                  className="relative mx-1 my-1 flex h-9 items-center rounded-md overflow-hidden"
                >
                  <div
                    className="absolute left-0 top-0 h-full rounded-md bg-neutral-900"
                    style={{ width: `${(gauge.pct / maxPct) * 100}%` }}
                  />
                  <div className="relative z-10 flex w-full items-center justify-between px-3">
                    <div className="truncate text-xs font-medium text-white max-w-[180px] sm:max-w-none">
                      {gauge.name}
                    </div>
                    <div className="flex items-center gap-2.5">
                      <span className="text-xs font-mono font-bold text-emerald-400">
                        {gauge.level}
                      </span>
                      <span className="text-[10px] text-neutral-400 font-mono">
                        {gauge.discharge}
                      </span>
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        </div>

        <div className="border-t border-neutral-800/80 px-4 py-2.5 text-xs text-neutral-400 flex items-center justify-between">
          <span>Central Water Commission Telemetry Feed</span>
          <span className="font-mono text-emerald-400">100% Station Sync</span>
        </div>
      </div>
    </section>
  );
}
