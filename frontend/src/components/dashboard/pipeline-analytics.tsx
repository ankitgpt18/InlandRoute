"use client";

import React from "react";
import { Database } from "lucide-react";

const pipelineFeeds = [
  { feed: "Sentinel-2 MSI Optical Swath", latency: "5-day Revisit", status: "Nominal", score: 99 },
  { feed: "CWC River Stage Telemetry", latency: "1-hour Polling", status: "Active", score: 100 },
  { feed: "AIS Transponder Vessel Position", latency: "Real-Time (10s)", status: "Active", score: 96 },
  { feed: "PostGIS Hydrodynamic Database", latency: "42 ms", status: "Healthy", score: 98 },
  { feed: "Swin-Transformer Deep Inference", latency: "1.2 s per tile", status: "Ready", score: 94 },
];

export default function PipelineAnalytics() {
  const maxScore = Math.max(...pipelineFeeds.map((p) => p.score));

  return (
    <section className="w-full">
      <div className="relative flex h-[285px] flex-col justify-between overflow-hidden rounded-lg border border-neutral-800 bg-black shadow">
        <div>
          <div className="flex items-center justify-between border-b border-neutral-800 px-4 py-4">
            <div className="flex items-center gap-2">
              <Database className="h-4 w-4 text-emerald-400" />
              <h2 className="text-sm font-semibold text-white">Telemetry Pipeline</h2>
            </div>
            <span className="text-xs font-semibold text-neutral-400 uppercase tracking-wider font-mono">
              STATUS & CADENCE
            </span>
          </div>

          <div className="p-2">
            <ul className="space-y-1">
              {pipelineFeeds.map((item) => (
                <li
                  key={item.feed}
                  className="relative mx-1 my-1 flex h-8 items-center rounded-md overflow-hidden"
                >
                  <div
                    className="absolute left-0 top-0 h-full rounded-md bg-neutral-900"
                    style={{ width: `${(item.score / maxScore) * 100}%` }}
                  />
                  <div className="relative z-10 flex w-full items-center justify-between px-3">
                    <div className="truncate text-xs font-medium text-white max-w-[170px] sm:max-w-none">
                      {item.feed}
                    </div>
                    <span className="text-xs font-mono font-bold text-emerald-400">
                      {item.latency}
                    </span>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        </div>

        <div className="border-t border-neutral-800/80 px-4 py-2 text-[11px] text-neutral-500 flex justify-between">
          <span>Google Earth Engine Engine Integration</span>
          <span className="text-emerald-400">100% Online</span>
        </div>
      </div>
    </section>
  );
}
