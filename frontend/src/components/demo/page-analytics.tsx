"use client";

import React from "react";

const shoals = [
  { name: "Varanasi Shoal #4", lad: "2.10 m", risk: "Critical", score: 92 },
  { name: "Kahalgaon Braided Bend", lad: "2.25 m", risk: "Caution", score: 78 },
  { name: "Barh Cutoff Channel", lad: "2.35 m", risk: "Caution", score: 65 },
  { name: "Ghazipur Downstream", lad: "2.60 m", risk: "Safe", score: 45 },
];

export default function PagesAnalytics() {
  return (
    <section>
      <div className="overflow-hidden rounded-lg border border-neutral-800 bg-black shadow">
        <div className="flex items-center justify-between border-b border-neutral-800 px-4 py-4">
          <h2 className="text-sm font-semibold text-white">Critical Fairway Shoals</h2>
          <span className="text-xs font-semibold text-neutral-400">
            LEAST AVAILABLE DEPTH
          </span>
        </div>
        <div className="relative p-2">
          <ul className="space-y-1">
            {shoals.map((shoal) => (
              <li
                key={shoal.name}
                className="relative flex h-8 items-center rounded-md"
              >
                <div
                  className="absolute left-0 top-0 h-full rounded-md bg-neutral-900"
                  style={{ width: `${shoal.score}%` }}
                />
                <div className="relative z-10 flex w-full items-center justify-between px-3">
                  <div className="truncate text-xs font-medium text-white">
                    {shoal.name}
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-mono font-semibold text-cyan-300">
                      {shoal.lad}
                    </span>
                    <span className="text-[10px] text-neutral-400">
                      {shoal.risk}
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
