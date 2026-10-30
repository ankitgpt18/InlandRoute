"use client";

import React from "react";
import { ShieldCheck } from "lucide-react";

const sectors = [
  { name: "Farakka to Haldia (Tidal)", ratio: "98.2%", score: 98 },
  { name: "Patna to Barh Reach", ratio: "92.4%", score: 92 },
  { name: "Varanasi to Ghazipur", ratio: "88.6%", score: 88 },
  { name: "Prayagraj to Varanasi", ratio: "82.1%", score: 82 },
  { name: "Dhubri to Pandu (NW-2)", ratio: "86.5%", score: 86 },
];

export default function SectorAnalytics() {
  const maxScore = Math.max(...sectors.map((s) => s.score));

  return (
    <section className="w-full">
      <div className="relative flex h-[285px] flex-col justify-between overflow-hidden rounded-lg border border-neutral-800 bg-black shadow">
        <div>
          <div className="flex items-center justify-between border-b border-neutral-800 px-4 py-4">
            <div className="flex items-center gap-2">
              <ShieldCheck className="h-4 w-4 text-cyan-400" />
              <h2 className="text-sm font-semibold text-white">Waterway Sectors</h2>
            </div>
            <span className="text-xs font-semibold text-neutral-400 uppercase tracking-wider font-mono">
              NAVIGABLE RATIO
            </span>
          </div>

          <div className="p-2">
            <ul className="space-y-1">
              {sectors.map((sector) => (
                <li
                  key={sector.name}
                  className="relative mx-1 my-1 flex h-8 items-center rounded-md overflow-hidden"
                >
                  <div
                    className="absolute left-0 top-0 h-full rounded-md bg-neutral-900"
                    style={{ width: `${(sector.score / maxScore) * 100}%` }}
                  />
                  <div className="relative z-10 flex w-full items-center justify-between px-3">
                    <div className="truncate text-xs font-medium text-white max-w-[170px] sm:max-w-none">
                      {sector.name}
                    </div>
                    <span className="text-xs font-mono font-bold text-neutral-200">
                      {sector.ratio}
                    </span>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        </div>

        <div className="border-t border-neutral-800/80 px-4 py-2 text-[11px] text-neutral-500 flex justify-between">
          <span>Corridor Reach Division</span>
          <span>10 km Operational Slices</span>
        </div>
      </div>
    </section>
  );
}
