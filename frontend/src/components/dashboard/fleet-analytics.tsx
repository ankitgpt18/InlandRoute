"use client";

import React from "react";
import { Ship } from "lucide-react";

const fleetClasses = [
  { vessel: "Class I Barges (Draft < 1.5m)", permitted: "100%", count: 98 },
  { vessel: "Class II Vessels (Draft 1.5-2.2m)", permitted: "96.4%", count: 94 },
  { vessel: "Class III Standard (Draft 2.2-2.8m)", permitted: "89.1%", count: 86 },
  { vessel: "Heavy Bulk Carriers (Draft > 2.8m)", permitted: "74.5%", count: 72 },
  { vessel: "Container Barges (Twin Propeller)", permitted: "88.0%", count: 85 },
];

export default function FleetAnalytics() {
  const maxCount = Math.max(...fleetClasses.map((f) => f.count));

  return (
    <section className="w-full">
      <div className="relative flex h-[285px] flex-col justify-between overflow-hidden rounded-lg border border-neutral-800 bg-black shadow">
        <div>
          <div className="flex items-center justify-between border-b border-neutral-800 px-4 py-4">
            <div className="flex items-center gap-2">
              <Ship className="h-4 w-4 text-blue-400" />
              <h2 className="text-sm font-semibold text-white">Fleet Draft Clearance</h2>
            </div>
            <span className="text-xs font-semibold text-neutral-400 uppercase tracking-wider font-mono">
              PERMITTED TRANSIT
            </span>
          </div>

          <div className="p-2">
            <ul className="space-y-1">
              {fleetClasses.map((fleet) => (
                <li
                  key={fleet.vessel}
                  className="relative mx-1 my-1 flex h-8 items-center rounded-md overflow-hidden"
                >
                  <div
                    className="absolute left-0 top-0 h-full rounded-md bg-neutral-900"
                    style={{ width: `${(fleet.count / maxCount) * 100}%` }}
                  />
                  <div className="relative z-10 flex w-full items-center justify-between px-3">
                    <div className="truncate text-xs font-medium text-white max-w-[170px] sm:max-w-none">
                      {fleet.vessel}
                    </div>
                    <span className="text-xs font-mono font-bold text-emerald-400">
                      {fleet.permitted}
                    </span>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        </div>

        <div className="border-t border-neutral-800/80 px-4 py-2 text-[11px] text-neutral-500 flex justify-between">
          <span>IWAI Standard Navigation Specs</span>
          <span>DWT Range: 800 - 3,000</span>
        </div>
      </div>
    </section>
  );
}
