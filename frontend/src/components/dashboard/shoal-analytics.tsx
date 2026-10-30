"use client";

import React from "react";
import { AlertCircle } from "lucide-react";

const shoals = [
  { name: "Varanasi Shoal #4", lad: "2.10 m", status: "Critical", score: 92, notice: "Dredger Assigned" },
  { name: "Kahalgaon Braided Bend", lad: "2.25 m", status: "Caution", score: 78, notice: "Daylight Transit" },
  { name: "Barh Cutoff Channel", lad: "2.35 m", status: "Caution", score: 68, notice: "One-Way Traffic" },
  { name: "Ghazipur Downstream", lad: "2.60 m", status: "Safe", score: 48, notice: "Standard Fairway" },
  { name: "Mirzapur Meander", lad: "2.70 m", status: "Safe", score: 40, notice: "Standard Fairway" },
  { name: "Patna Digha Reach", lad: "2.90 m", status: "Safe", score: 30, notice: "Unrestricted" },
];

export default function ShoalAnalytics() {
  const maxScore = Math.max(...shoals.map((s) => s.score));

  return (
    <section className="w-full">
      <div className="relative flex h-[365px] flex-col justify-between overflow-hidden rounded-lg border border-neutral-800 bg-black">
        <div>
          <div className="flex items-center justify-between border-b border-neutral-800 px-4 py-3.5">
            <div className="flex items-center gap-2">
              <AlertCircle className="h-4 w-4 text-neutral-400" />
              <h2 className="text-sm font-semibold text-white">Critical Fairway Shoals</h2>
            </div>
            <span className="text-[11px] font-semibold text-neutral-400 uppercase tracking-wider font-mono">
              LEAST AVAILABLE DEPTH
            </span>
          </div>

          <div className="p-2">
            <ul className="space-y-1">
              {shoals.map((shoal) => (
                <li
                  key={shoal.name}
                  className="relative mx-1 my-1 flex h-9 items-center rounded-md overflow-hidden"
                >
                  <div
                    className="absolute left-0 top-0 h-full rounded-md bg-neutral-900/70"
                    style={{ width: `${(shoal.score / maxScore) * 100}%` }}
                  />
                  <div className="relative z-10 flex w-full items-center justify-between px-3">
                    <div className="truncate text-xs font-medium text-white max-w-[180px] sm:max-w-none">
                      {shoal.name}
                    </div>
                    <div className="flex items-center gap-3">
                      <span className="text-xs font-mono font-semibold text-neutral-200">
                        {shoal.lad}
                      </span>
                      <span className="inline-flex items-center gap-1.5 text-[11px] font-medium text-neutral-400">
                        <span
                          className={`h-1.5 w-1.5 rounded-full ${
                            shoal.status === "Critical"
                              ? "bg-rose-500"
                              : shoal.status === "Caution"
                              ? "bg-amber-400"
                              : "bg-neutral-600"
                          }`}
                        />
                        <span>{shoal.notice}</span>
                      </span>
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        </div>

        <div className="border-t border-neutral-800 px-4 py-2.5 text-xs text-neutral-400 flex items-center justify-between">
          <span>Continuous Bathymetric Inversion</span>
          <span className="font-mono text-neutral-300">Class III UKC Margin: 0.30m</span>
        </div>
      </div>
    </section>
  );
}
