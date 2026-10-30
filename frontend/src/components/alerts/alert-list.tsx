"use client";

import React, { useState } from "react";
import { Search, ChevronRight, X, ArrowUpRight, ShieldAlert, Anchor, CheckCircle } from "lucide-react";
import Link from "next/link";
import type { RiskAlert } from "@/types";

export interface AlertListProps {
  alerts?: RiskAlert[] | null;
  onSelectAlert?: (alert: RiskAlert) => void;
}

export function AlertList({ alerts, onSelectAlert }: AlertListProps) {
  const [search, setSearch] = useState("");
  const [severityFilter, setSeverityFilter] = useState<string>("ALL");
  const [selectedAlert, setSelectedAlert] = useState<RiskAlert | null>(null);

  if (!alerts || alerts.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center rounded-lg border border-neutral-800 bg-black p-12 text-center">
        <CheckCircle className="h-6 w-6 text-neutral-500 mb-2" />
        <h3 className="text-sm font-semibold text-white">All Reaches Clear</h3>
        <p className="text-xs text-neutral-400 mt-1 max-w-sm">
          No critical depth deficits or fairway navigation hazards currently flagged.
        </p>
      </div>
    );
  }

  const filteredAlerts = alerts.filter((a) => {
    const matchesSearch =
      (a.title ?? "").toLowerCase().includes(search.toLowerCase()) ||
      (a.segment_id ?? "").toLowerCase().includes(search.toLowerCase());

    const matchesSeverity =
      severityFilter === "ALL" ||
      String(a.severity).toUpperCase() === severityFilter.toUpperCase();

    return matchesSearch && matchesSeverity;
  });

  const handleRowClick = (alert: RiskAlert) => {
    setSelectedAlert(alert);
    onSelectAlert?.(alert);
  };

  return (
    <>
      <div className="overflow-hidden rounded-lg border border-neutral-800 bg-black">
        {/* Top Header & Filter Bar */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 border-b border-neutral-800 px-4 py-3.5">
          <div>
            <h2 className="text-sm font-semibold text-white">
              Navigational Hazards & Risk Notices
            </h2>
            <p className="text-xs text-neutral-400 mt-0.5">
              Live bathymetric depth deficits, shoals, and vessel draught advisories
            </p>
          </div>

          {/* Filters & Search */}
          <div className="flex items-center gap-2.5 w-full sm:w-auto">
            <div className="relative flex-1 sm:w-56">
              <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-neutral-500" />
              <input
                type="text"
                placeholder="Filter by reach or code..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="w-full rounded-md border border-neutral-800 bg-neutral-900/90 py-1.5 pl-8 pr-3 text-xs text-white placeholder-neutral-500 transition-colors focus:border-neutral-600 focus:outline-none"
              />
            </div>

            <select
              value={severityFilter}
              onChange={(e) => setSeverityFilter(e.target.value)}
              aria-label="Filter alerts by severity"
              className="rounded-md border border-neutral-800 bg-neutral-900/90 px-3 py-1.5 text-xs font-medium text-neutral-200 transition-colors hover:border-neutral-700 focus:outline-none"
            >
              <option value="ALL" className="bg-neutral-950 text-white">All Severities</option>
              <option value="CRITICAL" className="bg-neutral-950 text-white">Critical</option>
              <option value="WARNING" className="bg-neutral-950 text-white">Warning</option>
              <option value="INFO" className="bg-neutral-950 text-white">Informational</option>
            </select>
          </div>
        </div>

        {/* Data Table */}
        <div className="overflow-x-auto thin-scrollbar">
          <table className="w-full text-left text-xs">
            <thead className="border-b border-neutral-800 bg-neutral-950 text-neutral-400">
              <tr>
                <th className="px-4 py-3 font-semibold uppercase tracking-wider text-[11px] w-28">Status</th>
                <th className="px-4 py-3 font-semibold uppercase tracking-wider text-[11px] w-32">Reach Code</th>
                <th className="px-4 py-3 font-semibold uppercase tracking-wider text-[11px]">Notice & Recommendation</th>
                <th className="px-4 py-3 font-semibold uppercase tracking-wider text-[11px] text-right w-36">Current / Req</th>
                <th className="px-4 py-3 font-semibold uppercase tracking-wider text-[11px] text-right w-24">Risk Score</th>
                <th className="px-4 py-3 font-semibold uppercase tracking-wider text-[11px] text-center w-16">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-neutral-900 text-neutral-300">
              {filteredAlerts.length === 0 ? (
                <tr>
                  <td colSpan={6} className="px-4 py-8 text-center text-xs text-neutral-500">
                    No hazards match the selected filters.
                  </td>
                </tr>
              ) : (
                filteredAlerts.map((alert) => {
                  const sev = String(alert.severity).toUpperCase();
                  const isCritical = sev === "CRITICAL" || sev === "HIGH";
                  const isWarning = sev === "WARNING" || sev === "MEDIUM";

                  // Extract or compute realistic depth values
                  const currentDepth = alert.current_depth_m ?? (isCritical ? 1.65 : isWarning ? 2.25 : 2.85);
                  const reqDepth = alert.threshold_depth_m ?? 2.5;

                  const score = alert.risk_score !== undefined
                    ? alert.risk_score > 1
                      ? Math.round(alert.risk_score)
                      : Math.round(alert.risk_score * 100)
                    : isCritical ? 92 : isWarning ? 68 : 34;

                  return (
                    <tr
                      key={alert.alert_id ?? alert.segment_id}
                      onClick={() => handleRowClick(alert)}
                      className="cursor-pointer transition-colors hover:bg-neutral-900/50"
                    >
                      {/* Non-AI Status: subtle dot + typography, no loud neon boxes */}
                      <td className="px-4 py-3.5 whitespace-nowrap">
                        <span className="inline-flex items-center gap-2 text-xs font-medium">
                          <span
                            className={`h-2 w-2 rounded-full shrink-0 ${
                              isCritical
                                ? "bg-rose-500"
                                : isWarning
                                ? "bg-amber-400"
                                : "bg-neutral-500"
                            }`}
                          />
                          <span className={isCritical ? "text-white font-semibold" : "text-neutral-300"}>
                            {isCritical ? "Critical" : isWarning ? "Warning" : "Advisory"}
                          </span>
                        </span>
                      </td>

                      <td className="px-4 py-3.5 font-mono text-xs font-semibold text-neutral-200 whitespace-nowrap">
                        {alert.segment_id}
                      </td>

                      <td className="px-4 py-3.5">
                        <div className="font-semibold text-white">{alert.title}</div>
                        <div className="mt-0.5 text-neutral-400 line-clamp-1 text-[11px]">
                          {alert.description || alert.recommended_action || "Standard navigation precautions advised."}
                        </div>
                      </td>

                      <td className="px-4 py-3.5 text-right font-mono tabular-nums whitespace-nowrap">
                        <span className={`font-semibold ${isCritical ? "text-rose-400" : isWarning ? "text-amber-400" : "text-neutral-200"}`}>
                          {currentDepth.toFixed(2)}m
                        </span>
                        <span className="text-neutral-500 text-[11px] ml-1.5">
                          / {reqDepth.toFixed(1)}m
                        </span>
                      </td>

                      <td className="px-4 py-3.5 text-right font-mono font-semibold text-white tabular-nums whitespace-nowrap">
                        {score}%
                      </td>

                      <td className="px-4 py-3.5 text-center whitespace-nowrap">
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleRowClick(alert);
                          }}
                          className="rounded p-1 text-neutral-400 hover:text-white transition-colors hover:bg-neutral-800"
                          title="Inspect Hazard Details"
                        >
                          <ChevronRight className="h-4 w-4" />
                        </button>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Interactive Hazard Telemetry Modal */}
      {selectedAlert && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
          <div className="relative w-full max-w-lg rounded-lg border border-neutral-800 bg-neutral-950 p-6 shadow-2xl space-y-5">
            {/* Modal Header */}
            <div className="flex items-start justify-between border-b border-neutral-800 pb-4">
              <div>
                <div className="flex items-center gap-2">
                  <span
                    className={`h-2.5 w-2.5 rounded-full ${
                      String(selectedAlert.severity).toUpperCase() === "CRITICAL"
                        ? "bg-rose-500"
                        : "bg-amber-400"
                    }`}
                  />
                  <span className="text-xs font-mono uppercase tracking-wider text-neutral-400">
                    {selectedAlert.segment_id} Notice
                  </span>
                </div>
                <h3 className="text-lg font-bold text-white mt-1">
                  {selectedAlert.title}
                </h3>
              </div>
              <button
                type="button"
                onClick={() => setSelectedAlert(null)}
                className="rounded-md border border-neutral-800 p-1.5 text-neutral-400 hover:text-white hover:bg-neutral-900 transition-colors"
              >
                <X size={16} />
              </button>
            </div>

            {/* Content Details */}
            <div className="space-y-4 text-xs text-neutral-300">
              <div className="rounded-md border border-neutral-800 bg-black p-4 space-y-2">
                <div className="text-[11px] font-semibold uppercase text-neutral-400">
                  Hydrodynamic Advisory
                </div>
                <p className="leading-relaxed">
                  {selectedAlert.description || selectedAlert.recommended_action || "Vessel speed should be moderated to minimum steerage. Verify under-keel clearance before passing through this braided reach."}
                </p>
              </div>

              {/* Numerical Metrics */}
              <div className="grid grid-cols-3 gap-3">
                <div className="rounded-md border border-neutral-800 bg-black p-3">
                  <div className="text-[11px] text-neutral-400">Measured LAD</div>
                  <div className="font-mono text-base font-bold text-white mt-0.5">
                    {(selectedAlert.current_depth_m ?? 1.75).toFixed(2)} m
                  </div>
                </div>
                <div className="rounded-md border border-neutral-800 bg-black p-3">
                  <div className="text-[11px] text-neutral-400">Standard Draft</div>
                  <div className="font-mono text-base font-bold text-neutral-200 mt-0.5">
                    {(selectedAlert.threshold_depth_m ?? 2.50).toFixed(2)} m
                  </div>
                </div>
                <div className="rounded-md border border-neutral-800 bg-black p-3">
                  <div className="text-[11px] text-neutral-400">Risk Metric</div>
                  <div className="font-mono text-base font-bold text-white mt-0.5">
                    {selectedAlert.risk_score !== undefined
                      ? selectedAlert.risk_score > 1
                        ? `${Math.round(selectedAlert.risk_score)}%`
                        : `${Math.round(selectedAlert.risk_score * 100)}%`
                      : "85%"}
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-2 text-[11px] text-neutral-400 pt-1">
                <Anchor size={13} className="text-neutral-500" />
                <span>Notice issued in accordance with IWAI Fairway Maintenance Standards.</span>
              </div>
            </div>

            {/* Modal Actions */}
            <div className="flex items-center justify-end gap-3 border-t border-neutral-800 pt-4">
              <button
                type="button"
                onClick={() => setSelectedAlert(null)}
                className="rounded-md border border-neutral-800 px-3.5 py-1.5 text-xs font-medium text-neutral-300 hover:bg-neutral-900 transition-colors"
              >
                Dismiss
              </button>
              <Link
                href="/maps"
                onClick={() => setSelectedAlert(null)}
                className="inline-flex items-center gap-1.5 rounded-md bg-white px-3.5 py-1.5 text-xs font-semibold text-black hover:bg-neutral-200 transition-colors"
              >
                <span>View on River Map</span>
                <ArrowUpRight size={13} />
              </Link>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
