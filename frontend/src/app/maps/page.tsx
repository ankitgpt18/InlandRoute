"use client";

import React, { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import { motion, AnimatePresence } from "framer-motion";
import { X, Activity } from "lucide-react";
import { useAppStore } from "@/store/app-store";
import ApiService from "@/lib/api";
import type { NavigabilityMap, NavigabilityPrediction } from "@/types";
import { NavigabilityBadge } from "@/components/ui/navigability-badge";
import { DashboardShell } from "@/components/layout/dashboard-shell";

// Dynamic import for Mapbox / Leaflet component
const RiverMap = dynamic(
  () => import("@/components/maps/river-map").then((mod) => mod.RiverMap),
  {
    ssr: false,
    loading: () => (
      <div className="w-full h-full min-h-[600px] bg-neutral-950 flex items-center justify-center text-xs font-semibold text-neutral-400">
        Loading Mapbox GeoJSON Engine...
      </div>
    ),
  }
);

export default function MapsPage() {
  const selectedWaterway = useAppStore((s) => s.selectedWaterway);
  const selectedMonth = useAppStore((s) => s.selectedMonth);
  const selectedYear = useAppStore((s) => s.selectedYear);
  const selectedSegmentId = useAppStore((s) => s.selectedSegmentId);
  const setSelectedSegmentId = useAppStore((s) => s.setSelectedSegmentId);

  const [navMap, setNavMap] = useState<NavigabilityMap | null>(null);
  const [segmentDetail, setSegmentDetail] = useState<NavigabilityPrediction | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    let mounted = true;
    setLoading(true);

    ApiService.getNavigabilityMap(selectedWaterway, selectedMonth, selectedYear)
      .then((data) => {
        if (mounted) {
          setNavMap(data);
          setLoading(false);
        }
      })
      .catch(() => {
        if (mounted) setLoading(false);
      });

    return () => {
      mounted = false;
    };
  }, [selectedWaterway, selectedMonth, selectedYear]);

  useEffect(() => {
    if (!selectedSegmentId) {
      setSegmentDetail(null);
      return;
    }

    let mounted = true;
    ApiService.getSegmentPrediction(
      selectedWaterway,
      selectedSegmentId,
      selectedMonth,
      selectedYear
    )
      .then((pred) => {
        if (mounted) setSegmentDetail(pred);
      })
      .catch(() => {
        const found = navMap?.predictions.find(
          (p) => p.segment_id === selectedSegmentId
        );
        if (mounted && found) setSegmentDetail(found);
      });

    return () => {
      mounted = false;
    };
  }, [selectedWaterway, selectedSegmentId, selectedMonth, selectedYear, navMap]);

  const predictions = navMap?.predictions ?? [];
  const navPct =
    navMap?.overall_navigability_pct ?? navMap?.navigable_pct ?? 0;

  return (
    <DashboardShell>
      <div className="space-y-4">
        {/* Floating Telemetry Stats Bar */}
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-neutral-800 bg-black p-3.5">
          <div className="flex items-center gap-3 text-xs">
            <span className="flex items-center gap-1.5 font-semibold text-white">
              <Activity size={14} className="text-neutral-300" />
              {selectedWaterway} Route Telemetry
            </span>
            <span className="text-neutral-700">|</span>
            <span className="font-semibold text-emerald-400">
              {navPct.toFixed(1)}% Navigable
            </span>
            <span className="text-neutral-700">|</span>
            <span className="text-neutral-400">
              {predictions.length} Monitored River Reaches
            </span>
          </div>

          <div className="flex items-center gap-2 text-[11px] text-neutral-400">
            <span className="inline-block h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
            <span>Interactive Segment Inspection Enabled</span>
          </div>
        </div>

        {/* Main Map Container */}
        <div className="relative h-[650px] w-full overflow-hidden rounded-lg border border-neutral-800 bg-black">
          <RiverMap navMap={navMap} loading={loading} />

          {/* Slide-over River Segment Inspector Drawer */}
          <AnimatePresence>
            {selectedSegmentId && (
              <motion.div
                initial={{ x: "100%" }}
                animate={{ x: 0 }}
                exit={{ x: "100%" }}
                transition={{ type: "spring", damping: 25, stiffness: 200 }}
                className="absolute right-0 top-0 z-30 flex h-full w-96 flex-col space-y-5 overflow-y-auto border-l border-neutral-800 bg-black/95 p-6 shadow-2xl backdrop-blur-md thin-scrollbar"
              >
                {/* Header */}
                <div className="flex items-start justify-between border-b border-neutral-800 pb-4">
                  <div>
                    <span className="text-xs font-semibold uppercase tracking-wider text-neutral-400">
                      Segment Telemetry Inspector
                    </span>
                    <h2 className="text-xl font-bold text-white">
                      {selectedSegmentId}
                    </h2>
                  </div>
                  <button
                    onClick={() => setSelectedSegmentId(null)}
                    className="rounded border border-neutral-800 bg-neutral-900 p-1.5 text-neutral-400 hover:text-white transition-colors"
                  >
                    <X size={15} />
                  </button>
                </div>

                {/* Status & Prediction Details */}
                {segmentDetail ? (
                  <div className="space-y-5 text-xs">
                    {/* Status Badge */}
                    <div className="flex items-center justify-between">
                      <span className="font-medium text-neutral-400">
                        Predicted Status
                      </span>
                      <NavigabilityBadge
                        navigabilityClass={segmentDetail.navigability_class}
                        size="lg"
                      />
                    </div>

                    {/* Numerical Physical Metrics */}
                    <div className="grid grid-cols-2 gap-3">
                      <div className="rounded-md border border-neutral-800 bg-neutral-950 p-3">
                        <div className="text-[11px] font-medium text-neutral-400 mb-1">
                          Predicted Depth
                        </div>
                        <div className="font-mono text-xl font-semibold text-white">
                          {segmentDetail.predicted_depth_m.toFixed(2)} m
                        </div>
                      </div>

                      <div className="rounded-md border border-neutral-800 bg-neutral-950 p-3">
                        <div className="text-[11px] font-medium text-neutral-400 mb-1">
                          Channel Width
                        </div>
                        <div className="font-mono text-xl font-semibold text-white">
                          {(segmentDetail.width_m ?? 50).toFixed(0)} m
                        </div>
                      </div>
                    </div>

                    {/* Risk Score */}
                    <div className="rounded-md border border-neutral-800 bg-neutral-950 p-4 space-y-2">
                      <div className="flex justify-between font-medium">
                        <span className="text-neutral-300">Composite Risk Index</span>
                        <span className="font-mono text-emerald-400 font-semibold">
                          {((segmentDetail.risk_score ?? 0.1) * 100).toFixed(0)} / 100
                        </span>
                      </div>
                      <div className="h-1.5 w-full overflow-hidden rounded-full bg-neutral-900">
                        <div
                          className="h-full rounded-full bg-emerald-400"
                          style={{
                            width: `${(segmentDetail.risk_score ?? 0.1) * 100}%`,
                          }}
                        />
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="py-12 text-center text-xs text-neutral-400">
                    Loading segment prediction telemetry...
                  </div>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </DashboardShell>
  );
}
