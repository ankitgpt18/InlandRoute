// ============================================================
// InlandRoute — Minimalist Pure Dark River Navigability Map Page
// ============================================================

'use client';

import React, { useEffect, useState } from 'react';
import dynamic from 'next/dynamic';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Activity } from 'lucide-react';
import { useAppStore } from '@/store/app-store';
import ApiService from '@/lib/api';
import type { NavigabilityMap, NavigabilityPrediction } from '@/types';
import { NavigabilityBadge } from '@/components/ui/navigability-badge';

// Dynamic import for Mapbox GL component with ssr: false for fast page load
const RiverMap = dynamic(
  () => import('@/components/maps/river-map').then((mod) => mod.RiverMap),
  {
    ssr: false,
    loading: () => (
      <div className="w-full h-full bg-zinc-950 flex items-center justify-center text-xs font-semibold text-zinc-400">
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
  const [filterClass, setFilterClass] = useState<string>('ALL');

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

    return () => { mounted = false; };
  }, [selectedWaterway, selectedMonth, selectedYear]);

  useEffect(() => {
    if (!selectedSegmentId) {
      setSegmentDetail(null);
      return;
    }

    let mounted = true;
    ApiService.getSegmentPrediction(selectedWaterway, selectedSegmentId, selectedMonth, selectedYear)
      .then((pred) => { if (mounted) setSegmentDetail(pred); })
      .catch(() => {
        const found = navMap?.predictions.find((p) => p.segment_id === selectedSegmentId);
        if (mounted && found) setSegmentDetail(found);
      });

    return () => { mounted = false; };
  }, [selectedWaterway, selectedSegmentId, selectedMonth, selectedYear, navMap]);

  const predictions = navMap?.predictions ?? [];
  const navPct = navMap?.overall_navigability_pct ?? navMap?.navigable_pct ?? 0;

  return (
    <div className="relative w-full h-[calc(100vh-4rem)] overflow-hidden bg-black flex text-white">
      {/* Floating Top Telemetry Bar */}
      <div className="absolute top-4 left-6 z-20 flex items-center gap-3">
        <div className="saas-card px-3.5 py-1.5 flex items-center gap-2.5 text-xs font-semibold text-zinc-200 bg-black/90 border-zinc-800 shadow-xl backdrop-blur-md">
          <span className="font-bold text-white flex items-center gap-1.5">
            <Activity size={14} className="text-sky-400" />
            {selectedWaterway} Route
          </span>
          <span className="text-zinc-700">|</span>
          <span className="text-sky-400 font-bold">{navPct.toFixed(1)}% Navigable</span>
          <span className="text-zinc-700">|</span>
          <span className="text-zinc-400">{predictions.length} Reaches</span>
        </div>
      </div>

      {/* Main Map Canvas Component */}
      <div className="flex-1 h-full relative bg-black">
        <RiverMap navMap={navMap} loading={loading} />
      </div>

      {/* Slide-over River Segment Inspector Drawer */}
      <AnimatePresence>
        {selectedSegmentId && (
          <motion.div
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', damping: 25, stiffness: 200 }}
            className="w-96 h-full bg-zinc-900 border-l border-zinc-800 shadow-2xl z-30 flex flex-col p-6 space-y-6 overflow-y-auto thin-scrollbar"
          >
            {/* Header */}
            <div className="flex items-start justify-between pb-4 border-b border-zinc-800">
              <div>
                <span className="text-xs font-bold uppercase tracking-wider text-zinc-400">
                  Segment Telemetry Inspector
                </span>
                <h2 className="text-xl font-extrabold text-white">
                  {selectedSegmentId}
                </h2>
              </div>
              <button
                onClick={() => setSelectedSegmentId(null)}
                className="p-1.5 rounded-lg border border-zinc-700 text-zinc-400 hover:bg-zinc-800 hover:text-white"
              >
                <X size={16} />
              </button>
            </div>

            {/* Status & Prediction Details */}
            {segmentDetail ? (
              <div className="space-y-6 text-xs">
                {/* Status Badge */}
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-zinc-400">Predicted Status</span>
                  <NavigabilityBadge navigabilityClass={segmentDetail.navigability_class} size="lg" />
                </div>

                {/* Numerical Physical Metrics */}
                <div className="grid grid-cols-2 gap-3">
                  <div className="p-3 bg-zinc-800/80 border border-zinc-700 rounded-xl">
                    <div className="text-zinc-400 font-semibold mb-1">Water Depth</div>
                    <div className="text-xl font-bold text-white font-mono">
                      {segmentDetail.predicted_depth_m.toFixed(2)} m
                    </div>
                  </div>

                  <div className="p-3 bg-zinc-800/80 border border-zinc-700 rounded-xl">
                    <div className="text-zinc-400 font-semibold mb-1">Channel Width</div>
                    <div className="text-xl font-bold text-white font-mono">
                      {(segmentDetail.width_m ?? 50).toFixed(0)} m
                    </div>
                  </div>
                </div>

                {/* Risk Score */}
                <div className="p-4 bg-zinc-800 text-white rounded-xl space-y-2 border border-zinc-700">
                  <div className="flex justify-between font-semibold">
                    <span>Composite Risk Index</span>
                    <span className="font-mono text-emerald-400">
                      {((segmentDetail.risk_score ?? 0.1) * 100).toFixed(0)} / 100
                    </span>
                  </div>
                  <div className="w-full h-1.5 bg-zinc-700 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-emerald-400 rounded-full"
                      style={{ width: `${(segmentDetail.risk_score ?? 0.1) * 100}%` }}
                    />
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center py-12 text-zinc-400 text-xs">
                Loading segment prediction telemetry...
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
