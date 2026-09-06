// ============================================================
// InlandRoute — Interactive Vessel Navigation Route Pathfinder
// (A* / Dijkstra Least-Cost Waterway Routing & Telemetry Engine)
// ============================================================

'use client';

import React, { useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import {
  Navigation,
  Ship,
  Gauge,
  Fuel,
  Clock,
  AlertTriangle,
  CheckCircle2,
  Anchor,
  Compass,
  ArrowRight,
  ShieldAlert,
} from 'lucide-react';
import { useAppStore } from '@/store/app-store';
import { RiverMap } from '@/components/maps/river-map';
import { computeVesselRoute } from '@/lib/pathfinder';
import {
  NW1_PORTS,
  NW2_PORTS,
  NW3_PORTS,
  NW4_PORTS,
  NW5_PORTS,
} from '@/lib/mock-data';
import {
  NW1_GANGA_BELT,
  NW2_BRAHMAPUTRA_BELT,
  NW3_WESTCOAST_BELT,
  NW4_GODAVARI_BELT,
  NW5_BRAHMANI_BELT,
} from '@/components/maps/river-map';
import type { WaterwayId } from '@/types';

export default function NavigationPage() {
  const selectedWaterway = useAppStore((s) => s.selectedWaterway);
  const setSelectedWaterway = useAppStore((s) => s.setSelectedWaterway);

  const [bargeDraftM, setBargeDraftM] = useState<number>(2.2);
  const [cruisingSpeedKmH, setCruisingSpeedKmH] = useState<number>(15);

  const ports = useMemo(() => {
    if (selectedWaterway === 'NW-1') return NW1_PORTS;
    if (selectedWaterway === 'NW-2') return NW2_PORTS;
    if (selectedWaterway === 'NW-3') return NW3_PORTS;
    if (selectedWaterway === 'NW-4') return NW4_PORTS;
    return NW5_PORTS;
  }, [selectedWaterway]);

  const beltPoints = useMemo(() => {
    if (selectedWaterway === 'NW-1') return NW1_GANGA_BELT;
    if (selectedWaterway === 'NW-2') return NW2_BRAHMAPUTRA_BELT;
    if (selectedWaterway === 'NW-3') return NW3_WESTCOAST_BELT;
    if (selectedWaterway === 'NW-4') return NW4_GODAVARI_BELT;
    return NW5_BRAHMANI_BELT;
  }, [selectedWaterway]);

  const [originPortId, setOriginPortId] = useState<string>(ports[0].id);
  const [destinationPortId, setDestinationPortId] = useState<string>(ports[ports.length - 1].id);

  // Re-sync origin/destination when waterway changes
  React.useEffect(() => {
    setOriginPortId(ports[0].id);
    setDestinationPortId(ports[ports.length - 1].id);
  }, [ports]);

  const routeResult = useMemo(() => {
    return computeVesselRoute(
      {
        waterwayId: selectedWaterway,
        originPortId,
        destinationPortId,
        bargeDraftM,
        vesselCruisingSpeedKmH: cruisingSpeedKmH,
      },
      ports,
      beltPoints
    );
  }, [selectedWaterway, originPortId, destinationPortId, bargeDraftM, cruisingSpeedKmH, ports, beltPoints]);

  return (
    <div className="flex flex-col min-h-screen bg-zinc-950 text-white p-6 space-y-6">
      {/* Header Title */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-zinc-800 pb-4">
        <div>
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-400">
              <Navigation size={20} />
            </div>
            <h1 className="text-xl font-bold tracking-tight text-white">Vessel Navigation Pathfinder</h1>
          </div>
          <p className="text-xs text-zinc-400 mt-1">
            Real-time least-cost pathfinding, draft clearance verification, and fuel burn optimization across National Waterways.
          </p>
        </div>

        {/* Waterway Selector Pills */}
        <div className="flex items-center bg-zinc-900 p-1 rounded-xl border border-zinc-800 overflow-x-auto">
          {[
            { id: 'NW-1', label: 'NW-1 · Ganga' },
            { id: 'NW-2', label: 'NW-2 · Brahmaputra' },
            { id: 'NW-3', label: 'NW-3 · West Coast' },
            { id: 'NW-4', label: 'NW-4 · Godavari/Krishna' },
            { id: 'NW-5', label: 'NW-5 · Brahmani' },
          ].map((ww) => (
            <button
              key={ww.id}
              onClick={() => setSelectedWaterway(ww.id as WaterwayId)}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold whitespace-nowrap transition-all ${
                selectedWaterway === ww.id
                  ? 'bg-emerald-500 text-zinc-950 shadow-sm font-bold'
                  : 'text-zinc-400 hover:text-white'
              }`}
            >
              {ww.label}
            </button>
          ))}
        </div>
      </div>

      {/* Control Panel: Route Configuration */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 bg-zinc-900/80 p-4 rounded-2xl border border-zinc-800/80 shadow-xl backdrop-blur-md">
        {/* Origin Port */}
        <div className="flex flex-col gap-1.5">
          <label className="text-xs font-semibold text-zinc-400 flex items-center gap-1.5">
            <Anchor size={13} className="text-emerald-400" /> Origin Terminal
          </label>
          <select
            value={originPortId}
            onChange={(e) => setOriginPortId(e.target.value)}
            className="bg-zinc-800 border border-zinc-700 text-xs text-white rounded-xl p-2.5 focus:outline-none focus:border-emerald-500"
          >
            {ports.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name} (Km {p.km})
              </option>
            ))}
          </select>
        </div>

        {/* Destination Port */}
        <div className="flex flex-col gap-1.5">
          <label className="text-xs font-semibold text-zinc-400 flex items-center gap-1.5">
            <Compass size={13} className="text-emerald-400" /> Destination Port
          </label>
          <select
            value={destinationPortId}
            onChange={(e) => setDestinationPortId(e.target.value)}
            className="bg-zinc-800 border border-zinc-700 text-xs text-white rounded-xl p-2.5 focus:outline-none focus:border-emerald-500"
          >
            {ports.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name} (Km {p.km})
              </option>
            ))}
          </select>
        </div>

        {/* Barge Draft */}
        <div className="flex flex-col gap-1.5">
          <label className="text-xs font-semibold text-zinc-400 flex items-center gap-1.5">
            <Ship size={13} className="text-emerald-400" /> Vessel Draft Requirement
          </label>
          <select
            value={bargeDraftM}
            onChange={(e) => setBargeDraftM(parseFloat(e.target.value))}
            className="bg-zinc-800 border border-zinc-700 text-xs text-white rounded-xl p-2.5 focus:outline-none focus:border-emerald-500"
          >
            <option value={1.5}>1.5 m — Light Coastal Barge (800 DWT)</option>
            <option value={2.2}>2.2 m — Standard Inland Vessel (1500 DWT)</option>
            <option value={3.0}>3.0 m — Heavy Bulk Carrier (3000 DWT)</option>
          </select>
        </div>

        {/* Cruising Speed */}
        <div className="flex flex-col gap-1.5">
          <div className="flex justify-between items-center text-xs">
            <span className="font-semibold text-zinc-400 flex items-center gap-1.5">
              <Gauge size={13} className="text-emerald-400" /> Speed (km/h)
            </span>
            <span className="text-emerald-400 font-bold">{cruisingSpeedKmH} km/h</span>
          </div>
          <input
            type="range"
            min={10}
            max={25}
            step={1}
            value={cruisingSpeedKmH}
            onChange={(e) => setCruisingSpeedKmH(parseInt(e.target.value))}
            className="w-full accent-emerald-500 cursor-pointer mt-2"
          />
        </div>
      </div>

      {/* Primary Telemetry Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {/* Total Distance */}
        <div className="bg-zinc-900 border border-zinc-800 p-4 rounded-2xl flex items-center gap-3">
          <div className="p-3 bg-zinc-800 text-emerald-400 rounded-xl">
            <Compass size={20} />
          </div>
          <div>
            <div className="text-xs text-zinc-400 font-medium">Total Distance</div>
            <div className="text-xl font-bold text-white mt-0.5">{routeResult.totalDistanceKm} km</div>
          </div>
        </div>

        {/* Travel Time */}
        <div className="bg-zinc-900 border border-zinc-800 p-4 rounded-2xl flex items-center gap-3">
          <div className="p-3 bg-zinc-800 text-sky-400 rounded-xl">
            <Clock size={20} />
          </div>
          <div>
            <div className="text-xs text-zinc-400 font-medium">Estimated Transit Time</div>
            <div className="text-xl font-bold text-white mt-0.5">{routeResult.estimatedTravelHours} hours</div>
          </div>
        </div>

        {/* Fuel Burn */}
        <div className="bg-zinc-900 border border-zinc-800 p-4 rounded-2xl flex items-center gap-3">
          <div className="p-3 bg-zinc-800 text-amber-400 rounded-xl">
            <Fuel size={20} />
          </div>
          <div>
            <div className="text-xs text-zinc-400 font-medium">Estimated Fuel Burn</div>
            <div className="text-xl font-bold text-white mt-0.5">{routeResult.fuelConsumptionLiters} Liters</div>
          </div>
        </div>

        {/* Navigation Clearance Status */}
        <div className="bg-zinc-900 border border-zinc-800 p-4 rounded-2xl flex items-center gap-3">
          <div
            className={`p-3 rounded-xl ${
              routeResult.isDraftCompliant
                ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                : 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
            }`}
          >
            {routeResult.isDraftCompliant ? <CheckCircle2 size={20} /> : <AlertTriangle size={20} />}
          </div>
          <div>
            <div className="text-xs text-zinc-400 font-medium">Channel Clearance</div>
            <div
              className={`text-sm font-bold mt-0.5 ${
                routeResult.isDraftCompliant ? 'text-emerald-400' : 'text-amber-400'
              }`}
            >
              {routeResult.navigationRecommendation}
            </div>
          </div>
        </div>
      </div>

      {/* Main Split-View Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Interactive GIS Map */}
        <div className="lg:col-span-2 bg-zinc-900 border border-zinc-800 rounded-2xl overflow-hidden h-[540px] relative">
          <RiverMap />
        </div>

        {/* Right Column: Route Breakdown & Bottlenecks */}
        <div className="bg-zinc-900 border border-zinc-800 p-5 rounded-2xl flex flex-col justify-between space-y-4">
          <div>
            <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <Ship size={16} className="text-emerald-400" /> Route Waypoint Telemetry
              </h3>
              <span className="text-xs px-2.5 py-1 rounded-full bg-zinc-800 text-zinc-300 border border-zinc-700">
                LAD Threshold: {bargeDraftM}m
              </span>
            </div>

            {/* Waypoint Sequence List */}
            <div className="mt-4 space-y-3 max-h-[300px] overflow-y-auto pr-1">
              <div className="flex items-center justify-between bg-zinc-800/60 p-3 rounded-xl border border-zinc-700/60">
                <div>
                  <div className="text-xs font-bold text-emerald-400">ORIGIN: {routeResult.originPortName}</div>
                  <div className="text-[11px] text-zinc-400">Starting Terminal</div>
                </div>
                <div className="text-xs font-semibold text-zinc-300">Km 0</div>
              </div>

              {routeResult.shallowBottlenecks.map((b, i) => (
                <div
                  key={i}
                  className="flex items-center justify-between bg-amber-500/10 p-3 rounded-xl border border-amber-500/20"
                >
                  <div className="flex items-center gap-2">
                    <ShieldAlert size={15} className="text-amber-400 flex-shrink-0" />
                    <div>
                      <div className="text-xs font-bold text-amber-300">{b.name}</div>
                      <div className="text-[11px] text-amber-400/80">Depth: {b.depthM}m (Requires Care)</div>
                    </div>
                  </div>
                  <div className="text-xs font-bold text-amber-400">Km {b.km}</div>
                </div>
              ))}

              <div className="flex items-center justify-between bg-zinc-800/60 p-3 rounded-xl border border-zinc-700/60">
                <div>
                  <div className="text-xs font-bold text-emerald-400">DESTINATION: {routeResult.destinationPortName}</div>
                  <div className="text-[11px] text-zinc-400">Terminal Arrival</div>
                </div>
                <div className="text-xs font-semibold text-zinc-300">Km {routeResult.totalDistanceKm}</div>
              </div>
            </div>
          </div>

          {/* Action Footer */}
          <div className="pt-3 border-t border-zinc-800 flex items-center justify-between">
            <div className="text-xs text-zinc-400">
              Min Channel Depth: <span className="text-white font-bold">{routeResult.minDepthEncounteredM}m</span>
            </div>
            <button
              onClick={() => alert(`Vessel Navigation Plan exported for ${routeResult.originPortName} to ${routeResult.destinationPortName}.`)}
              className="px-4 py-2 bg-emerald-500 hover:bg-emerald-400 text-zinc-950 font-bold rounded-xl text-xs transition-all shadow-lg flex items-center gap-2"
            >
              Export Route Plan <ArrowRight size={14} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
