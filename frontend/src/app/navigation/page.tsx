"use client";

import React, { useState, useMemo } from "react";
import {
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
  X,
  Printer,
  FileCheck,
} from "lucide-react";
import { useAppStore } from "@/store/app-store";
import { RiverMap } from "@/components/maps/river-map";
import { computeVesselRoute } from "@/lib/pathfinder";
import {
  NW1_PORTS,
  NW2_PORTS,
  NW3_PORTS,
  NW4_PORTS,
  NW5_PORTS,
} from "@/lib/mock-data";
import {
  NW1_GANGA_BELT,
  NW2_BRAHMAPUTRA_BELT,
  NW3_WESTCOAST_BELT,
  NW4_GODAVARI_BELT,
  NW5_BRAHMANI_BELT,
} from "@/components/maps/river-map";
import { DashboardShell } from "@/components/layout/dashboard-shell";

export default function NavigationPage() {
  const selectedWaterway = useAppStore((s) => s.selectedWaterway);

  const [bargeDraftM, setBargeDraftM] = useState<number>(2.2);
  const [cruisingSpeedKmH, setCruisingSpeedKmH] = useState<number>(15);
  const [showExportModal, setShowExportModal] = useState<boolean>(false);

  const ports = useMemo(() => {
    if (selectedWaterway === "NW-1") return NW1_PORTS;
    if (selectedWaterway === "NW-2") return NW2_PORTS;
    if (selectedWaterway === "NW-3") return NW3_PORTS;
    if (selectedWaterway === "NW-4") return NW4_PORTS;
    return NW5_PORTS;
  }, [selectedWaterway]);

  const beltPoints = useMemo(() => {
    if (selectedWaterway === "NW-1") return NW1_GANGA_BELT;
    if (selectedWaterway === "NW-2") return NW2_BRAHMAPUTRA_BELT;
    if (selectedWaterway === "NW-3") return NW3_WESTCOAST_BELT;
    if (selectedWaterway === "NW-4") return NW4_GODAVARI_BELT;
    return NW5_BRAHMANI_BELT;
  }, [selectedWaterway]);

  const [originPortId, setOriginPortId] = useState<string>(ports[0].id);
  const [destinationPortId, setDestinationPortId] = useState<string>(
    ports[ports.length - 1].id
  );

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
  }, [
    selectedWaterway,
    originPortId,
    destinationPortId,
    bargeDraftM,
    cruisingSpeedKmH,
    ports,
    beltPoints,
  ]);

  return (
    <DashboardShell>
      <div className="space-y-6">
        {/* Control Panel: Route Configuration */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 rounded-lg border border-neutral-800 bg-black p-4">
          {/* Origin Port */}
          <div className="flex flex-col gap-1.5">
            <label className="text-xs font-semibold text-neutral-400 flex items-center gap-1.5">
              <Anchor size={13} className="text-neutral-400" /> Origin Terminal
            </label>
            <select
              value={originPortId}
              onChange={(e) => setOriginPortId(e.target.value)}
              className="rounded-md border border-neutral-800 bg-neutral-900/90 p-2 text-xs text-white focus:outline-none focus:border-neutral-600"
            >
              {ports.map((p) => (
                <option key={p.id} value={p.id} className="bg-neutral-950 text-white">
                  {p.name} (Km {p.km})
                </option>
              ))}
            </select>
          </div>

          {/* Destination Port */}
          <div className="flex flex-col gap-1.5">
            <label className="text-xs font-semibold text-neutral-400 flex items-center gap-1.5">
              <Compass size={13} className="text-neutral-400" /> Destination Port
            </label>
            <select
              value={destinationPortId}
              onChange={(e) => setDestinationPortId(e.target.value)}
              className="rounded-md border border-neutral-800 bg-neutral-900/90 p-2 text-xs text-white focus:outline-none focus:border-neutral-600"
            >
              {ports.map((p) => (
                <option key={p.id} value={p.id} className="bg-neutral-950 text-white">
                  {p.name} (Km {p.km})
                </option>
              ))}
            </select>
          </div>

          {/* Barge Draft */}
          <div className="flex flex-col gap-1.5">
            <label className="text-xs font-semibold text-neutral-400 flex items-center gap-1.5">
              <Ship size={13} className="text-neutral-400" /> Vessel Draft Requirement
            </label>
            <select
              value={bargeDraftM}
              onChange={(e) => setBargeDraftM(parseFloat(e.target.value))}
              className="rounded-md border border-neutral-800 bg-neutral-900/90 p-2 text-xs text-white focus:outline-none focus:border-neutral-600"
            >
              <option value={1.5} className="bg-neutral-950 text-white">
                1.5 m - Light Coastal Barge (800 DWT)
              </option>
              <option value={2.2} className="bg-neutral-950 text-white">
                2.2 m - Standard Inland Vessel (1500 DWT)
              </option>
              <option value={3.0} className="bg-neutral-950 text-white">
                3.0 m - Heavy Bulk Carrier (3000 DWT)
              </option>
            </select>
          </div>

          {/* Cruising Speed */}
          <div className="flex flex-col gap-1.5">
            <div className="flex justify-between items-center text-xs">
              <span className="font-semibold text-neutral-400 flex items-center gap-1.5">
                <Gauge size={13} className="text-neutral-400" /> Speed (km/h)
              </span>
              <span className="text-white font-mono font-semibold">
                {cruisingSpeedKmH} km/h
              </span>
            </div>
            <input
              type="range"
              min={10}
              max={25}
              step={1}
              value={cruisingSpeedKmH}
              onChange={(e) => setCruisingSpeedKmH(parseInt(e.target.value))}
              className="w-full accent-neutral-300 cursor-pointer mt-3"
            />
          </div>
        </div>

        {/* Primary Telemetry Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Total Distance */}
          <div className="rounded-lg border border-neutral-800 bg-black p-4 flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-md border border-neutral-800 bg-neutral-900 text-neutral-400">
              <Compass size={18} />
            </div>
            <div>
              <div className="text-xs text-neutral-400">Total Distance</div>
              <div className="text-xl font-semibold text-white mt-0.5 font-mono">
                {routeResult.totalDistanceKm} km
              </div>
            </div>
          </div>

          {/* Travel Time */}
          <div className="rounded-lg border border-neutral-800 bg-black p-4 flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-md border border-neutral-800 bg-neutral-900 text-neutral-400">
              <Clock size={18} />
            </div>
            <div>
              <div className="text-xs text-neutral-400">Transit Duration</div>
              <div className="text-xl font-semibold text-white mt-0.5 font-mono">
                {routeResult.estimatedTravelHours} hours
              </div>
            </div>
          </div>

          {/* Fuel Burn */}
          <div className="rounded-lg border border-neutral-800 bg-black p-4 flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-md border border-neutral-800 bg-neutral-900 text-neutral-400">
              <Fuel size={18} />
            </div>
            <div>
              <div className="text-xs text-neutral-400">Fuel Burn</div>
              <div className="text-xl font-semibold text-white mt-0.5 font-mono">
                {routeResult.fuelConsumptionLiters} Liters
              </div>
            </div>
          </div>

          {/* Channel Clearance: Non-AI clean status */}
          <div className="rounded-lg border border-neutral-800 bg-black p-4 flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-md border border-neutral-800 bg-neutral-900 text-neutral-400">
              {routeResult.isDraftCompliant ? (
                <CheckCircle2 size={18} className="text-emerald-400" />
              ) : (
                <AlertTriangle size={18} className="text-amber-400" />
              )}
            </div>
            <div>
              <div className="text-xs text-neutral-400">Channel Clearance</div>
              <div className="flex items-center gap-1.5 mt-0.5">
                <span
                  className={`h-1.5 w-1.5 rounded-full ${
                    routeResult.isDraftCompliant ? "bg-emerald-500" : "bg-amber-400"
                  }`}
                />
                <span className="text-xs font-semibold text-white">
                  {routeResult.navigationRecommendation}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Main Split-View Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column: Interactive GIS Map */}
          <div className="lg:col-span-2 rounded-lg border border-neutral-800 bg-black overflow-hidden h-[540px] relative">
            <RiverMap />
          </div>

          {/* Right Column: Route Breakdown & Bottlenecks */}
          <div className="rounded-lg border border-neutral-800 bg-black p-5 flex flex-col justify-between space-y-4">
            <div>
              <div className="flex items-center justify-between border-b border-neutral-800 pb-3">
                <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                  <Ship size={15} className="text-neutral-400" /> Route Waypoint Telemetry
                </h3>
                <span className="text-[11px] px-2 py-0.5 rounded border border-neutral-800 bg-neutral-900 text-neutral-300 font-mono">
                  LAD: {bargeDraftM}m
                </span>
              </div>

              {/* Waypoint Sequence List */}
              <div className="mt-4 space-y-2.5 max-h-[300px] overflow-y-auto pr-1 thin-scrollbar">
                <div className="flex items-center justify-between rounded-md border border-neutral-800 bg-neutral-950 p-3">
                  <div>
                    <div className="text-xs font-semibold text-white">
                      ORIGIN: {routeResult.originPortName}
                    </div>
                    <div className="text-[11px] text-neutral-400 mt-0.5">
                      Departure Terminal
                    </div>
                  </div>
                  <div className="text-xs font-mono font-medium text-neutral-400">
                    Km 0
                  </div>
                </div>

                {routeResult.shallowBottlenecks.map((b, i) => (
                  <div
                    key={i}
                    className="flex items-center justify-between rounded-md border border-neutral-800 bg-neutral-950 p-3"
                  >
                    <div className="flex items-center gap-2.5">
                      <span className="h-2 w-2 rounded-full bg-amber-400 shrink-0" />
                      <div>
                        <div className="text-xs font-semibold text-white">
                          {b.name}
                        </div>
                        <div className="text-[11px] text-neutral-400 mt-0.5">
                          Depth: {b.depthM}m (Cautionary Navigation)
                        </div>
                      </div>
                    </div>
                    <div className="text-xs font-mono font-medium text-amber-400">
                      Km {b.km}
                    </div>
                  </div>
                ))}

                <div className="flex items-center justify-between rounded-md border border-neutral-800 bg-neutral-950 p-3">
                  <div>
                    <div className="text-xs font-semibold text-white">
                      DESTINATION: {routeResult.destinationPortName}
                    </div>
                    <div className="text-[11px] text-neutral-400 mt-0.5">
                      Arrival Terminal
                    </div>
                  </div>
                  <div className="text-xs font-mono font-medium text-neutral-400">
                    Km {routeResult.totalDistanceKm}
                  </div>
                </div>
              </div>
            </div>

            {/* Action Footer */}
            <div className="pt-3 border-t border-neutral-800 flex items-center justify-between">
              <div className="text-xs text-neutral-400">
                Min Channel:{" "}
                <span className="text-white font-mono font-semibold">
                  {routeResult.minDepthEncounteredM}m
                </span>
              </div>
              <button
                type="button"
                onClick={() => setShowExportModal(true)}
                className="group flex items-center justify-center gap-2 rounded-md bg-white px-3.5 py-1.5 text-xs font-semibold text-black transition-all hover:bg-neutral-200"
              >
                <span>Export Plan</span>
                <ArrowRight size={13} />
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Functioning Voyage Manifest Modal */}
      {showExportModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
          <div className="relative w-full max-w-lg rounded-lg border border-neutral-800 bg-neutral-950 p-6 shadow-2xl space-y-5">
            <div className="flex items-start justify-between border-b border-neutral-800 pb-3">
              <div className="flex items-center gap-2">
                <FileCheck className="h-5 w-5 text-neutral-300" />
                <div>
                  <h3 className="text-base font-bold text-white">
                    Vessel Navigation Plan Manifest
                  </h3>
                  <p className="text-xs text-neutral-400">
                    Corridor {selectedWaterway} · Operational Routing Certificate
                  </p>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setShowExportModal(false)}
                className="rounded-md border border-neutral-800 p-1.5 text-neutral-400 hover:text-white hover:bg-neutral-900 transition-colors"
              >
                <X size={15} />
              </button>
            </div>

            <div className="space-y-3 text-xs text-neutral-300">
              <div className="grid grid-cols-2 gap-3">
                <div className="rounded-md border border-neutral-800 bg-black p-3 space-y-1">
                  <div className="text-[11px] text-neutral-400">Departure Port</div>
                  <div className="font-semibold text-white">{routeResult.originPortName}</div>
                </div>
                <div className="rounded-md border border-neutral-800 bg-black p-3 space-y-1">
                  <div className="text-[11px] text-neutral-400">Arrival Destination</div>
                  <div className="font-semibold text-white">{routeResult.destinationPortName}</div>
                </div>
              </div>

              <div className="grid grid-cols-3 gap-3">
                <div className="rounded-md border border-neutral-800 bg-black p-3">
                  <div className="text-[11px] text-neutral-400">Voyage Distance</div>
                  <div className="font-mono text-sm font-bold text-white mt-0.5">
                    {routeResult.totalDistanceKm} km
                  </div>
                </div>
                <div className="rounded-md border border-neutral-800 bg-black p-3">
                  <div className="text-[11px] text-neutral-400">Est. Duration</div>
                  <div className="font-mono text-sm font-bold text-white mt-0.5">
                    {routeResult.estimatedTravelHours} hrs
                  </div>
                </div>
                <div className="rounded-md border border-neutral-800 bg-black p-3">
                  <div className="text-[11px] text-neutral-400">Fuel Estimate</div>
                  <div className="font-mono text-sm font-bold text-white mt-0.5">
                    {routeResult.fuelConsumptionLiters} L
                  </div>
                </div>
              </div>

              <div className="rounded-md border border-neutral-800 bg-black p-3 space-y-1.5">
                <div className="flex items-center justify-between text-[11px]">
                  <span className="text-neutral-400">Safety & Clearance Index</span>
                  <span className="font-mono font-semibold text-emerald-400">
                    {routeResult.isDraftCompliant ? "Fully Compliant" : "Conditional Navigation"}
                  </span>
                </div>
                <p className="text-[11px] text-neutral-400 leading-relaxed">
                  {routeResult.navigationRecommendation}. Verified against Central Water Commission stage discharge telemetry and Sentinel-2 fairway profiles.
                </p>
              </div>
            </div>

            <div className="flex items-center justify-end gap-3 border-t border-neutral-800 pt-3">
              <button
                type="button"
                onClick={() => setShowExportModal(false)}
                className="rounded-md border border-neutral-800 px-3 py-1.5 text-xs font-medium text-neutral-300 hover:bg-neutral-900 transition-colors"
              >
                Close
              </button>
              <button
                type="button"
                onClick={() => {
                  window.print();
                }}
                className="inline-flex items-center gap-1.5 rounded-md bg-white px-3 py-1.5 text-xs font-semibold text-black hover:bg-neutral-200 transition-colors"
              >
                <Printer size={13} />
                <span>Print Route Manifest</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </DashboardShell>
  );
}
