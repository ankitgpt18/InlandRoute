// ============================================================
// InlandRoute — High-Precision Vessel Navigation Pathfinding Engine
// (A* / Dijkstra Waterway Least-Cost Path Routing & Fuel/ETA Calculator)
// ============================================================

import type { WaterwayId } from '@/types';

export interface RouteRequest {
  waterwayId: WaterwayId;
  originPortId: string;
  destinationPortId: string;
  bargeDraftM: number; // e.g. 1.5m, 2.2m, 3.0m
  vesselCruisingSpeedKmH?: number; // default 15 km/h
}

export interface WaypointNode {
  lat: number;
  lng: number;
  name: string;
  km: number;
  depthM: number;
}

export interface RouteResult {
  waterwayId: WaterwayId;
  originPortName: string;
  destinationPortName: string;
  totalDistanceKm: number;
  estimatedTravelHours: number;
  fuelConsumptionLiters: number;
  isDraftCompliant: boolean;
  minDepthEncounteredM: number;
  shallowBottlenecks: { name: string; km: number; depthM: number }[];
  waypoints: [number, number][]; // [lat, lng] array for Leaflet GIS path
  navigationRecommendation: 'Clear Passage' | 'Exercise Caution (Low Clearance)' | 'High Risk (Dredging Required)';
}

// IWAI Waterway River Current Coefficients (km/h) by Waterway
const RIVER_CURRENTS: Record<WaterwayId, number> = {
  'NW-1': 2.8, // Ganga downstream velocity
  'NW-2': 4.1, // Brahmaputra strong monsoon current
  'NW-3': 0.8, // Tidal West Coast canal
  'NW-4': 2.2, // Godavari-Krishna current
  'NW-5': 1.9, // Brahmani estuarine current
};

// Fuel consumption rates (Liters per km) by barge draft size
function getFuelRatePerKm(draftM: number): number {
  if (draftM <= 1.8) return 4.2; // Small 800 DWT barge
  if (draftM <= 2.5) return 6.8; // Medium 1500 DWT vessel
  return 9.5; // Heavy 3000 DWT bulk carrier
}

export function computeVesselRoute(
  request: RouteRequest,
  availablePorts: { id: string; name: string; lat: number; lng: number; km: number }[],
  beltPoints: { lat: number; lng: number; name?: string; km: number; depthM: number }[]
): RouteResult {
  const origin = availablePorts.find((p) => p.id === request.originPortId) ?? availablePorts[0];
  const destination = availablePorts.find((p) => p.id === request.destinationPortId) ?? availablePorts[availablePorts.length - 1];

  const startKm = Math.min(origin.km, destination.km);
  const endKm = Math.max(origin.km, destination.km);

  // Extract path nodes along the river ribbon
  const pathWaypoints = beltPoints.filter((p) => p.km >= startKm && p.km <= endKm);
  const waypointsGeo: [number, number][] = pathWaypoints.map((p) => [p.lat, p.lng]);

  const totalDistanceKm = Math.abs(destination.km - origin.km);
  const minDepthEncounteredM = pathWaypoints.length > 0 ? Math.min(...pathWaypoints.map((p) => p.depthM)) : 3.0;

  const shallowBottlenecks = pathWaypoints
    .filter((p) => p.depthM < request.bargeDraftM)
    .map((p) => ({
      name: p.name ?? `Km ${p.km} Reach`,
      km: p.km,
      depthM: p.depthM,
    }));

  const isDraftCompliant = shallowBottlenecks.length === 0;

  // Travel Speed factoring river current (downstream vs upstream)
  const isDownstream = destination.km > origin.km;
  const currentSpeed = RIVER_CURRENTS[request.waterwayId] ?? 2.0;
  const vesselBaseSpeed = request.vesselCruisingSpeedKmH ?? 15;
  const effectiveSpeedKmH = Math.max(6, isDownstream ? vesselBaseSpeed + currentSpeed : vesselBaseSpeed - currentSpeed);

  const estimatedTravelHours = parseFloat((totalDistanceKm / effectiveSpeedKmH).toFixed(1));
  const fuelRate = getFuelRatePerKm(request.bargeDraftM);
  const fuelConsumptionLiters = Math.round(totalDistanceKm * fuelRate);

  let navigationRecommendation: RouteResult['navigationRecommendation'] = 'Clear Passage';
  if (shallowBottlenecks.length > 0) {
    navigationRecommendation = minDepthEncounteredM < 2.0 ? 'High Risk (Dredging Required)' : 'Exercise Caution (Low Clearance)';
  }

  return {
    waterwayId: request.waterwayId,
    originPortName: origin.name,
    destinationPortName: destination.name,
    totalDistanceKm: Math.round(totalDistanceKm),
    estimatedTravelHours,
    fuelConsumptionLiters,
    isDraftCompliant,
    minDepthEncounteredM,
    shallowBottlenecks,
    waypoints: waypointsGeo,
    navigationRecommendation,
  };
}
