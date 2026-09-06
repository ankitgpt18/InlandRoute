// ============================================================
// InlandRoute — Bold Electric-Blue River Highway Belt Renderer
// (Ultra-High-Visibility 36px Electric Blue Ribbon for NW-1 & NW-2)
// ============================================================

'use client';

import React, { useEffect, useMemo, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Satellite,
  Moon,
  Sun,
  Layers,
} from 'lucide-react';
import { useAppStore } from '@/store/app-store';
import { MapLegendCard } from '@/components/ui/navigability-badge';
import { cn } from '@/lib/utils';
import type { NavigabilityClass, WaterwayId, MapStyle, NavigabilityMap } from '@/types';

const TILE_PROVIDERS: Record<MapStyle, { url: string; attribution: string }> = {
  dark: {
    url: 'https://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
    attribution: '&copy; OpenStreetMap &copy; CARTO',
  },
  satellite: {
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attribution: '&copy; Esri, Maxar, Earthstar Geographics',
  },
  light: {
    url: 'https://basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png',
    attribution: '&copy; OpenStreetMap &copy; CARTO',
  },
};

export interface BeltPoint {
  lat: number;
  lng: number;
  name?: string;
  km: number;
  depthM: number;
  class: NavigabilityClass;
  isTerminal?: boolean;
}

// ─── High-Precision Coordinates for Electric Blue NW Strips ──────────────────

export const NW1_GANGA_BELT: BeltPoint[] = [
  { lat: 25.3176, lng: 83.0062, name: 'Varanasi Multimodal Terminal', km: 0, depthM: 3.4, class: 'navigable', isTerminal: true },
  { lat: 25.5600, lng: 83.5800, name: 'Ghazipur Reach', km: 130, depthM: 3.2, class: 'navigable' },
  { lat: 25.5700, lng: 83.9800, name: 'Buxar Reach', km: 210, depthM: 2.8, class: 'conditional' },
  { lat: 25.6100, lng: 85.1400, name: 'Patna Inland Port', km: 380, depthM: 3.5, class: 'navigable', isTerminal: true },
  { lat: 25.3700, lng: 86.4700, name: 'Munger Reach', km: 480, depthM: 3.1, class: 'navigable' },
  { lat: 25.2425, lng: 87.0139, name: 'Bhagalpur Port', km: 540, depthM: 2.9, class: 'conditional', isTerminal: true },
  { lat: 25.2504, lng: 87.6477, name: 'Sahibganj Multimodal Terminal', km: 650, depthM: 3.6, class: 'navigable', isTerminal: true },
  { lat: 24.8143, lng: 87.9304, name: 'Farakka Barrage & Lock', km: 780, depthM: 1.8, class: 'non_navigable', isTerminal: true },
  { lat: 24.0800, lng: 88.2500, name: 'Berhampore Reach', km: 920, depthM: 3.2, class: 'navigable' },
  { lat: 23.4100, lng: 88.3800, name: 'Nabadwip Reach', km: 1100, depthM: 3.4, class: 'navigable' },
  { lat: 22.9000, lng: 88.4000, name: 'Tribeni Hooghly Reach', km: 1220, depthM: 3.8, class: 'navigable' },
  { lat: 22.5726, lng: 88.3639, name: 'Kolkata Port (GRJ Jetty)', km: 1350, depthM: 4.2, class: 'navigable', isTerminal: true },
  { lat: 22.0257, lng: 88.0583, name: 'Haldia Floating Terminal', km: 1540, depthM: 4.5, class: 'navigable', isTerminal: true },
];

export const NW2_BRAHMAPUTRA_BELT: BeltPoint[] = [
  { lat: 26.0203, lng: 89.9744, name: 'Dhubri Inland Port', km: 0, depthM: 3.2, class: 'navigable', isTerminal: true },
  { lat: 26.1700, lng: 90.6200, name: 'Goalpara Reach', km: 90, depthM: 2.7, class: 'conditional' },
  { lat: 26.1667, lng: 91.7000, name: 'Pandu (Guwahati) Terminal', km: 260, depthM: 3.5, class: 'navigable', isTerminal: true },
  { lat: 26.2800, lng: 92.0500, name: 'Morigaon Silghat Reach', km: 340, depthM: 1.9, class: 'non_navigable' },
  { lat: 26.6338, lng: 92.7926, name: 'Tezpur Inland Terminal', km: 430, depthM: 3.1, class: 'navigable', isTerminal: true },
  { lat: 26.8500, lng: 94.2167, name: 'Neamati Ghat (Majuli) Jetty', km: 620, depthM: 2.8, class: 'conditional', isTerminal: true },
  { lat: 27.4800, lng: 94.9000, name: 'Dibrugarh Port', km: 760, depthM: 3.3, class: 'navigable', isTerminal: true },
  { lat: 27.8333, lng: 95.6667, name: 'Sadiya Terminal', km: 891, depthM: 3.0, class: 'navigable', isTerminal: true },
];

export const NW3_WESTCOAST_BELT: BeltPoint[] = [
  { lat: 10.2000, lng: 76.2000, name: 'Kottapuram Terminal', km: 0, depthM: 3.5, class: 'navigable', isTerminal: true },
  { lat: 10.0800, lng: 76.2900, name: 'Udyogmandal Industrial Lock', km: 35, depthM: 3.2, class: 'navigable', isTerminal: true },
  { lat: 9.4900, lng: 76.3300, name: 'Alappuzha Floating Jetty', km: 110, depthM: 3.0, class: 'navigable', isTerminal: true },
  { lat: 9.1700, lng: 76.5000, name: 'Kayamkulam Reach Terminal', km: 160, depthM: 2.8, class: 'conditional', isTerminal: true },
  { lat: 8.8800, lng: 76.5800, name: 'Kollam Multimodal Port', km: 205, depthM: 3.6, class: 'navigable', isTerminal: true },
];

export const NW4_GODAVARI_BELT: BeltPoint[] = [
  { lat: 16.9800, lng: 82.2400, name: 'Kakinada Deepwater Lock', km: 0, depthM: 4.2, class: 'navigable', isTerminal: true },
  { lat: 17.0000, lng: 81.7800, name: 'Rajahmundry Godavari Port', km: 210, depthM: 3.8, class: 'navigable', isTerminal: true },
  { lat: 16.5100, lng: 80.6200, name: 'Vijayawada Krishna Terminal', km: 540, depthM: 3.6, class: 'navigable', isTerminal: true },
  { lat: 16.8200, lng: 80.0500, name: 'Muktyala Mineral Terminal', km: 720, depthM: 2.9, class: 'conditional', isTerminal: true },
  { lat: 11.9300, lng: 79.8300, name: 'Puducherry Southern Port', km: 1078, depthM: 3.4, class: 'navigable', isTerminal: true },
];

export const NW5_BRAHMANI_BELT: BeltPoint[] = [
  { lat: 20.9500, lng: 85.2200, name: 'Talcher Coal Industrial Port', km: 0, depthM: 4.5, class: 'navigable', isTerminal: true },
  { lat: 20.7800, lng: 86.1200, name: 'Mangalgadi River Jetty', km: 180, depthM: 3.5, class: 'navigable', isTerminal: true },
  { lat: 20.8000, lng: 86.9200, name: 'Dhamra Feeder Terminal', km: 380, depthM: 4.8, class: 'navigable', isTerminal: true },
  { lat: 20.2700, lng: 86.6700, name: 'Paradip Lock Gate', km: 588, depthM: 4.2, class: 'navigable', isTerminal: true },
];

const WATERWAY_BOUNDS: Record<WaterwayId, [[number, number], [number, number]]> = {
  'NW-1': [
    [21.8, 82.6],
    [25.8, 88.5],
  ],
  'NW-2': [
    [25.8, 89.6],
    [28.0, 95.8],
  ],
  'NW-3': [
    [8.80, 76.10],
    [10.30, 76.70],
  ],
  'NW-4': [
    [11.80, 79.50],
    [17.10, 82.50],
  ],
  'NW-5': [
    [20.10, 85.10],
    [21.10, 87.10],
  ],
};

export interface RiverMapProps {
  navMap?: NavigabilityMap | null;
  loading?: boolean;
  fullscreen?: boolean;
  className?: string;
  onSegmentClick?: (segmentId: string) => void;
}

export function RiverMap({
  navMap,
  loading = false,
  className,
  onSegmentClick,
}: RiverMapProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const leafletMapRef = useRef<any>(null);
  const tileLayerRef = useRef<any>(null);
  const beltGroupRef = useRef<any>(null);
  const terminalsGroupRef = useRef<any>(null);

  const selectedWaterway = useAppStore((s) => s.selectedWaterway);
  const mapStyle = useAppStore((s) => s.mapStyle);
  const setMapStyle = useAppStore((s) => s.setMapStyle);

  const [showLegend, setShowLegend] = useState(true);

  // Active belt points for selected waterway
  const activeBelt = useMemo(() => {
    if (selectedWaterway === 'NW-1') return NW1_GANGA_BELT;
    if (selectedWaterway === 'NW-2') return NW2_BRAHMAPUTRA_BELT;
    if (selectedWaterway === 'NW-3') return NW3_WESTCOAST_BELT;
    if (selectedWaterway === 'NW-4') return NW4_GODAVARI_BELT;
    return NW5_BRAHMANI_BELT;
  }, [selectedWaterway]);

  // 1. Initialize Leaflet Map Instance
  useEffect(() => {
    if (typeof window === 'undefined' || !containerRef.current) return;
    let isCancelled = false;

    import('leaflet').then((L) => {
      if (isCancelled || leafletMapRef.current || !containerRef.current) return;

      const bounds = WATERWAY_BOUNDS[selectedWaterway];
      const map = L.map(containerRef.current, {
        zoomControl: false,
        minZoom: 5,
      });

      map.fitBounds(bounds, { padding: [60, 60] });

      const provider = TILE_PROVIDERS[mapStyle] ?? TILE_PROVIDERS.dark;
      const tileLayer = L.tileLayer(provider.url, {
        maxZoom: 19,
        attribution: provider.attribution,
      }).addTo(map);

      const beltGroup = L.layerGroup().addTo(map);
      const terminalsGroup = L.layerGroup().addTo(map);

      leafletMapRef.current = map;
      tileLayerRef.current = tileLayer;
      beltGroupRef.current = beltGroup;
      terminalsGroupRef.current = terminalsGroup;
    });

    return () => {
      isCancelled = true;
      if (leafletMapRef.current) {
        leafletMapRef.current.remove();
        leafletMapRef.current = null;
        tileLayerRef.current = null;
        beltGroupRef.current = null;
        terminalsGroupRef.current = null;
      }
    };
  }, []);

  // 2. Update Tile Layer on mapStyle change
  useEffect(() => {
    if (!leafletMapRef.current || !tileLayerRef.current) return;
    import('leaflet').then(() => {
      if (!leafletMapRef.current) return;
      const provider = TILE_PROVIDERS[mapStyle] ?? TILE_PROVIDERS.dark;
      tileLayerRef.current.setUrl(provider.url);
    });
  }, [mapStyle]);

  // 3. Auto-Fit Map Bounds on Waterway Change
  useEffect(() => {
    if (!leafletMapRef.current) return;
    const bounds = WATERWAY_BOUNDS[selectedWaterway];
    leafletMapRef.current.fitBounds(bounds, { padding: [60, 60], animate: true });
  }, [selectedWaterway]);

  // 4. Render Bold Electric Blue Highway Strip Along Entire NW Route
  useEffect(() => {
    if (!leafletMapRef.current || !beltGroupRef.current || !terminalsGroupRef.current) return;

    import('leaflet').then((L) => {
      if (!beltGroupRef.current || !terminalsGroupRef.current) return;
      beltGroupRef.current.clearLayers();
      terminalsGroupRef.current.clearLayers();

      const positions: [number, number][] = activeBelt.map((pt) => [pt.lat, pt.lng]);
      
      // ELECTRIC BLUE PALETTE (Visible Even Zoomed Out Across India)
      const primaryBlue = '#0284c7'; // Deep Electric Blue Outer Ribbon
      const coreBlue = '#38bdf8';    // Bright Neon Cyan Core Strip
      const highlightWhite = '#ffffff';

      // 4a. Ultra-Wide Glowing Outer Buffer Ribbon (36px wide)
      const outerGlow = L.polyline(positions, {
        color: primaryBlue,
        weight: 36,
        opacity: 0.55,
        lineCap: 'round',
        lineJoin: 'round',
      });

      // 4b. Bright Electric Blue Core Strip (18px wide)
      const innerStrip = L.polyline(positions, {
        color: coreBlue,
        weight: 18,
        opacity: 1,
        lineCap: 'round',
        lineJoin: 'round',
      });

      // 4c. Center High-Contrast White Baseline (4px wide)
      const centerLine = L.polyline(positions, {
        color: highlightWhite,
        weight: 4,
        opacity: 0.9,
        lineCap: 'round',
        lineJoin: 'round',
      });

      const routeLabel = selectedWaterway === 'NW-1'
        ? 'National Waterway 1 (Ganga Reach)'
        : 'National Waterway 2 (Brahmaputra Reach)';

      const tooltipContent = `
        <div style="background:#09090b; color:#fff; border:1.5px solid ${coreBlue}; padding:10px 14px; border-radius:10px; font-family:sans-serif; font-size:12px; box-shadow:0 12px 30px rgba(0,0,0,0.95);">
          <div style="font-weight:bold; font-size:13.5px; color:${coreBlue}; margin-bottom:4px;">
            ${routeLabel}
          </div>
          <div>Total Route Distance: <b>${selectedWaterway === 'NW-1' ? '1,620 km' : '891 km'}</b></div>
          <div>Inland Terminals: <b>${activeBelt.filter((p) => p.isTerminal).length} Multimodal Ports & Locks</b></div>
          <div style="font-size:11px; color:#34d399; font-weight:bold; margin-top:4px;">Channel Navigability: Operational (LAD ≥ 3.0m)</div>
        </div>
      `;

      outerGlow.bindTooltip(tooltipContent, { sticky: true, opacity: 1 });
      innerStrip.bindTooltip(tooltipContent, { sticky: true, opacity: 1 });
      centerLine.bindTooltip(tooltipContent, { sticky: true, opacity: 1 });

      beltGroupRef.current.addLayer(outerGlow);
      beltGroupRef.current.addLayer(innerStrip);
      beltGroupRef.current.addLayer(centerLine);

      // 4d. Render High-Contrast Terminal Anchor Circles Along the Blue Strip
      activeBelt.forEach((pt) => {
        if (!pt.isTerminal) return;

        const pin = L.circleMarker([pt.lat, pt.lng], {
          radius: 9,
          fillColor: '#ffffff',
          color: primaryBlue,
          weight: 4,
          fillOpacity: 1,
        });

        const pinTooltip = `
          <div style="background:#09090b; color:#fff; border:2px solid ${coreBlue}; padding:8px 12px; border-radius:8px; font-family:sans-serif; font-size:12px;">
            <div style="font-weight:bold; color:${coreBlue}; font-size:13px;">⚓ ${pt.name}</div>
            <div>Waterway Chainage: <b>km ${pt.km}</b></div>
            <div>Navigability Depth: <b style="color:#34d399;">${pt.depthM.toFixed(1)} m</b></div>
          </div>
        `;

        pin.bindTooltip(pinTooltip, { sticky: true, opacity: 1 });
        terminalsGroupRef.current.addLayer(pin);
      });
    });
  }, [activeBelt, selectedWaterway]);

  return (
    <div className={cn('relative w-full h-full bg-black overflow-hidden', className)}>
      {/* Native Browser Leaflet Canvas */}
      <div ref={containerRef} className="w-full h-full z-10 bg-black" />

      {/* Clean Top Right Controls (No Collision with Top Left Telemetry) */}
      <div className="absolute top-4 right-6 z-20 flex items-center gap-2">
        <div className="saas-card p-1 flex items-center gap-1 bg-black/90 border-zinc-800 shadow-2xl backdrop-blur-md">
          <button
            onClick={() => setMapStyle('dark')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              mapStyle === 'dark'
                ? 'bg-white text-zinc-900 shadow-sm'
                : 'text-zinc-400 hover:text-white'
            }`}
          >
            <Moon size={13} />
            <span>Dark GIS</span>
          </button>

          <button
            onClick={() => setMapStyle('satellite')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              mapStyle === 'satellite'
                ? 'bg-white text-zinc-900 shadow-sm'
                : 'text-zinc-400 hover:text-white'
            }`}
          >
            <Satellite size={13} />
            <span>Satellite HD</span>
          </button>

          <button
            onClick={() => setMapStyle('light')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              mapStyle === 'light'
                ? 'bg-white text-zinc-900 shadow-sm'
                : 'text-zinc-400 hover:text-white'
            }`}
          >
            <Sun size={13} />
            <span>Light</span>
          </button>
        </div>

        <button
          onClick={() => setShowLegend((v) => !v)}
          className="px-3 py-1.5 rounded-lg bg-black/90 border border-zinc-800 text-xs font-semibold text-zinc-300 hover:text-white shadow-xl backdrop-blur-md flex items-center gap-1.5"
        >
          <Layers size={13} />
          <span>{showLegend ? 'Hide Legend' : 'Show Legend'}</span>
        </button>
      </div>

      {/* Bottom Left Legend Card */}
      <AnimatePresence>
        {showLegend && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 10 }}
            className="absolute bottom-6 left-6 z-20"
          >
            <MapLegendCard />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
