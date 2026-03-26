// ============================================================
// AIDSTL — Inland Waterway Navigability Prediction System
// RiverMap — Interactive Mapbox navigability overlay
// ============================================================

"use client";

import type mapboxgl from "mapbox-gl";
import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import Map, {
  Layer,
  Source,
  NavigationControl,
  ScaleControl,
  type MapRef,
  type MapLayerMouseEvent,
  type ViewStateChangeEvent,
} from "react-map-gl";
import { motion, AnimatePresence } from "framer-motion";
import {
  Layers,
  Satellite,
  Moon,
  Sun,
  Maximize2,
  Minimize2,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Eye,
  EyeOff,
  Info,
  ChevronDown,
} from "lucide-react";
import { useAppStore } from "@/store/app-store";
import {
  getMockNavigabilityMap,
  buildNW1GeoJSON,
  buildNW2GeoJSON,
} from "@/lib/mock-data";
import { MapLegendCard } from "@/components/ui/navigability-badge";
import { cn } from "@/lib/utils";
import type { NavigabilityClass, WaterwayId, MapStyle } from "@/types";

// ─── Mapbox Token ──────────────────────────────────────────────────────────────

const MAPBOX_TOKEN = process.env.NEXT_PUBLIC_MAPBOX_TOKEN ?? "";

// ─── Map Styles ────────────────────────────────────────────────────────────────

const MAP_STYLES: Record<MapStyle, string> = {
  dark: "mapbox://styles/mapbox/dark-v11",
  satellite: "mapbox://styles/mapbox/satellite-streets-v12",
  light: "mapbox://styles/mapbox/light-v11",
};

// ─── Navigability colour lookup ────────────────────────────────────────────────

const NAV_COLORS: Record<NavigabilityClass, string> = {
  navigable: "#22c55e",
  conditional: "#f59e0b",
  non_navigable: "#ef4444",
};

// Mapbox expression: maps the "navigability_class" feature property → colour
const COLOR_EXPRESSION: mapboxgl.Expression = [
  "match",
  ["get", "navigability_class"],
  "navigable",
  "#22c55e",
  "conditional",
  "#f59e0b",
  "non_navigable",
  "#ef4444",
  /* fallback */ "#64748b",
];

// Width expression — thicker lines for more navigable segments
const WIDTH_EXPRESSION: mapboxgl.Expression = [
  "interpolate",
  ["linear"],
  ["zoom"],
  5,
  [
    "match",
    ["get", "navigability_class"],
    "navigable",
    3,
    "conditional",
    2,
    1.5,
  ],
  10,
  ["match", ["get", "navigability_class"], "navigable", 6, "conditional", 5, 4],
  14,
  [
    "match",
    ["get", "navigability_class"],
    "navigable",
    10,
    "conditional",
    8,
    6,
  ],
];

// ─── Default viewports ─────────────────────────────────────────────────────────

const WATERWAY_VIEWS: Record<
  WaterwayId,
  { longitude: number; latitude: number; zoom: number }
> = {
  "NW-1": { longitude: 84.0, latitude: 25.4, zoom: 6.5 },
  "NW-2": { longitude: 92.5, latitude: 26.5, zoom: 6.8 },
};

// ─── Types ─────────────────────────────────────────────────────────────────────

interface HoverInfo {
  x: number;
  y: number;
  segmentId: string;
  waterwayId: string;
  kmStart: number;
  kmEnd: number;
  navigabilityClass: NavigabilityClass;
  depthM: number;
  widthM: number;
  confidence: number;
  velocityMs?: number;
  state?: string;
}

interface RiverMapProps {
  /** If true the map occupies the full viewport (maps page) */
  fullscreen?: boolean;
  className?: string;
  /** Called when the user clicks on a segment */
  onSegmentClick?: (segmentId: string) => void;
}

// ─── Segment Hover Tooltip ─────────────────────────────────────────────────────

function SegmentTooltip({ info }: { info: HoverInfo }) {
  const cls = info.navigabilityClass;
  const colorMap: Record<NavigabilityClass, string> = {
    navigable: "text-emerald-400",
    conditional: "text-amber-400",
    non_navigable: "text-red-400",
  };
  const bgMap: Record<NavigabilityClass, string> = {
    navigable: "border-emerald-500/30",
    conditional: "border-amber-500/30",
    non_navigable: "border-red-500/30",
  };
  const labels: Record<NavigabilityClass, string> = {
    navigable: "Navigable",
    conditional: "Conditional",
    non_navigable: "Non-Navigable",
  };

  return (
    <motion.div
      key={info.segmentId}
      initial={{ opacity: 0, scale: 0.92, y: 4 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.92, y: -4 }}
      transition={{ duration: 0.12, ease: [0.16, 1, 0.3, 1] }}
      className={cn(
        "absolute z-50 pointer-events-none",
        "bg-slate-900/95 backdrop-blur-xl",
        "border rounded-xl",
        "px-3 py-2.5 min-w-[210px]",
        "shadow-2xl shadow-black/60",
        bgMap[cls],
      )}
      style={{
        left: info.x + 14,
        top: info.y - 14,
        // Flip to the left if too close to right edge
        transform:
          info.x > window.innerWidth - 260
            ? "translateX(calc(-100% - 28px))"
            : undefined,
      }}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-2 pb-2 border-b border-white/[0.07]">
        <div>
          <div className="text-[12px] font-bold text-slate-100">
            {info.waterwayId} · Seg {info.segmentId.split("-").pop()}
          </div>
          <div className="text-[10px] text-slate-500 mt-0.5 font-medium tabular-nums">
            {info.kmStart.toFixed(0)}–{info.kmEnd.toFixed(0)} km
            {info.state ? ` · ${info.state}` : ""}
          </div>
        </div>
        {/* Status pill */}
        <span
          className={cn(
            "text-[9px] font-bold tracking-wider uppercase px-2 py-0.5 rounded-full",
            cls === "navigable" && "bg-emerald-500/15 text-emerald-400",
            cls === "conditional" && "bg-amber-500/15 text-amber-400",
            cls === "non_navigable" && "bg-red-500/15 text-red-400",
          )}
        >
          {labels[cls]}
        </span>
      </div>

      {/* Data rows */}
      <div className="grid grid-cols-2 gap-x-4 gap-y-1.5">
        <TooltipRow
          label="Depth"
          value={`${info.depthM.toFixed(2)} m`}
          highlight={colorMap[cls]}
        />
        <TooltipRow label="Width" value={`${info.widthM.toFixed(0)} m`} />
        <TooltipRow
          label="Confidence"
          value={`${(info.confidence * 100).toFixed(0)}%`}
        />
        {info.velocityMs !== undefined && (
          <TooltipRow
            label="Velocity"
            value={`${info.velocityMs.toFixed(2)} m/s`}
          />
        )}
      </div>

      {/* Click hint */}
      <div className="mt-2 pt-1.5 border-t border-white/[0.05]">
        <p className="text-[9px] text-slate-600">
          Click to view full segment details
        </p>
      </div>
    </motion.div>
  );
}

function TooltipRow({
  label,
  value,
  highlight,
}: {
  label: string;
  value: string;
  highlight?: string;
}) {
  return (
    <div>
      <div className="text-[10px] text-slate-600 font-medium">{label}</div>
      <div
        className={cn(
          "text-[12px] font-bold tabular-nums",
          highlight ? "" : "text-slate-200",
        )}
        style={highlight ? { color: highlight } : undefined}
      >
        {value}
      </div>
    </div>
  );
}

// ─── Map Style Toggle ──────────────────────────────────────────────────────────

function MapStyleToggle({
  current,
  onChange,
}: {
  current: MapStyle;
  onChange: (s: MapStyle) => void;
}) {
  const [open, setOpen] = useState(false);

  const OPTIONS: { value: MapStyle; label: string; icon: React.ElementType }[] =
    [
      { value: "dark", label: "Dark", icon: Moon },
      { value: "satellite", label: "Satellite", icon: Satellite },
      { value: "light", label: "Light", icon: Sun },
    ];

  const currentOpt = OPTIONS.find((o) => o.value === current) ?? OPTIONS[0];
  const CurrentIcon = currentOpt.icon;

  return (
    <div className="relative">
      <button
        onClick={() => setOpen((v) => !v)}
        className="
          flex items-center gap-1.5 px-2.5 py-2 rounded-xl
          bg-slate-900/90 backdrop-blur-sm
          border border-white/[0.1]
          text-slate-300 hover:text-white
          text-[11px] font-semibold
          transition-all duration-150
          shadow-lg
        "
        aria-label="Change map style"
      >
        <CurrentIcon size={13} />
        <span className="hidden sm:inline">{currentOpt.label}</span>
        <ChevronDown
          size={11}
          className={`transition-transform duration-200 ${open ? "rotate-180" : ""}`}
        />
      </button>

      <AnimatePresence>
        {open && (
          <>
            <div
              className="fixed inset-0 z-40"
              onClick={() => setOpen(false)}
            />
            <motion.div
              initial={{ opacity: 0, y: -6, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -6, scale: 0.95 }}
              transition={{ duration: 0.12, ease: [0.16, 1, 0.3, 1] }}
              className="
                absolute right-0 top-full mt-1.5 w-36 z-50
                bg-slate-900/95 backdrop-blur-xl
                border border-white/[0.1] rounded-xl
                shadow-2xl overflow-hidden
              "
            >
              {OPTIONS.map((opt) => {
                const OptionIcon = opt.icon;
                return (
                  <button
                    key={opt.value}
                    onClick={() => {
                      onChange(opt.value);
                      setOpen(false);
                    }}
                    className={cn(
                      "w-full flex items-center gap-2.5 px-3 py-2.5",
                      "text-[12px] font-medium transition-colors duration-100",
                      current === opt.value
                        ? "bg-blue-500/15 text-blue-300"
                        : "text-slate-400 hover:text-white hover:bg-white/[0.06]",
                    )}
                  >
                    <OptionIcon size={13} className="flex-shrink-0" />
                    {opt.label}
                    {current === opt.value && (
                      <span className="ml-auto text-[10px] text-blue-400 font-bold">
                        ✓
                      </span>
                    )}
                  </button>
                );
              })}
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  );
}

// ─── Layer Toggles Panel ───────────────────────────────────────────────────────

function LayerToggles({
  showDepth,
  showWidth,
  showAlerts,
  onToggleDepth,
  onToggleWidth,
  onToggleAlerts,
}: {
  showDepth: boolean;
  showWidth: boolean;
  showAlerts: boolean;
  onToggleDepth: () => void;
  onToggleWidth: () => void;
  onToggleAlerts: () => void;
}) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="flex flex-col items-end gap-1.5">
      {/* Toggle button */}
      <button
        onClick={() => setExpanded((v) => !v)}
        className="
          flex items-center gap-1.5 px-2.5 py-2 rounded-xl
          bg-slate-900/90 backdrop-blur-sm
          border border-white/[0.1]
          text-slate-300 hover:text-white
          text-[11px] font-semibold
          transition-all duration-150 shadow-lg
        "
        aria-label="Toggle map layers"
        aria-expanded={expanded}
      >
        <Layers size={13} />
        <span className="hidden sm:inline">Layers</span>
      </button>

      {/* Layer checkboxes */}
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ opacity: 0, y: -6, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -6, scale: 0.95 }}
            transition={{ duration: 0.15, ease: [0.16, 1, 0.3, 1] }}
            className="
              w-44 bg-slate-900/95 backdrop-blur-xl
              border border-white/[0.1] rounded-xl
              shadow-2xl overflow-hidden p-1
            "
          >
            {[
              {
                id: "depth",
                label: "Depth Overlay",
                active: showDepth,
                toggle: onToggleDepth,
              },
              {
                id: "width",
                label: "Width Indicators",
                active: showWidth,
                toggle: onToggleWidth,
              },
              {
                id: "alerts",
                label: "Risk Alerts",
                active: showAlerts,
                toggle: onToggleAlerts,
              },
            ].map((item) => (
              <button
                key={item.id}
                onClick={item.toggle}
                className="
                  w-full flex items-center justify-between gap-2 px-3 py-2.5
                  rounded-lg transition-colors duration-100
                  text-[12px] font-medium
                  hover:bg-white/[0.05]
                "
              >
                <span
                  className={item.active ? "text-slate-200" : "text-slate-500"}
                >
                  {item.label}
                </span>
                {item.active ? (
                  <Eye size={13} className="text-blue-400 flex-shrink-0" />
                ) : (
                  <EyeOff size={13} className="text-slate-600 flex-shrink-0" />
                )}
              </button>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ─── Loading Skeleton ──────────────────────────────────────────────────────────

function MapLoadingSkeleton() {
  return (
    <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-950 z-10">
      {/* Ripple animation */}
      <div className="relative flex items-center justify-center mb-6">
        {[0, 1, 2].map((i) => (
          <motion.div
            key={i}
            className="absolute rounded-full border border-blue-500/40"
            style={{ width: 40 + i * 32, height: 40 + i * 32 }}
            animate={{ opacity: [0.8, 0], scale: [0.9, 1.3] }}
            transition={{
              duration: 1.8,
              repeat: Infinity,
              delay: i * 0.5,
              ease: "easeOut",
            }}
          />
        ))}
        <div
          className="relative w-10 h-10 rounded-full flex items-center justify-center"
          style={{ background: "linear-gradient(135deg, #0369a1, #3b82f6)" }}
        >
          <svg
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="none"
            className="text-white"
          >
            <path
              d="M2 12C2 12 5 7 12 7C19 7 22 12 22 12C22 12 19 17 12 17C5 17 2 12 2 12Z"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
            />
            <circle
              cx="12"
              cy="12"
              r="3"
              stroke="currentColor"
              strokeWidth="2"
            />
          </svg>
        </div>
      </div>

      <motion.p
        animate={{ opacity: [0.5, 1, 0.5] }}
        transition={{ duration: 2, repeat: Infinity }}
        className="text-sm font-medium text-slate-400"
      >
        Loading satellite data…
      </motion.p>
      <p className="text-xs text-slate-600 mt-1">
        Compositing Sentinel-2 imagery
      </p>
    </div>
  );
}

// ─── Stats Overlay ─────────────────────────────────────────────────────────────

function MapStatsOverlay({
  waterwayId,
  month,
}: {
  waterwayId: WaterwayId;
  month: number;
}) {
  const navMap = getMockNavigabilityMap(waterwayId, month);
  const navPct = navMap.navigable_pct ?? 0;
  const criticalCount = navMap.non_navigable_count ?? 0;

  const items = [
    {
      label: "Navigable",
      value: `${navPct.toFixed(0)}%`,
      color: "text-emerald-400",
      bg: "bg-emerald-500/10",
      border: "border-emerald-500/20",
    },
    {
      label: "Segments",
      value: navMap.total_segments?.toString() ?? "—",
      color: "text-blue-400",
      bg: "bg-blue-500/10",
      border: "border-blue-500/20",
    },
    {
      label: "Critical",
      value: criticalCount.toString(),
      color: criticalCount > 0 ? "text-red-400" : "text-slate-500",
      bg: criticalCount > 0 ? "bg-red-500/10" : "bg-white/5",
      border: criticalCount > 0 ? "border-red-500/20" : "border-white/10",
    },
  ];

  return (
    <div className="flex flex-col gap-1.5">
      {items.map((item) => (
        <div
          key={item.label}
          className={cn(
            "flex items-center justify-between gap-3",
            "px-2.5 py-1.5 rounded-lg",
            "border backdrop-blur-sm",
            item.bg,
            item.border,
          )}
        >
          <span className="text-[10px] font-medium text-slate-500">
            {item.label}
          </span>
          <span
            className={cn("text-[12px] font-bold tabular-nums", item.color)}
          >
            {item.value}
          </span>
        </div>
      ))}
    </div>
  );
}

// ─── Main RiverMap Component ───────────────────────────────────────────────────

export function RiverMap({
  fullscreen = false,
  className,
  onSegmentClick,
}: RiverMapProps) {
  const mapRef = useRef<MapRef>(null);

  // ── Store ──────────────────────────────────────────────────────────────────
  const selectedWaterway = useAppStore((s) => s.selectedWaterway);
  const selectedMonth = useAppStore((s) => s.selectedMonth);
  const selectedYear = useAppStore((s) => s.selectedYear);
  const selectedSegmentId = useAppStore((s) => s.selectedSegmentId);
  const mapStyle = useAppStore((s) => s.mapStyle);
  const showDepthOverlay = useAppStore((s) => s.showDepthOverlay);
  const showWidthOverlay = useAppStore((s) => s.showWidthOverlay);
  const showAlerts = useAppStore((s) => s.showAlerts);
  const mapViewport = useAppStore((s) => s.mapViewport);
  const setMapStyle = useAppStore((s) => s.setMapStyle);
  const setShowDepthOverlay = useAppStore((s) => s.setShowDepthOverlay);
  const setShowWidthOverlay = useAppStore((s) => s.setShowWidthOverlay);
  const setShowAlerts = useAppStore((s) => s.setShowAlerts);
  const setMapViewport = useAppStore((s) => s.setMapViewport);
  const setSelectedSegmentId = useAppStore((s) => s.setSelectedSegmentId);
  const setHoveredSegmentId = useAppStore((s) => s.setHoveredSegmentId);

  // ── Local state ───────────────────────────────────────────────────────────
  const [mapLoaded, setMapLoaded] = useState(false);
  const [hoverInfo, setHoverInfo] = useState<HoverInfo | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(fullscreen);
  const [showLegend, setShowLegend] = useState(true);
  const [showStats, setShowStats] = useState(true);

  // ── GeoJSON data ─────────────────────────────────────────────────────────
  // In production this would come from React Query / API
  // For now we use the rich mock data builders
  const geojsonData = useMemo(() => {
    return selectedWaterway === "NW-1"
      ? buildNW1GeoJSON(selectedMonth)
      : buildNW2GeoJSON(selectedMonth);
  }, [selectedWaterway, selectedMonth]);

  // ── Fly to new waterway on selection change ───────────────────────────────
  useEffect(() => {
    if (!mapRef.current || !mapLoaded) return;
    const target = WATERWAY_VIEWS[selectedWaterway];
    mapRef.current.flyTo({
      center: [target.longitude, target.latitude],
      zoom: target.zoom,
      duration: 1800,
      essential: true,
    });
  }, [selectedWaterway, mapLoaded]);

  // ── Fly to selected segment ───────────────────────────────────────────────
  useEffect(() => {
    if (!mapRef.current || !mapLoaded || !selectedSegmentId || !geojsonData)
      return;
    const feature = geojsonData.features.find(
      (f) => f.properties?.segment_id === selectedSegmentId,
    );
    if (!feature?.geometry?.coordinates?.length) return;
    const coords = feature.geometry.coordinates as [number, number][];
    const midIdx = Math.floor(coords.length / 2);
    const [lng, lat] = coords[midIdx];
    mapRef.current.flyTo({
      center: [lng, lat],
      zoom: Math.max(mapViewport.zoom ?? 7, 9),
      duration: 1200,
      essential: true,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedSegmentId, mapLoaded]);

  // ── Handlers ─────────────────────────────────────────────────────────────

  const handleMapLoad = useCallback(() => {
    setMapLoaded(true);
  }, []);

  const handleMouseMove = useCallback(
    (e: MapLayerMouseEvent) => {
      const features = e.features;
      if (!features || features.length === 0) {
        setHoverInfo(null);
        setHoveredSegmentId(null);
        if (mapRef.current) {
          mapRef.current.getCanvas().style.cursor = "";
        }
        return;
      }
      const f = features[0];
      const p = f.properties as Record<string, unknown> | null;
      if (!p) return;

      setHoveredSegmentId(p["segment_id"] as string);
      setHoverInfo({
        x: e.point.x,
        y: e.point.y,
        segmentId: p["segment_id"] as string,
        waterwayId: p["waterway_id"] as string,
        kmStart: p["km_start"] as number,
        kmEnd: p["km_end"] as number,
        navigabilityClass: p["navigability_class"] as NavigabilityClass,
        depthM: p["depth_m"] as number,
        widthM: p["width_m"] as number,
        confidence: p["confidence"] as number,
        velocityMs: p["velocity_ms"] as number | undefined,
        state: p["state"] as string | undefined,
      });

      if (mapRef.current) {
        mapRef.current.getCanvas().style.cursor = "pointer";
      }
    },
    [setHoveredSegmentId],
  );

  const handleMouseLeave = useCallback(() => {
    setHoverInfo(null);
    setHoveredSegmentId(null);
    if (mapRef.current) {
      mapRef.current.getCanvas().style.cursor = "";
    }
  }, [setHoveredSegmentId]);

  const handleClick = useCallback(
    (e: MapLayerMouseEvent) => {
      const features = e.features;
      if (!features || features.length === 0) {
        setSelectedSegmentId(null);
        return;
      }
      const p = features[0].properties as Record<string, unknown> | null;
      if (!p) return;
      const segId = p["segment_id"] as string;
      setSelectedSegmentId(segId);
      onSegmentClick?.(segId);
    },
    [setSelectedSegmentId, onSegmentClick],
  );

  const handleViewStateChange = useCallback(
    (e: ViewStateChangeEvent) => {
      setMapViewport({
        longitude: e.viewState.longitude,
        latitude: e.viewState.latitude,
        zoom: e.viewState.zoom,
        pitch: e.viewState.pitch,
        bearing: e.viewState.bearing,
      });
    },
    [setMapViewport],
  );

  const handleResetView = useCallback(() => {
    if (!mapRef.current) return;
    const target = WATERWAY_VIEWS[selectedWaterway];
    mapRef.current.flyTo({
      center: [target.longitude, target.latitude],
      zoom: target.zoom,
      pitch: 0,
      bearing: 0,
      duration: 1000,
    });
  }, [selectedWaterway]);

  // ── Layer paint properties ────────────────────────────────────────────────

  // Base segment line layer
  const segmentLinePaint: mapboxgl.LinePaint = {
    "line-color": COLOR_EXPRESSION,
    "line-width": WIDTH_EXPRESSION,
    "line-opacity": [
      "case",
      // Dim non-selected segments when one is selected
      [
        "all",
        ["!=", ["get", "segment_id"], selectedSegmentId ?? ""],
        ["!=", selectedSegmentId, null],
      ],
      0.45,
      // Full opacity otherwise
      0.92,
    ],
  };

  // Glow / halo layer behind the lines (subtle blur effect)
  const glowLinePaint: mapboxgl.LinePaint = {
    "line-color": COLOR_EXPRESSION,
    "line-width": ["interpolate", ["linear"], ["zoom"], 5, 6, 10, 14, 14, 22],
    "line-opacity": 0.15,
    "line-blur": 4,
  };

  // Selected segment highlight
  const selectedLinePaint: mapboxgl.LinePaint = {
    "line-color": "#ffffff",
    "line-width": ["interpolate", ["linear"], ["zoom"], 5, 5, 10, 9, 14, 14],
    "line-opacity": 0.8,
  };

  // Depth label layer (shown only when showDepthOverlay is true)
  const depthSymbolLayout: mapboxgl.SymbolLayout = {
    "text-field": ["concat", ["to-string", ["round", ["get", "depth_m"]]], "m"],
    "text-size": ["interpolate", ["linear"], ["zoom"], 8, 9, 12, 12],
    "text-font": ["DIN Offc Pro Medium", "Arial Unicode MS Regular"],
    "text-anchor": "center",
    "symbol-placement": "line",
    "text-max-angle": 30,
    "symbol-spacing": ["interpolate", ["linear"], ["zoom"], 8, 120, 12, 80],
  };

  const depthSymbolPaint: mapboxgl.SymbolPaint = {
    "text-color": COLOR_EXPRESSION as mapboxgl.Expression,
    "text-halo-color": "#020817",
    "text-halo-width": 2,
    "text-opacity": 0.9,
  };

  // ── Interactable layer IDs ─────────────────────────────────────────────────
  const interactiveLayers = ["river-segments", "river-glow"];

  // ── No token fallback ─────────────────────────────────────────────────────
  if (!MAPBOX_TOKEN) {
    return (
      <div
        className={cn(
          "relative flex flex-col items-center justify-center",
          "bg-slate-900 rounded-2xl border border-white/10",
          "text-center px-8 py-12",
          className,
        )}
      >
        <div className="w-12 h-12 rounded-xl bg-amber-500/15 border border-amber-500/30 flex items-center justify-center mb-4">
          <Info size={22} className="text-amber-400" />
        </div>
        <h3 className="text-base font-semibold text-slate-200 mb-2">
          Mapbox Token Required
        </h3>
        <p className="text-sm text-slate-500 max-w-xs leading-relaxed">
          Set{" "}
          <code className="text-amber-400 bg-amber-500/10 px-1.5 py-0.5 rounded text-xs">
            NEXT_PUBLIC_MAPBOX_TOKEN
          </code>{" "}
          in your{" "}
          <code className="text-slate-400 bg-white/5 px-1 rounded text-xs">
            .env.local
          </code>{" "}
          file to enable the interactive river map.
        </p>
        <div className="mt-6 grid grid-cols-3 gap-3 w-full max-w-xs">
          {(
            ["navigable", "conditional", "non_navigable"] as NavigabilityClass[]
          ).map((cls) => (
            <div
              key={cls}
              className="h-2 rounded-full"
              style={{ backgroundColor: NAV_COLORS[cls], opacity: 0.7 }}
            />
          ))}
        </div>
        <p className="text-[10px] text-slate-600 mt-2">
          Navigable · Conditional · Non-Navigable
        </p>
      </div>
    );
  }

  return (
    <div
      className={cn(
        "relative overflow-hidden rounded-2xl",
        isFullscreen
          ? "fixed inset-0 z-50 rounded-none"
          : "w-full h-full min-h-[400px]",
        className,
      )}
      style={!isFullscreen ? { height: "100%" } : undefined}
    >
      {/* ── Loading overlay ────────────────────────────────────── */}
      <AnimatePresence>
        {!mapLoaded && (
          <motion.div
            initial={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.4 }}
            className="absolute inset-0 z-20"
          >
            <MapLoadingSkeleton />
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── The Map ────────────────────────────────────────────── */}
      <Map
        ref={mapRef}
        mapboxAccessToken={MAPBOX_TOKEN}
        mapStyle={MAP_STYLES[mapStyle]}
        longitude={mapViewport.longitude}
        latitude={mapViewport.latitude}
        zoom={mapViewport.zoom}
        pitch={mapViewport.pitch ?? 0}
        bearing={mapViewport.bearing ?? 0}
        onMove={handleViewStateChange}
        onLoad={handleMapLoad}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        onClick={handleClick}
        interactiveLayerIds={interactiveLayers}
        reuseMaps
        style={{ width: "100%", height: "100%" }}
        attributionControl={false}
      >
        {/* ── Navigation control ──────────────────────────── */}
        <NavigationControl
          position="bottom-right"
          showCompass={true}
          showZoom={false}
          visualizePitch={true}
        />

        {/* ── Scale bar ───────────────────────────────────── */}
        <ScaleControl
          position="bottom-left"
          maxWidth={100}
          unit="metric"
          style={{ marginBottom: 8, marginLeft: 8 }}
        />

        {/* ── River segment data source ────────────────────── */}
        {geojsonData && (
          <Source
            id="river-segments"
            type="geojson"
            data={geojsonData}
            lineMetrics={true}
          >
            {/* Glow layer (rendered first = under) */}
            <Layer
              id="river-glow"
              type="line"
              paint={glowLinePaint}
              layout={{ "line-cap": "round", "line-join": "round" }}
            />

            {/* Main segment lines */}
            <Layer
              id="river-segments"
              type="line"
              paint={segmentLinePaint}
              layout={{ "line-cap": "round", "line-join": "round" }}
            />

            {/* Selected segment highlight */}
            {selectedSegmentId && (
              <Layer
                id="river-selected"
                type="line"
                filter={["==", ["get", "segment_id"], selectedSegmentId]}
                paint={selectedLinePaint}
                layout={{ "line-cap": "round", "line-join": "round" }}
              />
            )}

            {/* Depth labels (conditional on toggle) */}
            {showDepthOverlay && (
              <Layer
                id="river-depth-labels"
                type="symbol"
                layout={depthSymbolLayout}
                paint={depthSymbolPaint}
              />
            )}
          </Source>
        )}
      </Map>

      {/* ── Top-right controls ─────────────────────────────────── */}
      {mapLoaded && (
        <div className="absolute top-3 right-3 flex flex-col items-end gap-2 z-20">
          {/* Map style switcher */}
          <MapStyleToggle current={mapStyle} onChange={setMapStyle} />

          {/* Layer toggles */}
          <LayerToggles
            showDepth={showDepthOverlay}
            showWidth={showWidthOverlay}
            showAlerts={showAlerts}
            onToggleDepth={() => setShowDepthOverlay(!showDepthOverlay)}
            onToggleWidth={() => setShowWidthOverlay(!showWidthOverlay)}
            onToggleAlerts={() => setShowAlerts(!showAlerts)}
          />

          {/* Fullscreen toggle */}
          <button
            onClick={() => setIsFullscreen((v) => !v)}
            className="
              flex items-center justify-center w-9 h-9 rounded-xl
              bg-slate-900/90 backdrop-blur-sm
              border border-white/[0.1]
              text-slate-400 hover:text-white
              transition-colors duration-150 shadow-lg
            "
            aria-label={isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
          >
            {isFullscreen ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
          </button>

          {/* Reset view */}
          <button
            onClick={handleResetView}
            className="
              flex items-center justify-center w-9 h-9 rounded-xl
              bg-slate-900/90 backdrop-blur-sm
              border border-white/[0.1]
              text-slate-400 hover:text-white
              transition-colors duration-150 shadow-lg
            "
            aria-label="Reset map view"
            title="Reset to default view"
          >
            <RotateCcw size={14} />
          </button>
        </div>
      )}

      {/* ── Bottom-left legend ─────────────────────────────────── */}
      {mapLoaded && (
        <div className="absolute bottom-10 left-3 z-20 flex flex-col gap-2">
          {/* Toggle legend */}
          <button
            onClick={() => setShowLegend((v) => !v)}
            className="
              flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg
              bg-slate-900/90 backdrop-blur-sm
              border border-white/[0.1]
              text-slate-500 hover:text-slate-300
              text-[10px] font-semibold
              transition-colors duration-150 self-start
            "
          >
            {showLegend ? <EyeOff size={11} /> : <Eye size={11} />}
            Legend
          </button>

          <AnimatePresence>
            {showLegend && (
              <motion.div
                initial={{ opacity: 0, y: 8, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: 8, scale: 0.95 }}
                transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
              >
                <MapLegendCard />
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}

      {/* ── Top-left quick stats ───────────────────────────────── */}
      {mapLoaded && (
        <div className="absolute top-3 left-3 z-20">
          <AnimatePresence>
            {showStats && (
              <motion.div
                initial={{ opacity: 0, x: -12 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -12 }}
                transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
              >
                {/* Waterway name pill */}
                <div
                  className="
                  flex items-center gap-2 px-2.5 py-1.5 mb-2
                  bg-slate-900/90 backdrop-blur-sm
                  border border-white/[0.1] rounded-xl
                  text-[11px] font-semibold text-slate-300
                  shadow-lg
                "
                >
                  <span
                    className="w-2 h-2 rounded-full flex-shrink-0"
                    style={{
                      backgroundColor:
                        selectedWaterway === "NW-1" ? "#3b82f6" : "#8b5cf6",
                    }}
                  />
                  {selectedWaterway === "NW-1"
                    ? "NW-1 · Ganga"
                    : "NW-2 · Brahmaputra"}
                </div>

                {/* Quick stats */}
                <MapStatsOverlay
                  waterwayId={selectedWaterway}
                  month={selectedMonth}
                />
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}

      {/* ── Hover tooltip ──────────────────────────────────────── */}
      <AnimatePresence>
        {hoverInfo && <SegmentTooltip info={hoverInfo} />}
      </AnimatePresence>

      {/* ── Fullscreen ESC hint ─────────────────────────────────── */}
      {isFullscreen && mapLoaded && (
        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-20 pointer-events-none">
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="
              px-3 py-1.5 rounded-lg
              bg-slate-900/80 backdrop-blur-sm
              border border-white/[0.08]
              text-[11px] text-slate-500 font-medium
            "
          >
            Press{" "}
            <kbd className="px-1 py-0.5 bg-white/10 rounded text-[10px] font-mono">
              Esc
            </kbd>{" "}
            or use the
            <button
              onClick={() => setIsFullscreen(false)}
              className="pointer-events-auto ml-1 text-blue-400 hover:text-blue-300 underline underline-offset-2 transition-colors"
            >
              minimize button
            </button>{" "}
            to exit
          </motion.div>
        </div>
      )}

      {/* ── Mapbox attribution ─────────────────────────────────── */}
      <div className="absolute bottom-1 right-2 z-10 pointer-events-none">
        <span className="text-[9px] text-slate-700 font-medium">
          © Mapbox · © OpenStreetMap · Sentinel-2 / ESA Copernicus
        </span>
      </div>
    </div>
  );
}
