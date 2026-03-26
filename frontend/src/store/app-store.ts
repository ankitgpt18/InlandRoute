// ============================================================
// AIDSTL — Inland Waterway Navigability Prediction
// Zustand Global App Store
// ============================================================

import { create } from 'zustand';
import { devtools, persist, subscribeWithSelector } from 'zustand/middleware';
import type {
  WaterwayId,
  MapStyle,
  Month,
  MapViewport,
  NavigabilityClass,
} from '@/types';

// ─── State Shape ──────────────────────────────────────────────────────────────

export interface AppStore {
  // ── Waterway / Time Selection ────────────────────────────────────────────
  selectedWaterway: WaterwayId;
  selectedMonth: Month;
  selectedYear: number;

  // ── Segment Selection ────────────────────────────────────────────────────
  selectedSegmentId: string | null;
  hoveredSegmentId: string | null;

  // ── Map Controls ─────────────────────────────────────────────────────────
  mapStyle: MapStyle;
  showDepthOverlay: boolean;
  showWidthOverlay: boolean;
  showAlerts: boolean;
  showConfidenceLayer: boolean;
  mapViewport: MapViewport;

  // ── UI State ─────────────────────────────────────────────────────────────
  sidebarCollapsed: boolean;
  alertsPanelOpen: boolean;
  detailPanelOpen: boolean;
  isLoadingMap: boolean;

  // ── Filter / Alert State ─────────────────────────────────────────────────
  alertSeverityFilter: 'ALL' | 'CRITICAL' | 'WARNING' | 'INFO';
  navigabilityFilter: NavigabilityClass | 'ALL';

  // ── Actions: Waterway & Time ─────────────────────────────────────────────
  setSelectedWaterway: (waterway: WaterwayId) => void;
  setSelectedMonth: (month: Month) => void;
  setSelectedYear: (year: number) => void;
  setMonthYear: (month: Month, year: number) => void;
  goToPreviousMonth: () => void;
  goToNextMonth: () => void;

  // ── Actions: Segment ─────────────────────────────────────────────────────
  setSelectedSegmentId: (id: string | null) => void;
  setHoveredSegmentId: (id: string | null) => void;
  clearSegmentSelection: () => void;

  // ── Actions: Map ─────────────────────────────────────────────────────────
  setMapStyle: (style: MapStyle) => void;
  setShowDepthOverlay: (show: boolean) => void;
  setShowWidthOverlay: (show: boolean) => void;
  setShowAlerts: (show: boolean) => void;
  setShowConfidenceLayer: (show: boolean) => void;
  setMapViewport: (viewport: Partial<MapViewport>) => void;
  flyToWaterway: (waterway: WaterwayId) => void;
  flyToSegment: (lng: number, lat: number, zoom?: number) => void;

  // ── Actions: UI ──────────────────────────────────────────────────────────
  setSidebarCollapsed: (collapsed: boolean) => void;
  toggleSidebar: () => void;
  setAlertsPanelOpen: (open: boolean) => void;
  setDetailPanelOpen: (open: boolean) => void;
  setIsLoadingMap: (loading: boolean) => void;

  // ── Actions: Filters ─────────────────────────────────────────────────────
  setAlertSeverityFilter: (filter: 'ALL' | 'CRITICAL' | 'WARNING' | 'INFO') => void;
  setNavigabilityFilter: (filter: NavigabilityClass | 'ALL') => void;
  resetFilters: () => void;

  // ── Actions: Combined ────────────────────────────────────────────────────
  selectSegmentAndOpenPanel: (segmentId: string) => void;
  resetAll: () => void;
}

// ─── Default Viewports ───────────────────────────────────────────────────────

const NW1_VIEWPORT: MapViewport = {
  longitude: 84.0,
  latitude:  25.4,
  zoom:      6.5,
  pitch:     0,
  bearing:   0,
};

const NW2_VIEWPORT: MapViewport = {
  longitude: 92.5,
  latitude:  26.5,
  zoom:      6.8,
  pitch:     0,
  bearing:   0,
};

const INDIA_VIEWPORT: MapViewport = {
  longitude: 84.0,
  latitude:  24.0,
  zoom:      5.0,
  pitch:     0,
  bearing:   0,
};

// ─── Initial State ────────────────────────────────────────────────────────────

const currentDate   = new Date();
const currentMonth  = (currentDate.getMonth() + 1) as Month;
const currentYear   = currentDate.getFullYear();

const INITIAL_STATE = {
  // Waterway / Time
  selectedWaterway:     'NW-1' as WaterwayId,
  selectedMonth:        currentMonth,
  selectedYear:         currentYear,

  // Segment
  selectedSegmentId:    null as string | null,
  hoveredSegmentId:     null as string | null,

  // Map
  mapStyle:             'dark'   as MapStyle,
  showDepthOverlay:     true,
  showWidthOverlay:     false,
  showAlerts:           true,
  showConfidenceLayer:  false,
  mapViewport:          NW1_VIEWPORT,

  // UI
  sidebarCollapsed:     false,
  alertsPanelOpen:      false,
  detailPanelOpen:      false,
  isLoadingMap:         false,

  // Filters
  alertSeverityFilter:  'ALL' as const,
  navigabilityFilter:   'ALL' as const,
};

// ─── Store ────────────────────────────────────────────────────────────────────

export const useAppStore = create<AppStore>()(
  devtools(
    subscribeWithSelector(
      persist(
        (set, get) => ({
          ...INITIAL_STATE,

          // ── Waterway & Time Actions ─────────────────────────────────────

          setSelectedWaterway: (waterway) =>
            set(
              (state) => ({
                selectedWaterway:  waterway,
                selectedSegmentId: null,
                detailPanelOpen:   false,
                mapViewport:
                  waterway === 'NW-1' ? NW1_VIEWPORT : NW2_VIEWPORT,
              }),
              false,
              'setSelectedWaterway',
            ),

          setSelectedMonth: (month) =>
            set({ selectedMonth: month }, false, 'setSelectedMonth'),

          setSelectedYear: (year) =>
            set({ selectedYear: year }, false, 'setSelectedYear'),

          setMonthYear: (month, year) =>
            set({ selectedMonth: month, selectedYear: year }, false, 'setMonthYear'),

          goToPreviousMonth: () => {
            const { selectedMonth, selectedYear } = get();
            if (selectedMonth === 1) {
              set(
                { selectedMonth: 12, selectedYear: selectedYear - 1 },
                false,
                'goToPreviousMonth',
              );
            } else {
              set(
                { selectedMonth: (selectedMonth - 1) as Month },
                false,
                'goToPreviousMonth',
              );
            }
          },

          goToNextMonth: () => {
            const { selectedMonth, selectedYear } = get();
            // Don't go beyond current month
            const now = new Date();
            const isCurrentOrFuture =
              selectedYear > now.getFullYear() ||
              (selectedYear === now.getFullYear() &&
                selectedMonth >= now.getMonth() + 1);
            if (isCurrentOrFuture) return;

            if (selectedMonth === 12) {
              set(
                { selectedMonth: 1, selectedYear: selectedYear + 1 },
                false,
                'goToNextMonth',
              );
            } else {
              set(
                { selectedMonth: (selectedMonth + 1) as Month },
                false,
                'goToNextMonth',
              );
            }
          },

          // ── Segment Actions ─────────────────────────────────────────────

          setSelectedSegmentId: (id) =>
            set(
              { selectedSegmentId: id, detailPanelOpen: id !== null },
              false,
              'setSelectedSegmentId',
            ),

          setHoveredSegmentId: (id) =>
            set({ hoveredSegmentId: id }, false, 'setHoveredSegmentId'),

          clearSegmentSelection: () =>
            set(
              { selectedSegmentId: null, detailPanelOpen: false },
              false,
              'clearSegmentSelection',
            ),

          // ── Map Actions ─────────────────────────────────────────────────

          setMapStyle: (style) =>
            set({ mapStyle: style }, false, 'setMapStyle'),

          setShowDepthOverlay: (show) =>
            set({ showDepthOverlay: show }, false, 'setShowDepthOverlay'),

          setShowWidthOverlay: (show) =>
            set({ showWidthOverlay: show }, false, 'setShowWidthOverlay'),

          setShowAlerts: (show) =>
            set({ showAlerts: show }, false, 'setShowAlerts'),

          setShowConfidenceLayer: (show) =>
            set({ showConfidenceLayer: show }, false, 'setShowConfidenceLayer'),

          setMapViewport: (viewport) =>
            set(
              (state) => ({
                mapViewport: { ...state.mapViewport, ...viewport },
              }),
              false,
              'setMapViewport',
            ),

          flyToWaterway: (waterway) =>
            set(
              {
                mapViewport:
                  waterway === 'NW-1' ? NW1_VIEWPORT : NW2_VIEWPORT,
              },
              false,
              'flyToWaterway',
            ),

          flyToSegment: (lng, lat, zoom = 10) =>
            set(
              (state) => ({
                mapViewport: {
                  ...state.mapViewport,
                  longitude: lng,
                  latitude:  lat,
                  zoom,
                },
              }),
              false,
              'flyToSegment',
            ),

          // ── UI Actions ──────────────────────────────────────────────────

          setSidebarCollapsed: (collapsed) =>
            set({ sidebarCollapsed: collapsed }, false, 'setSidebarCollapsed'),

          toggleSidebar: () =>
            set(
              (state) => ({ sidebarCollapsed: !state.sidebarCollapsed }),
              false,
              'toggleSidebar',
            ),

          setAlertsPanelOpen: (open) =>
            set({ alertsPanelOpen: open }, false, 'setAlertsPanelOpen'),

          setDetailPanelOpen: (open) =>
            set(
              { detailPanelOpen: open, selectedSegmentId: open ? get().selectedSegmentId : null },
              false,
              'setDetailPanelOpen',
            ),

          setIsLoadingMap: (loading) =>
            set({ isLoadingMap: loading }, false, 'setIsLoadingMap'),

          // ── Filter Actions ──────────────────────────────────────────────

          setAlertSeverityFilter: (filter) =>
            set(
              { alertSeverityFilter: filter },
              false,
              'setAlertSeverityFilter',
            ),

          setNavigabilityFilter: (filter) =>
            set(
              { navigabilityFilter: filter },
              false,
              'setNavigabilityFilter',
            ),

          resetFilters: () =>
            set(
              { alertSeverityFilter: 'ALL', navigabilityFilter: 'ALL' },
              false,
              'resetFilters',
            ),

          // ── Combined Actions ────────────────────────────────────────────

          selectSegmentAndOpenPanel: (segmentId) =>
            set(
              {
                selectedSegmentId: segmentId,
                detailPanelOpen:   true,
                alertsPanelOpen:   false,
              },
              false,
              'selectSegmentAndOpenPanel',
            ),

          resetAll: () =>
            set(
              {
                ...INITIAL_STATE,
                selectedMonth: currentMonth,
                selectedYear:  currentYear,
              },
              false,
              'resetAll',
            ),
        }),

        // ── Persist config — only persist non-transient UI prefs ──────────
        {
          name:    'aidstl-app-store',
          version: 1,
          partialize: (state) => ({
            selectedWaterway:    state.selectedWaterway,
            selectedMonth:       state.selectedMonth,
            selectedYear:        state.selectedYear,
            mapStyle:            state.mapStyle,
            showDepthOverlay:    state.showDepthOverlay,
            showWidthOverlay:    state.showWidthOverlay,
            showAlerts:          state.showAlerts,
            sidebarCollapsed:    state.sidebarCollapsed,
          }),
        },
      ),
    ),
    {
      name:    'AIDSTL Store',
      enabled: process.env.NODE_ENV === 'development',
    },
  ),
);

// ─── Selector Hooks (memoisation helpers) ─────────────────────────────────────

/** Current waterway + month + year selection. */
export const useWaterwaySelection = () =>
  useAppStore((s) => ({
    selectedWaterway: s.selectedWaterway,
    selectedMonth:    s.selectedMonth,
    selectedYear:     s.selectedYear,
  }));

/** Map display settings. */
export const useMapSettings = () =>
  useAppStore((s) => ({
    mapStyle:            s.mapStyle,
    showDepthOverlay:    s.showDepthOverlay,
    showWidthOverlay:    s.showWidthOverlay,
    showAlerts:          s.showAlerts,
    showConfidenceLayer: s.showConfidenceLayer,
    mapViewport:         s.mapViewport,
  }));

/** Currently selected / hovered segment IDs. */
export const useSegmentSelection = () =>
  useAppStore((s) => ({
    selectedSegmentId: s.selectedSegmentId,
    hoveredSegmentId:  s.hoveredSegmentId,
  }));

/** Panel open/closed states. */
export const usePanelState = () =>
  useAppStore((s) => ({
    sidebarCollapsed: s.sidebarCollapsed,
    alertsPanelOpen:  s.alertsPanelOpen,
    detailPanelOpen:  s.detailPanelOpen,
  }));

/** Active filter values. */
export const useFilters = () =>
  useAppStore((s) => ({
    alertSeverityFilter: s.alertSeverityFilter,
    navigabilityFilter:  s.navigabilityFilter,
  }));

// ─── Viewport constants (re-exported for convenience) ────────────────────────
export { NW1_VIEWPORT, NW2_VIEWPORT, INDIA_VIEWPORT };
