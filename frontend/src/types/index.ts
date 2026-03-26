// ============================================================
// AIDSTL — Inland Waterway Navigability Prediction
// TypeScript Type Definitions
// ============================================================

// ------------------------------------------------------------
// Core Enumerations
// ------------------------------------------------------------

export type NavigabilityClass = 'navigable' | 'conditional' | 'non_navigable';

export type WaterwayId = 'NW-1' | 'NW-2';

export type MapStyle = 'satellite' | 'dark' | 'light';

export type AlertSeverity = 'CRITICAL' | 'WARNING' | 'INFO';

export type AlertType =
  | 'DEPTH_CRITICAL'
  | 'DEPTH_WARNING'
  | 'WIDTH_RESTRICTION'
  | 'VELOCITY_HIGH'
  | 'SEASONAL_CLOSURE'
  | 'OBSTACLE_DETECTED'
  | 'FLOOD_RISK'
  | 'DROUGHT_RISK';

export type Month = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12;

// ------------------------------------------------------------
// GeoJSON / Geometry
// ------------------------------------------------------------

export interface Coordinate {
  longitude: number;
  latitude: number;
}

export interface BoundingBox {
  minLng: number;
  minLat: number;
  maxLng: number;
  maxLat: number;
}

export interface LineStringGeometry {
  type: 'LineString';
  coordinates: [number, number][];
}

export interface FeatureProperties {
  segment_id: string;
  waterway_id: WaterwayId;
  km_start: number;
  km_end: number;
  navigability_class: NavigabilityClass;
  depth_m: number;
  width_m: number;
  confidence: number;
  velocity_ms?: number;
}

export interface RiverFeature {
  type: 'Feature';
  id: string;
  geometry: LineStringGeometry;
  properties: FeatureProperties;
}

export interface RiverGeoJSON {
  type: 'FeatureCollection';
  features: RiverFeature[];
}

// ------------------------------------------------------------
// River Segment
// ------------------------------------------------------------

export interface RiverSegment {
  segment_id: string;
  waterway_id: WaterwayId;
  name: string;
  km_start: number;
  km_end: number;
  km_length: number;

  // Physical characteristics
  mean_width_m: number;
  mean_depth_m: number;
  bed_elevation_m: number;
  bankfull_depth_m: number;

  // Location metadata
  upstream_landmark?: string;
  downstream_landmark?: string;
  state: string;
  district?: string;

  // Navigation infrastructure
  has_jetty: boolean;
  has_barge_terminal: boolean;
  has_navigation_lock: boolean;

  geometry: LineStringGeometry;
}

// ------------------------------------------------------------
// Navigability Prediction
// ------------------------------------------------------------

export interface SpectralIndices {
  mndwi: number;         // Modified Normalized Difference Water Index
  ndwi: number;          // Normalized Difference Water Index
  ndvi: number;          // Normalized Difference Vegetation Index
  stumpf_ratio: number;  // Stumpf band ratio for depth estimation
  awei_sh: number;       // Automated Water Extraction Index (shadow)
  awei_nsh: number;      // Automated Water Extraction Index (no shadow)
}

export interface ConfidenceInterval {
  lower_95: number;
  upper_95: number;
  lower_80: number;
  upper_80: number;
  std_dev: number;
}

export interface NavigabilityPrediction {
  prediction_id: string;
  segment_id: string;
  waterway_id: WaterwayId;
  month: Month;
  year: number;
  created_at: string;

  // Primary outputs
  navigability_class: NavigabilityClass;
  predicted_depth_m: number;
  predicted_width_m: number;
  probability: number;         // 0–1 confidence in the predicted class
  risk_score: number;          // 0–100 composite risk

  // Confidence intervals
  depth_ci: ConfidenceInterval;
  width_ci?: ConfidenceInterval;

  // Spectral indices used as inputs
  spectral_indices: SpectralIndices;

  // Derived thresholds
  depth_threshold_m: number;   // typically 3.0 m for NW-1
  is_above_threshold: boolean;
  margin_m: number;            // depth − threshold (negative means non-navigable)

  // Additional hydrology
  discharge_m3s?: number;
  velocity_ms?: number;
  water_surface_area_km2?: number;
}

export interface NavigabilityMap {
  waterway_id: WaterwayId;
  month: Month;
  year: number;
  generated_at: string;

  // Aggregate stats
  total_segments: number;
  navigable_count: number;
  conditional_count: number;
  non_navigable_count: number;
  navigable_km: number;
  conditional_km: number;
  non_navigable_km: number;
  total_km: number;
  navigable_pct: number;
  mean_confidence: number;

  predictions: NavigabilityPrediction[];
  geojson: RiverGeoJSON;
}

// ------------------------------------------------------------
// Seasonal Calendar
// ------------------------------------------------------------

export interface MonthlyCell {
  month: Month;
  navigability_class: NavigabilityClass;
  predicted_depth_m: number;
  probability: number;
  risk_score: number;
  is_monsoon: boolean;
  label: string; // 'Jan', 'Feb', …
}

export interface SegmentCalendarRow {
  segment_id: string;
  km_start: number;
  km_end: number;
  km_label: string;
  months: MonthlyCell[];
  navigable_months_count: number;
  best_month: Month;
  worst_month: Month;
}

export interface SeasonalCalendar {
  waterway_id: WaterwayId;
  year: number;
  generated_at: string;
  rows: SegmentCalendarRow[];
  month_summaries: MonthSummary[];
}

export interface MonthSummary {
  month: Month;
  label: string;
  navigable_pct: number;
  mean_depth_m: number;
  dominant_class: NavigabilityClass;
  alert_count: number;
}

// ------------------------------------------------------------
// Risk Alerts
// ------------------------------------------------------------

export interface RiskAlert {
  alert_id: string;
  waterway_id: WaterwayId;
  segment_id: string;
  km_start: number;
  km_end: number;
  severity: AlertSeverity;
  alert_type: AlertType;
  title: string;
  description: string;

  // Trigger values
  predicted_value: number;
  threshold_value: number;
  unit: string;
  risk_score: number;

  // Temporal
  valid_from: string;
  valid_until: string;
  created_at: string;
  is_active: boolean;

  // Recommended actions
  recommended_actions: string[];
  affected_vessels?: string[];
}

export interface AlertStats {
  total: number;
  critical: number;
  warning: number;
  info: number;
  by_type: Record<AlertType, number>;
  by_waterway: Record<WaterwayId, number>;
}

// ------------------------------------------------------------
// Depth Profile
// ------------------------------------------------------------

export interface DepthProfilePoint {
  km: number;
  depth_m: number;
  depth_lower_ci: number;
  depth_upper_ci: number;
  width_m: number;
  navigability_class: NavigabilityClass;
  segment_id: string;
  landmark?: string;
}

export interface DepthProfile {
  waterway_id: WaterwayId;
  month: Month;
  year: number;
  points: DepthProfilePoint[];
  min_depth_m: number;
  max_depth_m: number;
  mean_depth_m: number;
  navigable_threshold_m: number;
  conditional_threshold_m: number;
  bottleneck_km: number;
  bottleneck_depth_m: number;
}

// ------------------------------------------------------------
// Waterway Statistics
// ------------------------------------------------------------

export interface MonthlyStats {
  month: Month;
  label: string;
  navigable_pct: number;
  mean_depth_m: number;
  alert_count: number;
}

export interface WaterwayStats {
  waterway_id: WaterwayId;
  year: number;
  total_length_km: number;
  total_segments: number;

  // Annual aggregates
  annual_navigable_pct: number;
  annual_mean_depth_m: number;
  peak_navigability_month: Month;
  worst_navigability_month: Month;
  total_alerts_year: number;

  monthly_stats: MonthlyStats[];

  // Year-over-year comparison
  yoy_navigability_change_pct?: number;
  yoy_depth_change_m?: number;
}

// ------------------------------------------------------------
// Segment History
// ------------------------------------------------------------

export interface SegmentHistoryPoint {
  year: number;
  month: Month;
  depth_m: number;
  navigability_class: NavigabilityClass;
  probability: number;
  risk_score: number;
}

export interface SegmentHistory {
  segment_id: string;
  waterway_id: WaterwayId;
  km_start: number;
  km_end: number;
  history: SegmentHistoryPoint[];
  trend_depth_m_per_year: number;
  dominant_class: NavigabilityClass;
}

// ------------------------------------------------------------
// Analytics & Trends
// ------------------------------------------------------------

export interface YearlyTrendPoint {
  month: Month;
  label: string;
  navigable_pct: number;
  mean_depth_m: number;
}

export interface YearlyTrend {
  year: number;
  color: string;
  points: YearlyTrendPoint[];
  annual_mean_navigable_pct: number;
}

export interface AnalyticsTrends {
  waterway_id: WaterwayId;
  years: number[];
  trends: YearlyTrend[];
  generated_at: string;
}

// ------------------------------------------------------------
// Feature Importance (SHAP)
// ------------------------------------------------------------

export interface FeatureImportanceItem {
  feature_name: string;
  display_name: string;
  shap_value: number;         // mean |SHAP| across test set
  importance_pct: number;     // percentage of total importance
  direction: 'positive' | 'negative' | 'mixed';
  category: 'spectral' | 'temporal' | 'morphological' | 'hydrological';
  description: string;
}

export interface FeatureImportance {
  model_version: string;
  waterway_id: WaterwayId;
  total_samples: number;
  features: FeatureImportanceItem[];
  generated_at: string;
}

export interface SegmentSHAP {
  segment_id: string;
  month: Month;
  year: number;
  shap_values: Record<string, number>;
  base_value: number;
  predicted_depth_m: number;
}

// ------------------------------------------------------------
// Model Performance / Metrics
// ------------------------------------------------------------

export interface RegressionMetrics {
  r2: number;
  rmse_m: number;
  mae_m: number;
  mbe_m: number;        // Mean Bias Error
  mape_pct: number;     // Mean Absolute Percentage Error
  explained_variance: number;
}

export interface ClassificationMetrics {
  accuracy: number;
  f1_macro: number;
  f1_weighted: number;
  precision_macro: number;
  recall_macro: number;
  kappa: number;         // Cohen's kappa
  per_class: Record<NavigabilityClass, PerClassMetrics>;
}

export interface PerClassMetrics {
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface ConfusionMatrixEntry {
  actual: NavigabilityClass;
  predicted: NavigabilityClass;
  count: number;
}

export interface ModelMetrics {
  model_id: string;
  model_name: string;
  model_version: string;
  architecture: string;
  training_date: string;
  waterway_id: WaterwayId;

  // Evaluation split
  train_samples: number;
  val_samples: number;
  test_samples: number;
  train_years: number[];
  test_years: number[];

  regression: RegressionMetrics;
  classification: ClassificationMetrics;
  confusion_matrix: ConfusionMatrixEntry[];

  // Satellite data used
  satellites: string[];
  bands_used: string[];
  temporal_resolution: string;
  spatial_resolution_m: number;
}

// ------------------------------------------------------------
// UI / Store Types
// ------------------------------------------------------------

export interface AppState {
  // Navigation
  selectedWaterway: WaterwayId;
  selectedMonth: Month;
  selectedYear: number;
  selectedSegmentId: string | null;

  // Map
  mapStyle: MapStyle;
  showDepthOverlay: boolean;
  showWidthOverlay: boolean;
  showAlerts: boolean;
  mapViewport: MapViewport;

  // UI
  sidebarCollapsed: boolean;
  alertsPanelOpen: boolean;
  detailPanelOpen: boolean;
}

export interface MapViewport {
  longitude: number;
  latitude: number;
  zoom: number;
  pitch?: number;
  bearing?: number;
}

export interface TooltipData {
  x: number;
  y: number;
  segment: RiverSegment | null;
  prediction: NavigabilityPrediction | null;
}

// ------------------------------------------------------------
// API Response Wrappers
// ------------------------------------------------------------

export interface ApiResponse<T> {
  data: T;
  status: 'success' | 'error';
  message?: string;
  timestamp: string;
}

export interface PaginatedResponse<T> {
  data: T[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface ApiError {
  status: 'error';
  code: string;
  message: string;
  details?: Record<string, unknown>;
}

// ------------------------------------------------------------
// Utility / Helper Types
// ------------------------------------------------------------

export type ClassColorMap = Record<NavigabilityClass, string>;

export const NAVIGABILITY_COLORS: ClassColorMap = {
  navigable:     '#22c55e',
  conditional:   '#f59e0b',
  non_navigable: '#ef4444',
};

export const NAVIGABILITY_LABELS: Record<NavigabilityClass, string> = {
  navigable:     'Navigable',
  conditional:   'Conditional',
  non_navigable: 'Non-Navigable',
};

export const MONTH_LABELS: Record<Month, string> = {
  1:  'Jan', 2:  'Feb', 3:  'Mar',
  4:  'Apr', 5:  'May', 6:  'Jun',
  7:  'Jul', 8:  'Aug', 9:  'Sep',
  10: 'Oct', 11: 'Nov', 12: 'Dec',
};

export const WATERWAY_NAMES: Record<WaterwayId, string> = {
  'NW-1': 'Ganga (NW-1)',
  'NW-2': 'Brahmaputra (NW-2)',
};

export const WATERWAY_STATES: Record<WaterwayId, string[]> = {
  'NW-1': ['Uttar Pradesh', 'Bihar', 'Jharkhand', 'West Bengal'],
  'NW-2': ['Assam'],
};

export const DEPTH_THRESHOLDS = {
  navigable:   3.0,  // m — IWT Authority standard
  conditional: 2.0,  // m
  non_navigable: 0,
} as const;

export type Nullable<T> = T | null;
export type Optional<T> = T | undefined;
export type LoadingState = 'idle' | 'loading' | 'success' | 'error';
