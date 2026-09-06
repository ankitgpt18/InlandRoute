// ============================================================
// AIDSTL — Inland Waterway Navigability Prediction
// TypeScript Type Definitions (Aligned 1:1 with Backend Pydantic Schemas)
// ============================================================

export type NavigabilityClass = 'navigable' | 'conditional' | 'non_navigable';

export type WaterwayId = 'NW-1' | 'NW-2' | 'NW-3' | 'NW-4' | 'NW-5';

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

export const MONTH_LABELS = [
  'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
] as const;

export interface MapViewport {
  longitude: number;
  latitude: number;
  zoom: number;
  pitch: number;
  bearing: number;
}

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

export interface RiverSegment {
  segment_id: string;
  waterway_id: WaterwayId;
  name?: string;
  km_start: number;
  km_end: number;
  km_length?: number;
  length_km?: number;
  mean_width_m?: number;
  mean_depth_m?: number;
  bed_elevation_m?: number;
  bankfull_depth_m?: number;
  upstream_landmark?: string;
  downstream_landmark?: string;
  state?: string;
  district?: string;
  has_jetty?: boolean;
  has_barge_terminal?: boolean;
  has_navigation_lock?: boolean;
  geometry: LineStringGeometry | any;
}

export interface SpectralIndices {
  mndwi?: number;
  ndwi?: number;
  ndvi?: number;
  stumpf_ratio?: number;
  awei_sh?: number;
  awei_nsh?: number;
  [key: string]: any;
}

export interface ConfidenceInterval {
  lower_95?: number;
  upper_95?: number;
  lower_80?: number;
  upper_80?: number;
  std_dev?: number;
}

export interface NavigabilityPrediction {
  prediction_id?: string;
  segment_id: string;
  waterway_id: WaterwayId;
  month: Month;
  year: number;
  created_at?: string;

  navigability_class: NavigabilityClass;
  predicted_depth_m: number;
  predicted_width_m?: number;
  width_m?: number;
  depth_lower_ci?: number;
  depth_upper_ci?: number;
  probability?: number;
  risk_score?: number;
  confidence?: number;

  depth_ci?: ConfidenceInterval;
  width_ci?: ConfidenceInterval;

  spectral_indices?: SpectralIndices;
  features?: SpectralIndices;

  depth_threshold_m?: number;
  is_above_threshold?: boolean;
  margin_m?: number;

  discharge_m3s?: number;
  velocity_ms?: number;
  water_surface_area_km2?: number;
}

export interface NavigabilityMap {
  waterway_id: WaterwayId;
  month: Month;
  year: number;
  generated_at?: string;

  total_segments: number;
  navigable_count: number;
  conditional_count: number;
  non_navigable_count: number;

  navigable_km?: number;
  navigable_length_km?: number;
  conditional_km?: number;
  conditional_length_km?: number;
  non_navigable_km?: number;
  non_navigable_length_km?: number;
  total_km?: number;

  navigable_pct?: number;
  overall_navigability_pct?: number;

  mean_depth_m?: number;
  mean_width_m?: number;
  mean_risk_score?: number;
  mean_confidence?: number;

  predictions: NavigabilityPrediction[];
  geojson?: RiverGeoJSON;
}

export interface MonthlyCell {
  month: Month;
  navigability_class: NavigabilityClass;
  predicted_depth_m: number;
  probability?: number;
  risk_score?: number;
  is_monsoon?: boolean;
  label?: string;
}

export interface SegmentCalendarRow {
  segment_id: string;
  km_start?: number;
  km_end?: number;
  km_label?: string;
  months?: MonthlyCell[];
  monthly_outlooks?: MonthlyCell[];
  navigable_months_count?: number;
  best_month?: Month;
  worst_month?: Month;
  annual_navigability_pct?: number;
}

export interface SeasonalCalendar {
  waterway_id: WaterwayId;
  year: number;
  generated_at?: string;
  rows?: SegmentCalendarRow[];
  segment_outlooks?: SegmentCalendarRow[];
  month_summaries?: MonthSummary[];
}

export interface MonthSummary {
  month: Month;
  label: string;
  navigable_pct: number;
  mean_depth_m: number;
  dominant_class?: NavigabilityClass;
  alert_count?: number;
}

export interface RiskAlert {
  alert_id: string;
  waterway_id: WaterwayId;
  segment_id: string;
  km_start?: number;
  km_end?: number;
  severity: AlertSeverity;
  alert_type: AlertType;
  title: string;
  description: string;

  predicted_value?: number;
  threshold_value?: number;
  unit?: string;
  risk_score?: number;

  current_depth_m?: number;
  threshold_depth_m?: number;
  recommended_action?: string;

  valid_from?: string;
  valid_until?: string;
  created_at?: string;
  is_active?: boolean;

  recommended_actions?: string[];
  affected_vessels?: string[];
}

export interface AlertStats {
  total: number;
  critical: number;
  warning: number;
  info: number;
  by_type?: Record<string, number>;
  by_waterway?: Record<string, number>;
}

export interface DepthProfilePoint {
  km: number;
  chainage_km?: number;
  depth_m: number;
  predicted_depth_m?: number;
  depth_lower_ci?: number;
  depth_upper_ci?: number;
  width_m?: number;
  navigability_class: NavigabilityClass;
  segment_id: string;
  landmark?: string;
}

export interface DepthProfile {
  waterway_id: WaterwayId;
  month: Month;
  year: number;
  points?: DepthProfilePoint[];
  profile_points?: DepthProfilePoint[];
  min_depth_m?: number;
  max_depth_m?: number;
  mean_depth_m?: number;
  total_length_km?: number;
  navigable_threshold_m?: number;
  conditional_threshold_m?: number;
  bottleneck_km?: number;
  bottleneck_depth_m?: number;
}

export interface MonthlyStats {
  month: Month;
  label?: string;
  navigable_pct?: number;
  overall_navigability_pct?: number;
  mean_depth_m?: number;
  alert_count?: number;
}

export interface WaterwayStats {
  waterway_id: WaterwayId;
  year: number;
  total_length_km?: number;
  total_segments?: number;

  annual_navigable_pct?: number;
  mean_navigable_pct?: number;
  annual_mean_depth_m?: number;
  peak_depth_m?: number;
  min_depth_m?: number;
  gauge_count?: number;
  peak_navigability_month?: Month;
  worst_navigability_month?: Month;
  total_alerts_year?: number;

  monthly_stats?: MonthlyStats[];
  yoy_navigability_change_pct?: number;
  yoy_depth_change_m?: number;
}

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
  history: SegmentHistoryPoint[];
}

export interface SegmentSHAP {
  segment_id: string;
  waterway_id: WaterwayId;
  month: Month;
  year: number;
  base_value: number;
  prediction_value: number;
  feature_contributions: Record<string, number>;
}

export interface TrendYear {
  year: number;
  color?: string;
  points?: any[];
  monthly_aggregates?: MonthlyStats[];
  annual_mean_navigable_pct?: number;
  data?: MonthlyStats[];
}

export interface AnalyticsTrends {
  waterway_id: WaterwayId;
  years: TrendYear[] | any[];
  trends?: any[];
  generated_at?: string;
}

export interface FeatureImportanceItem {
  feature_name: string;
  display_name: string;
  importance_score?: number;
  shap_value?: number;
  importance_pct?: number;
  category?: string;
  description?: string;
  direction?: 'positive' | 'negative' | string;
}

export interface FeatureImportance {
  waterway_id?: WaterwayId;
  total_samples?: number;
  generated_at?: string;
  features: FeatureImportanceItem[];
  model_version?: string;
}

export interface ModelMetrics {
  waterway_id?: WaterwayId;
  model_id?: string;
  model_name?: string;
  model_version?: string;
  architecture?: any;
  regression?: any;
  classification?: any;
  confusion_matrix?: any;
  satellites?: any;
  features_count?: any;
  hyperparameters?: any;
  r2_score?: number;
  rmse?: number;
  mae?: number;
  f1_score?: number;
  accuracy?: number;
  precision?: number;
  recall?: number;
  trained_at?: string;
  training_date?: string;
  train_samples?: number;
  val_samples?: number;
  test_samples?: number;
  train_years?: number[];
  test_years?: number[];
  [key: string]: any;
}

export interface ApiResponse<T> {
  status: string;
  data: T;
  message?: string;
  error?: string;
}
