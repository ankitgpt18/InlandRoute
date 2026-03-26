// ============================================================
// AIDSTL — Inland Waterway Navigability Prediction
// Axios API Client — all endpoint functions
// ============================================================

import axios, {
  AxiosInstance,
  AxiosResponse,
  AxiosError,
  InternalAxiosRequestConfig,
} from 'axios';
import type {
  NavigabilityMap,
  SeasonalCalendar,
  RiskAlert,
  AlertStats,
  DepthProfile,
  WaterwayStats,
  RiverSegment,
  SegmentHistory,
  AnalyticsTrends,
  FeatureImportance,
  ModelMetrics,
  ApiResponse,
  WaterwayId,
  Month,
  NavigabilityPrediction,
  SegmentSHAP,
} from '@/types';

// ─── Base URL ─────────────────────────────────────────────────────────────────

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000/api/v1';

// ─── Axios Instance ───────────────────────────────────────────────────────────

const api: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30_000, // 30 s — satellite imagery endpoints can be slow
  headers: {
    'Content-Type': 'application/json',
    Accept: 'application/json',
  },
});

// ─── Request Interceptor ──────────────────────────────────────────────────────

api.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    // Attach auth token if present (future JWT support)
    if (typeof window !== 'undefined') {
      const token = localStorage.getItem('aidstl_token');
      if (token && config.headers) {
        config.headers.Authorization = `Bearer ${token}`;
      }
    }

    // Debug logging in development
    if (process.env.NODE_ENV === 'development') {
      console.debug(
        `[API] ${config.method?.toUpperCase()} ${config.baseURL}${config.url}`,
        config.params ?? '',
      );
    }

    return config;
  },
  (error: AxiosError) => Promise.reject(error),
);

// ─── Response Interceptor ─────────────────────────────────────────────────────

api.interceptors.response.use(
  (response: AxiosResponse) => response,
  (error: AxiosError) => {
    const status = error.response?.status;

    if (process.env.NODE_ENV === 'development') {
      console.error(
        `[API Error] ${status ?? 'Network'} —`,
        error.response?.data ?? error.message,
      );
    }

    // Surface meaningful error messages upstream
    const serverMessage =
      (error.response?.data as { detail?: string })?.detail ??
      (error.response?.data as { message?: string })?.message;

    if (serverMessage) {
      error.message = serverMessage;
    } else if (!error.response) {
      error.message =
        'Unable to reach the API server. Is the backend running on port 8000?';
    }

    return Promise.reject(error);
  },
);

// ─── Generic unwrapper ────────────────────────────────────────────────────────

/**
 * Unwraps `ApiResponse<T>` envelopes that the backend may return.
 * If the backend returns the object directly, it is passed through as-is.
 */
function unwrap<T>(response: AxiosResponse<ApiResponse<T> | T>): T {
  const payload = response.data;
  if (
    payload &&
    typeof payload === 'object' &&
    'data' in (payload as object) &&
    'status' in (payload as object)
  ) {
    return (payload as ApiResponse<T>).data;
  }
  return payload as T;
}

// ─── Types used only inside this module ──────────────────────────────────────

interface NavigabilityMapParams {
  month: Month;
  year: number;
  include_geojson?: boolean;
  include_spectral?: boolean;
}

interface AlertsParams {
  month?: Month;
  year?: number;
  severity?: string;
  is_active?: boolean;
  limit?: number;
  offset?: number;
}

interface SegmentHistoryParams {
  segment_id: string;
  years: number[];
}

interface TrendsParams {
  waterway_id: WaterwayId;
  years: number[];
}

// ============================================================
// 1. NAVIGABILITY MAP
// ============================================================

/**
 * Fetch the full navigability map (predictions + GeoJSON) for a waterway,
 * month, and year.
 *
 * GET /navigability/{waterway_id}/map
 */
export async function getNavigabilityMap(
  waterwayId: WaterwayId,
  month: Month,
  year: number,
): Promise<NavigabilityMap> {
  const params: NavigabilityMapParams = {
    month,
    year,
    include_geojson: true,
    include_spectral: true,
  };
  const response = await api.get<NavigabilityMap>(
    `/navigability/${encodeURIComponent(waterwayId)}/map`,
    { params },
  );
  return unwrap(response);
}

/**
 * Fetch a single segment's prediction for a given month/year.
 *
 * GET /navigability/{waterway_id}/segment/{segment_id}
 */
export async function getSegmentPrediction(
  waterwayId: WaterwayId,
  segmentId: string,
  month: Month,
  year: number,
): Promise<NavigabilityPrediction> {
  const response = await api.get<NavigabilityPrediction>(
    `/navigability/${encodeURIComponent(waterwayId)}/segment/${encodeURIComponent(segmentId)}`,
    { params: { month, year } },
  );
  return unwrap(response);
}

// ============================================================
// 2. SEASONAL CALENDAR
// ============================================================

/**
 * Fetch the 12-month seasonal navigability calendar for a waterway and year.
 * Returns a grid of segment × month navigability cells.
 *
 * GET /navigability/{waterway_id}/calendar
 */
export async function getSeasonalCalendar(
  waterwayId: WaterwayId,
  year: number,
): Promise<SeasonalCalendar> {
  const response = await api.get<SeasonalCalendar>(
    `/navigability/${encodeURIComponent(waterwayId)}/calendar`,
    { params: { year } },
  );
  return unwrap(response);
}

// ============================================================
// 3. RISK ALERTS
// ============================================================

/**
 * Fetch active risk alerts for a waterway, optionally filtered by month/year
 * and severity.
 *
 * GET /alerts/{waterway_id}
 */
export async function getRiskAlerts(
  waterwayId: WaterwayId,
  month?: Month,
  year?: number,
  options: { severity?: string; limit?: number } = {},
): Promise<RiskAlert[]> {
  const params: AlertsParams = {
    ...(month !== undefined && { month }),
    ...(year  !== undefined && { year }),
    is_active: true,
    limit: options.limit ?? 50,
    ...(options.severity && { severity: options.severity }),
  };
  const response = await api.get<RiskAlert[]>(
    `/alerts/${encodeURIComponent(waterwayId)}`,
    { params },
  );
  return unwrap(response);
}

/**
 * Fetch aggregated alert statistics for a waterway.
 *
 * GET /alerts/{waterway_id}/stats
 */
export async function getAlertStats(
  waterwayId: WaterwayId,
  year?: number,
): Promise<AlertStats> {
  const response = await api.get<AlertStats>(
    `/alerts/${encodeURIComponent(waterwayId)}/stats`,
    { params: year !== undefined ? { year } : {} },
  );
  return unwrap(response);
}

/**
 * Fetch all alerts across both waterways (for the global alerts page).
 *
 * GET /alerts
 */
export async function getAllAlerts(options: {
  severity?: string;
  waterway_id?: WaterwayId;
  month?: Month;
  year?: number;
  limit?: number;
  offset?: number;
} = {}): Promise<RiskAlert[]> {
  const response = await api.get<RiskAlert[]>('/alerts', { params: options });
  return unwrap(response);
}

// ============================================================
// 4. DEPTH PROFILE
// ============================================================

/**
 * Fetch the longitudinal depth profile (depth vs. km along river) for a
 * waterway, month, and year.
 *
 * GET /navigability/{waterway_id}/depth-profile
 */
export async function getDepthProfile(
  waterwayId: WaterwayId,
  month: Month,
  year: number,
): Promise<DepthProfile> {
  const response = await api.get<DepthProfile>(
    `/navigability/${encodeURIComponent(waterwayId)}/depth-profile`,
    { params: { month, year } },
  );
  return unwrap(response);
}

// ============================================================
// 5. WATERWAY STATISTICS
// ============================================================

/**
 * Fetch annual statistics (monthly aggregates, navigable %, alerts) for a
 * waterway and year.
 *
 * GET /waterways/{waterway_id}/stats
 */
export async function getWaterwayStats(
  waterwayId: WaterwayId,
  year: number,
): Promise<WaterwayStats> {
  const response = await api.get<WaterwayStats>(
    `/waterways/${encodeURIComponent(waterwayId)}/stats`,
    { params: { year } },
  );
  return unwrap(response);
}

// ============================================================
// 6. RIVER SEGMENTS
// ============================================================

/**
 * Fetch the list of all river segments for a waterway (static metadata,
 * no predictions).
 *
 * GET /waterways/{waterway_id}/segments
 */
export async function getSegments(
  waterwayId: WaterwayId,
): Promise<RiverSegment[]> {
  const response = await api.get<RiverSegment[]>(
    `/waterways/${encodeURIComponent(waterwayId)}/segments`,
  );
  return unwrap(response);
}

/**
 * Fetch metadata for a single segment.
 *
 * GET /waterways/{waterway_id}/segments/{segment_id}
 */
export async function getSegment(
  waterwayId: WaterwayId,
  segmentId: string,
): Promise<RiverSegment> {
  const response = await api.get<RiverSegment>(
    `/waterways/${encodeURIComponent(waterwayId)}/segments/${encodeURIComponent(segmentId)}`,
  );
  return unwrap(response);
}

// ============================================================
// 7. SEGMENT HISTORY
// ============================================================

/**
 * Fetch the multi-year monthly history for a single segment.
 * Returns a time-series of depth/navigability values.
 *
 * GET /navigability/segment/{segment_id}/history
 */
export async function getSegmentHistory(
  segmentId: string,
  years: number[],
): Promise<SegmentHistory> {
  const params: SegmentHistoryParams = { segment_id: segmentId, years };
  const response = await api.get<SegmentHistory>(
    `/navigability/segment/${encodeURIComponent(segmentId)}/history`,
    { params: { years: years.join(',') } },
  );
  return unwrap(response);
}

/**
 * Fetch SHAP explanation values for a single segment / month / year.
 *
 * GET /navigability/segment/{segment_id}/shap
 */
export async function getSegmentSHAP(
  segmentId: string,
  month: Month,
  year: number,
): Promise<SegmentSHAP> {
  const response = await api.get<SegmentSHAP>(
    `/navigability/segment/${encodeURIComponent(segmentId)}/shap`,
    { params: { month, year } },
  );
  return unwrap(response);
}

// ============================================================
// 8. ANALYTICS TRENDS
// ============================================================

/**
 * Fetch multi-year monthly navigability trend data for a waterway.
 * Used by the trend chart (one line per year).
 *
 * GET /analytics/{waterway_id}/trends
 */
export async function getAnalyticsTrends(
  waterwayId: WaterwayId,
  years: number[],
): Promise<AnalyticsTrends> {
  const params: TrendsParams = { waterway_id: waterwayId, years };
  const response = await api.get<AnalyticsTrends>(
    `/analytics/${encodeURIComponent(waterwayId)}/trends`,
    { params: { years: years.join(',') } },
  );
  return unwrap(response);
}

// ============================================================
// 9. FEATURE IMPORTANCE (SHAP Global)
// ============================================================

/**
 * Fetch global SHAP feature importance values computed across the full
 * test set. Optionally scoped to one waterway.
 *
 * GET /model/feature-importance
 */
export async function getFeatureImportance(
  waterwayId?: WaterwayId,
): Promise<FeatureImportance> {
  const response = await api.get<FeatureImportance>(
    '/model/feature-importance',
    { params: waterwayId ? { waterway_id: waterwayId } : {} },
  );
  return unwrap(response);
}

// ============================================================
// 10. MODEL PERFORMANCE METRICS
// ============================================================

/**
 * Fetch the full model performance report (R², RMSE, MAE, F1, confusion
 * matrix, per-class metrics) for a given waterway.
 *
 * GET /model/performance
 */
export async function getModelPerformance(
  waterwayId?: WaterwayId,
): Promise<ModelMetrics> {
  const response = await api.get<ModelMetrics>(
    '/model/performance',
    { params: waterwayId ? { waterway_id: waterwayId } : {} },
  );
  return unwrap(response);
}

/**
 * Fetch a list of all trained model versions (for the model info page).
 *
 * GET /model/versions
 */
export async function getModelVersions(): Promise<ModelMetrics[]> {
  const response = await api.get<ModelMetrics[]>('/model/versions');
  return unwrap(response);
}

// ============================================================
// 11. HEALTH CHECK
// ============================================================

/**
 * Ping the API server to check connectivity.
 *
 * GET /health
 */
export async function checkHealth(): Promise<{ status: string; version: string }> {
  const response = await api.get<{ status: string; version: string }>('/health');
  return unwrap(response);
}

// ─── Named export of the raw axios instance (for advanced use) ────────────────
export { api as apiClient };

// ─── Default export of all functions as a namespace ──────────────────────────
const ApiService = {
  // Navigability
  getNavigabilityMap,
  getSegmentPrediction,

  // Seasonal
  getSeasonalCalendar,

  // Alerts
  getRiskAlerts,
  getAlertStats,
  getAllAlerts,

  // Depth
  getDepthProfile,

  // Waterway stats & segments
  getWaterwayStats,
  getSegments,
  getSegment,

  // History & SHAP
  getSegmentHistory,
  getSegmentSHAP,

  // Analytics
  getAnalyticsTrends,

  // Model
  getFeatureImportance,
  getModelPerformance,
  getModelVersions,

  // Misc
  checkHealth,
};

export default ApiService;
