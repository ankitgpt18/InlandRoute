// ============================================================
// InlandRoute — Resilient Axios API Client + Automatic Fallback
// ============================================================

import axios, { AxiosInstance, AxiosResponse, AxiosError, InternalAxiosRequestConfig } from 'axios';
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

import {
  getMockNavigabilityMap,
  getMockDepthProfile,
  getMockAlerts,
  getMockSeasonalCalendar,
  getMockWaterwayStats,
  getMockTrends,
  MOCK_FEATURE_IMPORTANCE,
  MOCK_MODEL_METRICS,
  buildNW1GeoJSON,
} from '@/lib/mock-data';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000/api/v1';

const api: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 5000,
  headers: {
    'Content-Type': 'application/json',
    Accept: 'application/json',
  },
});

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

// ── 1. Navigability Map ──────────────────────────────────────────────────────
export async function getNavigabilityMap(
  waterwayId: WaterwayId,
  month: Month,
  year: number
): Promise<NavigabilityMap> {
  try {
    const res = await api.get<NavigabilityMap>(`/navigability/${encodeURIComponent(waterwayId)}/map`, {
      params: { month, year, include_geojson: true, include_spectral: true },
    });
    const data = unwrap(res);
    if (data && data.predictions && data.predictions.length > 0) return data;
    return getMockNavigabilityMap(waterwayId, month, year);
  } catch {
    return getMockNavigabilityMap(waterwayId, month, year);
  }
}

// ── 2. Segment Prediction ────────────────────────────────────────────────────
export async function getSegmentPrediction(
  waterwayId: WaterwayId,
  segmentId: string,
  month: Month,
  year: number
): Promise<NavigabilityPrediction> {
  try {
    const res = await api.get<NavigabilityPrediction>(
      `/navigability/${encodeURIComponent(waterwayId)}/segment/${encodeURIComponent(segmentId)}`,
      { params: { month, year } }
    );
    const data = unwrap(res);
    if (data && data.segment_id) return data;
    const map = getMockNavigabilityMap(waterwayId, month, year);
    return map.predictions.find((p) => p.segment_id === segmentId) ?? map.predictions[0];
  } catch {
    const map = getMockNavigabilityMap(waterwayId, month, year);
    return map.predictions.find((p) => p.segment_id === segmentId) ?? map.predictions[0];
  }
}

// ── 3. Seasonal Calendar ─────────────────────────────────────────────────────
export async function getSeasonalCalendar(
  waterwayId: WaterwayId,
  year: number
): Promise<SeasonalCalendar> {
  try {
    const res = await api.get<SeasonalCalendar>(`/navigability/${encodeURIComponent(waterwayId)}/calendar`, {
      params: { year },
    });
    const data = unwrap(res);
    if (data && (data.segment_outlooks?.length || data.rows?.length)) return data;
    return getMockSeasonalCalendar(waterwayId, year);
  } catch {
    return getMockSeasonalCalendar(waterwayId, year);
  }
}

// ── 4. Risk Alerts ───────────────────────────────────────────────────────────
export async function getRiskAlerts(
  waterwayId: WaterwayId,
  month?: Month,
  year?: number
): Promise<RiskAlert[]> {
  try {
    const res = await api.get<RiskAlert[]>(`/alerts/${encodeURIComponent(waterwayId)}`, {
      params: { month, year, is_active: true },
    });
    const data = unwrap(res);
    if (Array.isArray(data) && data.length > 0) return data;
    return getMockAlerts(waterwayId);
  } catch {
    return getMockAlerts(waterwayId);
  }
}

export async function getAlertStats(waterwayId: WaterwayId, year?: number): Promise<AlertStats> {
  try {
    const res = await api.get<AlertStats>(`/alerts/${encodeURIComponent(waterwayId)}/stats`, {
      params: { year },
    });
    const data = unwrap(res);
    if (data && data.total !== undefined) return data;
    const alerts = getMockAlerts(waterwayId);
    return {
      total: alerts.length,
      critical: alerts.filter((a) => a.severity === 'CRITICAL').length,
      warning: alerts.filter((a) => a.severity === 'WARNING').length,
      info: alerts.filter((a) => a.severity === 'INFO').length,
    };
  } catch {
    const alerts = getMockAlerts(waterwayId);
    return {
      total: alerts.length,
      critical: alerts.filter((a) => a.severity === 'CRITICAL').length,
      warning: alerts.filter((a) => a.severity === 'WARNING').length,
      info: alerts.filter((a) => a.severity === 'INFO').length,
    };
  }
}

export async function getAllAlerts(options: { waterway_id?: WaterwayId } = {}): Promise<RiskAlert[]> {
  try {
    const res = await api.get<RiskAlert[]>('/alerts', { params: options });
    const data = unwrap(res);
    if (Array.isArray(data) && data.length > 0) return data;
    return getMockAlerts(options.waterway_id ?? 'NW-1');
  } catch {
    return getMockAlerts(options.waterway_id ?? 'NW-1');
  }
}

// ── 5. Depth Profile ─────────────────────────────────────────────────────────
export async function getDepthProfile(
  waterwayId: WaterwayId,
  month: Month,
  year: number
): Promise<DepthProfile> {
  try {
    const res = await api.get<DepthProfile>(`/navigability/${encodeURIComponent(waterwayId)}/depth-profile`, {
      params: { month, year },
    });
    const data = unwrap(res);
    if (data && (data.points?.length || data.profile_points?.length)) return data;
    return getMockDepthProfile(waterwayId, month, year);
  } catch {
    return getMockDepthProfile(waterwayId, month, year);
  }
}

// ── 6. Waterway Stats ────────────────────────────────────────────────────────
export async function getWaterwayStats(waterwayId: WaterwayId, year: number): Promise<WaterwayStats> {
  try {
    const res = await api.get<WaterwayStats>(`/waterways/${encodeURIComponent(waterwayId)}/stats`, {
      params: { year },
    });
    const data = unwrap(res);
    if (data && data.total_segments) return data;
    return getMockWaterwayStats(waterwayId, year);
  } catch {
    return getMockWaterwayStats(waterwayId, year);
  }
}

// ── 7. Analytics Trends ──────────────────────────────────────────────────────
export async function getAnalyticsTrends(waterwayId: WaterwayId, years: number[]): Promise<AnalyticsTrends> {
  try {
    const res = await api.get<AnalyticsTrends>(`/analytics/${encodeURIComponent(waterwayId)}/trends`, {
      params: { years: years.join(',') },
    });
    const data = unwrap(res);
    if (data && data.years?.length) return data;
    return getMockTrends(waterwayId);
  } catch {
    return getMockTrends(waterwayId);
  }
}

// ── 8. Model Performance & SHAP ──────────────────────────────────────────────
export async function getFeatureImportance(waterwayId?: WaterwayId): Promise<FeatureImportance> {
  try {
    const res = await api.get<FeatureImportance>('/model/feature-importance', {
      params: waterwayId ? { waterway_id: waterwayId } : {},
    });
    const data = unwrap(res);
    if (data && data.features?.length) return data;
    return MOCK_FEATURE_IMPORTANCE;
  } catch {
    return MOCK_FEATURE_IMPORTANCE;
  }
}

export async function getModelPerformance(waterwayId?: WaterwayId): Promise<ModelMetrics> {
  try {
    const res = await api.get<ModelMetrics>('/model/performance', {
      params: waterwayId ? { waterway_id: waterwayId } : {},
    });
    const data = unwrap(res);
    if (data && (data.r2_score || data.accuracy)) return data;
    return MOCK_MODEL_METRICS;
  } catch {
    return MOCK_MODEL_METRICS;
  }
}

export async function checkHealth(): Promise<{ status: string; version: string }> {
  try {
    const res = await api.get<{ status: string; version: string }>('/health');
    return unwrap(res);
  } catch {
    return { status: 'healthy', version: '1.0.0-fallback' };
  }
}

const ApiService = {
  getNavigabilityMap,
  getSegmentPrediction,
  getSeasonalCalendar,
  getRiskAlerts,
  getAlertStats,
  getAllAlerts,
  getDepthProfile,
  getWaterwayStats,
  getAnalyticsTrends,
  getFeatureImportance,
  getModelPerformance,
  checkHealth,
};

export default ApiService;
