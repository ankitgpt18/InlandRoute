// ============================================================
// InlandRoute - Inland Waterway Navigability Prediction
// Comprehensive Mock Data for Development & Fallback
// ============================================================

import type {
  RiverGeoJSON,
  NavigabilityMap,
  SeasonalCalendar,
  RiskAlert,
  DepthProfile,
  WaterwayStats,
  AnalyticsTrends,
  FeatureImportance,
  ModelMetrics,
  NavigabilityClass,
  Month,
  WaterwayId,
  DepthProfilePoint,
  SegmentCalendarRow,
  MonthlyCell,
} from "@/types";

// ─── Helpers ──────────────────────────────────────────────────────────────────

function classifyDepth(d: number): NavigabilityClass {
  if (d >= 3.0) return "navigable";
  if (d >= 2.0) return "conditional";
  return "non_navigable";
}

const MONTH_LABELS = [
  "Jan",
  "Feb",
  "Mar",
  "Apr",
  "May",
  "Jun",
  "Jul",
  "Aug",
  "Sep",
  "Oct",
  "Nov",
  "Dec",
];

function monthLabel(m: number): string {
  return MONTH_LABELS[m - 1] ?? "";
}

// Monsoon depth multiplier: depth is highest Jun–Oct, lowest Mar–May
function monsoonMultiplier(month: number): number {
  const profile = [
    0.55, 0.5, 0.45, 0.48, 0.52, 0.75, 0.95, 1.0, 0.9, 0.8, 0.68, 0.6,
  ];
  return profile[month - 1] ?? 0.6;
}

// Pseudo-random deterministic generator for mock data to prevent hydration errors
function pseudoRandom(seed: number): number {
  const x = Math.sin(seed++) * 10000;
  return x - Math.floor(x);
}

const NOW_ISO = new Date("2024-08-15T12:00:00Z").toISOString(); // Static date for hydration

// ============================================================
// NW-1 GANGA — GeoJSON (20 segments, Allahabad → Kolkata corridor)
// Approximate centreline coordinates along Ganga
// ============================================================

// Base depths at peak monsoon (Aug) for each segment — realistic values in metres
const NW1_BASE_DEPTHS_AUG = [
  6.2, 5.8, 7.1, 4.9, 5.3, 6.8, 4.2, 3.8, 5.6, 6.4, 5.0, 4.5, 6.1, 5.7, 4.3,
  3.9, 5.2, 6.0, 4.8, 5.5,
];

// Approx coordinates along Ganga (lng, lat) — 20 waypoints Allahabad to Kolkata
const NW1_WAYPOINTS: [number, number][] = [
  [81.84, 25.45], // Allahabad / Prayagraj
  [82.0, 25.4],
  [82.17, 25.38],
  [82.56, 25.32], // Varanasi area
  [83.0, 25.28],
  [83.4, 25.55], // Ghazipur
  [83.59, 25.57],
  [84.0, 25.56],
  [84.35, 25.6], // Buxar
  [84.74, 25.58],
  [85.13, 25.6], // Patna
  [85.5, 25.55],
  [85.9, 25.52],
  [86.3, 25.47], // Bhagalpur
  [86.7, 25.42],
  [87.0, 25.28],
  [87.3, 25.1],
  [87.8, 24.85],
  [88.3, 24.55], // Farakka Barrage area
  [88.55, 24.1],
];

const NW1_KM_MARKERS = [
  0, 70, 140, 200, 260, 310, 360, 410, 460, 510, 565, 615, 660, 700, 745, 790,
  840, 900, 960, 1020,
];

const NW1_LANDMARKS = [
  "Prayagraj (Allahabad)",
  "Vindhyachal",
  "Chunar",
  "Varanasi Ghats",
  "Ramnagar",
  "Ghazipur",
  "Ballia",
  "Chhapra",
  "Buxar",
  "Arrah",
  "Patna",
  "Hajipur",
  "Mokameh",
  "Bhagalpur",
  "Kahalgaon",
  "Sahibganj",
  "Rajmahal",
  "Murshidabad",
  "Farakka Barrage",
  "Jangipur",
];

const NW1_STATES = [
  "UP",
  "UP",
  "UP",
  "UP",
  "UP",
  "UP",
  "UP",
  "Bihar",
  "Bihar",
  "Bihar",
  "Bihar",
  "Bihar",
  "Bihar",
  "Bihar",
  "Bihar",
  "Jharkhand",
  "Jharkhand",
  "West Bengal",
  "West Bengal",
  "West Bengal",
];

export function buildNW1GeoJSON(
  month: number = 8,
  year: number = 2024,
): RiverGeoJSON {
  const mult = monsoonMultiplier(month);
  const features = NW1_LANDMARKS.slice(0, 20).map((landmark, i) => {
    const nextIdx = Math.min(i + 1, NW1_WAYPOINTS.length - 1);
    const depth = NW1_BASE_DEPTHS_AUG[i] * mult;
    const cls = classifyDepth(depth);
    const segId = `NW1-SEG-${String(i + 1).padStart(2, "0")}`;

    return {
      type: "Feature" as const,
      id: segId,
      geometry: {
        type: "LineString" as const,
        coordinates: [NW1_WAYPOINTS[i], NW1_WAYPOINTS[nextIdx]] as [
          number,
          number,
        ][],
      },
      properties: {
        segment_id: segId,
        waterway_id: "NW-1" as WaterwayId,
        km_start: NW1_KM_MARKERS[i],
        km_end: NW1_KM_MARKERS[nextIdx] ?? NW1_KM_MARKERS[i] + 60,
        navigability_class: cls,
        depth_m: parseFloat(depth.toFixed(2)),
        width_m: 400 + i * 25 + (month >= 6 && month <= 9 ? 120 : 0),
        confidence: 0.82 + pseudoRandom(i + month * 10) * 0.15,
        velocity_ms: 0.4 + depth * 0.08,
        landmark,
        state: NW1_STATES[i],
      },
    };
  });

  return { type: "FeatureCollection", features };
}

// ============================================================
// NW-2 BRAHMAPUTRA — GeoJSON (15 segments, Dhubri → Dibrugarh)
// ============================================================

const NW2_BASE_DEPTHS_AUG = [
  8.5, 9.2, 7.8, 10.1, 8.9, 7.5, 9.8, 8.4, 6.9, 9.3, 8.1, 7.6, 9.0, 8.7, 7.2,
];

const NW2_WAYPOINTS: [number, number][] = [
  [89.97, 26.01], // Dhubri
  [90.3, 26.15],
  [90.65, 26.1], // Jogighopa
  [91.0, 26.18], // Guwahati
  [91.45, 26.22],
  [91.75, 26.45], // Tezpur approach
  [92.1, 26.63], // Tezpur
  [92.5, 26.78],
  [92.9, 26.9],
  [93.3, 27.1], // Bishnath Chariali
  [93.75, 27.2],
  [94.1, 27.3],
  [94.5, 27.38], // Jorhat approach
  [94.9, 27.48],
  [95.3, 27.48], // Dibrugarh
];

const NW2_KM_MARKERS = [
  0, 60, 115, 175, 240, 295, 355, 410, 465, 525, 585, 635, 680, 725, 770,
];

const NW2_LANDMARKS = [
  "Dhubri",
  "Bilasipara",
  "Jogighopa",
  "Guwahati",
  "Soalkuchi",
  "Tezpur (Kaliabar)",
  "Tezpur",
  "Biswanath Ghat",
  "Majuli Island",
  "Bishnath Chariali",
  "Bihpuria",
  "Gogamukh",
  "Jorhat",
  "Sibsagar",
  "Dibrugarh",
];

export function buildNW2GeoJSON(
  month: number = 8,
  year: number = 2024,
): RiverGeoJSON {
  const mult = monsoonMultiplier(month);
  const features = NW2_LANDMARKS.map((landmark, i) => {
    const nextIdx = Math.min(i + 1, NW2_WAYPOINTS.length - 1);
    const depth = NW2_BASE_DEPTHS_AUG[i] * mult;
    const cls = classifyDepth(depth);
    const segId = `NW2-SEG-${String(i + 1).padStart(2, "0")}`;

    return {
      type: "Feature" as const,
      id: segId,
      geometry: {
        type: "LineString" as const,
        coordinates: [NW2_WAYPOINTS[i], NW2_WAYPOINTS[nextIdx]] as [
          number,
          number,
        ][],
      },
      properties: {
        segment_id: segId,
        waterway_id: "NW-2" as WaterwayId,
        km_start: NW2_KM_MARKERS[i],
        km_end: NW2_KM_MARKERS[nextIdx] ?? NW2_KM_MARKERS[i] + 55,
        navigability_class: cls,
        depth_m: parseFloat(depth.toFixed(2)),
        width_m: 600 + i * 40 + (month >= 6 && month <= 9 ? 250 : 0),
        confidence: 0.85 + pseudoRandom(i + month * 20) * 0.12,
        velocity_ms: 0.6 + depth * 0.07,
        landmark,
        state: "Assam",
      },
    };
  });

  return { type: "FeatureCollection", features };
}

// ============================================================
// NAVIGABILITY MAP — NW-1
// ============================================================

export function buildNavigabilityMap(
  waterwayId: WaterwayId,
  month: number,
  year: number,
): NavigabilityMap {
  const isNW1 = waterwayId === "NW-1";
  const geojson = isNW1
    ? buildNW1GeoJSON(month, year)
    : buildNW2GeoJSON(month, year);

  const predictions = geojson.features.map((f, idx) => {
    const p = f.properties;
    const stdDev = 0.3 + pseudoRandom(1) * 0.4;
    return {
      prediction_id: `pred-${f.id}-${year}-${month}`,
      segment_id: f.id,
      waterway_id: waterwayId,
      month: month as Month,
      year,
      created_at: NOW_ISO,
      navigability_class: p.navigability_class as NavigabilityClass,
      predicted_depth_m: p.depth_m,
      predicted_width_m: p.width_m,
      probability: p.confidence,
      risk_score:
        p.navigability_class === "non_navigable"
          ? 70 + pseudoRandom(2) * 28
          : p.navigability_class === "conditional"
            ? 35 + pseudoRandom(3) * 30
            : 5 + pseudoRandom(4) * 25,
      depth_ci: {
        lower_95: parseFloat((p.depth_m - stdDev * 1.96).toFixed(2)),
        upper_95: parseFloat((p.depth_m + stdDev * 1.96).toFixed(2)),
        lower_80: parseFloat((p.depth_m - stdDev * 1.28).toFixed(2)),
        upper_80: parseFloat((p.depth_m + stdDev * 1.28).toFixed(2)),
        std_dev: parseFloat(stdDev.toFixed(3)),
      },
      spectral_indices: {
        mndwi: parseFloat((0.15 + pseudoRandom(5) * 0.65).toFixed(4)),
        ndwi: parseFloat((0.1 + pseudoRandom(6) * 0.55).toFixed(4)),
        ndvi: parseFloat((-0.05 + pseudoRandom(7) * 0.2).toFixed(4)),
        stumpf_ratio: parseFloat((1.02 + pseudoRandom(8) * 0.8).toFixed(4)),
        awei_sh: parseFloat((0.08 + pseudoRandom(9) * 0.5).toFixed(4)),
        awei_nsh: parseFloat((0.05 + pseudoRandom(10) * 0.45).toFixed(4)),
      },
      depth_threshold_m: 3.0,
      is_above_threshold: p.depth_m >= 3.0,
      margin_m: parseFloat((p.depth_m - 3.0).toFixed(2)),
      discharge_m3s: isNW1
        ? 4000 + idx * 200 + monsoonMultiplier(month) * 8000
        : 8000 + idx * 350 + monsoonMultiplier(month) * 20000,
      velocity_ms: p.velocity_ms,
      water_surface_area_km2: parseFloat(
        ((p.width_m * 0.06 * (p.km_end - p.km_start)) / 1000).toFixed(1),
      ),
    };
  });

  const navigable = predictions.filter(
    (p) => p.navigability_class === "navigable",
  );
  const conditional = predictions.filter(
    (p) => p.navigability_class === "conditional",
  );
  const nonNavigable = predictions.filter(
    (p) => p.navigability_class === "non_navigable",
  );

  const totalKm = isNW1 ? 1020 : 770;
  const navPct = navigable.length / predictions.length;

  return {
    waterway_id: waterwayId,
    month: month as Month,
    year,
    generated_at: NOW_ISO,
    total_segments: predictions.length,
    navigable_count: navigable.length,
    conditional_count: conditional.length,
    non_navigable_count: nonNavigable.length,
    navigable_km: parseFloat((totalKm * navPct).toFixed(1)),
    conditional_km: parseFloat(
      (totalKm * (conditional.length / predictions.length)).toFixed(1),
    ),
    non_navigable_km: parseFloat(
      (totalKm * (nonNavigable.length / predictions.length)).toFixed(1),
    ),
    total_km: totalKm,
    navigable_pct: parseFloat((navPct * 100).toFixed(1)),
    mean_confidence: parseFloat(
      (
        predictions.reduce((a, b) => a + b.probability, 0) / predictions.length
      ).toFixed(3),
    ),
    predictions,
    geojson,
  };
}

// ============================================================
// SEASONAL CALENDAR
// ============================================================

function buildCalendarRow(
  segId: string,
  kmStart: number,
  kmEnd: number,
  baseDepthAug: number,
  isNW2 = false,
): SegmentCalendarRow {
  const months: MonthlyCell[] = Array.from({ length: 12 }, (_, i) => {
    const m = (i + 1) as Month;
    const depth = baseDepthAug * monsoonMultiplier(m);
    const cls = classifyDepth(depth);
    const prob =
      cls === "navigable"
        ? 0.78 + pseudoRandom(11) * 0.2
        : cls === "conditional"
          ? 0.55 + pseudoRandom(12) * 0.25
          : 0.65 + pseudoRandom(13) * 0.3;
    return {
      month: m,
      navigability_class: cls,
      predicted_depth_m: parseFloat(depth.toFixed(2)),
      probability: parseFloat(prob.toFixed(3)),
      risk_score:
        cls === "non_navigable"
          ? 60 + pseudoRandom(14) * 35
          : cls === "conditional"
            ? 30 + pseudoRandom(15) * 30
            : 5 + pseudoRandom(16) * 20,
      is_monsoon: m >= 6 && m <= 9,
      label: monthLabel(m),
    };
  });

  const navMonths = months.filter((m) => m.navigability_class === "navigable");
  const worstMonth = months.reduce((a, b) =>
    a.predicted_depth_m < b.predicted_depth_m ? a : b,
  );
  const bestMonth = months.reduce((a, b) =>
    a.predicted_depth_m > b.predicted_depth_m ? a : b,
  );

  return {
    segment_id: segId,
    km_start: kmStart,
    km_end: kmEnd,
    km_label: `${kmStart}–${kmEnd} km`,
    months,
    navigable_months_count: navMonths.length,
    best_month: bestMonth.month,
    worst_month: worstMonth.month,
  };
}

export function buildSeasonalCalendar(
  waterwayId: WaterwayId,
  year: number,
): SeasonalCalendar {
  const isNW1 = waterwayId === "NW-1";
  const depths = isNW1 ? NW1_BASE_DEPTHS_AUG : NW2_BASE_DEPTHS_AUG;
  const kmMarkers = isNW1 ? NW1_KM_MARKERS : NW2_KM_MARKERS;
  const landmarks = isNW1 ? NW1_LANDMARKS : NW2_LANDMARKS;
  const prefix = isNW1 ? "NW1" : "NW2";

  const rows: SegmentCalendarRow[] = depths.map((d, i) => {
    const segId = `${prefix}-SEG-${String(i + 1).padStart(2, "0")}`;
    const kmStart = kmMarkers[i] ?? i * 55;
    const kmEnd = kmMarkers[i + 1] ?? kmStart + 55;
    return buildCalendarRow(segId, kmStart, kmEnd, d, !isNW1);
  });

  const monthSummaries = Array.from({ length: 12 }, (_, i) => {
    const m = (i + 1) as Month;
    const mult = monsoonMultiplier(m);
    const meanDepth =
      (depths.reduce((a, b) => a + b, 0) / depths.length) * mult;
    const navCount = depths.filter(
      (d) => classifyDepth(d * mult) === "navigable",
    ).length;
    const cls = classifyDepth(meanDepth);
    return {
      month: m,
      label: monthLabel(m),
      navigable_pct: parseFloat(((navCount / depths.length) * 100).toFixed(1)),
      mean_depth_m: parseFloat(meanDepth.toFixed(2)),
      dominant_class: cls,
      alert_count:
        cls === "non_navigable"
          ? 3 + Math.floor(pseudoRandom(17) * 4)
          : cls === "conditional"
            ? 1 + Math.floor(pseudoRandom(18) * 3)
            : 0,
    };
  });

  return {
    waterway_id: waterwayId,
    year,
    generated_at: NOW_ISO,
    rows,
    month_summaries: monthSummaries,
  };
}

// ============================================================
// RISK ALERTS
// ============================================================

export const MOCK_ALERTS_NW1: RiskAlert[] = [
  {
    alert_id: "ALT-NW1-001",
    waterway_id: "NW-1",
    segment_id: "NW1-SEG-07",
    km_start: 360,
    km_end: 410,
    severity: "CRITICAL",
    alert_type: "DEPTH_CRITICAL",
    title: "Critical Depth Deficiency — Ballia Reach",
    description:
      "Predicted depth of 1.41 m falls critically below the 2.0 m minimum threshold. " +
      "All commercial vessel movements should be suspended immediately.",
    predicted_value: 1.41,
    threshold_value: 2.0,
    unit: "m",
    risk_score: 91,
    valid_from: new Date(Date.now() - 3600_000).toISOString(),
    valid_until: new Date(Date.now() + 72 * 3600_000).toISOString(),
    created_at: new Date(Date.now() - 3600_000).toISOString(),
    is_active: true,
    recommended_actions: [
      "Suspend all commercial barge operations in this reach",
      "Issue NOTAM to vessel operators",
      "Deploy survey vessel for real-time depth monitoring",
      "Activate alternative road transport for perishable cargo",
    ],
    affected_vessels: ["Class-III Barges", "Push-Tow Convoys"],
  },
  {
    alert_id: "ALT-NW1-002",
    waterway_id: "NW-1",
    segment_id: "NW1-SEG-08",
    km_start: 410,
    km_end: 460,
    severity: "CRITICAL",
    alert_type: "SEASONAL_CLOSURE",
    title: "Seasonal Navigability Closure — Chhapra Reach",
    description:
      "Predicted depth of 1.71 m. Seasonal low-water conditions expected to persist for 6–8 weeks.",
    predicted_value: 1.71,
    threshold_value: 2.0,
    unit: "m",
    risk_score: 84,
    valid_from: new Date(Date.now() - 7200_000).toISOString(),
    valid_until: new Date(Date.now() + 15 * 24 * 3600_000).toISOString(),
    created_at: new Date(Date.now() - 7200_000).toISOString(),
    is_active: true,
    recommended_actions: [
      "Restrict vessels to Class-I and Class-II only",
      "Increase dredging operations at km 415–435",
      "Coordinate with Inland Waterways Authority of India (IWAI)",
    ],
    affected_vessels: ["Class-III Barges", "Class-IV Vessels"],
  },
  {
    alert_id: "ALT-NW1-003",
    waterway_id: "NW-1",
    segment_id: "NW1-SEG-04",
    km_start: 200,
    km_end: 260,
    severity: "WARNING",
    alert_type: "DEPTH_WARNING",
    title: "Depth Below Advisory Level — Varanasi Reach",
    description:
      "Predicted depth of 2.28 m is approaching the 3.0 m navigable threshold. " +
      "Vessel operators advised to exercise caution and reduce draft.",
    predicted_value: 2.28,
    threshold_value: 3.0,
    unit: "m",
    risk_score: 62,
    valid_from: new Date(Date.now() - 14400_000).toISOString(),
    valid_until: new Date(Date.now() + 48 * 3600_000).toISOString(),
    created_at: new Date(Date.now() - 14400_000).toISOString(),
    is_active: true,
    recommended_actions: [
      "Reduce vessel draft to ≤ 1.6 m",
      "Avoid night navigation in this reach",
      "Monitor IWAI daily depth bulletins",
    ],
    affected_vessels: ["Loaded Class-III Barges"],
  },
  {
    alert_id: "ALT-NW1-004",
    waterway_id: "NW-1",
    segment_id: "NW1-SEG-15",
    km_start: 745,
    km_end: 790,
    severity: "WARNING",
    alert_type: "VELOCITY_HIGH",
    title: "Elevated Current Velocity — Kahalgaon Reach",
    description:
      "River current velocity of 2.1 m/s exceeds safe navigation threshold (1.5 m/s) " +
      "following upstream discharge from Farakka Barrage.",
    predicted_value: 2.1,
    threshold_value: 1.5,
    unit: "m/s",
    risk_score: 55,
    valid_from: new Date(Date.now() - 18000_000).toISOString(),
    valid_until: new Date(Date.now() + 36 * 3600_000).toISOString(),
    created_at: new Date(Date.now() - 18000_000).toISOString(),
    is_active: true,
    recommended_actions: [
      "Increase engine power for upstream vessels",
      "Reduce convoy length for push-tow operations",
      "Standby tug assistance at Kahalgaon terminal",
    ],
    affected_vessels: ["Push-Tow Convoys", "Inland Vessels > 500 DWT"],
  },
  {
    alert_id: "ALT-NW1-005",
    waterway_id: "NW-1",
    segment_id: "NW1-SEG-18",
    km_start: 900,
    km_end: 960,
    severity: "INFO",
    alert_type: "FLOOD_RISK",
    title: "Elevated Flood Risk — Murshidabad Reach",
    description:
      "Water level 0.8 m above HFL recorded at Murshidabad gauge. " +
      "Floodplain inundation may affect bank infrastructure.",
    predicted_value: 8.4,
    threshold_value: 7.6,
    unit: "m (gauge)",
    risk_score: 38,
    valid_from: new Date(Date.now() - 10800_000).toISOString(),
    valid_until: new Date(Date.now() + 24 * 3600_000).toISOString(),
    created_at: new Date(Date.now() - 10800_000).toISOString(),
    is_active: true,
    recommended_actions: [
      "Secure vessels at jetties using additional mooring lines",
      "Monitor CWC flood forecasting portal",
      "Avoid anchorage in flood-prone areas",
    ],
  },
];

export const MOCK_ALERTS_NW2: RiskAlert[] = [
  {
    alert_id: "ALT-NW2-001",
    waterway_id: "NW-2",
    segment_id: "NW2-SEG-09",
    km_start: 465,
    km_end: 525,
    severity: "WARNING",
    alert_type: "OBSTACLE_DETECTED",
    title: "Sand Bar Formation — Majuli Island Reach",
    description:
      "Satellite imagery indicates dynamic sand bar formation at km 489. " +
      "Effective navigable width reduced to ~210 m.",
    predicted_value: 210,
    threshold_value: 300,
    unit: "m (width)",
    risk_score: 67,
    valid_from: new Date(Date.now() - 21600_000).toISOString(),
    valid_until: new Date(Date.now() + 96 * 3600_000).toISOString(),
    created_at: new Date(Date.now() - 21600_000).toISOString(),
    is_active: true,
    recommended_actions: [
      "Issue updated navigation chart for Majuli reach",
      "Dispatch survey vessel for bathymetric survey",
      "Post temporary navigation buoys marking safe channel",
    ],
    affected_vessels: ["Large Barges > 30 m beam"],
  },
  {
    alert_id: "ALT-NW2-002",
    waterway_id: "NW-2",
    segment_id: "NW2-SEG-03",
    km_start: 115,
    km_end: 175,
    severity: "INFO",
    alert_type: "DEPTH_WARNING",
    title: "Seasonal Depth Reduction — Jogighopa Reach",
    description:
      "Post-monsoon depth reduction trend detected. Depth forecast to approach " +
      "3.0 m threshold within 3 weeks.",
    predicted_value: 3.51,
    threshold_value: 3.0,
    unit: "m",
    risk_score: 28,
    valid_from: new Date(Date.now() - 5400_000).toISOString(),
    valid_until: new Date(Date.now() + 21 * 24 * 3600_000).toISOString(),
    created_at: new Date(Date.now() - 5400_000).toISOString(),
    is_active: true,
    recommended_actions: [
      "Begin pre-positioning of dredging equipment",
      "Advise vessel operators of evolving conditions",
    ],
  },
];

export function getMockAlerts(waterwayId?: WaterwayId): RiskAlert[] {
  if (waterwayId === "NW-1") return MOCK_ALERTS_NW1;
  if (waterwayId === "NW-2") return MOCK_ALERTS_NW2;
  return [...MOCK_ALERTS_NW1, ...MOCK_ALERTS_NW2];
}

// ============================================================
// DEPTH PROFILE
// ============================================================

export function buildDepthProfile(
  waterwayId: WaterwayId,
  month: number,
  year: number,
): DepthProfile {
  const isNW1 = waterwayId === "NW-1";
  const depths = isNW1 ? NW1_BASE_DEPTHS_AUG : NW2_BASE_DEPTHS_AUG;
  const kmMarkers = isNW1 ? NW1_KM_MARKERS : NW2_KM_MARKERS;
  const landmarks = isNW1 ? NW1_LANDMARKS : NW2_LANDMARKS;
  const prefix = isNW1 ? "NW1" : "NW2";
  const mult = monsoonMultiplier(month);

  const points: DepthProfilePoint[] = [];

  depths.forEach((baseDepth, i) => {
    const kmStart = kmMarkers[i];
    const kmEnd = kmMarkers[i + 1] ?? kmStart + 60;
    // Generate 3 sub-points per segment for smooth profile
    const subCount = 3;
    for (let j = 0; j <= subCount; j++) {
      const fraction = j / subCount;
      const km = kmStart + (kmEnd - kmStart) * fraction;
      // Add slight sinusoidal variation along segment
      const variation = Math.sin(fraction * Math.PI) * 0.4;
      const depth = Math.max(0.5, baseDepth * mult + variation - 0.2);
      const stdDev = 0.3 + pseudoRandom(19) * 0.3;
      points.push({
        km: parseFloat(km.toFixed(1)),
        depth_m: parseFloat(depth.toFixed(2)),
        depth_lower_ci: parseFloat(
          Math.max(0.3, depth - stdDev * 1.96).toFixed(2),
        ),
        depth_upper_ci: parseFloat((depth + stdDev * 1.96).toFixed(2)),
        width_m: isNW1
          ? 350 + i * 20 + monsoonMultiplier(month) * 150
          : 550 + i * 35 + monsoonMultiplier(month) * 300,
        navigability_class: classifyDepth(depth),
        segment_id: `${prefix}-SEG-${String(i + 1).padStart(2, "0")}`,
        landmark: j === 0 ? landmarks[i] : undefined,
      });
    }
  });

  const allDepths = points.map((p) => p.depth_m);
  const minDepth = Math.min(...allDepths);
  const maxDepth = Math.max(...allDepths);
  const meanDepth = allDepths.reduce((a, b) => a + b, 0) / allDepths.length;
  const bottleneck = points.reduce((a, b) => (a.depth_m < b.depth_m ? a : b));

  return {
    waterway_id: waterwayId,
    month: month as Month,
    year,
    points,
    min_depth_m: parseFloat(minDepth.toFixed(2)),
    max_depth_m: parseFloat(maxDepth.toFixed(2)),
    mean_depth_m: parseFloat(meanDepth.toFixed(2)),
    navigable_threshold_m: 3.0,
    conditional_threshold_m: 2.0,
    bottleneck_km: bottleneck.km,
    bottleneck_depth_m: bottleneck.depth_m,
  };
}

// ============================================================
// WATERWAY STATS
// ============================================================

export function buildWaterwayStats(
  waterwayId: WaterwayId,
  year: number,
): WaterwayStats {
  const isNW1 = waterwayId === "NW-1";
  const depths = isNW1 ? NW1_BASE_DEPTHS_AUG : NW2_BASE_DEPTHS_AUG;
  const total = isNW1 ? 1020 : 770;

  const monthly_stats = Array.from({ length: 12 }, (_, i) => {
    const m = (i + 1) as Month;
    const mult = monsoonMultiplier(m);
    const navCount = depths.filter((d) => d * mult >= 3.0).length;
    return {
      month: m,
      label: monthLabel(m),
      navigable_pct: parseFloat(((navCount / depths.length) * 100).toFixed(1)),
      mean_depth_m: parseFloat(
        ((depths.reduce((a, b) => a + b, 0) / depths.length) * mult).toFixed(2),
      ),
      alert_count:
        navCount < depths.length * 0.5
          ? 3 + Math.floor(pseudoRandom(20) * 4)
          : Math.floor(pseudoRandom(21) * 2),
    };
  });

  const annualNavPct =
    monthly_stats.reduce((a, b) => a + b.navigable_pct, 0) / 12;
  const annualMeanDepth =
    monthly_stats.reduce((a, b) => a + b.mean_depth_m, 0) / 12;
  const peakMonth = monthly_stats.reduce((a, b) =>
    a.navigable_pct > b.navigable_pct ? a : b,
  ).month;
  const worstMonth = monthly_stats.reduce((a, b) =>
    a.navigable_pct < b.navigable_pct ? a : b,
  ).month;

  return {
    waterway_id: waterwayId,
    year,
    total_length_km: total,
    total_segments: depths.length,
    annual_navigable_pct: parseFloat(annualNavPct.toFixed(1)),
    annual_mean_depth_m: parseFloat(annualMeanDepth.toFixed(2)),
    peak_navigability_month: peakMonth,
    worst_navigability_month: worstMonth,
    total_alerts_year: 23,
    monthly_stats,
    yoy_navigability_change_pct: 3.2,
    yoy_depth_change_m: 0.18,
  };
}

// ============================================================
// ANALYTICS TRENDS (multi-year)
// ============================================================

const YEAR_COLORS: Record<number, string> = {
  2020: "#818cf8",
  2021: "#34d399",
  2022: "#fb923c",
  2023: "#60a5fa",
  2024: "#f472b6",
};

export function buildAnalyticsTrends(
  waterwayId: WaterwayId,
  years: number[],
): AnalyticsTrends {
  const depths =
    waterwayId === "NW-1" ? NW1_BASE_DEPTHS_AUG : NW2_BASE_DEPTHS_AUG;

  // Simulate a gradual improvement trend across years due to climate / dredging
  const trends = years.map((year, yi) => {
    const improvement = yi * 0.015; // +1.5% per year
    const points = Array.from({ length: 12 }, (_, i) => {
      const m = (i + 1) as Month;
      const mult = monsoonMultiplier(m);
      const navCount = depths.filter(
        (d) => classifyDepth(d * mult) === "navigable",
      ).length;
      const navPct = Math.min(
        100,
        (navCount / depths.length) * 100 +
          improvement * 100 +
          (pseudoRandom(22) - 0.5) * 4,
      );
      return {
        month: m,
        label: monthLabel(m),
        navigable_pct: parseFloat(navPct.toFixed(1)),
        mean_depth_m: parseFloat(
          ((depths.reduce((a, b) => a + b, 0) / depths.length) * mult).toFixed(
            2,
          ),
        ),
      };
    });

    return {
      year,
      color: YEAR_COLORS[year] ?? "#94a3b8",
      points,
      annual_mean_navigable_pct: parseFloat(
        (points.reduce((a, b) => a + b.navigable_pct, 0) / 12).toFixed(1),
      ),
    };
  });

  return {
    waterway_id: waterwayId,
    years,
    trends,
    generated_at: NOW_ISO,
  };
}

// ============================================================
// FEATURE IMPORTANCE (SHAP)
// ============================================================

export const MOCK_FEATURE_IMPORTANCE: FeatureImportance = {
  model_version: "v2.4.1",
  waterway_id: "NW-1",
  total_samples: 12480,
  generated_at: NOW_ISO,
  features: [
    {
      feature_name: "mndwi",
      display_name: "MNDWI (Modified NDWI)",
      shap_value: 0.842,
      importance_pct: 22.3,
      direction: "positive",
      category: "spectral",
      description:
        "Most influential predictor — higher MNDWI strongly indicates greater water extent and depth.",
    },
    {
      feature_name: "stumpf_ratio",
      display_name: "Stumpf Band Ratio",
      shap_value: 0.718,
      importance_pct: 19.0,
      direction: "positive",
      category: "spectral",
      description:
        "Band ratio method for optically shallow water depth retrieval.",
    },
    {
      feature_name: "ndwi",
      display_name: "NDWI (Green–NIR)",
      shap_value: 0.631,
      importance_pct: 16.7,
      direction: "positive",
      category: "spectral",
      description: "Normalized water index highlighting water body extent.",
    },
    {
      feature_name: "awei_sh",
      display_name: "AWEI-SH Index",
      shap_value: 0.487,
      importance_pct: 12.9,
      direction: "positive",
      category: "spectral",
      description: "Shadow-resistant automated water extraction index.",
    },
    {
      feature_name: "month",
      display_name: "Calendar Month",
      shap_value: 0.412,
      importance_pct: 10.9,
      direction: "mixed",
      category: "temporal",
      description:
        "Captures strong seasonality — monsoon months (Jun–Sep) dominate navigability.",
    },
    {
      feature_name: "upstream_discharge",
      display_name: "Upstream Discharge (m³/s)",
      shap_value: 0.338,
      importance_pct: 8.9,
      direction: "positive",
      category: "hydrological",
      description:
        "CWC gauge discharge data; higher discharge correlates with greater depth.",
    },
    {
      feature_name: "ndvi",
      display_name: "NDVI (Vegetation Index)",
      shap_value: 0.195,
      importance_pct: 5.2,
      direction: "negative",
      category: "spectral",
      description:
        "Higher vegetation index on banks indicates lower water levels.",
    },
    {
      feature_name: "channel_width_m",
      display_name: "Channel Width (m)",
      shap_value: 0.142,
      importance_pct: 3.8,
      direction: "positive",
      category: "morphological",
      description:
        "Wider channels (from SAR) generally correlate with greater depth.",
    },
    {
      feature_name: "km_along_river",
      display_name: "Distance Along River (km)",
      shap_value: 0.036,
      importance_pct: 0.3,
      direction: "mixed",
      category: "morphological",
      description:
        "Position along the river captures longitudinal depth patterns.",
    },
  ],
};

// ============================================================
// MODEL PERFORMANCE METRICS
// ============================================================

export const MOCK_MODEL_METRICS: ModelMetrics = {
  model_id: "aidstl-cnn-lstm-v2",
  model_name: "CNN-LSTM Depth Prediction Model",
  model_version: "v2.4.1",
  architecture:
    "Convolutional Neural Network + Long Short-Term Memory (Hybrid)",
  training_date: "2024-03-15T00:00:00Z",
  waterway_id: "NW-1",

  train_samples: 9840,
  val_samples: 1560,
  test_samples: 1080,
  train_years: [2019, 2020, 2021, 2022],
  test_years: [2023, 2024],

  regression: {
    r2: 0.918,
    rmse_m: 0.512,
    mae_m: 0.371,
    mbe_m: -0.024,
    mape_pct: 8.7,
    explained_variance: 0.921,
  },

  classification: {
    accuracy: 0.934,
    f1_macro: 0.927,
    f1_weighted: 0.931,
    precision_macro: 0.933,
    recall_macro: 0.924,
    kappa: 0.891,
    per_class: {
      navigable: {
        precision: 0.951,
        recall: 0.963,
        f1: 0.957,
        support: 512,
      },
      conditional: {
        precision: 0.894,
        recall: 0.878,
        f1: 0.886,
        support: 308,
      },
      non_navigable: {
        precision: 0.953,
        recall: 0.931,
        f1: 0.942,
        support: 260,
      },
    },
  },

  confusion_matrix: [
    { actual: "navigable", predicted: "navigable", count: 493 },
    { actual: "navigable", predicted: "conditional", count: 17 },
    { actual: "navigable", predicted: "non_navigable", count: 2 },
    { actual: "conditional", predicted: "navigable", count: 22 },
    { actual: "conditional", predicted: "conditional", count: 270 },
    { actual: "conditional", predicted: "non_navigable", count: 16 },
    { actual: "non_navigable", predicted: "navigable", count: 1 },
    { actual: "non_navigable", predicted: "conditional", count: 17 },
    { actual: "non_navigable", predicted: "non_navigable", count: 242 },
  ],

  satellites: ["Sentinel-2 MSI", "Landsat-8 OLI", "Sentinel-1 SAR"],
  bands_used: ["Blue", "Green", "Red", "NIR", "SWIR-1", "SWIR-2", "VV", "VH"],
  temporal_resolution: "10-day composite",
  spatial_resolution_m: 10,
};

// ============================================================
// CONVENIENCE EXPORTS — pre-built for current month/year
// ============================================================

const CURRENT_MONTH = (new Date().getMonth() + 1) as Month;
const CURRENT_YEAR = new Date().getFullYear();

export const MOCK_NW1_MAP = buildNavigabilityMap(
  "NW-1",
  CURRENT_MONTH,
  CURRENT_YEAR,
);
export const MOCK_NW2_MAP = buildNavigabilityMap(
  "NW-2",
  CURRENT_MONTH,
  CURRENT_YEAR,
);
export const MOCK_NW1_CAL = buildSeasonalCalendar("NW-1", CURRENT_YEAR);
export const MOCK_NW2_CAL = buildSeasonalCalendar("NW-2", CURRENT_YEAR);
export const MOCK_NW1_DEPTH = buildDepthProfile(
  "NW-1",
  CURRENT_MONTH,
  CURRENT_YEAR,
);
export const MOCK_NW2_DEPTH = buildDepthProfile(
  "NW-2",
  CURRENT_MONTH,
  CURRENT_YEAR,
);
export const MOCK_NW1_STATS = buildWaterwayStats("NW-1", CURRENT_YEAR);
export const MOCK_NW2_STATS = buildWaterwayStats("NW-2", CURRENT_YEAR);
export const MOCK_NW1_TRENDS = buildAnalyticsTrends(
  "NW-1",
  [2020, 2021, 2022, 2023, 2024],
);
export const MOCK_NW2_TRENDS = buildAnalyticsTrends(
  "NW-2",
  [2020, 2021, 2022, 2023, 2024],
);

/**
 * Returns the appropriate mock map for a waterway / month / year combo.
 * Falls back to re-building with the correct seasonal multiplier.
 */
const CURRENT_YEAR_DEFAULT = new Date().getFullYear();
const DEFAULT_YEARS = [2020, 2021, 2022, 2023, 2024];

export function getMockNavigabilityMap(
  waterwayId: WaterwayId,
  month: number,
  year: number = CURRENT_YEAR_DEFAULT,
): NavigabilityMap {
  return buildNavigabilityMap(waterwayId, month, year);
}

export function getMockSeasonalCalendar(
  waterwayId: WaterwayId,
  year: number = CURRENT_YEAR_DEFAULT,
): SeasonalCalendar {
  return buildSeasonalCalendar(waterwayId, year);
}

export function getMockDepthProfile(
  waterwayId: WaterwayId,
  month: number,
  year: number = CURRENT_YEAR_DEFAULT,
): DepthProfile {
  return buildDepthProfile(waterwayId, month, year);
}

export function getMockWaterwayStats(
  waterwayId: WaterwayId,
  year: number = CURRENT_YEAR_DEFAULT,
): WaterwayStats {
  return buildWaterwayStats(waterwayId, year);
}

export function getMockTrends(
  waterwayId: WaterwayId,
  years: number[] = DEFAULT_YEARS,
): AnalyticsTrends {
  return buildAnalyticsTrends(waterwayId, years);
}
