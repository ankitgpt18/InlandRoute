// ============================================================
// AIDSTL — Inland Waterway Navigability Prediction System
// Analytics Page — Model metrics, feature importance, trends
// ============================================================

'use client';

import React, { useMemo, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  RadialBarChart,
  RadialBar,
  PieChart,
  Pie,
  Cell,
  Legend,
  type TooltipProps,
} from 'recharts';
import {
  BrainCircuit,
  Zap,
  Target,
  BarChart3,
  TrendingUp,
  Layers,
  Cpu,
  Database,
  FlaskConical,
  CheckCircle2,
  Info,
  ChevronDown,
  ArrowUpRight,
  Satellite,
  Activity,
  Award,
  GitBranch,
} from 'lucide-react';
import { useAppStore } from '@/store/app-store';
import { MOCK_FEATURE_IMPORTANCE, MOCK_MODEL_METRICS, getMockTrends } from '@/lib/mock-data';
import { TrendChart } from '@/components/charts/trend-chart';
import { StatCard, StatCardGrid } from '@/components/ui/stat-card';
import { cn } from '@/lib/utils';

// ─── Types ────────────────────────────────────────────────────────────────────

interface FeatureImportanceItem {
  feature_name:   string;
  display_name:   string;
  shap_value:     number;
  importance_pct: number;
  direction:      'positive' | 'negative';
  category:       string;
  description:    string;
}

// ─── Colour helpers ────────────────────────────────────────────────────────────

const CATEGORY_COLORS: Record<string, string> = {
  spectral:       '#3b82f6',
  hydrological:   '#22c55e',
  meteorological: '#0ea5e9',
  geomorphological:'#8b5cf6',
  temporal:       '#f59e0b',
};

const CATEGORY_ICONS: Record<string, React.ElementType> = {
  spectral:        Satellite,
  hydrological:    Activity,
  meteorological:  TrendingUp,
  geomorphological:Layers,
  temporal:        BarChart3,
};

// ─── Section Heading ──────────────────────────────────────────────────────────

function SectionHeading({
  title,
  subtitle,
  icon: Icon,
  badge,
  className,
}: {
  title:     string;
  subtitle?: string;
  icon?:     React.ElementType;
  badge?:    string;
  className?: string;
}) {
  return (
    <div className={cn('flex items-center gap-3 mb-5', className)}>
      {Icon && (
        <div className="w-8 h-8 rounded-xl bg-blue-500/15 border border-blue-500/25 flex items-center justify-center flex-shrink-0">
          <Icon size={15} className="text-slate-400" />
        </div>
      )}
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2 flex-wrap">
          <h2 className="text-base font-bold text-slate-900 leading-tight">{title}</h2>
          {badge && (
            <span className="
              text-[9px] font-bold tracking-wider uppercase
              px-2 py-0.5 rounded-full
              bg-blue-500/15 text-slate-400 border border-blue-500/25
            ">
              {badge}
            </span>
          )}
        </div>
        {subtitle && (
          <p className="text-[12px] text-slate-500 mt-0.5">{subtitle}</p>
        )}
      </div>
    </div>
  );
}

// ─── Model Architecture Card ──────────────────────────────────────────────────

function ModelArchitectureCard() {
  const [expanded, setExpanded] = useState(false);

  const components = [
    {
      name:    'Temporal Fusion Transformer',
      abbr:    'TFT',
      role:    'Primary depth estimator from time-series spectral data',
      icon:    BrainCircuit,
      color:   '#3b82f6',
      params:  '12.4M',
      details: 'Variable Selection Networks + Gated Residual Networks + Multi-Head Attention (8 heads, d_model=128) + LSTM encoder-decoder (hidden=128, layers=2) + Quantile output head (10th, 50th, 90th percentiles)',
    },
    {
      name:    'Swin Spectral Encoder',
      abbr:    'Swin-T',
      role:    'Satellite patch feature extraction (12-band Sentinel-2)',
      icon:    Satellite,
      color:   '#8b5cf6',
      params:  '28.3M',
      details: 'Swin Transformer (swin_tiny_patch4_window7_224) adapted for 12-channel input with learned spatial attention. Outputs 64-dim patch embeddings via projection head.',
    },
    {
      name:    'Cross-Modal Attention Fusion',
      abbr:    'CMA',
      role:    'Merges TFT temporal + Swin-T spatial representations',
      icon:    GitBranch,
      color:   '#f59e0b',
      params:  '2.1M',
      details: 'Multi-head cross-attention (4 heads) fusing temporal TFT representation with Swin spatial embeddings. Residual gating for stable training.',
    },
    {
      name:    'LightGBM + XGBoost',
      abbr:    'LGBM + XGB',
      role:    'Gradient boosting on engineered spectral features',
      icon:    Layers,
      color:   '#22c55e',
      params:  '—',
      details: 'LightGBM: leaf-wise growth, 500 leaves. XGBoost: 1000 estimators, lr=0.05, L2 reg=1.0. Both trained on the same feature matrix as HydroFormer.',
    },
    {
      name:    'Ridge Meta-Learner',
      abbr:    'Stacking',
      role:    'Combines all base model predictions via cross-validated stacking',
      icon:    Cpu,
      color:   '#0ea5e9',
      params:  '—',
      details: '5-fold spatial block cross-validation (prevents spatial data leakage). Out-of-fold predictions from all base models fed to RidgeCV meta-learner. Final output includes conformal prediction intervals.',
    },
  ];

  return (
    <div className="glass-card p-5 rounded-2xl">
      <SectionHeading
        title="HydroFormer Architecture"
        subtitle="TFT + Swin Transformer ensemble with stacking meta-learner"
        icon={BrainCircuit}
        badge="v1.0"
      />

      {/* Architecture diagram (ASCII-art style visual) */}
      <div className="relative mb-5 p-4 rounded-xl bg-slate-50/60 border border-slate-900/[0.06] overflow-x-auto">
        <div className="flex items-center justify-center gap-2 min-w-[500px]">
          {/* Input block */}
          <div className="flex flex-col gap-1.5">
            <div className="px-3 py-2 rounded-lg bg-slate-100/80 border border-slate-900/[0.08] text-center">
              <div className="text-[9px] font-bold text-slate-400 uppercase tracking-wider">Input</div>
              <div className="text-[11px] font-semibold text-slate-800 mt-0.5">Sentinel-2</div>
              <div className="text-[9px] text-slate-500">12 bands · 10m</div>
            </div>
            <div className="px-3 py-2 rounded-lg bg-slate-100/80 border border-slate-900/[0.08] text-center">
              <div className="text-[9px] font-bold text-slate-400 uppercase tracking-wider">Input</div>
              <div className="text-[11px] font-semibold text-slate-800 mt-0.5">CWC + ERA5</div>
              <div className="text-[9px] text-slate-500">T=12 months</div>
            </div>
          </div>

          {/* Arrow */}
          <div className="text-slate-400 text-lg">→</div>

          {/* Backbone block */}
          <div className="flex flex-col gap-1.5">
            <div className="px-3 py-2 rounded-lg border text-center" style={{ background: 'rgba(139,92,246,0.12)', borderColor: 'rgba(139,92,246,0.3)' }}>
              <div className="text-[9px] font-bold uppercase tracking-wider" style={{ color: '#8b5cf6' }}>Swin-T</div>
              <div className="text-[11px] font-semibold text-slate-800 mt-0.5">Spatial</div>
              <div className="text-[9px] text-slate-500">64-dim</div>
            </div>
            <div className="px-3 py-2 rounded-lg border text-center" style={{ background: 'rgba(59,130,246,0.12)', borderColor: 'rgba(59,130,246,0.3)' }}>
              <div className="text-[9px] font-bold uppercase tracking-wider" style={{ color: '#3b82f6' }}>TFT</div>
              <div className="text-[11px] font-semibold text-slate-800 mt-0.5">Temporal</div>
              <div className="text-[9px] text-slate-500">128-dim</div>
            </div>
          </div>

          {/* Arrow */}
          <div className="text-slate-400 text-lg">→</div>

          {/* Fusion */}
          <div className="px-3 py-3 rounded-lg border text-center" style={{ background: 'rgba(245,158,11,0.12)', borderColor: 'rgba(245,158,11,0.3)' }}>
            <div className="text-[9px] font-bold uppercase tracking-wider" style={{ color: '#f59e0b' }}>Cross-Modal</div>
            <div className="text-[11px] font-semibold text-slate-800 mt-0.5">Fusion</div>
            <div className="text-[9px] text-slate-500">Attention</div>
          </div>

          {/* Arrow */}
          <div className="text-slate-400 text-lg">→</div>

          {/* Ensemble */}
          <div className="px-3 py-3 rounded-lg border text-center" style={{ background: 'rgba(34,197,94,0.10)', borderColor: 'rgba(34,197,94,0.25)' }}>
            <div className="text-[9px] font-bold uppercase tracking-wider" style={{ color: '#22c55e' }}>Stack</div>
            <div className="text-[11px] font-semibold text-slate-800 mt-0.5">Ensemble</div>
            <div className="text-[9px] text-slate-500">Ridge CV</div>
          </div>

          {/* Arrow */}
          <div className="text-slate-400 text-lg">→</div>

          {/* Output */}
          <div className="flex flex-col gap-1.5">
            <div className="px-3 py-2 rounded-lg bg-emerald-500/12 border border-emerald-500/25 text-center">
              <div className="text-[9px] font-bold text-slate-400 uppercase tracking-wider">Output A</div>
              <div className="text-[11px] font-semibold text-slate-800 mt-0.5">Depth (m)</div>
              <div className="text-[9px] text-slate-500">+ 90% CI</div>
            </div>
            <div className="px-3 py-2 rounded-lg bg-blue-500/12 border border-blue-500/25 text-center">
              <div className="text-[9px] font-bold text-slate-400 uppercase tracking-wider">Output B</div>
              <div className="text-[11px] font-semibold text-slate-800 mt-0.5">Nav Class</div>
              <div className="text-[9px] text-slate-500">Prob + SHAP</div>
            </div>
          </div>
        </div>
      </div>

      {/* Component list */}
      <div className="space-y-2">
        {components.slice(0, expanded ? components.length : 3).map((comp, i) => {
          const Icon = comp.icon;
          return (
            <motion.div
              key={comp.abbr}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.05, duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
              className="flex items-start gap-3 p-3 rounded-xl bg-white/[0.03] border border-slate-900/[0.05]"
            >
              <div
                className="flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center mt-0.5"
                style={{ background: `${comp.color}18`, border: `1px solid ${comp.color}35` }}
              >
                <Icon size={14} style={{ color: comp.color }} />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-[12px] font-bold text-slate-800">{comp.name}</span>
                  <span
                    className="text-[9px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded-full"
                    style={{ background: `${comp.color}18`, color: comp.color, border: `1px solid ${comp.color}30` }}
                  >
                    {comp.abbr}
                  </span>
                  {comp.params !== '—' && (
                    <span className="text-[9px] text-slate-400 font-medium">{comp.params} params</span>
                  )}
                </div>
                <p className="text-[11px] text-slate-500 mt-0.5 leading-snug">{comp.role}</p>
                {expanded && (
                  <motion.p
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="text-[10px] text-slate-400 mt-1.5 leading-relaxed font-mono"
                  >
                    {comp.details}
                  </motion.p>
                )}
              </div>
            </motion.div>
          );
        })}
      </div>

      <button
        onClick={() => setExpanded(v => !v)}
        className="
          mt-3 flex items-center gap-1.5 w-full justify-center py-2 rounded-xl
          text-[11px] font-semibold text-slate-500 hover:text-slate-700
          bg-white/[0.03] border border-slate-900/[0.06]
          hover:bg-white/[0.06]
          transition-all duration-150
        "
      >
        <motion.span
          animate={{ rotate: expanded ? 180 : 0 }}
          transition={{ duration: 0.2 }}
          className="flex items-center"
        >
          <ChevronDown size={13} />
        </motion.span>
        {expanded ? 'Show less' : `Show all ${components.length} components`}
      </button>
    </div>
  );
}

// ─── Confusion Matrix ─────────────────────────────────────────────────────────

function ConfusionMatrix({ data }: { data: { actual: string; predicted: string; count: number }[] }) {
  const classes = ['navigable', 'conditional', 'non_navigable'];
  const labels  = ['Navigable', 'Conditional', 'Non-Nav'];

  const colors: Record<string, string> = {
    navigable:     '#22c55e',
    conditional:   '#f59e0b',
    non_navigable: '#ef4444',
  };

  const getCount = (actual: string, predicted: string) =>
    data.find(d => d.actual === actual && d.predicted === predicted)?.count ?? 0;

  const rowTotals = classes.map(cls =>
    classes.reduce((s, p) => s + getCount(cls, p), 0)
  );

  const maxCount = Math.max(...data.map(d => d.count));

  return (
    <div>
      {/* Column headers */}
      <div className="flex items-center mb-2">
        <div className="w-24 flex-shrink-0" />
        <div className="flex-1 flex gap-1">
          {labels.map((l, i) => (
            <div key={i} className="flex-1 text-center">
              <span
                className="text-[9px] font-bold uppercase tracking-wider"
                style={{ color: colors[classes[i]] }}
              >
                {l}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Predicted label (rotated) */}
      <div className="flex items-start gap-2">
        {/* Actual label */}
        <div className="flex flex-col justify-around h-full gap-1 w-24 flex-shrink-0">
          {classes.map((cls, ri) => (
            <div key={cls} className="flex items-center justify-end pr-2 h-14">
              <span
                className="text-[9px] font-bold uppercase tracking-wider text-right leading-tight"
                style={{ color: colors[cls] }}
              >
                {labels[ri]}
              </span>
            </div>
          ))}
        </div>

        {/* Matrix cells */}
        <div className="flex-1 flex flex-col gap-1">
          {classes.map((actual, ri) => (
            <div key={actual} className="flex gap-1">
              {classes.map((predicted, ci) => {
                const count   = getCount(actual, predicted);
                const isDiag  = ri === ci;
                const pct     = rowTotals[ri] > 0 ? (count / rowTotals[ri]) * 100 : 0;
                const alpha   = Math.max(0.08, count / maxCount);
                const color   = isDiag ? colors[actual] : '#ef4444';

                return (
                  <motion.div
                    key={predicted}
                    initial={{ opacity: 0, scale: 0.85 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: (ri * 3 + ci) * 0.04, duration: 0.25 }}
                    className="flex-1 h-14 rounded-lg flex flex-col items-center justify-center border"
                    style={{
                      background: isDiag ? `${color}${Math.round(alpha * 255).toString(16).padStart(2, '0')}` : `rgba(239,68,68,${alpha * 0.3})`,
                      borderColor: isDiag ? `${color}40` : 'rgba(239,68,68,0.15)',
                    }}
                    title={`Actual: ${labels[ri]}, Predicted: ${labels[ci]}: ${count} samples (${pct.toFixed(1)}%)`}
                  >
                    <span
                      className="text-base font-extrabold tabular-nums leading-none"
                      style={{ color: isDiag ? color : 'rgba(239,68,68,0.7)' }}
                    >
                      {count}
                    </span>
                    <span
                      className="text-[9px] font-semibold tabular-nums mt-0.5"
                      style={{ color: isDiag ? `${color}aa` : 'rgba(239,68,68,0.4)' }}
                    >
                      {pct.toFixed(0)}%
                    </span>
                    {isDiag && (
                      <CheckCircle2
                        size={8}
                        style={{ color: `${color}80` }}
                        className="mt-0.5"
                      />
                    )}
                  </motion.div>
                );
              })}
            </div>
          ))}
        </div>
      </div>

      {/* Legend */}
      <div className="flex items-center gap-4 mt-3 text-[10px] text-slate-400">
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded bg-emerald-500/30 border border-emerald-500/40" />
          <span>Correct prediction (diagonal)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded bg-red-500/15 border border-red-500/20" />
          <span>Misclassification</span>
        </div>
      </div>
    </div>
  );
}

// ─── Feature Importance Chart ─────────────────────────────────────────────────

interface FITooltipProps extends TooltipProps<number, string> {
  features: FeatureImportanceItem[];
}

function FITooltip({ active, payload, features }: FITooltipProps) {
  if (!active || !payload || payload.length === 0) return null;
  const entry = payload[0];
  const feat  = features.find(f => f.display_name === entry.payload?.display_name);
  if (!feat) return null;
  const color = CATEGORY_COLORS[feat.category] ?? '#94a3b8';

  return (
    <div className="
      bg-white/98 backdrop-blur-xl border border-slate-300 rounded-xl
      px-3.5 py-3 min-w-[220px] shadow-2xl pointer-events-none
    " style={{ borderColor: `${color}30` }}>
      <div className="text-[12px] font-bold text-slate-900 mb-1">{feat.display_name}</div>
      <div className="flex items-center gap-2 mb-2">
        <span
          className="text-[9px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded-full"
          style={{ background: `${color}20`, color, border: `1px solid ${color}35` }}
        >
          {feat.category}
        </span>
        <span className="text-[10px] text-slate-500">{feat.direction === 'positive' ? '↑ Increases depth' : '↓ Decreases depth'}</span>
      </div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-[11px]">
        <div>
          <div className="text-slate-400">SHAP value</div>
          <div className="font-bold text-slate-800 tabular-nums">{feat.shap_value.toFixed(4)}</div>
        </div>
        <div>
          <div className="text-slate-400">Importance</div>
          <div className="font-bold tabular-nums" style={{ color }}>{feat.importance_pct.toFixed(1)}%</div>
        </div>
      </div>
      <p className="text-[10px] text-slate-400 mt-2 leading-relaxed border-t border-slate-900/[0.06] pt-2">
        {feat.description}
      </p>
    </div>
  );
}

function FeatureImportanceChart() {
  const [sortBy, setSortBy] = useState<'importance' | 'category'>('importance');
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);

  const features: FeatureImportanceItem[] = MOCK_FEATURE_IMPORTANCE.features as FeatureImportanceItem[];

  const sorted = useMemo(() => {
    let list = [...features];
    if (selectedCategory) {
      list = list.filter(f => f.category === selectedCategory);
    }
    if (sortBy === 'importance') {
      list.sort((a, b) => b.importance_pct - a.importance_pct);
    } else {
      list.sort((a, b) => a.category.localeCompare(b.category) || b.importance_pct - a.importance_pct);
    }
    return list;
  }, [features, sortBy, selectedCategory]);

  const categories = Array.from(new Set(features.map(f => f.category)));

  // Category distribution for pie chart
  const categoryPie = useMemo(() => {
    return categories.map(cat => ({
      name:  cat.charAt(0).toUpperCase() + cat.slice(1),
      value: features.filter(f => f.category === cat).reduce((s, f) => s + f.importance_pct, 0),
      color: CATEGORY_COLORS[cat] ?? '#94a3b8',
    }));
  }, [features, categories]);

  const customBarShape = (props: {
    x?: number;
    y?: number;
    width?: number;
    height?: number;
    display_name?: string;
  }) => {
    const { x = 0, y = 0, width = 0, height = 0 } = props;
    const feat  = sorted.find(f => f.display_name === props.display_name);
    const color = feat ? (CATEGORY_COLORS[feat.category] ?? '#94a3b8') : '#94a3b8';
    return (
      <g>
        <rect x={x} y={y + 2} width={width} height={Math.max(0, height - 4)} rx={4} fill={`${color}30`} />
        <rect x={x} y={y + 2} width={Math.max(0, width - 2)} height={Math.max(0, height - 4)} rx={4} fill={color} opacity={0.85} />
      </g>
    );
  };

  return (
    <div>
      {/* Controls */}
      <div className="flex items-center gap-3 mb-4 flex-wrap">
        {/* Sort toggle */}
        <div className="flex items-center rounded-lg border border-slate-900/[0.08] overflow-hidden text-[11px]">
          {(['importance', 'category'] as const).map((opt, i) => (
            <React.Fragment key={opt}>
              {i > 0 && <div className="w-px h-5 bg-white/[0.08]" />}
              <button
                onClick={() => setSortBy(opt)}
                className={cn(
                  'px-2.5 py-1.5 font-semibold transition-all duration-150',
                  sortBy === opt ? 'bg-blue-500/20 text-slate-300' : 'text-slate-500 hover:text-slate-700',
                )}
              >
                {opt === 'importance' ? 'By Importance' : 'By Category'}
              </button>
            </React.Fragment>
          ))}
        </div>

        {/* Category filter */}
        <div className="flex items-center gap-1.5 flex-wrap">
          <button
            onClick={() => setSelectedCategory(null)}
            className={cn(
              'px-2.5 py-1 rounded-full text-[10px] font-bold uppercase tracking-wider border transition-all duration-150',
              selectedCategory === null
                ? 'bg-slate-500/20 text-slate-700 border-slate-500/40'
                : 'border-slate-900/[0.07] text-slate-400 hover:text-slate-400',
            )}
          >
            All
          </button>
          {categories.map(cat => {
            const color = CATEGORY_COLORS[cat] ?? '#94a3b8';
            const Icon  = CATEGORY_ICONS[cat];
            return (
              <button
                key={cat}
                onClick={() => setSelectedCategory(cat === selectedCategory ? null : cat)}
                className={cn(
                  'flex items-center gap-1 px-2.5 py-1 rounded-full text-[10px] font-bold uppercase tracking-wider border transition-all duration-150',
                  selectedCategory === cat ? 'opacity-100' : 'opacity-50 hover:opacity-80',
                )}
                style={selectedCategory === cat ? {
                  background:   `${color}20`,
                  borderColor:  `${color}40`,
                  color,
                } : {
                  borderColor: 'rgba(15,23,42,0.07)',
                  color: '#64748b',
                }}
              >
                {Icon && <Icon size={9} />}
                {cat}
              </button>
            );
          })}
        </div>

        <div className="ml-auto text-[10px] text-slate-400">
          {sorted.length} features · SHAP values
        </div>
      </div>

      {/* Two-column layout: bar chart + pie chart */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        {/* Bar chart (takes 2 cols) */}
        <div className="lg:col-span-2" style={{ height: Math.max(280, sorted.length * 30 + 40) }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={sorted.map(f => ({
                display_name:   f.display_name,
                importance_pct: f.importance_pct,
                category:       f.category,
              }))}
              layout="vertical"
              margin={{ top: 4, right: 40, left: 0, bottom: 4 }}
              barCategoryGap="20%"
            >
              <CartesianGrid
                strokeDasharray="3 4"
                stroke="rgba(15,23,42,0.04)"
                horizontal={false}
              />
              <XAxis
                type="number"
                domain={[0, Math.ceil(Math.max(...sorted.map(f => f.importance_pct)) * 1.1)]}
                tick={{ fill: '#475569', fontSize: 10, fontFamily: 'Inter, sans-serif' }}
                axisLine={{ stroke: 'rgba(15,23,42,0.06)' }}
                tickLine={false}
                tickFormatter={v => `${v}%`}
              />
              <YAxis
                type="category"
                dataKey="display_name"
                width={130}
                tick={{ fill: '#94a3b8', fontSize: 10, fontFamily: 'Inter, sans-serif', fontWeight: 500 }}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip
                content={<FITooltip features={features} />}
                cursor={{ fill: 'rgba(15,23,42,0.03)' }}
              />
              <Bar
                dataKey="importance_pct"
                name="Importance"
                radius={[0, 4, 4, 0]}
                maxBarSize={22}
              >
                {sorted.map((feat) => (
                  <Cell
                    key={feat.feature_name}
                    fill={CATEGORY_COLORS[feat.category] ?? '#94a3b8'}
                    fillOpacity={0.85}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Pie chart (takes 1 col) */}
        <div className="flex flex-col gap-3">
          <div style={{ height: 200 }}>
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={categoryPie}
                  cx="50%"
                  cy="50%"
                  innerRadius={55}
                  outerRadius={80}
                  paddingAngle={3}
                  dataKey="value"
                  isAnimationActive={true}
                  animationDuration={800}
                >
                  {categoryPie.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} opacity={0.85} />
                  ))}
                </Pie>
                <Tooltip
                  formatter={(val: number, name: string) => [`${val.toFixed(1)}%`, name]}
                  contentStyle={{
                    background: 'rgba(15,23,42,0.95)',
                    border: '1px solid rgba(15, 23, 42, 0.1)',
                    borderRadius: 10,
                    fontSize: 12,
                    color: '#e2e8f0',
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>

          {/* Category legend */}
          <div className="flex flex-col gap-2">
            {categoryPie.map(cat => (
              <div key={cat.name} className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ backgroundColor: cat.color }} />
                  <span className="text-[11px] text-slate-400 font-medium capitalize">{cat.name}</span>
                </div>
                <span className="text-[11px] font-bold tabular-nums" style={{ color: cat.color }}>
                  {cat.value.toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Per-Class Metrics ────────────────────────────────────────────────────────

function PerClassMetrics({ metrics }: { metrics: typeof MOCK_MODEL_METRICS }) {
  const classes = [
    { key: 'navigable',     label: 'Navigable',     color: '#22c55e' },
    { key: 'conditional',   label: 'Conditional',   color: '#f59e0b' },
    { key: 'non_navigable', label: 'Non-Navigable', color: '#ef4444' },
  ] as const;

  const perClass = metrics.classification.per_class as Record<
    string,
    { precision: number; recall: number; f1: number; support: number }
  >;

  return (
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
      {classes.map(cls => {
        const data   = perClass[cls.key];
        const radial = [
          { name: 'Precision', value: data.precision * 100, fill: cls.color },
          { name: 'Recall',    value: data.recall * 100,    fill: `${cls.color}88` },
          { name: 'F1',        value: data.f1 * 100,        fill: `${cls.color}50` },
        ];

        return (
          <motion.div
            key={cls.key}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
            className="flex flex-col gap-3 p-4 rounded-2xl border"
            style={{
              background:   `${cls.color}08`,
              borderColor:  `${cls.color}25`,
            }}
          >
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ backgroundColor: cls.color }} />
              <span className="text-[12px] font-bold" style={{ color: cls.color }}>{cls.label}</span>
              <span className="ml-auto text-[10px] text-slate-500 tabular-nums">
                n={data.support.toLocaleString()}
              </span>
            </div>

            {/* Radial bars */}
            <div style={{ height: 120 }}>
              <ResponsiveContainer width="100%" height="100%">
                <RadialBarChart
                  cx="50%"
                  cy="50%"
                  innerRadius={20}
                  outerRadius={55}
                  data={radial}
                  startAngle={90}
                  endAngle={-270}
                >
                  <RadialBar
                    dataKey="value"
                    cornerRadius={4}
                    background={{ fill: 'rgba(15,23,42,0.04)' }}
                  />
                </RadialBarChart>
              </ResponsiveContainer>
            </div>

            {/* Metric rows */}
            {[
              { label: 'Precision', value: data.precision, color: cls.color },
              { label: 'Recall',    value: data.recall,    color: `${cls.color}cc` },
              { label: 'F1 Score',  value: data.f1,        color: `${cls.color}99` },
            ].map(m => (
              <div key={m.label} className="flex items-center gap-2">
                <span className="text-[10px] text-slate-500 w-16 flex-shrink-0">{m.label}</span>
                <div className="flex-1 h-1.5 rounded-full bg-white/[0.06] overflow-hidden">
                  <motion.div
                    className="h-full rounded-full"
                    style={{ backgroundColor: m.color }}
                    initial={{ width: 0 }}
                    animate={{ width: `${m.value * 100}%` }}
                    transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
                  />
                </div>
                <span
                  className="text-[11px] font-bold tabular-nums w-10 text-right"
                  style={{ color: m.color }}
                >
                  {(m.value * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </motion.div>
        );
      })}
    </div>
  );
}

// ─── Data Split Info ──────────────────────────────────────────────────────────

function DataSplitCard({ metrics }: { metrics: typeof MOCK_MODEL_METRICS }) {
  const total = metrics.train_samples + metrics.val_samples + metrics.test_samples;

  const splits = [
    { label: 'Training',   n: metrics.train_samples,   pct: (metrics.train_samples / total) * 100,   color: '#3b82f6', years: metrics.train_years },
    { label: 'Validation', n: metrics.val_samples,     pct: (metrics.val_samples / total) * 100,     color: '#8b5cf6', years: [] },
    { label: 'Test',       n: metrics.test_samples,    pct: (metrics.test_samples / total) * 100,    color: '#22c55e', years: metrics.test_years },
  ];

  return (
    <div className="p-4 rounded-2xl border border-slate-900/[0.06] bg-white/[0.02]">
      <div className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-3">
        Dataset Split
      </div>

      {/* Stacked bar */}
      <div className="flex h-3 rounded-full overflow-hidden mb-3 gap-0.5">
        {splits.map(s => (
          <motion.div
            key={s.label}
            className="h-full rounded-sm"
            style={{ backgroundColor: s.color }}
            initial={{ flex: 0 }}
            animate={{ flex: s.pct }}
            transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
          />
        ))}
      </div>

      {/* Split details */}
      <div className="space-y-2">
        {splits.map(s => (
          <div key={s.label} className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: s.color }} />
            <span className="text-[11px] text-slate-400 w-20">{s.label}</span>
            <span className="text-[11px] font-bold text-slate-800 tabular-nums flex-1">
              {s.n.toLocaleString()} samples
            </span>
            <span className="text-[10px] text-slate-400 tabular-nums">{s.pct.toFixed(0)}%</span>
            {s.years.length > 0 && (
              <span className="text-[9px] text-slate-400">
                {s.years[0]}–{s.years[s.years.length - 1]}
              </span>
            )}
          </div>
        ))}
      </div>

      {/* Validation strategy */}
      <div className="mt-3 pt-3 border-t border-slate-900/[0.06]">
        <div className="text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-1.5">
          Validation Strategy
        </div>
        <div className="flex flex-wrap gap-1.5">
          {[
            '5-fold spatial block CV',
            'No adjacent-segment leakage',
            'Temporal hold-out: 2024',
            'IWAI LAD cross-validation',
          ].map(item => (
            <span
              key={item}
              className="text-[9px] font-medium text-slate-400 px-2 py-0.5 rounded-full bg-white/[0.04] border border-slate-900/[0.06]"
            >
              {item}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

// ─── Main Analytics Page ──────────────────────────────────────────────────────

export default function AnalyticsPage() {
  const selectedWaterway = useAppStore(s => s.selectedWaterway);
  const selectedYear     = useAppStore(s => s.selectedYear);

  const metrics = MOCK_MODEL_METRICS;

  // Animation
  const containerVariants = {
    hidden:   { opacity: 0 },
    visible:  { opacity: 1, transition: { staggerChildren: 0.08, delayChildren: 0.05 } },
  };
  const itemVariants = {
    hidden:   { opacity: 0, y: 20 },
    visible:  { opacity: 1, y: 0, transition: { duration: 0.45, ease: [0.16, 1, 0.3, 1] } },
  };

  return (
    <motion.div
      className="p-5 space-y-6 max-w-[1600px] mx-auto"
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      {/* ── Page heading ─────────────────────────────────────────────────── */}
      <motion.div variants={itemVariants}>
        <div className="flex items-start justify-between gap-4 flex-wrap">
          <div>
            <div className="flex items-center gap-3">
              <h1 className="text-2xl font-extrabold text-slate-900 tracking-tight">Analytics</h1>
              <span className="
                text-[10px] font-bold tracking-widest uppercase
                px-2.5 py-1 rounded-full
                bg-blue-500/12 border border-blue-500/25 text-slate-400
              ">
                HydroFormer v1.0
              </span>
            </div>
            <p className="text-[13px] text-slate-500 mt-1">
              Model performance · Feature importance · Multi-year navigability trends
            </p>
          </div>
          <div className="flex items-center gap-2 text-[11px] text-slate-400">
            <Database size={11} />
            <span>
              {metrics.train_samples.toLocaleString()} training ·{' '}
              {metrics.test_samples.toLocaleString()} test ·{' '}
              {metrics.satellites.join(', ')}
            </span>
          </div>
        </div>
      </motion.div>

      {/* ── Row 1: Top-level regression metrics ──────────────────────────── */}
      <motion.div variants={itemVariants}>
        <SectionHeading
          title="Depth Estimation Performance"
          subtitle="Task A — Regression · HydroFormer ensemble vs. CWC gauge ground truth"
          icon={Target}
          badge="Regression"
        />
        <StatCardGrid cols={4}>
          <StatCard
            label="R² Score"
            value={metrics.regression.r2 * 100}
            unit="%"
            decimals={2}
            subtitle="Variance explained by ensemble depth model"
            trend={8.3}
            trendDirection="up"
            trendLabel="vs RF baseline"
            trendUpIsGood={true}
            icon={Award}
            variant="navigable"
            animationDelay={0}
            tooltip="Coefficient of determination comparing predicted depth vs CWC gauge readings on held-out test segments"
          />
          <StatCard
            label="RMSE"
            value={metrics.regression.rmse_m}
            unit="m"
            decimals={3}
            subtitle="Root mean squared error on test segments"
            trend={-22.5}
            trendDirection="down"
            trendLabel="vs RF baseline"
            trendUpIsGood={false}
            icon={Activity}
            variant="accent"
            animationDelay={0.07}
            tooltip="Root Mean Squared Error in metres. Target was RMSE < 1.5m; achieved 1.24m."
          />
          <StatCard
            label="MAE"
            value={metrics.regression.mae_m}
            unit="m"
            decimals={3}
            subtitle="Mean absolute error — median segment-level"
            trend={-18.9}
            trendDirection="down"
            trendLabel="vs RF baseline"
            trendUpIsGood={false}
            icon={BarChart3}
            variant="info"
            animationDelay={0.14}
          />
          <StatCard
            label="MBE"
            value={Math.abs(metrics.regression.mbe_m)}
            unit="m"
            decimals={3}
            subtitle={`${metrics.regression.mbe_m >= 0 ? 'Slight over' : 'Slight under'}prediction bias`}
            trend={-45.2}
            trendDirection="down"
            trendLabel="vs RF baseline"
            trendUpIsGood={false}
            icon={Zap}
            variant="conditional"
            animationDelay={0.21}
            tooltip="Mean Bias Error. Values close to 0 indicate no systematic over/under-prediction."
          />
        </StatCardGrid>
      </motion.div>

      {/* ── Row 2: Classification metrics + Confusion matrix ─────────────── */}
      <motion.div variants={itemVariants}>
        <SectionHeading
          title="Navigability Classification"
          subtitle="Task B — 3-class classifier · Macro-averaged metrics on test set"
          icon={FlaskConical}
          badge="Classification"
        />

        <div className="grid grid-cols-1 xl:grid-cols-5 gap-5">
          {/* Metrics (3 cols) */}
          <div className="xl:col-span-3 space-y-4">
            {/* Top classification stats */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {[
                { label: 'Accuracy',  value: metrics.classification.accuracy,  unit: '%', mul: 100, color: '#22c55e' },
                { label: 'F1 Macro', value: metrics.classification.f1_macro,  unit: '%', mul: 100, color: '#3b82f6' },
                { label: 'Precision',value: metrics.classification.precision_macro, unit: '%', mul: 100, color: '#8b5cf6' },
                { label: 'Kappa',    value: metrics.classification.kappa,      unit: '',  mul: 1,   color: '#f59e0b' },
              ].map((m, i) => (
                <motion.div
                  key={m.label}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.06, duration: 0.3 }}
                  className="flex flex-col gap-0.5 p-3.5 rounded-xl border bg-white/[0.03]"
                  style={{ borderColor: `${m.color}25` }}
                >
                  <span className="text-[9px] font-bold uppercase tracking-wider text-slate-500">{m.label}</span>
                  <span className="text-2xl font-extrabold tracking-tight tabular-nums leading-tight" style={{ color: m.color }}>
                    {(m.value * m.mul).toFixed(m.mul === 1 ? 3 : 1)}{m.unit}
                  </span>
                </motion.div>
              ))}
            </div>

            {/* Per-class breakdown */}
            <PerClassMetrics metrics={metrics} />
          </div>

          {/* Confusion matrix (2 cols) */}
          <div className="xl:col-span-2 p-4 rounded-2xl border border-slate-900/[0.06] bg-white/[0.02]">
            <div className="mb-3">
              <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Confusion Matrix</div>
              <p className="text-[11px] text-slate-400 mt-0.5">
                Rows = actual · Columns = predicted
              </p>
            </div>
            <ConfusionMatrix data={metrics.confusion_matrix} />

            {/* Data split info */}
            <div className="mt-4">
              <DataSplitCard metrics={metrics} />
            </div>
          </div>
        </div>
      </motion.div>

      {/* ── Row 3: Feature Importance ─────────────────────────────────────── */}
      <motion.div variants={itemVariants}>
        <div className="glass-card p-5 rounded-2xl">
          <SectionHeading
            title="Feature Importance"
            subtitle="SHAP values — mean absolute impact on depth prediction across all test segments"
            icon={Layers}
            badge="SHAP"
          />
          <FeatureImportanceChart />
        </div>
      </motion.div>

      {/* ── Row 4: Multi-year trend ───────────────────────────────────────── */}
      <motion.div variants={itemVariants}>
        <div className="glass-card p-5 rounded-2xl">
          <TrendChart
            waterwayId={selectedWaterway}
            height={300}
            showTypeToggle={true}
            showModeToggle={true}
            showYearStats={true}
          />
        </div>
      </motion.div>

      {/* ── Row 5: Model architecture ─────────────────────────────────────── */}
      <motion.div variants={itemVariants}>
        <ModelArchitectureCard />
      </motion.div>

      {/* ── Footer: Data sources ──────────────────────────────────────────── */}
      <motion.div variants={itemVariants}>
        <div className="p-4 rounded-2xl border border-slate-900/[0.06] bg-white/[0.02]">
          <div className="text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-3">
            Data Sources & Specifications
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
            {[
              { label: 'Sentinel-2 L2A', sub: '10m · 5-day revisit', color: '#3b82f6', icon: Satellite },
              { label: 'CWC Gauge Data', sub: 'Daily water levels', color: '#22c55e', icon: Activity },
              { label: 'ERA5 Reanalysis', sub: '0.25° · hourly', color: '#0ea5e9', icon: TrendingUp },
              { label: 'SRTM DEM', sub: '30m elevation', color: '#8b5cf6', icon: Layers },
              { label: 'IWAI LAD Reports', sub: 'Monthly surveys', color: '#f59e0b', icon: Database },
              { label: 'Sentinel-1 SAR', sub: 'VV+VH backscatter', color: '#ef4444', icon: Zap },
            ].map(src => {
              const Icon = src.icon;
              return (
                <div
                  key={src.label}
                  className="flex items-start gap-2.5 p-3 rounded-xl border bg-white/[0.02]"
                  style={{ borderColor: `${src.color}20` }}
                >
                  <div
                    className="w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0 mt-0.5"
                    style={{ background: `${src.color}18`, border: `1px solid ${src.color}30` }}
                  >
                    <Icon size={13} style={{ color: src.color }} />
                  </div>
                  <div>
                    <div className="text-[11px] font-bold text-slate-700 leading-tight">{src.label}</div>
                    <div className="text-[10px] text-slate-400 mt-0.5">{src.sub}</div>
                  </div>
                </div>
              );
            })}
          </div>
          <div className="mt-3 flex items-center gap-2 text-[10px] text-slate-700">
            <Info size={10} />
            <span>
              Training data: 2019–2023 · Test data: 2024 (temporal out-of-sample) ·
              Spatial block CV prevents leakage between adjacent river segments ·
              Bands used: {metrics.bands_used.join(', ')}
            </span>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
}
