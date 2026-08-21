// ============================================================
// InlandRoute — Minimalist Light/Dark SaaS Metric Card
// ============================================================

'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { ArrowUpRight, ArrowDownRight, Minus } from 'lucide-react';

export interface StatCardProps {
  label: string;
  value: string | number;
  unit?: string;
  subtitle?: string;
  trend?: number;
  trendLabel?: string;
  icon?: React.ElementType;
  loading?: boolean;
  variant?: 'default' | 'navigable' | 'conditional' | 'non_navigable';
  onClick?: () => void;
}

export function StatCard({
  label,
  value,
  unit,
  subtitle,
  trend,
  trendLabel = 'this period',
  icon: Icon,
  loading = false,
  onClick,
}: StatCardProps) {
  if (loading) {
    return (
      <div className="saas-card p-5 animate-pulse">
        <div className="h-3 w-24 bg-slate-200 dark:bg-zinc-800 rounded mb-4" />
        <div className="h-8 w-32 bg-slate-200 dark:bg-zinc-800 rounded mb-2" />
        <div className="h-3 w-40 bg-slate-100 dark:bg-zinc-800/60 rounded" />
      </div>
    );
  }

  const isPositive = trend !== undefined && trend > 0;
  const isNegative = trend !== undefined && trend < 0;

  return (
    <motion.div
      whileHover={onClick ? { y: -2 } : undefined}
      onClick={onClick}
      className={`saas-card saas-card-hover p-5 flex flex-col justify-between ${
        onClick ? 'cursor-pointer' : ''
      }`}
    >
      {/* Category Header */}
      <div className="flex items-center justify-between gap-2 mb-3">
        <span className="text-xs font-semibold text-slate-500 dark:text-zinc-400 uppercase tracking-wider">
          {label}
        </span>
        {Icon && (
          <div className="w-8 h-8 rounded-lg bg-slate-100 dark:bg-zinc-800 flex items-center justify-center text-slate-700 dark:text-zinc-300">
            <Icon size={16} />
          </div>
        )}
      </div>

      {/* Main Metric & Trend Pill */}
      <div className="flex items-baseline justify-between gap-2 my-1">
        <div className="flex items-baseline gap-1">
          <span className="text-3xl font-extrabold text-slate-900 dark:text-white tracking-tight tabular-nums">
            {value}
          </span>
          {unit && (
            <span className="text-sm font-semibold text-slate-500 dark:text-zinc-400">
              {unit}
            </span>
          )}
        </div>

        {trend !== undefined && (
          <div
            className={`pill ${
              isPositive
                ? 'pill-trend-up'
                : isNegative
                ? 'pill-trend-down'
                : 'bg-slate-100 dark:bg-zinc-800 text-slate-600 dark:text-zinc-300 border border-slate-200 dark:border-zinc-700'
            }`}
          >
            {isPositive && <ArrowUpRight size={12} />}
            {isNegative && <ArrowDownRight size={12} />}
            {!isPositive && !isNegative && <Minus size={12} />}
            <span>{isPositive ? `+${trend}%` : `${trend}%`}</span>
          </div>
        )}
      </div>

      {/* Subtext Description */}
      {subtitle && (
        <p className="text-xs text-slate-500 dark:text-zinc-400 font-medium mt-2">
          {subtitle} {trendLabel ? `· ${trendLabel}` : ''}
        </p>
      )}
    </motion.div>
  );
}

export function StatCardGrid({ children }: { children: React.ReactNode }) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
      {children}
    </div>
  );
}
