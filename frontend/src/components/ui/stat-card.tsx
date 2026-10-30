"use client";

import React from "react";
import { ArrowUpRight, ArrowDownRight, Minus } from "lucide-react";

export interface StatCardProps {
  label: string;
  value: string | number;
  unit?: string;
  subtitle?: string;
  trend?: number;
  trendLabel?: string;
  icon?: React.ElementType;
  loading?: boolean;
  variant?: "default" | "navigable" | "conditional" | "non_navigable";
  onClick?: () => void;
}

export function StatCard({
  label,
  value,
  unit,
  subtitle,
  trend,
  trendLabel = "this cycle",
  icon: Icon,
  loading = false,
  onClick,
}: StatCardProps) {
  if (loading) {
    return (
      <div className="rounded-lg border border-neutral-800 bg-black p-5 animate-pulse">
        <div className="h-3 w-24 bg-neutral-800 rounded mb-4" />
        <div className="h-7 w-32 bg-neutral-800 rounded mb-2" />
        <div className="h-3 w-40 bg-neutral-900 rounded" />
      </div>
    );
  }

  const isPositive = trend !== undefined && trend > 0;
  const isNegative = trend !== undefined && trend < 0;

  return (
    <div
      onClick={onClick}
      className={`relative flex flex-col justify-between overflow-hidden rounded-lg border border-neutral-800 bg-black p-5 transition hover:border-neutral-700 hover:bg-neutral-950/80 ${
        onClick ? "cursor-pointer" : ""
      }`}
    >
      {/* Category Header */}
      <div className="flex items-center justify-between gap-2 mb-3">
        <span className="text-xs font-semibold uppercase tracking-wider text-neutral-400">
          {label}
        </span>
        {Icon && (
          <div className="flex h-7 w-7 items-center justify-center rounded-md border border-neutral-800 bg-neutral-900 text-neutral-300">
            <Icon size={15} />
          </div>
        )}
      </div>

      {/* Main Metric & Trend Pill */}
      <div className="flex items-baseline justify-between gap-2 my-1">
        <div className="flex items-baseline gap-1.5">
          <span className="text-2xl sm:text-3xl font-semibold tracking-tight text-white tabular-nums">
            {value}
          </span>
          {unit && (
            <span className="text-xs font-medium text-neutral-400">
              {unit}
            </span>
          )}
        </div>

        {trend !== undefined && (
          <div
            className={`inline-flex items-center gap-1 text-xs font-mono tabular-nums ${
              isPositive
                ? "text-emerald-400"
                : isNegative
                ? "text-rose-400"
                : "text-neutral-400"
            }`}
          >
            {isPositive && <ArrowUpRight size={13} />}
            {isNegative && <ArrowDownRight size={13} />}
            {!isPositive && !isNegative && <Minus size={13} />}
            <span>{isPositive ? `+${trend}%` : `${trend}%`}</span>
          </div>
        )}
      </div>

      {/* Subtext Description */}
      {subtitle && (
        <p className="mt-2 text-xs text-neutral-400 font-normal">
          {subtitle} {trendLabel ? `· ${trendLabel}` : ""}
        </p>
      )}
    </div>
  );
}

export function StatCardGrid({ children }: { children: React.ReactNode }) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      {children}
    </div>
  );
}
