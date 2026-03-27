// ============================================================
// AIDSTL — Inland Waterway Navigability Prediction System
// Sidebar Navigation Component
// ============================================================

'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { motion, AnimatePresence } from 'framer-motion';
import {
  LayoutDashboard,
  Map,
  BarChart3,
  Bell,
  BrainCircuit,
  ChevronLeft,
  ChevronRight,
  ChevronUp,
  ChevronDown,
  Waves,
  Droplets,
  Calendar,
  Info,
  ExternalLink,
} from 'lucide-react';
import { useAppStore } from '@/store/app-store';
import type { WaterwayId, Month } from '@/types';

// ─── Constants ────────────────────────────────────────────────────────────────

const MONTH_NAMES = [
  'January', 'February', 'March', 'April',
  'May', 'June', 'July', 'August',
  'September', 'October', 'November', 'December',
] as const;

const SHORT_MONTH = [
  'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
] as const;

const NAV_ITEMS = [
  {
    href: '/dashboard',
    label: 'Dashboard',
    icon: LayoutDashboard,
    description: 'Overview & stats',
  },
  {
    href: '/maps',
    label: 'River Maps',
    icon: Map,
    description: 'Interactive navigability map',
  },
  {
    href: '/analytics',
    label: 'Analytics',
    icon: BarChart3,
    description: 'Trends & model metrics',
  },
  {
    href: '/alerts',
    label: 'Risk Alerts',
    icon: Bell,
    description: 'Active warnings',
    badge: true,
  },
  {
    href: '/model',
    label: 'Model Info',
    icon: BrainCircuit,
    description: 'HydroFormer architecture',
  },
] as const;

const WATERWAYS: { id: WaterwayId; name: string; river: string; length: string; color: string }[] = [
  {
    id: 'NW-1',
    name: 'NW-1',
    river: 'Ganga',
    length: '1,390 km',
    color: '#3b82f6',
  },
  {
    id: 'NW-2',
    name: 'NW-2',
    river: 'Brahmaputra',
    length: '891 km',
    color: '#8b5cf6',
  },
];

// ─── Animations ───────────────────────────────────────────────────────────────

const sidebarVariants = {
  expanded: {
    width: 256,
    transition: { duration: 0.3, ease: [0.16, 1, 0.3, 1] },
  },
  collapsed: {
    width: 64,
    transition: { duration: 0.3, ease: [0.16, 1, 0.3, 1] },
  },
};

const labelVariants = {
  visible: {
    opacity: 1,
    x: 0,
    width: 'auto',
    transition: { duration: 0.2, delay: 0.1 },
  },
  hidden: {
    opacity: 0,
    x: -8,
    width: 0,
    transition: { duration: 0.15 },
  },
};

// ─── Sub-Components ───────────────────────────────────────────────────────────

/** Animated wave logo mark */
function LogoMark({ size = 28 }: { size?: number }) {
  return (
    <div
      className="relative flex-shrink-0 flex items-center justify-center rounded-xl"
      style={{
        width: size,
        height: size,
        background: 'linear-gradient(135deg, #0369a1 0%, #3b82f6 100%)',
        boxShadow: '0 0 16px rgba(59,130,246,0.4)',
      }}
    >
      <Waves size={size * 0.55} className="text-slate-900" strokeWidth={2.5} />
    </div>
  );
}

/** Single navigation item */
function NavItem({
  href,
  label,
  icon: Icon,
  description,
  badge,
  isActive,
  isCollapsed,
  alertCount = 0,
}: {
  href: string;
  label: string;
  icon: React.ElementType;
  description: string;
  badge?: boolean;
  isActive: boolean;
  isCollapsed: boolean;
  alertCount?: number;
}) {
  return (
    <Link href={href as any} className="block relative group">
      <motion.div
        className={`
          relative flex items-center gap-3 px-3 py-2.5 rounded-xl cursor-pointer
          transition-colors duration-150
          ${isActive
            ? 'bg-blue-50 text-slate-700'
            : 'text-slate-500 hover:text-slate-900 hover:bg-slate-50'
          }
        `}
        whileHover={{ x: isActive ? 0 : 2 }}
        whileTap={{ scale: 0.97 }}
        transition={{ duration: 0.15 }}
      >
        {/* Active indicator bar */}
        {isActive && (
          <motion.div
            layoutId="nav-indicator"
            className="absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-5 bg-blue-400 rounded-full"
            transition={{ type: 'spring', bounce: 0.2, duration: 0.4 }}
          />
        )}

        {/* Icon */}
        <div className="relative flex-shrink-0 w-5 h-5 flex items-center justify-center">
          <Icon
            size={18}
            strokeWidth={isActive ? 2.5 : 2}
            className={isActive ? 'text-slate-600' : 'text-slate-400 group-hover:text-slate-600'}
          />
          {/* Alert badge on icon (collapsed mode) */}
          {badge && alertCount > 0 && isCollapsed && (
            <span className="absolute -top-1.5 -right-1.5 w-4 h-4 bg-red-500 text-slate-900 text-[9px] font-bold rounded-full flex items-center justify-center leading-none">
              {alertCount > 9 ? '9+' : alertCount}
            </span>
          )}
        </div>

        {/* Label + badge (expanded mode) */}
        <AnimatePresence>
          {!isCollapsed && (
            <motion.div
              variants={labelVariants}
              initial="hidden"
              animate="visible"
              exit="hidden"
              className="flex-1 min-w-0 flex items-center justify-between overflow-hidden"
            >
              <span className="text-sm font-medium whitespace-nowrap">{label}</span>
              {badge && alertCount > 0 && (
                <span className="ml-2 px-1.5 py-0.5 bg-red-500/20 text-slate-400 text-[10px] font-bold rounded-full border border-red-500/30 whitespace-nowrap">
                  {alertCount}
                </span>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>

      {/* Tooltip for collapsed mode */}
      {isCollapsed && (
        <div className="
          absolute left-full ml-3 top-1/2 -translate-y-1/2
          bg-slate-100 border border-slate-300 text-slate-800
          text-xs font-medium px-2.5 py-1.5 rounded-lg whitespace-nowrap
          pointer-events-none opacity-0 group-hover:opacity-100
          transition-opacity duration-150 z-50
          shadow-lg
        ">
          {label}
          <span className="block text-[10px] text-slate-400 font-normal mt-0.5">{description}</span>
        </div>
      )}
    </Link>
  );
}

// ─── Month/Year Picker ────────────────────────────────────────────────────────

function MonthYearPicker({ isCollapsed }: { isCollapsed: boolean }) {
  const selectedMonth = useAppStore((s) => s.selectedMonth);
  const selectedYear  = useAppStore((s) => s.selectedYear);
  const goToPreviousMonth = useAppStore((s) => s.goToPreviousMonth);
  const goToNextMonth     = useAppStore((s) => s.goToNextMonth);
  const setSelectedYear   = useAppStore((s) => s.setSelectedYear);

  const now = new Date();
  const isCurrentOrFuture =
    selectedYear > now.getFullYear() ||
    (selectedYear === now.getFullYear() && selectedMonth >= now.getMonth() + 1);

  if (isCollapsed) {
    return (
      <div className="flex flex-col items-center gap-1 px-2">
        <button
          onClick={goToPreviousMonth}
          className="w-8 h-6 flex items-center justify-center rounded-md text-slate-400 hover:text-slate-800 hover:bg-slate-200/5 transition-colors"
          title="Previous month"
        >
          <ChevronUp size={14} />
        </button>
        <div className="text-center">
          <div className="text-[10px] font-bold text-slate-700 tabular-nums">
            {SHORT_MONTH[selectedMonth - 1]}
          </div>
          <div className="text-[9px] text-slate-500 tabular-nums">{selectedYear}</div>
        </div>
        <button
          onClick={goToNextMonth}
          disabled={isCurrentOrFuture}
          className="w-8 h-6 flex items-center justify-center rounded-md text-slate-400 hover:text-slate-800 hover:bg-slate-200/5 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
          title="Next month"
        >
          <ChevronDown size={14} />
        </button>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ delay: 0.1 }}
      className="px-3"
    >
      {/* Section label */}
      <div className="flex items-center gap-1.5 mb-2">
        <Calendar size={11} className="text-slate-500" />
        <span className="text-[10px] font-semibold tracking-widest text-slate-500 uppercase">
          Analysis Period
        </span>
      </div>

      {/* Month selector */}
      <div className="bg-white p-2 rounded-xl border border-slate-200 shadow-sm">
        <div className="flex items-center justify-between mb-2">
          <button
            onClick={goToPreviousMonth}
            className="w-7 h-7 flex items-center justify-center rounded-lg text-slate-400 hover:text-slate-800 hover:bg-slate-200/8 transition-colors"
            aria-label="Previous month"
          >
            <ChevronLeft size={14} />
          </button>

          <div className="text-center">
            <div className="text-sm font-semibold text-slate-800 tabular-nums">
              {MONTH_NAMES[selectedMonth - 1]}
            </div>
          </div>

          <button
            onClick={goToNextMonth}
            disabled={isCurrentOrFuture}
            className="w-7 h-7 flex items-center justify-center rounded-lg text-slate-400 hover:text-slate-800 hover:bg-slate-200/8 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
            aria-label="Next month"
          >
            <ChevronRight size={14} />
          </button>
        </div>

        {/* Year selector */}
        <div className="flex items-center justify-between">
          <button
            onClick={() => setSelectedYear(selectedYear - 1)}
            disabled={selectedYear <= 2019}
            className="w-6 h-6 flex items-center justify-center rounded-md text-slate-500 hover:text-slate-700 hover:bg-slate-200/5 transition-colors disabled:opacity-30 disabled:cursor-not-allowed text-[10px] font-bold"
            aria-label="Previous year"
          >
            ‹
          </button>
          <span className="text-xs font-bold text-slate-400 tabular-nums">{selectedYear}</span>
          <button
            onClick={() => setSelectedYear(selectedYear + 1)}
            disabled={selectedYear >= now.getFullYear()}
            className="w-6 h-6 flex items-center justify-center rounded-md text-slate-500 hover:text-slate-700 hover:bg-slate-200/5 transition-colors disabled:opacity-30 disabled:cursor-not-allowed text-[10px] font-bold"
            aria-label="Next year"
          >
            ›
          </button>
        </div>
      </div>
    </motion.div>
  );
}

// ─── Waterway Selector ────────────────────────────────────────────────────────

function WaterwaySelector({ isCollapsed }: { isCollapsed: boolean }) {
  const selectedWaterway    = useAppStore((s) => s.selectedWaterway);
  const setSelectedWaterway = useAppStore((s) => s.setSelectedWaterway);

  if (isCollapsed) {
    return (
      <div className="flex flex-col gap-1.5 px-2">
        {WATERWAYS.map((w) => (
          <button
            key={w.id}
            onClick={() => setSelectedWaterway(w.id)}
            title={`${w.name} — ${w.river} (${w.length})`}
            className={`
              relative w-10 h-10 rounded-xl flex items-center justify-center
              text-[10px] font-bold transition-all duration-200
              ${selectedWaterway === w.id
                ? 'text-white shadow-md'
                : 'text-slate-500 hover:text-slate-900 bg-white border border-slate-200 hover:bg-slate-50'
              }
            `}
            style={
              selectedWaterway === w.id
                ? {
                    background: w.color,
                    boxShadow: `0 4px 12px ${w.color}50`,
                  }
                : {}
            }
          >
            {w.id}
          </button>
        ))}
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ delay: 0.05 }}
      className="px-3"
    >
      {/* Section label */}
      <div className="flex items-center gap-1.5 mb-2">
        <Droplets size={11} className="text-slate-500" />
        <span className="text-[10px] font-semibold tracking-widest text-slate-500 uppercase">
          Waterway
        </span>
      </div>

      {/* Waterway cards */}
      <div className="flex flex-col gap-1.5">
        {WATERWAYS.map((w) => {
          const isSelected = selectedWaterway === w.id;
          return (
            <motion.button
              key={w.id}
              onClick={() => setSelectedWaterway(w.id)}
              whileHover={{ scale: 1.01 }}
              whileTap={{ scale: 0.98 }}
              className={`
                relative w-full text-left px-3 py-2.5 rounded-xl
                border transition-all duration-200 overflow-hidden
                ${isSelected
                  ? 'bg-blue-50 border-blue-100'
                  : 'bg-white border-slate-200 text-slate-500 hover:text-slate-900 hover:bg-slate-50 shadow-sm'
                }
              `}
              style={
                isSelected
                  ? { boxShadow: `0 4px 12px ${w.color}15` }
                  : {}
              }
            >
              {/* Active glow strip */}
              {isSelected && (
                <motion.div
                  layoutId="waterway-indicator"
                  className="absolute left-0 inset-y-0 w-0.5 rounded-r-full"
                  style={{ background: w.color }}
                  transition={{ type: 'spring', bounce: 0.2, duration: 0.4 }}
                />
              )}

              <div className="flex items-center justify-between">
                <div>
                  <div
                    className="text-xs font-bold"
                    style={{ color: isSelected ? w.color : '#0f172a' }}
                  >
                    {w.name}
                  </div>
                  <div className={`text-[11px] font-medium mt-0.5 ${isSelected ? 'text-slate-700/70' : 'text-slate-500'}`}>
                    {w.river}
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-[10px] text-slate-500">{w.length}</div>
                  {isSelected && (
                    <div
                      className="text-sm font-bold tracking-tight text-slate-700"
                    >
                      Active
                    </div>
                  )}
                </div>
              </div>
            </motion.button>
          );
        })}
      </div>
    </motion.div>
  );
}

// ─── Main Sidebar ─────────────────────────────────────────────────────────────

export function Sidebar() {
  const sidebarCollapsed = useAppStore((s) => s.sidebarCollapsed);
  const toggleSidebar    = useAppStore((s) => s.toggleSidebar);
  const pathname         = usePathname();

  // Mock alert count — in production this comes from React Query
  const alertCount = 3;

  return (
    <motion.aside
      variants={sidebarVariants}
      animate={sidebarCollapsed ? 'collapsed' : 'expanded'}
      initial={false}
      className="
        relative flex flex-col h-full flex-shrink-0
        bg-white
        border-r border-slate-200
        overflow-hidden z-40
      "
    >
      {/* ── Header / Logo ─────────────────────────────────────── */}
      <div
        className={`
          flex items-center h-16 flex-shrink-0 px-4 border-b border-slate-200
          ${sidebarCollapsed ? 'justify-center' : 'justify-between'}
        `}
      >
        <Link href="/dashboard" className="flex items-center gap-3 group">
          <motion.div whileHover={{ rotate: [0, -5, 5, 0] }} transition={{ duration: 0.4 }}>
            <LogoMark size={32} />
          </motion.div>
          <AnimatePresence>
            {!sidebarCollapsed && (
              <motion.div
                variants={labelVariants}
                initial="hidden"
                animate="visible"
                exit="hidden"
                className="overflow-hidden whitespace-nowrap"
              >
                <div className="text-sm font-bold text-slate-900 tracking-tight leading-tight">
                  AIDSTL
                </div>
                <div className="text-[10px] text-slate-500 font-medium tracking-wide">
                  Waterway Intelligence
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </Link>

        {/* Collapse toggle — shown only in expanded mode */}
        <AnimatePresence>
          {!sidebarCollapsed && (
            <motion.button
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.8 }}
              onClick={toggleSidebar}
              className="
                w-7 h-7 rounded-lg flex items-center justify-center
                text-slate-500 hover:text-slate-700 hover:bg-slate-200/6
                border border-slate-300 transition-colors duration-150
                flex-shrink-0
              "
              aria-label="Collapse sidebar"
            >
              <ChevronLeft size={14} />
            </motion.button>
          )}
        </AnimatePresence>
      </div>

      {/* ── Scrollable Content ────────────────────────────────── */}
      <div className="flex flex-col flex-1 overflow-y-auto overflow-x-hidden thin-scrollbar py-4 gap-6">

        {/* Waterway Selector */}
        <WaterwaySelector isCollapsed={sidebarCollapsed} />

        {/* Divider */}
        <div className="mx-3 h-px bg-slate-100" />

        {/* Navigation */}
        <nav className="px-3 flex flex-col gap-1" aria-label="Main navigation">
          {!sidebarCollapsed && (
            <div className="flex items-center gap-1.5 mb-1 px-1">
              <span className="text-[10px] font-semibold tracking-widest text-slate-500 uppercase">
                Navigation
              </span>
            </div>
          )}
          {NAV_ITEMS.map((item) => (
            <NavItem
              key={item.href}
              href={item.href}
              label={item.label}
              icon={item.icon}
              description={item.description}
              badge={'badge' in item ? (item as any).badge : false}
              isActive={
                item.href === '/dashboard'
                  ? pathname === '/' || pathname === '/dashboard'
                  : pathname.startsWith(item.href)
              }
              isCollapsed={sidebarCollapsed}
              alertCount={'badge' in item && (item as any).badge ? alertCount : 0}
            />
          ))}
        </nav>

        {/* Divider */}
        <div className="mx-3 h-px bg-slate-100" />

        {/* Month/Year Picker */}
        <MonthYearPicker isCollapsed={sidebarCollapsed} />
      </div>

      {/* ── Footer ────────────────────────────────────────────── */}
      <div className="flex-shrink-0 border-t border-slate-200 p-3 bg-slate-50/50">
        {sidebarCollapsed ? (
          /* Expand button in collapsed mode */
          <button
            onClick={toggleSidebar}
            className="
              w-10 h-10 mx-auto flex items-center justify-center rounded-xl
              text-slate-500 hover:text-slate-700 hover:bg-slate-200/6
              border border-slate-300 transition-colors duration-150
            "
            aria-label="Expand sidebar"
          >
            <ChevronRight size={16} />
          </button>
        ) : (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="flex flex-col gap-2"
          >
            {/* Version / Info */}
            <div className="flex items-center justify-between px-1">
              <div>
                <div className="text-[10px] font-semibold text-slate-400">
                  HydroFormer v1.0
                </div>
                <div className="text-[9px] text-slate-400 mt-0.5">
                  Gati Shakti Vishwavidyalaya
                </div>
              </div>
              <a
                href="https://github.com"
                target="_blank"
                rel="noopener noreferrer"
                className="text-slate-400 hover:text-slate-400 transition-colors"
                aria-label="View on GitHub"
              >
                <ExternalLink size={13} />
              </a>
            </div>

            {/* Model status pill */}
            <div className="flex items-center gap-2 px-2.5 py-1.5 rounded-lg bg-green-500/8 border border-green-500/15">
              <span className="relative flex h-2 w-2 flex-shrink-0">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500" />
              </span>
              <span className="text-[10px] font-semibold text-slate-400">Model Online</span>
              <span className="ml-auto text-[9px] text-slate-600">R² 0.918</span>
            </div>
          </motion.div>
        )}
      </div>
    </motion.aside>
  );
}
