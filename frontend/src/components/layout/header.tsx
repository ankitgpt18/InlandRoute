// ============================================================
// InlandRoute — Minimalist Pure Dark SaaS Header
// ============================================================

'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Bell,
  Download,
  RefreshCw,
  Satellite,
  ChevronDown,
  FileText,
  Share2,
  Sun,
  Moon,
} from 'lucide-react';
import { useAppStore } from '@/store/app-store';
import ApiService from '@/lib/api';

const MONTH_NAMES = [
  'January', 'February', 'March', 'April',
  'May', 'June', 'July', 'August',
  'September', 'October', 'November', 'December',
] as const;

export function Header() {
  const selectedWaterway = useAppStore((s) => s.selectedWaterway);
  const setSelectedWaterway = useAppStore((s) => s.setSelectedWaterway);
  const selectedMonth = useAppStore((s) => s.selectedMonth);
  const selectedYear = useAppStore((s) => s.selectedYear);
  const setSelectedMonth = useAppStore((s) => s.setSelectedMonth);
  const setSelectedYear = useAppStore((s) => s.setSelectedYear);
  const setAlertsPanelOpen = useAppStore((s) => s.setAlertsPanelOpen);
  const themeMode = useAppStore((s) => s.themeMode);
  const toggleThemeMode = useAppStore((s) => s.toggleThemeMode);

  const [apiOnline, setApiOnline] = useState<boolean>(true);
  const [exportOpen, setExportOpen] = useState<boolean>(false);
  const [refreshing, setRefreshing] = useState<boolean>(false);

  useEffect(() => {
    if (typeof document !== 'undefined') {
      if (themeMode === 'dark') {
        document.documentElement.classList.add('dark');
      } else {
        document.documentElement.classList.remove('dark');
      }
    }
  }, [themeMode]);

  useEffect(() => {
    let mounted = true;
    ApiService.checkHealth()
      .then(() => { if (mounted) setApiOnline(true); })
      .catch(() => { if (mounted) setApiOnline(false); });
    return () => { mounted = false; };
  }, []);

  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      await ApiService.checkHealth();
      setApiOnline(true);
    } catch {
      setApiOnline(false);
    } finally {
      setTimeout(() => setRefreshing(false), 600);
    }
  };

  return (
    <header className="relative flex items-center justify-between h-16 px-6 bg-zinc-900 border-b border-zinc-800 z-40">
      {/* Left Section: Waterway Selector */}
      <div className="flex items-center gap-4">
        {/* Waterway Pill Selector */}
        <div className="flex items-center bg-zinc-800 p-1 rounded-xl border border-zinc-700">
          <button
            onClick={() => setSelectedWaterway('NW-1')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              selectedWaterway === 'NW-1'
                ? 'bg-white text-zinc-900 shadow-sm'
                : 'text-zinc-400 hover:text-white'
            }`}
          >
            NW-1 · Ganga
          </button>
          <button
            onClick={() => setSelectedWaterway('NW-2')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              selectedWaterway === 'NW-2'
                ? 'bg-white text-zinc-900 shadow-sm'
                : 'text-zinc-400 hover:text-white'
            }`}
          >
            NW-2 · Brahmaputra
          </button>
        </div>
      </div>

      {/* Center Section: Date & Satellite Context */}
      <div className="hidden md:flex items-center gap-3">
        <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-zinc-800 border border-zinc-700 text-xs font-medium text-zinc-300">
          <Satellite size={13} className="text-zinc-400" />
          <span>Sentinel-2:</span>
          <select
            value={selectedMonth}
            onChange={(e) => setSelectedMonth(Number(e.target.value) as any)}
            className="bg-transparent font-semibold text-white focus:outline-none cursor-pointer"
          >
            {MONTH_NAMES.map((name, idx) => (
              <option key={idx + 1} value={idx + 1} className="bg-zinc-800 text-white">
                {name}
              </option>
            ))}
          </select>
          <select
            value={selectedYear}
            onChange={(e) => setSelectedYear(Number(e.target.value))}
            className="bg-transparent font-semibold text-white focus:outline-none cursor-pointer ml-1"
          >
            {[2026, 2025, 2024, 2023, 2022, 2021].map((yr) => (
              <option key={yr} value={yr} className="bg-zinc-800 text-white">
                {yr}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Right Section: Actions & Controls */}
      <div className="flex items-center gap-2">
        {/* Light / Dark Mode Toggle */}
        <button
          onClick={toggleThemeMode}
          className="p-2 rounded-lg border border-zinc-700 text-zinc-300 hover:text-white hover:bg-zinc-800 transition-all"
          title={`Switch to ${themeMode === 'dark' ? 'Light' : 'Dark'} Mode`}
        >
          {themeMode === 'dark' ? (
            <Sun size={15} className="text-amber-400" />
          ) : (
            <Moon size={15} className="text-zinc-300" />
          )}
        </button>

        {/* Refresh */}
        <button
          onClick={handleRefresh}
          disabled={refreshing}
          className="p-2 rounded-lg border border-zinc-700 text-zinc-300 hover:text-white hover:bg-zinc-800 transition-all"
          title="Refresh Telemetry"
        >
          <RefreshCw size={15} className={refreshing ? 'animate-spin' : ''} />
        </button>

        {/* Alerts Toggle */}
        <button
          onClick={() => setAlertsPanelOpen(true)}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-zinc-700 bg-zinc-800 text-xs font-semibold text-zinc-300 hover:bg-zinc-700 transition-all shadow-sm"
        >
          <Bell size={14} className="text-zinc-400" />
          <span>Alerts</span>
        </button>

        {/* Export Button */}
        <div className="relative">
          <button
            onClick={() => setExportOpen((v) => !v)}
            className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg bg-white text-zinc-900 text-xs font-semibold hover:bg-zinc-200 transition-all shadow-sm"
          >
            <Download size={14} />
            <span>Export</span>
            <ChevronDown size={12} />
          </button>

          <AnimatePresence>
            {exportOpen && (
              <motion.div
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 6 }}
                className="absolute right-0 top-full mt-2 w-48 bg-zinc-800 border border-zinc-700 rounded-xl shadow-lg z-50 p-1 text-xs"
              >
                <button
                  onClick={() => { setExportOpen(false); window.print(); }}
                  className="w-full text-left px-3 py-2 rounded-lg text-zinc-200 hover:bg-zinc-700 transition-colors flex items-center gap-2"
                >
                  <FileText size={14} /> Print Executive Summary
                </button>
                <button
                  onClick={() => { setExportOpen(false); }}
                  className="w-full text-left px-3 py-2 rounded-lg text-zinc-200 hover:bg-zinc-700 transition-colors flex items-center gap-2"
                >
                  <Share2 size={14} /> Export GeoJSON Data
                </button>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </header>
  );
}
