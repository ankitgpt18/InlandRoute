"use client";

import React from "react";
import Image from "next/image";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { CiGlobe } from "react-icons/ci";
import { BiLinkExternal } from "react-icons/bi";
import {
  LayoutDashboard,
  Map,
  BarChart3,
  Navigation,
  AlertTriangle,
  Cpu,
  ChevronRight,
} from "lucide-react";
import { useAppStore } from "@/store/app-store";
import type { WaterwayId, Month } from "@/types";

export const WATERWAYS: {
  code: WaterwayId;
  name: string;
  shortName: string;
  corridor: string;
  length: string;
}[] = [
  {
    code: "NW-1",
    name: "National Waterway 1 (NW-1 Ganga)",
    shortName: "NW-1 · Ganga",
    corridor: "Prayagraj to Haldia",
    length: "1,620 km",
  },
  {
    code: "NW-2",
    name: "National Waterway 2 (NW-2 Brahmaputra)",
    shortName: "NW-2 · Brahmaputra",
    corridor: "Dhubri to Sadiya",
    length: "891 km",
  },
  {
    code: "NW-3",
    name: "National Waterway 3 (NW-3 West Coast Canal)",
    shortName: "NW-3 · West Coast",
    corridor: "Kottapuram to Kollam",
    length: "205 km",
  },
  {
    code: "NW-4",
    name: "National Waterway 4 (NW-4 Godavari & Krishna)",
    shortName: "NW-4 · Godavari / Krishna",
    corridor: "Kakinada to Puducherry",
    length: "1,078 km",
  },
  {
    code: "NW-5",
    name: "National Waterway 5 (NW-5 Brahmani & Mahanadi)",
    shortName: "NW-5 · Brahmani Delta",
    corridor: "Talcher to Dhamra",
    length: "588 km",
  },
];

const MONTHS: { num: Month; label: string }[] = [
  { num: 1, label: "January (Lean)" },
  { num: 2, label: "February (Lean)" },
  { num: 3, label: "March (Pre-Monsoon)" },
  { num: 4, label: "April (Pre-Monsoon)" },
  { num: 5, label: "May (Pre-Monsoon)" },
  { num: 6, label: "June (Monsoon Onset)" },
  { num: 7, label: "July (Monsoon Peak)" },
  { num: 8, label: "August (Monsoon Peak)" },
  { num: 9, label: "September (Receding)" },
  { num: 10, label: "October (Post-Monsoon)" },
  { num: 11, label: "November (Post-Monsoon)" },
  { num: 12, label: "December (Winter Lean)" },
];

const MODULE_TABS = [
  { href: "/dashboard", label: "Executive Dashboard", icon: LayoutDashboard },
  { href: "/maps", label: "River Map", icon: Map },
  { href: "/analytics", label: "Hydrology Trends", icon: BarChart3 },
  { href: "/navigation", label: "Route Pathfinder", icon: Navigation },
  { href: "/alerts", label: "Risk Alerts", icon: AlertTriangle },
  { href: "/model", label: "AI DL Pipeline", icon: Cpu },
];

export default function SiteHeader() {
  const pathname = usePathname();
  const selectedWaterway = useAppStore((s) => s.selectedWaterway);
  const setSelectedWaterway = useAppStore((s) => s.setSelectedWaterway);
  const selectedMonth = useAppStore((s) => s.selectedMonth);
  const setSelectedMonth = useAppStore((s) => s.setSelectedMonth);
  const selectedYear = useAppStore((s) => s.selectedYear);
  const setSelectedYear = useAppStore((s) => s.setSelectedYear);

  const currentWw = WATERWAYS.find((w) => w.code === selectedWaterway) ?? WATERWAYS[0];
  const currentTab = MODULE_TABS.find((t) => t.href === pathname) ?? MODULE_TABS[0];

  return (
    <div className="flex flex-col gap-4">
      {/* Statsio-style Breadcrumb */}
      <div className="flex items-center space-x-1.5 text-xs text-neutral-400">
        <Link href="/" className="hover:text-white transition-colors">
          Home
        </Link>
        <ChevronRight className="h-3.5 w-3.5 text-neutral-600" />
        <Link href="/dashboard" className="hover:text-white transition-colors">
          Dashboard
        </Link>
        <ChevronRight className="h-3.5 w-3.5 text-neutral-600" />
        <span className="text-neutral-200 font-medium">{currentTab.label}</span>
      </div>

      {/* Main Title & Controls Bar */}
      <div className="flex flex-col gap-4 border-b border-neutral-800/80 pb-5 pt-1 lg:flex-row lg:items-center lg:justify-between">
        <div className="flex flex-col space-y-2">
          <div className="flex flex-row items-center gap-3">
            <Image
              src="/logo.svg"
              width={32}
              height={32}
              alt="InlandRoute"
              className="h-8 w-8 rounded-lg object-contain"
            />
            <h1 className="text-2xl font-bold tracking-tight text-white sm:text-3xl">
              {currentWw.name}
            </h1>
          </div>
          <div className="flex flex-wrap items-center gap-2 text-xs text-neutral-400 sm:text-sm">
            <span className="text-neutral-300 font-medium">
              {currentWw.corridor} ({currentWw.length})
            </span>
            <span className="text-neutral-600">|</span>
            <CiGlobe className="h-4 w-4 text-cyan-400" />
            <Link
              href="https://iwai.nic.in"
              target="_blank"
              rel="noopener noreferrer"
              className="flex flex-row items-center gap-1 font-semibold text-white hover:underline"
            >
              iwai.nic.in
              <BiLinkExternal className="h-3 w-3 text-neutral-400" />
            </Link>
            <span className="text-neutral-600">|</span>
            <span className="inline-flex items-center gap-1.5 text-xs text-neutral-400">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-500"></span>
              Sentinel-2 MSI Orbit Synced
            </span>
          </div>
        </div>

        {/* Filters & Selectors */}
        <div className="flex flex-wrap items-center gap-2.5">
          {/* Waterway Pill Switcher */}
          <div className="flex items-center rounded-lg border border-neutral-800 bg-neutral-950 p-1">
            {WATERWAYS.map((w) => (
              <button
                key={w.code}
                type="button"
                onClick={() => setSelectedWaterway(w.code)}
                className={`rounded px-2.5 py-1 text-xs font-semibold transition-all ${
                  selectedWaterway === w.code
                    ? "bg-neutral-800 text-white shadow-sm"
                    : "text-neutral-400 hover:text-white"
                }`}
                title={w.name}
              >
                {w.code}
              </button>
            ))}
          </div>

          {/* Month selector */}
          <select
            value={selectedMonth}
            onChange={(e) => setSelectedMonth(Number(e.target.value) as Month)}
            aria-label="Select Forecast Month"
            className="rounded-lg border border-neutral-800 bg-neutral-900/90 px-3 py-1.5 text-xs font-semibold text-white transition-colors hover:border-neutral-700 focus:outline-none focus:ring-1 focus:ring-cyan-500"
          >
            {MONTHS.map((m) => (
              <option key={m.num} value={m.num} className="bg-neutral-950 text-white">
                {m.label}
              </option>
            ))}
          </select>

          {/* Year selector */}
          <select
            value={selectedYear}
            onChange={(e) => setSelectedYear(Number(e.target.value))}
            aria-label="Select Year"
            className="rounded-lg border border-neutral-800 bg-neutral-900/90 px-3 py-1.5 text-xs font-semibold text-white transition-colors hover:border-neutral-700 focus:outline-none focus:ring-1 focus:ring-cyan-500"
          >
            {[2026, 2025, 2024, 2023].map((yr) => (
              <option key={yr} value={yr} className="bg-neutral-950 text-white">
                {yr}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Statsio-Style Horizontal Module Navigation Tabs */}
      <div className="flex items-center overflow-x-auto thin-scrollbar border-b border-neutral-800">
        <div className="flex items-center gap-1">
          {MODULE_TABS.map((tab) => {
            const Icon = tab.icon;
            const isActive = pathname === tab.href;
            return (
              <Link
                key={tab.href}
                href={tab.href as any}
                className={`relative inline-flex h-9 items-center justify-center gap-2 whitespace-nowrap rounded-none border-b-2 px-3.5 pb-2 pt-1 text-xs font-medium transition-colors ${
                  isActive
                    ? "border-b-white text-white"
                    : "border-b-transparent text-neutral-400 hover:text-white"
                }`}
              >
                <Icon className="h-3.5 w-3.5" />
                <span>{tab.label}</span>
              </Link>
            );
          })}
        </div>
      </div>
    </div>
  );
}
