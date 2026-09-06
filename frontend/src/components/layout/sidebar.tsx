// ============================================================
// InlandRoute — Minimalist Pure Dark SaaS Sidebar
// ============================================================

'use client';

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  LayoutDashboard,
  Map,
  BarChart3,
  Bell,
  Navigation,
  BrainCircuit,
  ChevronLeft,
  ChevronRight,
  Waves,
} from 'lucide-react';
import { useAppStore } from '@/store/app-store';

interface NavItem {
  href: string;
  label: string;
  icon: any;
  badge?: boolean;
}

interface NavGroup {
  title: string;
  items: NavItem[];
}

const NAV_GROUPS: NavGroup[] = [
  {
    title: 'Overview',
    items: [
      { href: '/dashboard', label: 'Executive Dashboard', icon: LayoutDashboard },
    ],
  },
  {
    title: 'Geospatial & Telemetry',
    items: [
      { href: '/maps', label: 'Interactive River Map', icon: Map },
      { href: '/analytics', label: 'Hydrological Trends', icon: BarChart3 },
    ],
  },
  {
    title: 'Operations',
    items: [
      { href: '/navigation', label: 'Vessel Route Pathfinder', icon: Navigation },
      { href: '/alerts', label: 'Risk & Early Warning', icon: Bell, badge: true },
    ],
  },
];

export function Sidebar() {
  const sidebarCollapsed = useAppStore((s) => s.sidebarCollapsed);
  const toggleSidebar = useAppStore((s) => s.toggleSidebar);
  const pathname = usePathname();

  return (
    <aside
      className={`relative flex flex-col h-full bg-zinc-900 border-r border-zinc-800 transition-all duration-300 z-30 ${
        sidebarCollapsed ? 'w-16' : 'w-64'
      }`}
    >
      {/* Brand Header */}
      <div className="flex items-center justify-between h-16 px-4 border-b border-zinc-800">
        <Link href="/dashboard" className="flex items-center gap-3 min-w-0">
          <div className="w-8 h-8 rounded-lg bg-white flex items-center justify-center text-zinc-900 shadow-sm flex-shrink-0">
            <Waves size={18} />
          </div>
          {!sidebarCollapsed && (
            <div className="flex flex-col min-w-0">
              <span className="font-bold text-sm text-white tracking-tight leading-none truncate">
                InlandRoute
              </span>
              <span className="text-[10px] text-zinc-400 font-medium tracking-wide mt-0.5 truncate">
                IWAI Intelligence
              </span>
            </div>
          )}
        </Link>
        {!sidebarCollapsed && (
          <button
            onClick={toggleSidebar}
            className="p-1.5 rounded-lg border border-zinc-700 text-zinc-400 hover:bg-zinc-800 hover:text-white transition-all flex-shrink-0 ml-2"
            aria-label="Collapse Sidebar"
          >
            <ChevronLeft size={14} />
          </button>
        )}
      </div>

      {/* Navigation Sections */}
      <div className="flex-1 overflow-y-auto thin-scrollbar p-3 space-y-6">
        {NAV_GROUPS.map((group, idx) => (
          <div key={idx} className="space-y-1">
            {!sidebarCollapsed && (
              <h2 className="px-2 text-[10px] font-semibold uppercase tracking-wider text-zinc-400">
                {group.title}
              </h2>
            )}
            {group.items.map((item) => {
              const Icon = item.icon;
              const isActive = pathname === item.href;

              return (
                <Link
                  key={item.href}
                  href={item.href as any}
                  className={`flex items-center gap-3 py-2 rounded-xl text-xs font-semibold transition-all ${
                    sidebarCollapsed ? 'justify-center px-0' : 'px-3'
                  } ${
                    isActive
                      ? 'bg-white text-zinc-900 shadow-sm'
                      : 'text-zinc-400 hover:bg-zinc-800 hover:text-white'
                  }`}
                  title={sidebarCollapsed ? item.label : undefined}
                >
                  <Icon size={16} className={isActive ? 'text-zinc-900' : 'text-zinc-400'} />
                  {!sidebarCollapsed && (
                    <span className="flex-1 truncate">{item.label}</span>
                  )}
                  {!sidebarCollapsed && item.badge && (
                    <span className="px-1.5 py-0.5 text-[9px] font-bold rounded-full bg-rose-500 text-white flex-shrink-0">
                      Alerts
                    </span>
                  )}
                </Link>
              );
            })}
          </div>
        ))}
      </div>

      {/* Footer / Expand Button */}
      <div className="p-3 border-t border-zinc-800 bg-zinc-900/50">
        <button
          onClick={toggleSidebar}
          className="w-full flex items-center justify-center p-2 rounded-lg border border-zinc-700 text-zinc-400 hover:bg-zinc-800 transition-all"
          aria-label={sidebarCollapsed ? 'Expand Sidebar' : 'Collapse Sidebar'}
        >
          {sidebarCollapsed ? <ChevronRight size={14} /> : <ChevronLeft size={14} />}
        </button>
      </div>
    </aside>
  );
}
