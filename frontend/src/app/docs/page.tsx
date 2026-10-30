"use client";

import Link from "next/link";
import Footer from "@/components/landingpage/footer";
import Navbar from "@/components/landingpage/navbar-shrink";
import { cn } from "@/lib/utils";
import { Waves, Terminal, Database, Bell } from "lucide-react";

export default function DocumentationPage() {
  return (
    <div className="bg-black text-white min-h-screen">
      <Navbar />
      <div className="min-h-screen">
        <div className="mx-auto max-w-4xl px-6 pb-20 pt-32">
          <div className="mb-16 flex flex-col items-center text-center">
            <div className="mb-4 flex max-w-fit items-center justify-center gap-2 rounded-full bg-neutral-900 border border-neutral-800 px-3 py-1">
              <div className="relative flex h-1.5 w-1.5 items-center justify-center rounded-full bg-emerald-500/40">
                <div className="flex h-2 w-2 animate-ping items-center justify-center rounded-full bg-emerald-500">
                  <div className="flex h-2 w-2 animate-ping items-center justify-center rounded-full bg-emerald-500"></div>
                </div>
                <div className="absolute left-1/2 top-1/2 flex h-1.5 w-1.5 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full bg-emerald-400"></div>
              </div>
              <span className="bg-clip-text text-xs font-medium text-zinc-300">
                Telemetry Documentation
              </span>
            </div>
            <h1 className="bg-gradient-to-br from-white via-white via-30% to-white/40 bg-clip-text text-4xl font-medium tracking-tight text-transparent md:text-5xl mb-4">
              Waterway Navigability Guide
            </h1>
            <p className="mx-auto max-w-2xl text-balance text-sm md:text-base leading-relaxed text-neutral-400">
              Technical documentation for interpreting fairway depth forecasts, accessing geospatial telemetry layers, and configuring safe navigation thresholds.
            </p>
          </div>

          <div className="space-y-10">
            {/* Section 1 */}
            <div className="space-y-4 rounded-xl border border-neutral-800/80 bg-neutral-950 p-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2.5">
                  <Waves className="h-5 w-5 text-cyan-400" />
                  <h3 className="text-lg font-semibold text-white">
                    1. Waterway Corridor Selection
                  </h3>
                </div>
                <span className="text-xs text-neutral-500 uppercase tracking-wider font-mono">
                  NW-1 / NW-2
                </span>
              </div>
              <p className="text-sm leading-relaxed text-neutral-400">
                The platform monitors designated stretches of India&apos;s National Waterways. Choose between NW-1 Ganga (1,620 km corridor spanning Prayagraj to Haldia) and NW-2 Brahmaputra (891 km corridor from Dhubri to Sadiya). Each corridor is segmented into 10 km operational reaches with localized LAD tracking.
              </p>
              <div className="pt-2">
                <Link href="/maps">
                  <button
                    className={cn(
                      "flex items-center justify-center gap-2 rounded-md bg-neutral-900 border border-neutral-700 px-4 py-2 text-xs font-semibold text-white transition-all duration-200 hover:bg-neutral-800",
                    )}
                  >
                    Open River Map
                  </button>
                </Link>
              </div>
            </div>

            {/* Section 2 */}
            <div className="space-y-4 rounded-xl border border-neutral-800/80 bg-neutral-950 p-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2.5">
                  <Terminal className="h-5 w-5 text-blue-400" />
                  <h3 className="text-lg font-semibold text-white">
                    2. Least Available Depth (LAD) and Draft Rules
                  </h3>
                </div>
                <span className="text-xs text-neutral-500 uppercase tracking-wider font-mono">
                  Standards
                </span>
              </div>
              <p className="text-sm leading-relaxed text-neutral-400">
                LAD values are reported in meters relative to the low water datum (LWD). Vessels operating under IWAI Class III specifications require a minimum LAD of 2.50 meters to maintain a safe under-keel clearance (UKC) of 0.30 meters. Reaches falling below 2.20 meters trigger automatic cautionary notices.
              </p>
              <div className="overflow-x-auto rounded-lg border border-neutral-800 bg-black p-4 font-mono text-xs text-neutral-300">
                <p className="text-neutral-500">// Classification Thresholds</p>
                <p><span className="text-emerald-400">Navigable</span>: LAD &gt;= 2.50 m (Full draft transit permitted)</p>
                <p><span className="text-amber-400">Conditional</span>: 2.00 m &lt;= LAD &lt; 2.50 m (Daylight navigation only, reduced cargo)</p>
                <p><span className="text-rose-400">Non-Navigable</span>: LAD &lt; 2.00 m (Shoal bottleneck, dredging required)</p>
              </div>
            </div>

            {/* Section 3 */}
            <div className="space-y-4 rounded-xl border border-neutral-800/80 bg-neutral-950 p-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2.5">
                  <Database className="h-5 w-5 text-emerald-400" />
                  <h3 className="text-lg font-semibold text-white">
                    3. Telemetry Ingestion and API Integration
                  </h3>
                </div>
                <span className="text-xs text-neutral-500 uppercase tracking-wider font-mono">
                  FastAPI REST
                </span>
              </div>
              <p className="text-sm leading-relaxed text-neutral-400">
                Logistics software and vessel management systems can query real-time fairway forecasts through the backend API. Pass the waterway code and station ID to retrieve latest sounding bathymetry and 12-month depth calendar forecasts.
              </p>
              <div className="rounded-lg border border-neutral-800 bg-black p-4 font-mono text-xs text-neutral-300 overflow-x-auto">
                <p className="text-neutral-500"># Query fairway depth by river reach</p>
                <p className="text-cyan-400">curl -X GET &quot;http://localhost:8000/api/waterways/NW1/depth?station=patna&quot; \</p>
                <p className="text-cyan-400 pl-4">-H &quot;Accept: application/json&quot;</p>
              </div>
            </div>

            {/* Section 4 */}
            <div className="space-y-4 rounded-xl border border-neutral-800/80 bg-neutral-950 p-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2.5">
                  <Bell className="h-5 w-5 text-amber-400" />
                  <h3 className="text-lg font-semibold text-white">
                    4. Automated Risk Early Warning
                  </h3>
                </div>
                <span className="text-xs text-neutral-500 uppercase tracking-wider font-mono">
                  Alerts
                </span>
              </div>
              <p className="text-sm leading-relaxed text-neutral-400">
                Subscribe to automated shoal notifications and seasonal monsoon flood advisories. When hydrological gauges report abnormal discharge surges or sediment deposition along braided river bends, warning bulletins are dispatched to registered port controllers.
              </p>
              <div className="pt-2">
                <Link href="/alerts">
                  <button
                    className={cn(
                      "flex items-center justify-center gap-2 rounded-md bg-neutral-900 border border-neutral-700 px-4 py-2 text-xs font-semibold text-white transition-all duration-200 hover:bg-neutral-800",
                    )}
                  >
                    View Active Advisories
                  </button>
                </Link>
              </div>
            </div>
          </div>
        </div>
      </div>
      <Footer />
    </div>
  );
}
