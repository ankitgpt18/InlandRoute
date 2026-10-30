import { cn } from "@/lib/utils";
import { IoIosTrendingUp } from "react-icons/io";
import { Waves, Navigation, ShieldCheck, Compass } from "lucide-react";

const FeatureInsights = () => {
  return (
    <div
      className={cn(
        "group relative",
        "flex flex-col justify-between",
        "h-[20rem] space-y-4",
        "rounded-xl border border-neutral-800/80 bg-neutral-950",
      )}
    >
      {/* Insight preview cards */}
      <div
        className={cn(
          "relative overflow-hidden",
          "flex items-center justify-center",
          "h-[14.7rem] w-full",
        )}
      >
        {/* Left floating card */}
        <div className="absolute bottom-0 left-4 h-[8.5rem] w-[13.5rem] rotate-[-2deg] rounded-t-lg bg-gradient-to-br from-neutral-800/50 to-neutral-850 px-4 py-3 transition-all duration-300 group-hover:bottom-2 border border-neutral-800">
          <div className="flex h-full w-full flex-col justify-between">
            <div className="flex flex-col items-start gap-1">
              <div className="flex items-center gap-1.5 text-xs text-neutral-400">
                <Waves className="h-3 w-3 text-cyan-400" />
                Fairway LAD (Oct)
              </div>
              <div className="flex items-end gap-2">
                <div className="flex items-center gap-1 text-[0.95rem] font-bold text-white">
                  <p>2.85 m</p>
                  <IoIosTrendingUp className="text-sm text-emerald-400" />
                </div>
              </div>
            </div>
            <div className="flex flex-col items-start gap-1">
              <div className="flex items-center gap-1.5 text-xs text-neutral-400">
                <Compass className="h-3 w-3 text-blue-400" />
                Surface Velocity
              </div>
              <div className="flex items-end gap-2">
                <div className="flex items-center gap-1 text-[0.95rem] font-bold text-white">
                  <p>1.35 m/s</p>
                  <span className="text-[10px] text-neutral-400">Steady</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right floating card */}
        <div className="absolute left-[30.5rem] top-[4rem] h-[12.5rem] w-[20rem] rotate-[2deg] rounded-t-lg bg-gradient-to-br from-neutral-800/50 to-neutral-900 px-4 py-3 transition-all duration-300 group-hover:top-[3rem] border border-neutral-800">
          <div className="flex h-full w-full flex-col gap-2">
            <div className="flex w-full items-center text-xs font-semibold uppercase tracking-wider text-neutral-400">
              Sector Navigability
            </div>
            <div className="flex w-full flex-1 flex-col gap-1.5">
              <div className="relative h-[32px] w-full rounded-md px-2 py-1.5">
                <div className="absolute inset-x-0 top-0 h-full w-[96%] rounded-md bg-neutral-800" />
                <div className="absolute left-2 flex items-center gap-1.5 text-xs text-neutral-100">
                  <ShieldCheck className="h-3 w-3 text-emerald-400" />
                  <p>Farakka to Haldia</p>
                </div>
                <p className="absolute right-2 text-xs font-mono text-neutral-300">98%</p>
              </div>
              <div className="relative h-[32px] w-full rounded-md px-2 py-1.5">
                <div className="absolute inset-x-0 top-0 h-full w-[91%] rounded-md bg-neutral-800" />
                <div className="absolute left-2 flex items-center gap-1.5 text-xs text-neutral-100">
                  <ShieldCheck className="h-3 w-3 text-emerald-400" />
                  <p>Patna to Barh</p>
                </div>
                <p className="absolute right-2 text-xs font-mono text-neutral-300">92%</p>
              </div>
              <div className="relative h-[32px] w-full rounded-md px-2 py-1.5">
                <div className="absolute inset-x-0 top-0 h-full w-[84%] rounded-md bg-neutral-800" />
                <div className="absolute left-2 flex items-center gap-1.5 text-xs text-neutral-100">
                  <ShieldCheck className="h-3 w-3 text-emerald-400" />
                  <p>Varanasi Reach</p>
                </div>
                <p className="absolute right-2 text-xs font-mono text-neutral-300">88%</p>
              </div>
            </div>
          </div>
        </div>

        {/* Center main card */}
        <div className="absolute bottom-0 left-[10rem] h-[12.5rem] w-[21rem] rounded-t-lg bg-neutral-900 p-4 shadow-2xl shadow-black border-t border-x border-neutral-800">
          <div className="flex h-full w-full flex-col gap-2">
            <div className="flex w-full items-center text-xs font-semibold uppercase tracking-wider text-neutral-400">
              Active Gauge Stations
            </div>
            <div className="flex w-full flex-1 flex-col gap-1.5">
              <div className="relative h-[34px] w-full rounded-md px-2 py-2">
                <div className="absolute inset-x-0 top-0 h-full w-[100%] rounded-md bg-neutral-800" />
                <div className="absolute left-2 flex items-center gap-1.5 text-xs text-neutral-100">
                  <Navigation className="h-3 w-3 text-cyan-400" />
                  <p>CWC Patna Gandhighat</p>
                </div>
                <p className="absolute right-2 text-xs font-mono text-neutral-300">+46.2m</p>
              </div>
              <div className="relative h-[34px] w-full rounded-md px-2 py-2">
                <div className="absolute inset-x-0 top-0 h-full w-[88%] rounded-md bg-neutral-800" />
                <div className="absolute left-2 flex items-center gap-1.5 text-xs text-neutral-100">
                  <Navigation className="h-3 w-3 text-cyan-400" />
                  <p>CWC Varanasi Rajghat</p>
                </div>
                <p className="absolute right-2 text-xs font-mono text-neutral-300">+68.4m</p>
              </div>
              <div className="relative h-[34px] w-full rounded-md px-2 py-2">
                <div className="absolute inset-x-0 top-0 h-full w-[72%] rounded-md bg-neutral-800" />
                <div className="absolute left-2 flex items-center gap-1.5 text-xs text-neutral-100">
                  <Navigation className="h-3 w-3 text-cyan-400" />
                  <p>CWC Farakka Barrage</p>
                </div>
                <p className="absolute right-2 text-xs font-mono text-neutral-300">+19.1m</p>
              </div>
            </div>
          </div>
        </div>

        <div className="absolute bottom-0 left-0 h-5 w-full bg-gradient-to-t from-neutral-950 to-transparent" />
      </div>

      <div className="px-5 pb-5">
        <div className="text-sm font-semibold text-white">Fairway Navigability Insights</div>
        <div className="mt-1.5 text-xs text-neutral-400 leading-relaxed">
          Real-time telemetry on riverbed shoal formations, gauge discharge levels, and seasonal navigable draft calendars for safe vessel logistics.
        </div>
      </div>
    </div>
  );
};

export default FeatureInsights;
