"use client";

import { cn } from "@/lib/utils";
import FrameworkCard from "./frameworkcard";

const FeatureFramework = () => {
  return (
    <div
      className={cn(
        "relative",
        "flex flex-col justify-between",
        "h-[20rem] space-y-4",
        "rounded-xl border border-neutral-800/80 bg-neutral-950",
      )}
    >
      <FrameworkCard />
      <div className="px-5 pb-5">
        <div className="text-sm font-semibold text-white">
          Comprehensive River Basin Coverage
        </div>
        <div className="mt-1.5 text-xs text-neutral-400 leading-relaxed">
          Unified navigability telemetry spanning NW-1 Ganga, NW-2 Brahmaputra, and critical deltaic tributary networks across northern and eastern India.
        </div>
      </div>
    </div>
  );
};

export default FeatureFramework;
