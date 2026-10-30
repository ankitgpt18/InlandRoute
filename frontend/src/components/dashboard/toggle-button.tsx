"use client";

import React from "react";
import { cn } from "@/lib/utils";

interface ToggleButtonProps {
  activeTab: "overview" | "forecast";
  onTabChange: (tab: "overview" | "forecast") => void;
}

export default function ToggleButton({ activeTab, onTabChange }: ToggleButtonProps) {
  return (
    <div className="flex w-fit items-center gap-1 rounded-lg border border-neutral-800 bg-neutral-950 p-1 text-xs">
      <button
        onClick={() => onTabChange("overview")}
        className={cn(
          "rounded-md px-4 py-2 font-semibold transition-all duration-200",
          activeTab === "overview"
            ? "bg-neutral-800 text-white shadow-sm"
            : "text-neutral-400 hover:text-neutral-200",
        )}
      >
        Fairway Navigability Overview
      </button>
      <button
        onClick={() => onTabChange("forecast")}
        className={cn(
          "rounded-md px-4 py-2 font-semibold transition-all duration-200",
          activeTab === "forecast"
            ? "bg-neutral-800 text-white shadow-sm"
            : "text-neutral-400 hover:text-neutral-200",
        )}
      >
        Seasonal Depth Calendar & Profile
      </button>
    </div>
  );
}
