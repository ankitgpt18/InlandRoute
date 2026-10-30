"use client";

import { cn } from "@/lib/utils";
import { motion } from "framer-motion";
import { useEffect, useState } from "react";
import { IoMdCheckmark } from "react-icons/io";
import { LuLoader } from "react-icons/lu";

interface OnboardCardProps {
  duration?: number;
  step1?: string;
  step2?: string;
  step3?: string;
}

const OnboardCard = ({
  duration = 3000,
  step1 = "Sentinel-2 Imagery Ingested",
  step2 = "Hydrodynamic Bathymetry Inferred",
  step3 = "Fairway LAD Advisory Published",
}: OnboardCardProps) => {
  const [progress, setProgress] = useState(0);
  const [animateKey, setAnimateKey] = useState(0);

  useEffect(() => {
    const forward = setTimeout(() => setProgress(100), 100);
    const reset = setTimeout(() => {
      setAnimateKey((k) => k + 1);
    }, duration + 2000);

    return () => {
      clearTimeout(forward);
      clearTimeout(reset);
    };
  }, [animateKey, duration]);

  return (
    <div
      className={cn(
        "relative",
        "mt-4 flex flex-col items-center justify-center gap-1.5 p-1",
      )}
    >
      <div className="flex w-full max-w-[260px] scale-[0.9] flex-col justify-center gap-2 rounded-lg border border-neutral-800 bg-gradient-to-br from-neutral-800 to-neutral-950 py-2.5 pl-3 pr-10 opacity-75">
        <div className="flex items-center justify-start gap-2 text-xs text-white">
          <LuLoader className="animate-spin text-neutral-400" />
          <span className="truncate">{step3}</span>
        </div>
        <div className="ml-5 h-1.5 w-[100%] overflow-hidden rounded-full bg-neutral-700"></div>
      </div>

      <div className="flex w-full max-w-[260px] flex-col justify-center gap-2 rounded-lg border border-neutral-700 bg-gradient-to-br from-neutral-800 to-neutral-950 py-2.5 pl-3 pr-10 shadow-md">
        <div className="flex items-center justify-start gap-2 text-xs text-white">
          <LuLoader className="animate-spin text-cyan-400" />
          <span className="truncate font-medium">{step2}</span>
        </div>
        <div className="ml-5 h-1.5 w-[100%] overflow-hidden rounded-full bg-neutral-700">
          <motion.div
            key={animateKey}
            className="h-full bg-cyan-400"
            initial={{ width: 0 }}
            animate={{ width: `${progress}%` }}
            transition={{ duration: duration / 1000, ease: "easeInOut" }}
          />
        </div>
      </div>

      <div className="flex w-full max-w-[260px] scale-[0.9] flex-col justify-center gap-2 rounded-lg border border-neutral-800 bg-gradient-to-br from-neutral-800 to-neutral-950 py-2.5 pl-3 pr-10 opacity-75">
        <div className="flex items-center justify-start gap-2 text-xs text-white">
          <div className="relative flex items-center justify-center">
            <svg width="18" height="18">
              <circle cx="9" cy="9" r="6" fill="#06b6d4" />
            </svg>
            <div className="absolute inset-0 flex items-center justify-center text-neutral-950">
              <IoMdCheckmark className="size-2.5 font-bold" />
            </div>
          </div>
          <span className="truncate">{step1}</span>
        </div>
        <div className="ml-5 h-1.5 w-[100%] overflow-hidden rounded-full bg-cyan-500"></div>
      </div>

      <div className="absolute top-0 h-[40%] w-full pointer-events-none [background-image:linear-gradient(to_bottom,rgba(10,10,10,1)_15%,transparent_100%)]" />
      <div className="absolute bottom-0 h-[40%] w-full pointer-events-none [background-image:linear-gradient(to_top,rgba(10,10,10,1)_15%,transparent_100%)]" />
    </div>
  );
};

export default OnboardCard;
