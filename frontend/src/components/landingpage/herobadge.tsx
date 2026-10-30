"use client";

import { motion } from "framer-motion";

const HeroBadge = () => {
  return (
    <>
      <motion.div
        initial={{
          opacity: 0,
          y: 8,
          filter: "blur(10px)",
        }}
        whileInView={{
          opacity: 1,
          y: 0,
          filter: "blur(0px)",
        }}
        transition={{
          duration: 0.3,
          delay: 0.1,
          ease: "easeInOut",
        }}
        viewport={{ once: true }}
        className="relative flex max-w-fit items-center justify-center gap-3 rounded-full border border-neutral-700/80 bg-black px-3.5 py-1.5"
      >
        <div className="absolute inset-x-0 bottom-full h-20 w-[165px]">
          <svg
            className="h-full w-full"
            width="100%"
            height="100%"
            viewBox="0 0 50 50"
            fill="none"
          >
            <g mask="url(#ml-mask-1)">
              <circle
                className="multiline ml-light-1"
                cx="0"
                cy="0"
                r="20"
                fill="url(#ml-white-grad)"
              />
            </g>
            <defs>
              <mask id="ml-mask-1">
                <path
                  d="M 69 49.8 h -30 q -3 0 -3 -3 v -13 q 0 -3 -3 -3 h -23 q -3 0 -3 -3 v -13 q 0 -3 -3 -3 h -30"
                  strokeWidth="0.6"
                  stroke="white"
                />
              </mask>
              <radialGradient id="ml-white-grad" fx="1">
                <stop offset="0%" stopColor={"#38bdf8"} />
                <stop offset="20%" stopColor={"#0284c7"} />
                <stop offset="100%" stopColor="transparent" />
              </radialGradient>
            </defs>
          </svg>
        </div>
        <div className="relative flex h-1.5 w-1.5 items-center justify-center rounded-full bg-cyan-500/40">
          <div className="flex h-2.5 w-2.5 animate-ping items-center justify-center rounded-full bg-cyan-500">
            <div className="flex h-2.5 w-2.5 animate-ping items-center justify-center rounded-full bg-cyan-500"></div>
          </div>
          <div className="absolute left-1/2 top-1/2 flex h-1.5 w-1.5 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full bg-cyan-400"></div>
        </div>
        <span className="bg-clip-text text-xs font-medium text-zinc-300">
          Sentinel-2 Remote Sensing and Hydrodynamic AI
        </span>
      </motion.div>
    </>
  );
};

export default HeroBadge;
