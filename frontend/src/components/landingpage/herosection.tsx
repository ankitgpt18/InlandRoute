"use client";

import { cn } from "@/lib/utils";
import { motion } from "framer-motion";
import AnalyticsPreview from "../demo/analyticspreview";
import Herobuttons from "./hero-buttons";
import TechStackSection from "./techstacksection";
import HeroBadge from "./herobadge";

const HeroSection = () => {
  return (
    <div className="mx-auto flex min-h-[100vh] flex-col items-center justify-center">
      <div
        className={cn(
          "flex h-full min-h-[99.5vh] w-full max-w-[99.5%] justify-center rounded-b-2xl",
          "bg-[radial-gradient(125%_125%_at_50%_101%,rgba(80,140,240,0.7)_0%,rgba(60,120,220,0.6)_15%,rgba(40,90,200,0.5)_30%,rgba(0,0,0,0.5)_50%,rgba(0,0,0,1)_70%,rgba(0,0,0,1)_100%)]",
        )}
      >
        <div
          className={cn(
            "mt-tab mt-24 flex h-full min-h-[99.5vh] w-full flex-col justify-center rounded-b-2xl sm:mt-48 md:mt-48 lg:mt-0 lg:flex-row",
          )}
        >
          <div className="flex lg:flex-[2]">
            <div className="flex flex-col justify-center space-y-5 pl-6 lg:mx-auto lg:max-w-[34rem] xl:pl-4">
              <HeroBadge />
              <motion.h1
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
                  ease: "easeInOut",
                }}
                viewport={{ once: true }}
                className="max-w-lg whitespace-pre-wrap text-balance font-sans text-4xl font-bold tracking-tight text-white md:max-w-2xl md:text-5xl"
              >
                Predict Inland Waterway Navigability with Satellite Remote Sensing
              </motion.h1>
              <motion.p
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
                className="max-w-sm text-balance text-[0.95rem] text-neutral-300 sm:max-w-lg md:text-[1.05rem] lg:max-w-md leading-relaxed"
              >
                High-resolution fairway depth forecasts and shoal risk detection for India&apos;s National Waterways, combining multitemporal Sentinel-2 imagery with CWC hydrological gauge models.
              </motion.p>
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
                  delay: 0.2,
                  ease: "easeInOut",
                }}
                viewport={{ once: true }}
                className="flex items-start"
              >
                <Herobuttons />
              </motion.div>
              <TechStackSection />
            </div>
          </div>

          <div className="relative hidden items-center overflow-hidden lg:flex lg:flex-[3]">
            <div className="relative w-[calc(100%+8rem)] max-w-none origin-right skew-x-[-0.05rad] skew-y-[-0.02rad]">
              <motion.div
                initial={{
                  opacity: 0,
                  filter: "blur(6px)",
                }}
                whileInView={{
                  opacity: 1,
                  filter: "blur(0px)",
                }}
                transition={{
                  duration: 0.6,
                  ease: "easeInOut",
                }}
                viewport={{ once: true }}
                className="min-w-[600px] scale-75 rounded-xl"
              >
                <AnalyticsPreview />
              </motion.div>
            </div>
          </div>

          {/* Mobile preview */}
          <div className="mx-auto mt-6 block max-w-7xl overflow-hidden px-4 pb-12 lg:hidden">
            <motion.div
              initial={{
                opacity: 0,
                filter: "blur(10px)",
              }}
              whileInView={{
                opacity: 1,
                filter: "blur(0px)",
              }}
              transition={{
                duration: 0.5,
                delay: 0.1,
                ease: "easeInOut",
              }}
              viewport={{ once: true }}
              className="w-full scale-95 overflow-hidden rounded-xl border border-neutral-800 bg-neutral-950/80 p-2 shadow-2xl shadow-black"
            >
              <div className="scale-[0.8] origin-top -mb-20">
                <AnalyticsPreview />
              </div>
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HeroSection;
