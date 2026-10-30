"use client";

import { motion } from "framer-motion";
import React from "react";
import FeatureIcon from "../icons/featureicon";

const FeatureContent = () => {
  return (
    <>
      <motion.div
        initial={{
          y: 10,
          filter: "blur(10px)",
          opacity: 0,
        }}
        whileInView={{
          y: 0,
          filter: "blur(0px)",
          opacity: 1,
        }}
        transition={{
          duration: 0.3,
          delay: 0.1,
          ease: "easeInOut",
        }}
        viewport={{ once: true }}
        className="mb-2"
      >
        <FeatureIcon />
      </motion.div>
      <motion.h2
        initial={{
          y: 10,
          filter: "blur(10px)",
          opacity: 0,
        }}
        whileInView={{
          y: 0,
          filter: "blur(0px)",
          opacity: 1,
        }}
        transition={{
          duration: 0.4,
          delay: 0.1,
          ease: "easeInOut",
        }}
        viewport={{ once: true }}
        className="mx-auto mb-2 max-w-3xl text-balance bg-gradient-to-br from-neutral-100 via-neutral-100 via-50% to-neutral-100/30 bg-clip-text py-2 text-center text-3xl font-medium leading-[1.15] tracking-tight text-transparent md:text-5xl"
      >
        Engineered for Port Authorities and Fleet Navigators
      </motion.h2>
      <motion.p
        initial={{
          y: 10,
          filter: "blur(10px)",
          opacity: 0,
        }}
        whileInView={{
          y: 0,
          filter: "blur(0px)",
          opacity: 1,
        }}
        transition={{
          duration: 0.3,
          delay: 0.2,
          ease: "easeInOut",
        }}
        viewport={{ once: true }}
        className="mb-8 max-w-sm text-balance bg-gradient-to-br from-white/70 via-white/70 to-white/30 bg-clip-text text-center text-[0.85rem] text-transparent sm:max-w-[34rem] sm:text-[0.92rem] lg:text-[1rem] leading-relaxed"
      >
        InlandRoute delivers predictive bathymetry and shoal classification, transforming multispectral satellite telemetry into reliable fairway clearance forecasts.
      </motion.p>
    </>
  );
};

export default FeatureContent;
