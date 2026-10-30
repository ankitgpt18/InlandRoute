"use client";

import { motion } from "framer-motion";
import { RiNextjsFill } from "react-icons/ri";
import { BiLogoPostgresql } from "react-icons/bi";
import { SiPytorch, SiFastapi, SiTailwindcss, SiTypescript } from "react-icons/si";
import { FaSatellite } from "react-icons/fa6";

const TechStackSection = () => {
  return (
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
        delay: 0.25,
        ease: "easeInOut",
      }}
      viewport={{ once: true }}
      className="flex items-center gap-4 pt-3 text-neutral-400"
    >
      <p className="text-xs text-neutral-400 uppercase tracking-wider font-medium md:text-sm">
        Telemetry Stack
      </p>
      <div title="Satellite Remote Sensing">
        <FaSatellite className="size-4 md:size-5 text-neutral-400 hover:text-white transition-colors" />
      </div>
      <div title="PyTorch Deep Learning">
        <SiPytorch className="size-4 md:size-5 text-neutral-400 hover:text-white transition-colors" />
      </div>
      <div title="FastAPI Engine">
        <SiFastapi className="size-4 md:size-5 text-neutral-400 hover:text-white transition-colors" />
      </div>
      <div title="PostGIS Geospatial DB">
        <BiLogoPostgresql className="size-5 md:size-6 text-neutral-400 hover:text-white transition-colors" />
      </div>
      <div title="Next.js 14">
        <RiNextjsFill className="size-5 md:size-6 text-neutral-400 hover:text-white transition-colors" />
      </div>
      <div title="TypeScript">
        <SiTypescript className="size-4 md:size-5 text-neutral-400 hover:text-white transition-colors" />
      </div>
      <div title="Tailwind CSS">
        <SiTailwindcss className="size-4 md:size-5 text-neutral-400 hover:text-white transition-colors" />
      </div>
    </motion.div>
  );
};

export default TechStackSection;
