"use client";

import React from "react";
import Image from "next/image";
import Link from "next/link";
import { CiGlobe } from "react-icons/ci";
import { BiLinkExternal } from "react-icons/bi";

export default function SiteHeader() {
  return (
    <div className="flex flex-row items-start justify-between px-1 pb-3 pt-1">
      <div className="flex flex-col space-y-3">
        <div className="flex flex-row items-center gap-3">
          <Image
            src="/logo.svg"
            width={28}
            height={28}
            alt="InlandRoute"
            className="h-7 w-7 rounded-lg"
          />
          <h1 className="text-2xl font-semibold text-white tracking-tight sm:text-3xl">
            National Waterway 1 (NW-1 Ganga)
          </h1>
        </div>
        <div className="flex items-center space-x-2 text-sm text-neutral-400">
          <CiGlobe className="h-4 w-4 text-cyan-400" />
          <span className="text-neutral-400">Prayagraj to Haldia Fairway (1,620 km)</span>
          <span className="text-neutral-600">|</span>
          <Link
            href="/maps"
            className="flex flex-row items-center gap-1 font-semibold text-white hover:underline"
          >
            Live AIS Telemetry
            <BiLinkExternal className="mt-0.5 h-3.5 w-3.5 text-neutral-400" />
          </Link>
        </div>
      </div>
    </div>
  );
}
