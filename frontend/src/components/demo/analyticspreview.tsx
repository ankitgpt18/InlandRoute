"use client";

import React from "react";
import SiteHeader from "./site-header";
import PagesAnalytics from "./page-analytics";
import ReffererAnalytics from "./refferer-analytics";
import AnalyticsChart from "./analytics-chart";
import ToggleButton from "./togglebutton";

const AnalyticsPreview = () => {
  return (
    <div className="mx-auto flex h-auto w-[80rem] items-center justify-center rounded-lg bg-[#060606] px-6 py-5 text-white shadow-2xl shadow-black border border-neutral-800/80">
      <div className="flex w-full flex-col space-y-5">
        <SiteHeader />
        <ToggleButton />
        <AnalyticsChart />
        <div className="flex w-full flex-row gap-4">
          <div className="w-1/2">
            <PagesAnalytics />
          </div>
          <div className="w-1/2">
            <ReffererAnalytics />
          </div>
        </div>
      </div>
    </div>
  );
};

export default AnalyticsPreview;
