import { cn } from "@/lib/utils";
import OnboardCard from "./onboard-card";

const FeatureOnboard = () => {
  return (
    <div
      className={cn(
        "flex h-[20rem] flex-col justify-between space-y-4 rounded-xl border border-neutral-800/80 bg-neutral-950 p-5",
      )}
    >
      <OnboardCard />
      <div>
        <div className="text-sm font-semibold text-white">
          Automated Telemetry Pipeline
        </div>
        <div className="mt-1.5 text-xs text-neutral-400 leading-relaxed">
          Continuous Earth Engine satellite ingestion and depth inference delivering automated fairway clearance advisories directly to fleet operators.
        </div>
      </div>
    </div>
  );
};

export default FeatureOnboard;
