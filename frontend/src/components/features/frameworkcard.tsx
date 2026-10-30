"use client";

import { cn } from "@/lib/utils";
import { useState, useEffect } from "react";
import { Waves, Ship, Navigation } from "lucide-react";

const FrameworkCard = () => {
  const [card1Transform, setCard1Transform] = useState("none");
  const [card2Transform, setCard2Transform] = useState("none");
  const [card3Transform, setCard3Transform] = useState("none");

  useEffect(() => {
    let isMounted = true;
    const cycleAnimations = async () => {
      const upStyle = "translateY(-3.71px) rotateX(10.71deg) translateZ(20px)";
      const downStyle = "none";

      const transitionDuration = 1100;
      const durationOfUpState = 1200;
      const delayBetweenCards = 600;

      while (isMounted) {
        setCard1Transform(upStyle);
        await new Promise((resolve) => setTimeout(resolve, durationOfUpState));
        if (!isMounted) break;
        setCard1Transform(downStyle);
        await new Promise((resolve) =>
          setTimeout(resolve, transitionDuration + delayBetweenCards),
        );
        if (!isMounted) break;

        setCard2Transform(upStyle);
        await new Promise((resolve) => setTimeout(resolve, durationOfUpState));
        if (!isMounted) break;
        setCard2Transform(downStyle);
        await new Promise((resolve) =>
          setTimeout(resolve, transitionDuration + delayBetweenCards),
        );
        if (!isMounted) break;

        setCard3Transform(upStyle);
        await new Promise((resolve) => setTimeout(resolve, durationOfUpState));
        if (!isMounted) break;
        setCard3Transform(downStyle);
        await new Promise((resolve) =>
          setTimeout(resolve, transitionDuration + delayBetweenCards),
        );
      }
    };

    cycleAnimations();
    return () => {
      isMounted = false;
    };
  }, []);

  const cardClasses =
    "flex aspect-square flex-col items-center justify-center rounded-lg border border-neutral-800 bg-gradient-to-b from-neutral-800 to-neutral-950 p-3 " +
    "w-24 sm:w-32 " +
    "transition-transform duration-1000 ease-out will-change-transform shadow-lg";

  return (
    <>
      <div
        className={cn(
          "relative",
          "flex flex-col items-center justify-center gap-1",
          "h-[14.5rem] w-full",
        )}
      >
        <div className="absolute flex h-full w-full items-center justify-center pointer-events-none">
          <div className="h-full w-[15rem]">
            <svg
              className="h-full w-full"
              width="100%"
              height="100%"
              viewBox="0 0 100 100"
              fill="none"
            >
              <g stroke="#737373" strokeWidth="0.1">
                <path d="M 1 0 v 5 q 0 5 5 5 h 39 q 5 0 5 5 v 71 q 0 5 5 5 h 39 q 5 0 5 5 v 5" />
              </g>
              <g mask="url(#framework-mask)">
                <circle
                  className="frameworkline framework-line"
                  cx="0"
                  cy="0"
                  r="12"
                  fill="url(#framework-blue-grad)"
                />
              </g>
              <defs>
                <mask id="framework-mask">
                  <path
                    d="M 1 0 v 5 q 0 5 5 5 h 39 q 5 0 5 5 v 71 q 0 5 5 5 h 39 q 5 0 5 5 v 5"
                    strokeWidth="0.3"
                    stroke="white"
                  />
                </mask>
                <radialGradient id="framework-blue-grad" fx="1">
                  <stop offset="0%" stopColor={"#38bdf8"} />
                  <stop offset="100%" stopColor="transparent" />
                </radialGradient>
              </defs>
            </svg>
          </div>
        </div>
        <div
          className={cn(
            "flex items-center justify-center gap-3 sm:gap-4",
            "[perspective:1000px] [transform-style:preserve-3d]",
          )}
        >
          <div className={cardClasses} style={{ transform: card1Transform }}>
            <Waves className="size-6 sm:size-7 text-cyan-400 mb-2" />
            <span className="text-xs font-bold text-white">NW-1</span>
            <span className="text-[10px] text-neutral-400">Ganga (1,620 km)</span>
          </div>
          <div className={cardClasses} style={{ transform: card2Transform }}>
            <Ship className="size-6 sm:size-7 text-blue-400 mb-2" />
            <span className="text-xs font-bold text-white">NW-2</span>
            <span className="text-[10px] text-neutral-400">Brahmaputra (891 km)</span>
          </div>
          <div className={cardClasses} style={{ transform: card3Transform }}>
            <Navigation className="size-6 sm:size-7 text-emerald-400 mb-2" />
            <span className="text-xs font-bold text-white">NW-16</span>
            <span className="text-[10px] text-neutral-400">Barak (121 km)</span>
          </div>
        </div>

        <div className="absolute bottom-0 left-0 h-3 w-full bg-gradient-to-t from-neutral-950 to-transparent" />
      </div>
    </>
  );
};

export default FrameworkCard;
