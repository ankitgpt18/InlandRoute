"use client";

import { cn } from "@/lib/utils";
import Link from "next/link";
import { usePathname } from "next/navigation";
import React from "react";

interface AnchorProps {
  href: any;
  absolute?: boolean;
  activeClassName?: string;
  className?: string;
  disabled?: boolean;
  children: React.ReactNode;
}

export default function AnchorNav({
  href,
  absolute,
  className = "",
  disabled,
  children,
}: AnchorProps) {
  const path = usePathname();
  const hrefStr = typeof href === "string" ? href : "";
  const isMatch = absolute
    ? hrefStr.split("/")[1] === path.split("/")[1]
    : path === hrefStr;

  if (disabled) {
    return <div className={cn(className, "cursor-not-allowed")}>{children}</div>;
  }

  return (
    <Link
      href={href as any}
      className={cn(
        className,
        "block text-neutral-400 transition-colors duration-150 hover:text-white",
        isMatch && "font-semibold text-white",
      )}
    >
      {children}
    </Link>
  );
}
