"use client";

import React from "react";
import Navbar from "@/components/landingpage/navbar-shrink";
import Footer from "@/components/landingpage/footer";
import SiteHeader from "@/components/dashboard/site-header";

interface DashboardShellProps {
  children: React.ReactNode;
}

export function DashboardShell({ children }: DashboardShellProps) {
  return (
    <div className="bg-black text-white min-h-screen selection:bg-neutral-800 selection:text-white flex flex-col justify-between font-sans">
      <Navbar />
      <main className="mx-auto flex w-full max-w-7xl flex-1 flex-col gap-5 px-4 pb-16 pt-28">
        <SiteHeader />
        {children}
      </main>
      <Footer />
    </div>
  );
}

export default DashboardShell;
