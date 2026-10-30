"use client";

import React from "react";
import Navbar from "@/components/landingpage/navbar-shrink";
import Footer from "@/components/landingpage/footer";

export default function PrivacyPolicyPage() {
  return (
    <div className="bg-black text-white min-h-screen">
      <Navbar />
      <div className="container mx-auto max-w-4xl px-6 pt-32 pb-24">
        <div className="mb-10">
          <h1 className="bg-[radial-gradient(61.17%_178.53%_at_38.83%_-13.54%,#3B3B3B_0%,#888787_12.61%,#FFFFFF_50%,#888787_80%,#3B3B3B_100%)] bg-clip-text text-4xl font-bold tracking-tight text-transparent py-2">
            Privacy Policy
          </h1>
          <p className="text-sm text-neutral-400 mt-2">
            Last updated: October 2026
          </p>
        </div>

        <div className="space-y-8 rounded-xl border border-neutral-800 bg-neutral-950 p-8 text-neutral-300 leading-relaxed text-sm">
          <section className="space-y-3">
            <h2 className="text-lg font-semibold text-white">1. Overview</h2>
            <p className="text-neutral-400">
              InlandRoute provides waterway navigability telemetry and remote sensing bathymetry services for port authorities, commercial navigators, and logistics planners. We are committed to processing operational data securely, transparently, and in compliance with applicable information technology regulations.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-lg font-semibold text-white">2. Geospatial and Hydrological Data</h2>
            <p className="text-neutral-400">
              The platform ingests publicly accessible multispectral imagery from the European Space Agency Sentinel-2 constellation and river stage records published by the Central Water Commission (CWC). Vessel position telemetry (AIS) shared by registered operators is used solely for route risk assessment and is not shared with unverified third parties.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-lg font-semibold text-white">3. Information Security</h2>
            <p className="text-neutral-400">
              All telemetry transmissions, depth sounding archives, and model outputs are transmitted using TLS 1.3 encryption and stored in hardened database instances. System access logs are maintained to protect navigation advisories against unauthorized tampering.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-lg font-semibold text-white">4. Inquiries and Contact</h2>
            <p className="text-neutral-400">
              For questions concerning telemetry data handling, audit compliance, or data deletion requests, contact the project maintainers via the InlandRoute repository on GitHub.
            </p>
          </section>
        </div>
      </div>
      <Footer />
    </div>
  );
}
