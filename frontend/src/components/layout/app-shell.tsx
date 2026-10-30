'use client';

import React from 'react';

interface AppShellProps {
  children: React.ReactNode;
}

export function AppShell({ children }: AppShellProps) {
  return (
    <div className="min-h-screen w-full bg-black text-white antialiased selection:bg-cyan-500 selection:text-black">
      {children}
    </div>
  );
}
