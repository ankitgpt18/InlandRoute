// ============================================================
// AIDSTL — Inland Waterway Navigability Prediction System
// Client-side Providers Wrapper
// ============================================================

'use client';

import React, { useState } from 'react';
import {
  QueryClient,
  QueryClientProvider,
  isServer,
} from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';

// ─── QueryClient Factory ───────────────────────────────────────────────────────

function makeQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: {
        // Data considered fresh for 5 minutes — satellite data doesn't change second-to-second
        staleTime: 5 * 60 * 1000,
        // Keep in cache for 30 minutes after all observers unmount
        gcTime: 30 * 60 * 1000,
        // Retry failed requests up to 2 times with exponential back-off
        retry: 2,
        retryDelay: (attempt) => Math.min(1000 * 2 ** attempt, 30_000),
        // Refetch on window focus only for fresh data checks
        refetchOnWindowFocus: false,
        refetchOnReconnect: true,
        // Show stale data while re-fetching (better UX)
        placeholderData: (previousData: unknown) => previousData,
      },
      mutations: {
        retry: 1,
        retryDelay: 1000,
      },
    },
  });
}

// ─── Singleton for SSR ────────────────────────────────────────────────────────
// On the server we always make a new client (never share between requests).
// On the client we make one client and reuse it across the app lifetime.

let browserQueryClient: QueryClient | undefined;

function getQueryClient(): QueryClient {
  if (isServer) {
    return makeQueryClient();
  }
  if (!browserQueryClient) {
    browserQueryClient = makeQueryClient();
  }
  return browserQueryClient;
}

// ─── Providers Component ───────────────────────────────────────────────────────

interface ProvidersProps {
  children: React.ReactNode;
}

/**
 * Wraps the entire application with all necessary client-side providers:
 *
 * - TanStack QueryClientProvider   — server-state caching & synchronisation
 * - ReactQueryDevtools              — visible only in development
 *
 * Add additional providers here (e.g. Toaster, Theme) as the app grows.
 */
export function Providers({ children }: ProvidersProps) {
  // NOTE: Avoid useState when initialising QueryClient on the server,
  // because useState will suspend on the server. On the client, useState
  // ensures we don't recreate the client on every render.
  const [queryClient] = useState<QueryClient>(() => getQueryClient());

  return (
    <QueryClientProvider client={queryClient}>
      {children}

      {/* React Query DevTools — only rendered in development builds */}
      {process.env.NODE_ENV === 'development' && (
        <ReactQueryDevtools
          initialIsOpen={false}
          buttonPosition="bottom-right"
          position="bottom"
        />
      )}
    </QueryClientProvider>
  );
}
