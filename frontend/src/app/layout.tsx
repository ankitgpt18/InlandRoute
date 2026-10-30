// ============================================================
// InlandRoute - Inland Waterway Navigability Prediction System
// Root Layout
// ============================================================

import type { Metadata, Viewport } from 'next';
import { Inter, JetBrains_Mono } from 'next/font/google';
import './globals.css';
import { Providers } from '@/components/layout/providers';
import { AppShell } from '@/components/layout/app-shell';

// ─── Fonts ────────────────────────────────────────────────────────────────────

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
  display: 'swap',
  weight: ['300', '400', '500', '600', '700', '800', '900'],
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-mono',
  display: 'swap',
  weight: ['400', '500', '600'],
});

// ─── Metadata ─────────────────────────────────────────────────────────────────

export const metadata: Metadata = {
  title: {
    default: 'InlandRoute - Inland Waterway Navigability',
    template: '%s | InlandRoute',
  },
  description:
    'Inland waterway navigability prediction for India\'s National Waterways using Sentinel-2 satellite imagery and deep learning.',
  keywords: [
    'inland waterways',
    'navigability prediction',
    'satellite remote sensing',
    'bathymetry',
    'deep learning',
    'Ganga',
    'Brahmaputra',
    'NW-1',
    'NW-2',
    'IWAI',
    'India',
  ],
  authors: [
    { name: 'InlandRoute Team' },
  ],
  creator: 'InlandRoute',
  openGraph: {
    type: 'website',
    locale: 'en_IN',
    title: 'InlandRoute - Inland Waterway Navigability Prediction',
    description:
      'Predict navigability of India\'s inland waterways using satellite imagery and hydrodynamics.',
    siteName: 'InlandRoute',
  },
  icons: {
    icon: '/logo.svg',
    shortcut: '/favicon.svg',
    apple: '/logo.png',
  },
  robots: {
    index: true,
    follow: true,
  },
};

export const viewport: Viewport = {
  themeColor: '#000000',
  width: 'device-width',
  initialScale: 1,
  colorScheme: 'dark',
};

// ─── Layout ───────────────────────────────────────────────────────────────────

interface RootLayoutProps {
  children: React.ReactNode;
}

export default function RootLayout({ children }: RootLayoutProps) {
  return (
    <html
      lang="en"
      className={`${inter.variable} ${jetbrainsMono.variable} dark`}
      suppressHydrationWarning
    >
      <body className="bg-black text-white antialiased font-sans">
        <Providers>
          <AppShell>{children}</AppShell>
        </Providers>
      </body>
    </html>
  );
}

