// ============================================================
// InlandRoute - Inland Waterway Navigability Prediction System
// Root Layout
// ============================================================

import type { Metadata, Viewport } from 'next';
import { Inter, JetBrains_Mono } from 'next/font/google';
import './globals.css';
import { Providers } from '@/components/layout/providers';
import { Sidebar } from '@/components/layout/sidebar';
import { Header } from '@/components/layout/header';

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
    'AI-powered inland waterway navigability prediction for India\'s National Waterways using Sentinel-2 satellite imagery and deep learning.',
  keywords: [
    'inland waterways',
    'navigability prediction',
    'satellite remote sensing',
    'deep learning',
    'Ganga',
    'Brahmaputra',
    'NW-1',
    'NW-2',
    'IWAI',
    'India',
  ],
  authors: [
    { name: 'Dev Yadav' },
    { name: 'Chakshu Vashisth' },
    { name: 'Ankit Gupta' },
  ],
  creator: 'Gati Shakti Vishwavidyalaya, Vadodara',
  openGraph: {
    type: 'website',
    locale: 'en_IN',
    title: 'InlandRoute - Inland Waterway Navigability Prediction',
    description:
      'Predict navigability of India\'s inland waterways using satellite imagery and AI.',
    siteName: 'InlandRoute',
  },
  icons: {
    icon: '/favicon.svg',
    shortcut: '/favicon.svg',
  },
  robots: {
    index: true,
    follow: true,
  },
};

export const viewport: Viewport = {
  themeColor: '#020817',
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
      <body className="bg-slate-50 text-slate-900 antialiased font-sans overflow-hidden">
        <Providers>
          {/* App Shell */}
          <div className="flex h-screen w-screen overflow-hidden">
            {/* Sidebar */}
            <Sidebar />

            {/* Main Content Area */}
            <div className="flex flex-col flex-1 min-w-0 overflow-hidden">
              {/* Top Header */}
              <Header />

              {/* Page Content */}
              <main className="flex-1 overflow-y-auto overflow-x-hidden thin-scrollbar relative">
                {/* Subtle background pattern */}
                <div
                  className="absolute inset-0 pointer-events-none"
                  style={{
                    backgroundImage:
                      'radial-gradient(rgba(15,23,42,0.025) 1px, transparent 1px)',
                    backgroundSize: '28px 28px',
                  }}
                  aria-hidden="true"
                />
                {/* Waterway gradient accent — top */}
                <div
                  className="absolute top-0 left-0 right-0 h-48 pointer-events-none"
                  style={{
                    background:
                      'linear-gradient(180deg, rgba(3,105,161,0.06) 0%, transparent 100%)',
                  }}
                  aria-hidden="true"
                />

                <div className="relative z-10">{children}</div>
              </main>
            </div>
          </div>
        </Providers>
      </body>
    </html>
  );
}
