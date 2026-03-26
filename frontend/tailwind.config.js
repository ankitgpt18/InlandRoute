/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: ['class'],
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // Navigability status colors
        navigable: {
          DEFAULT: '#22c55e',
          50:  '#f0fdf4',
          100: '#dcfce7',
          200: '#bbf7d0',
          300: '#86efac',
          400: '#4ade80',
          500: '#22c55e',
          600: '#16a34a',
          700: '#15803d',
          800: '#166534',
          900: '#14532d',
          950: '#052e16',
        },
        conditional: {
          DEFAULT: '#f59e0b',
          50:  '#fffbeb',
          100: '#fef3c7',
          200: '#fde68a',
          300: '#fcd34d',
          400: '#fbbf24',
          500: '#f59e0b',
          600: '#d97706',
          700: '#b45309',
          800: '#92400e',
          900: '#78350f',
          950: '#451a03',
        },
        'non-navigable': {
          DEFAULT: '#ef4444',
          50:  '#fef2f2',
          100: '#fee2e2',
          200: '#fecaca',
          300: '#fca5a5',
          400: '#f87171',
          500: '#ef4444',
          600: '#dc2626',
          700: '#b91c1c',
          800: '#991b1b',
          900: '#7f1d1d',
          950: '#450a0a',
        },
        // UI accent
        accent: {
          DEFAULT: '#3b82f6',
          50:  '#eff6ff',
          100: '#dbeafe',
          200: '#bfdbfe',
          300: '#93c5fd',
          400: '#60a5fa',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
          800: '#1e40af',
          900: '#1e3a8a',
          950: '#172554',
        },
        // Dark theme background palette
        surface: {
          base:   '#020817',  // bg-slate-950 equivalent
          card:   '#0f172a',  // bg-slate-900
          raised: '#1e293b',  // bg-slate-800
          border: 'rgba(255,255,255,0.08)',
        },
        // River / water theme
        river: {
          deep:    '#0c4a6e',
          mid:     '#0369a1',
          shallow: '#38bdf8',
          surface: '#7dd3fc',
          foam:    '#e0f2fe',
        },
      },

      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },

      backgroundImage: {
        // Subtle water-ripple gradient used for cards
        'river-gradient':
          'linear-gradient(135deg, rgba(3,105,161,0.15) 0%, rgba(12,74,110,0.05) 100%)',
        'hero-gradient':
          'linear-gradient(135deg, #020817 0%, #0c4a6e 50%, #020817 100%)',
        // Glassmorphism helper
        'glass':
          'linear-gradient(135deg, rgba(255,255,255,0.06) 0%, rgba(255,255,255,0.02) 100%)',
      },

      backdropBlur: {
        xs: '2px',
      },

      boxShadow: {
        'glass':     '0 4px 16px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.06)',
        'glass-lg':  '0 8px 32px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.08)',
        'glow-green':'0 0 20px rgba(34,197,94,0.35)',
        'glow-amber':'0 0 20px rgba(245,158,11,0.35)',
        'glow-red':  '0 0 20px rgba(239,68,68,0.35)',
        'glow-blue': '0 0 20px rgba(59,130,246,0.35)',
        'card':      '0 2px 8px rgba(0,0,0,0.6)',
        'card-hover':'0 8px 24px rgba(0,0,0,0.7)',
      },

      borderRadius: {
        '2xl': '1rem',
        '3xl': '1.5rem',
        '4xl': '2rem',
      },

      keyframes: {
        // Pulsing glow for critical alerts
        'pulse-glow': {
          '0%, 100%': { boxShadow: '0 0 8px rgba(239,68,68,0.6)' },
          '50%':       { boxShadow: '0 0 20px rgba(239,68,68,0.9)' },
        },
        // Subtle shimmer for skeleton loaders
        shimmer: {
          '0%':   { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition:  '200% 0' },
        },
        // Floating animation for logo / decorative elements
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%':      { transform: 'translateY(-6px)' },
        },
        // Water wave
        wave: {
          '0%':   { transform: 'rotate(0deg)' },
          '100%': { transform: 'rotate(360deg)' },
        },
        // Counter tick-up for stat cards
        'count-up': {
          from: { opacity: '0', transform: 'translateY(8px)' },
          to:   { opacity: '1', transform: 'translateY(0)' },
        },
        // Slide-in from right for detail panel
        'slide-in-right': {
          from: { transform: 'translateX(100%)', opacity: '0' },
          to:   { transform: 'translateX(0)',    opacity: '1' },
        },
        // Fade-up generic entrance
        'fade-up': {
          from: { opacity: '0', transform: 'translateY(16px)' },
          to:   { opacity: '1', transform: 'translateY(0)' },
        },
      },

      animation: {
        'pulse-glow':     'pulse-glow 2s ease-in-out infinite',
        shimmer:          'shimmer 2s linear infinite',
        float:            'float 3s ease-in-out infinite',
        wave:             'wave 8s linear infinite',
        'count-up':       'count-up 0.4s ease-out forwards',
        'slide-in-right': 'slide-in-right 0.3s cubic-bezier(0.16,1,0.3,1) forwards',
        'fade-up':        'fade-up 0.5s ease-out forwards',
      },

      transitionTimingFunction: {
        spring: 'cubic-bezier(0.16, 1, 0.3, 1)',
      },

      typography: {
        DEFAULT: {
          css: {
            color: '#e2e8f0',
            a:     { color: '#3b82f6' },
            strong:{ color: '#f8fafc' },
            code:  { color: '#7dd3fc' },
          },
        },
      },
    },
  },
  plugins: [],
};
