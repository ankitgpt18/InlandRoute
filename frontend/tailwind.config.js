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
        // Navigability status colors - SaaS soft tones
        navigable: {
          DEFAULT: '#10b981',
          50:  '#ecfdf5',
          100: '#d1fae5',
          200: '#a7f3d0',
          300: '#6ee7b7',
          400: '#34d399',
          500: '#10b981',
          600: '#059669',
          700: '#047857',
          800: '#065f46',
          900: '#064e3b',
          950: '#022c22',
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
          DEFAULT: '#0f172a',
          50:  '#f8fafc',
          100: '#f1f5f9',
          200: '#e2e8f0',
          300: '#cbd5e1',
          400: '#94a3b8',
          500: '#64748b',
          600: '#475569',
          700: '#334155',
          800: '#1e293b',
          900: '#0f172a',
          950: '#020817',
        },
        // Clean White-Grey SaaS Surfaces
        surface: {
          base:   '#f8fafc',  // Ultra-clean light neutral bg
          card:   '#ffffff',  // Pure white card bg
          raised: '#f1f5f9',  // Subtle grey highlight / hover
          dark:   '#09090b',  // Dark accent panel
          border: '#e2e8f0',  // Minimal 1px border
          borderHover: '#cbd5e1',
        },
        // River / water theme
        river: {
          deep:    '#0369a1',
          mid:     '#0284c7',
          shallow: '#38bdf8',
          surface: '#0ea5e9',
          foam:    '#f0f9ff',
        },
      },

      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },

      backgroundImage: {
        'river-gradient':
          'linear-gradient(135deg, rgba(2,132,199,0.06) 0%, rgba(3,105,161,0.02) 100%)',
        'glass-gradient':
          'linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(255,255,255,0.7) 100%)',
      },

      boxShadow: {
        'subtle':    '0 1px 2px 0 rgba(0, 0, 0, 0.03), 0 1px 3px 0 rgba(0, 0, 0, 0.02)',
        'card':      '0 1px 3px 0 rgba(15, 23, 42, 0.04), 0 1px 2px -1px rgba(15, 23, 42, 0.02)',
        'card-hover':'0 10px 15px -3px rgba(15, 23, 42, 0.05), 0 4px 6px -2px rgba(15, 23, 42, 0.025)',
        'glass':     '0 4px 16px rgba(15,23,42,0.05), 0 1px 2px rgba(15,23,42,0.02)',
        'float':     '0 12px 32px -4px rgba(15,23,42,0.08)',
      },

      borderRadius: {
        'xl': '0.75rem',
        '2xl': '1rem',
        '3xl': '1.5rem',
      },

      keyframes: {
        'pulse-glow': {
          '0%, 100%': { boxShadow: '0 0 4px rgba(239,68,68,0.4)' },
          '50%':       { boxShadow: '0 0 12px rgba(239,68,68,0.7)' },
        },
        shimmer: {
          '0%':   { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition:  '200% 0' },
        },
        'fade-in': {
          from: { opacity: '0', transform: 'translateY(4px)' },
          to:   { opacity: '1', transform: 'translateY(0)' },
        },
        'slide-in-right': {
          from: { transform: 'translateX(100%)', opacity: '0' },
          to:   { transform: 'translateX(0)',    opacity: '1' },
        },
      },

      animation: {
        'pulse-glow':     'pulse-glow 2s ease-in-out infinite',
        shimmer:          'shimmer 2s linear infinite',
        'fade-in':        'fade-in 0.3s ease-out forwards',
        'slide-in-right': 'slide-in-right 0.3s cubic-bezier(0.16,1,0.3,1) forwards',
      },

      transitionTimingFunction: {
        spring: 'cubic-bezier(0.16, 1, 0.3, 1)',
      },
    },
  },
  plugins: [],
};
