const fs = require('fs');
const path = require('path');

const srcDir = path.join(__dirname, 'src');

// Map of dark Tailwind classes to their light equivalents
const REPLACEMENTS = [
  // First, target transparent whites -> transparent slates
  { from: /bg-white\/\[0\.(\d+)\]/g, to: 'bg-slate-900/[0.$1]' },
  { from: /text-white\/(\d+)/g, to: 'text-slate-900/$1' },
  { from: /border-white\/\[0\.(\d+)\]/g, to: 'border-slate-900/[0.$1]' },
  { from: /border-white\/(\d+)/g, to: 'border-slate-300' },
  { from: /ring-white\/(\d+)/g, to: 'ring-slate-900/$1' },
  { from: /bg-white\/(\d+)/g, to: 'bg-slate-200/$1' }, // Instead of 900, use 200 for overlays to be subtle
  
  // Then target specific slates -> whites/light slates
  { from: /bg-slate-950/g, to: 'bg-slate-50' },
  { from: /bg-slate-900/g, to: 'bg-white' },
  { from: /bg-slate-800/g, to: 'bg-slate-100' },
  { from: /bg-slate-700/g, to: 'bg-slate-200' },
  { from: /bg-\[\#0B1120\]/g, to: 'bg-slate-50' },
  { from: /bg-black/g, to: 'bg-white' },

  // Texts
  { from: /text-white(?![\/\w])/g, to: 'text-slate-900' }, // Only exact 
  { from: /text-slate-100/g, to: 'text-slate-900' },
  { from: /text-slate-200/g, to: 'text-slate-800' },
  { from: /text-slate-300/g, to: 'text-slate-700' },
  { from: /text-slate-400/g, to: 'text-slate-600' },
  { from: /text-slate-500/g, to: 'text-slate-500' }, // neutral
  { from: /text-slate-600/g, to: 'text-slate-400' },

  // Borders
  { from: /border-slate-800/g, to: 'border-slate-200' },
  { from: /border-slate-700/g, to: 'border-slate-300' },

  // Shadows
  { from: /shadow-white/g, to: 'shadow-slate-900' },
];

function processFile(filePath) {
  let content = fs.readFileSync(filePath, 'utf8');
  let originalContent = content;

  for (const { from, to } of REPLACEMENTS) {
    content = content.replace(from, to);
  }

  // Handle specific rgba replacements for borders/bgs in inline styles
  content = content.replace(/rgba\(255,255,255,0\.0/g, 'rgba(15,23,42,0.0');
  content = content.replace(/rgba\(255,\s*255,\s*255,\s*0\.(\d+)\)/g, 'rgba(15, 23, 42, 0.$1)');

  if (content !== originalContent) {
    fs.writeFileSync(filePath, content, 'utf8');
    console.log(`Updated: ${filePath.replace(__dirname, '')}`);
  }
}

function walkDir(dir) {
  const files = fs.readdirSync(dir);
  for (const file of files) {
    const fullPath = path.join(dir, file);
    if (fs.statSync(fullPath).isDirectory()) {
      if (!fullPath.includes('node_modules') && !fullPath.includes('.next')) {
        walkDir(fullPath);
      }
    } else if (fullPath.endsWith('.tsx') || fullPath.endsWith('.ts') || fullPath.endsWith('.css')) {
      processFile(fullPath);
    }
  }
}

console.log('Starting theme migration...');
walkDir(srcDir);
console.log('Done!');
