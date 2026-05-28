import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ["var(--font-inter)", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "monospace"],
      },
      colors: {
        // Clinical workstation palette — restrained, trustworthy
        clinical: {
          bg: "#f0f3f7",
          surface: "#ffffff",
          "surface-muted": "#f8fafc",
          "surface-raised": "#ffffff",
          border: "#dde3ec",
          "border-strong": "#b8c4d8",
          primary: "#1d3461",
          "primary-light": "#2a4a8a",
          accent: "#2563eb",
          "accent-muted": "#dbeafe",
          safe: "#059669",
          "safe-muted": "#d1fae5",
          warning: "#d97706",
          "warning-muted": "#fef3c7",
          alert: "#dc2626",
          "alert-muted": "#fee2e2",
          "text-primary": "#0f172a",
          "text-secondary": "#475569",
          "text-muted": "#94a3b8",
          "text-inverse": "#ffffff",
        },
      },
      boxShadow: {
        clinical: "0 1px 3px 0 rgba(0,0,0,0.06), 0 1px 2px -1px rgba(0,0,0,0.04)",
        "clinical-md": "0 4px 6px -1px rgba(0,0,0,0.07), 0 2px 4px -2px rgba(0,0,0,0.04)",
        "clinical-lg": "0 10px 15px -3px rgba(0,0,0,0.08), 0 4px 6px -4px rgba(0,0,0,0.04)",
      },
      animation: {
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "fade-in": "fadeIn 0.2s ease-out",
      },
      keyframes: {
        fadeIn: {
          "0%": { opacity: "0", transform: "translateY(2px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
    },
  },
  plugins: [],
};

export default config;
