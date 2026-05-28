/**
 * app/layout.tsx
 * ===============
 * Root Next.js layout — font loading, metadata, global stylesheet.
 * Keeps the viewport locked to full-height to prevent scroll on the workstation.
 */

import type { Metadata, Viewport } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

// ─── Typography ───────────────────────────────────────────────────────────────

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
  // Weights used across the workstation
  weight: ["400", "500", "600", "700"],
});

// ─── Metadata ─────────────────────────────────────────────────────────────────

export const metadata: Metadata = {
  title: "CASDRE — Dermatological Reasoning Workstation",
  description:
    "Certainty-Aware Symbolic Dermatological Reasoning Engine: an interpretable clinical decision-support workstation for inflammatory skin disease differential diagnosis.",
  keywords: [
    "dermatology",
    "clinical reasoning",
    "symbolic inference",
    "diagnostic workstation",
    "inflammatory skin disease",
  ],
  authors: [{ name: "Clinical Informatics Research Group" }],
  robots: { index: false, follow: false }, // Research prototype — not indexed
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  // Disable zooming on a clinical workstation UI
  maximumScale: 1,
  userScalable: false,
};

// ─── Root layout ──────────────────────────────────────────────────────────────

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="en"
      className={inter.variable}
      // Prevent flash on initial load
      suppressHydrationWarning
    >
      <head>
        {/* Prefetch the React Flow stylesheet for faster first paint */}
        <link
          rel="preload"
          href="/_next/static/css/app/globals.css"
          as="style"
        />
      </head>
      <body
        className="font-sans antialiased overflow-hidden"
        style={{ fontFamily: "var(--font-inter), Inter, system-ui, sans-serif" }}
      >
        {children}
      </body>
    </html>
  );
}
