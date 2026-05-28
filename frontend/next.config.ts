import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Strict mode for better development experience
  reactStrictMode: true,

  // Resolve @xyflow/react correctly (ESM-only package)
  transpilePackages: ["@xyflow/react"],

  // Disable powered-by header
  poweredByHeader: false,

  // Experimental: allow server actions
  experimental: {},
};

export default nextConfig;
