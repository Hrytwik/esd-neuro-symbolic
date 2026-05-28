/**
 * app/page.tsx
 * =============
 * Root page — renders the full-screen clinical reasoning workstation.
 * No additional chrome; the workspace is the entire application.
 */

import { ClinicalReasoningWorkspace } from "@/components/workspace/ClinicalReasoningWorkspace";

export default function WorkstationPage() {
  return <ClinicalReasoningWorkspace />;
}
