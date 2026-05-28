/**
 * app/page.tsx
 * =============
 * Root page — clinical workflow.
 *
 * Default experience: simple clinical input → results (doctor-facing).
 * Advanced reasoning is accessible via the drawer in the results view.
 */

import { ClinicalWorkflow } from "@/components/clinical/ClinicalWorkflow";

export default function WorkstationPage() {
  return <ClinicalWorkflow />;
}
