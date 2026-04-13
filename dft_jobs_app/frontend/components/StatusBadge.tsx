import { Badge } from "@/components/ui/badge";
import type { JobStatus } from "@/lib/api";
import { cn } from "@/lib/utils";

const STYLES: Record<JobStatus, string> = {
  pending: "bg-slate-200 text-slate-800 hover:bg-slate-200",
  running: "bg-blue-100 text-blue-800 hover:bg-blue-100",
  completed: "bg-green-100 text-green-800 hover:bg-green-100",
  failed: "bg-red-100 text-red-800 hover:bg-red-100",
};

export function StatusBadge({ status }: { status: JobStatus }) {
  return (
    <Badge
      variant="secondary"
      className={cn("uppercase tracking-wide", STYLES[status])}
    >
      {status}
    </Badge>
  );
}
