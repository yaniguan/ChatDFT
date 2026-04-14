"use client";

import Link from "next/link";

import { StatusBadge } from "@/components/StatusBadge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { Job } from "@/lib/api";

function fmtTime(value: string | null): string {
  if (!value) return "-";
  return new Date(value).toLocaleString();
}

function fmtEnergy(value: number | null): string {
  return value === null ? "-" : `${value.toFixed(4)} eV`;
}

export function JobTable({ jobs }: { jobs: Job[] }) {
  if (jobs.length === 0) {
    return (
      <div className="rounded-md border p-12 text-center text-muted-foreground">
        No jobs yet. Click &quot;New Job&quot; to submit one.
      </div>
    );
  }

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Name</TableHead>
          <TableHead>Formula</TableHead>
          <TableHead>Status</TableHead>
          <TableHead>Energy</TableHead>
          <TableHead>Steps</TableHead>
          <TableHead>Created</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {jobs.map((job) => (
          <TableRow key={job.id} className="cursor-pointer">
            <TableCell className="font-medium">
              <Link href={`/jobs/${job.id}`} className="hover:underline">
                {job.name}
              </Link>
            </TableCell>
            <TableCell>{job.formula || "-"}</TableCell>
            <TableCell>
              <StatusBadge status={job.status} />
            </TableCell>
            <TableCell>{fmtEnergy(job.energy)}</TableCell>
            <TableCell>{job.convergence.length}</TableCell>
            <TableCell>{fmtTime(job.created_at)}</TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
