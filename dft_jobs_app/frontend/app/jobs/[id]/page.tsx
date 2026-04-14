"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";

import { ConvergenceChart } from "@/components/ConvergenceChart";
import { StatusBadge } from "@/components/StatusBadge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useJobStream, type StreamMessage } from "@/hooks/useJobStream";
import { api, type Job } from "@/lib/api";

const StructureViewer = dynamic(
  () => import("@/components/StructureViewer"),
  { ssr: false, loading: () => <div className="h-80 rounded-md border bg-muted/30" /> },
);

function fmtTime(value: string | null): string {
  if (!value) return "-";
  return new Date(value).toLocaleString();
}

function duration(start: string | null, end: string | null): string {
  if (!start) return "-";
  const s = new Date(start).getTime();
  const e = end ? new Date(end).getTime() : Date.now();
  const secs = Math.max(0, Math.round((e - s) / 1000));
  return `${secs}s`;
}

export default function JobDetailPage() {
  const params = useParams<{ id: string }>();
  const router = useRouter();
  const jobId = params.id;
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    let active = true;
    api
      .getJob(jobId)
      .then((j) => {
        if (active) setJob(j);
      })
      .catch((e) =>
        setError(e instanceof Error ? e.message : String(e)),
      );
    return () => {
      active = false;
    };
  }, [jobId]);

  const onMessage = useCallback(
    (msg: StreamMessage) => {
      if (msg.type === "deleted") {
        router.push("/");
        return;
      }
      const incoming = msg.job as Job | undefined;
      if (incoming && incoming.id === jobId) {
        setJob(incoming);
      }
    },
    [jobId, router],
  );

  const state = useJobStream(`/ws/jobs/${jobId}`, onMessage);

  const onDelete = async (): Promise<void> => {
    if (!confirm("Delete this job?")) return;
    setDeleting(true);
    try {
      await api.deleteJob(jobId);
      router.push("/");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setDeleting(false);
    }
  };

  if (error) {
    return (
      <div className="rounded-md border border-destructive/40 bg-destructive/10 p-4 text-sm text-destructive">
        {error}
      </div>
    );
  }
  if (!job) {
    return <div className="text-muted-foreground">Loading job...</div>;
  }

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-semibold">{job.name}</h1>
            <StatusBadge status={job.status} />
          </div>
          <p className="mt-1 text-sm text-muted-foreground">
            {job.formula || "no formula"} &middot; id {job.id.slice(0, 8)}
            &hellip; &middot; stream {state}
          </p>
        </div>
        <div className="flex gap-2">
          <Button asChild variant="outline">
            <Link href="/">Back</Link>
          </Button>
          <Button variant="destructive" onClick={onDelete} disabled={deleting}>
            {deleting ? "Deleting..." : "Delete"}
          </Button>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm text-muted-foreground">
              Energy
            </CardTitle>
          </CardHeader>
          <CardContent className="text-xl font-mono">
            {job.energy === null ? "-" : `${job.energy.toFixed(4)} eV`}
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="text-sm text-muted-foreground">
              Steps
            </CardTitle>
          </CardHeader>
          <CardContent className="text-xl font-mono">
            {job.convergence.length}
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="text-sm text-muted-foreground">
              Started
            </CardTitle>
          </CardHeader>
          <CardContent className="text-sm">
            {fmtTime(job.started_at)}
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="text-sm text-muted-foreground">
              Duration
            </CardTitle>
          </CardHeader>
          <CardContent className="text-sm">
            {duration(job.started_at, job.finished_at)}
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Structure</CardTitle>
          </CardHeader>
          <CardContent>
            {job.structure_xyz ? (
              <StructureViewer xyz={job.structure_xyz} />
            ) : (
              <pre className="max-h-80 overflow-auto rounded-md border bg-muted/40 p-3 text-xs">
                {job.poscar || "No POSCAR provided"}
              </pre>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Convergence</CardTitle>
          </CardHeader>
          <CardContent>
            <ConvergenceChart data={job.convergence} />
          </CardContent>
        </Card>
      </div>

      {job.error && (
        <Card>
          <CardHeader>
            <CardTitle className="text-destructive">Error</CardTitle>
          </CardHeader>
          <CardContent className="font-mono text-sm">{job.error}</CardContent>
        </Card>
      )}
    </div>
  );
}
