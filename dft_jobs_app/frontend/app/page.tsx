"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import { JobTable } from "@/components/JobTable";
import { Button } from "@/components/ui/button";
import { useJobStream, type StreamMessage } from "@/hooks/useJobStream";
import { api, type Job } from "@/lib/api";

function mergeJob(list: Job[], next: Job): Job[] {
  const idx = list.findIndex((j) => j.id === next.id);
  if (idx === -1) return [next, ...list];
  const clone = list.slice();
  clone[idx] = next;
  return clone;
}

export default function HomePage() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const data = await api.listJobs(1, 100);
      setJobs(data.items);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const onMessage = useCallback((msg: StreamMessage) => {
    const type = msg.type as string | undefined;
    if (type === "deleted") {
      const jobId = msg.job_id as string | undefined;
      if (jobId) setJobs((prev) => prev.filter((j) => j.id !== jobId));
      return;
    }
    const job = msg.job as Job | undefined;
    if (job) setJobs((prev) => mergeJob(prev, job));
  }, []);

  const state = useJobStream("/ws/jobs", onMessage);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Jobs</h1>
          <p className="text-sm text-muted-foreground">
            Live stream: {state}
          </p>
        </div>
        <Button asChild>
          <Link href="/jobs/new">New Job</Link>
        </Button>
      </div>

      {error && (
        <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
          {error}
        </div>
      )}

      {loading ? (
        <div className="rounded-md border p-12 text-center text-muted-foreground">
          Loading jobs...
        </div>
      ) : (
        <JobTable jobs={jobs} />
      )}
    </div>
  );
}
