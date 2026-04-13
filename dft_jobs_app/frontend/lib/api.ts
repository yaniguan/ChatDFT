export type JobStatus = "pending" | "running" | "completed" | "failed";

export interface ConvergenceStep {
  step: number;
  energy: number;
  force: number;
}

export interface Job {
  id: string;
  name: string;
  status: JobStatus;
  formula: string;
  poscar: string;
  created_at: string | null;
  updated_at: string | null;
  started_at: string | null;
  finished_at: string | null;
  energy: number | null;
  error: string | null;
  convergence: ConvergenceStep[];
  structure_xyz: string | null;
}

export interface JobList {
  items: Job[];
  total: number;
  page: number;
  page_size: number;
}

export interface JobCreateInput {
  name: string;
  formula: string;
  poscar: string;
}

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export const WS_BASE =
  process.env.NEXT_PUBLIC_WS_URL ??
  API_BASE.replace(/^http/, "ws");

async function request<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    cache: "no-store",
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`API ${res.status}: ${text || res.statusText}`);
  }
  if (res.status === 204) {
    return undefined as unknown as T;
  }
  return (await res.json()) as T;
}

export const api = {
  listJobs: (page = 1, pageSize = 50): Promise<JobList> =>
    request(`/api/jobs?page=${page}&page_size=${pageSize}`),
  getJob: (id: string): Promise<Job> => request(`/api/jobs/${id}`),
  createJob: (input: JobCreateInput): Promise<Job> =>
    request(`/api/jobs`, { method: "POST", body: JSON.stringify(input) }),
  deleteJob: (id: string): Promise<void> =>
    request(`/api/jobs/${id}`, { method: "DELETE" }),
};
