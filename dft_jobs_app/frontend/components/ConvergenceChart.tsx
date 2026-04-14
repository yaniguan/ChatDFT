"use client";

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { ConvergenceStep } from "@/lib/api";

export function ConvergenceChart({ data }: { data: ConvergenceStep[] }) {
  if (data.length === 0) {
    return (
      <div className="flex h-64 items-center justify-center text-sm text-muted-foreground">
        Waiting for convergence data...
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={320}>
      <LineChart data={data} margin={{ top: 16, right: 32, left: 0, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
        <XAxis
          dataKey="step"
          label={{ value: "Step", position: "insideBottom", offset: -4 }}
        />
        <YAxis
          yAxisId="left"
          domain={["auto", "auto"]}
          tickFormatter={(v: number) => v.toFixed(2)}
          label={{ value: "Energy (eV)", angle: -90, position: "insideLeft" }}
        />
        <YAxis
          yAxisId="right"
          orientation="right"
          domain={[0, "auto"]}
          tickFormatter={(v: number) => v.toFixed(2)}
          label={{ value: "Force (eV/Å)", angle: 90, position: "insideRight" }}
        />
        <Tooltip
          formatter={(value: number, name: string) => [value.toFixed(4), name]}
        />
        <Legend />
        <Line
          yAxisId="left"
          type="monotone"
          dataKey="energy"
          stroke="#2563eb"
          dot={false}
          strokeWidth={2}
          name="Energy"
          isAnimationActive={false}
        />
        <Line
          yAxisId="right"
          type="monotone"
          dataKey="force"
          stroke="#dc2626"
          dot={false}
          strokeWidth={2}
          name="Max force"
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
