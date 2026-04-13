import type { Metadata } from "next";
import Link from "next/link";

import "./globals.css";

export const metadata: Metadata = {
  title: "DFT Jobs",
  description: "Submit, monitor, and visualize DFT jobs",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-background text-foreground antialiased">
        <header className="border-b">
          <div className="container flex h-14 items-center justify-between">
            <Link href="/" className="font-semibold">
              DFT Jobs
            </Link>
            <nav className="text-sm text-muted-foreground">
              FastAPI + Next.js + RQ
            </nav>
          </div>
        </header>
        <main className="container py-8">{children}</main>
      </body>
    </html>
  );
}
