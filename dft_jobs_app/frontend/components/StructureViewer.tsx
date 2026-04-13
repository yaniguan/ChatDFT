"use client";

import { useEffect, useRef } from "react";

interface StructureViewerProps {
  xyz: string;
}

import * as $3DmolNS from "3dmol";
const $3Dmol = $3DmolNS as unknown as {
  createViewer: (el: HTMLElement, cfg: Record<string, unknown>) => {
    addModel: (data: string, fmt: string) => void;
    setStyle: (sel: Record<string, unknown>, style: Record<string, unknown>) => void;
    zoomTo: () => void;
    render: () => void;
    resize: () => void;
    clear: () => void;
  };
};

export default function StructureViewer({ xyz }: StructureViewerProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const el = containerRef.current;
    el.innerHTML = "";

    const viewer = $3Dmol.createViewer(el, {
      backgroundColor: "white",
    });
    viewer.addModel(xyz, "xyz");
    viewer.setStyle({}, { stick: { radius: 0.12 }, sphere: { scale: 0.28 } });
    viewer.zoomTo();
    viewer.render();

    const handleResize = (): void => viewer.resize();
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      try {
        viewer.clear();
      } catch {
        // ignore teardown errors
      }
    };
  }, [xyz]);

  return (
    <div
      ref={containerRef}
      className="relative h-80 w-full rounded-md border bg-white"
    />
  );
}
