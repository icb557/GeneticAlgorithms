import React from "react";

interface CardProps {
  children: React.ReactNode;
}

export function Card({ children }: CardProps) {
  return <div className="border rounded-lg p-4 shadow">{children}</div>;
}

export function CardContent({ children }: CardProps) {
  return <div className="p-2">{children}</div>;
}

export function CardHeader({ children }: CardProps) {
  return <div className="border-b p-2">{children}</div>;
}

export function CardTitle({ children }: CardProps) {
  return <h2 className="text-lg font-bold">{children}</h2>;
}
