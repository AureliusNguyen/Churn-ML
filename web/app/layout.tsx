import type { Metadata } from "next";
import { fraunces, inter, mono } from "@/lib/fonts";
import { ScrollProgress } from "@/components/ui/ScrollProgress";
import "./globals.css";

export const metadata: Metadata = {
  title: "The Churn Report",
  description:
    "An editorial dashboard for predicting and explaining bank customer churn.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${fraunces.variable} ${inter.variable} ${mono.variable} bg-cream text-ink antialiased`}
      >
        <ScrollProgress />
        {children}
      </body>
    </html>
  );
}
