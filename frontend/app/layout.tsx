import type { Metadata } from "next";
import { Special_Elite } from "next/font/google";
import "./globals.css";

// Load the free Special Elite typewriter font
const typewriterFont = Special_Elite({
  weight: "400",
  subsets: ["latin"],
  variable: "--font-typewriter",
});

export const metadata: Metadata = {
  title: "Vocalis — Meeting Assistant",
  description: "Local AI meeting transcription and speaker identification",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${typewriterFont.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}