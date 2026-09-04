interface Artifact {
  text: string
  top: string
  left: string
  size: number
  opacity: number
  duration: number
  delay: number
}

// Curated, not randomized -- fixed positions/timings chosen for a spread
// that clears the hero's headline and command bar, mirroring the reference
// design's scattered hexagon shapes but standing in real packet telemetry:
// the kind of values a SOC analyst actually stares at (IPs, TTL/flag
// bytes, raw binary octets, well-known ports).
const ARTIFACTS: Artifact[] = [
  { text: "172.16.0.1", top: "8%", left: "6%", size: 13, opacity: 0.5, duration: 9, delay: 0 },
  { text: "0x1A3F", top: "14%", left: "88%", size: 12, opacity: 0.4, duration: 7, delay: 0.6 },
  { text: "SYN·ACK", top: "22%", left: "4%", size: 11, opacity: 0.35, duration: 8.5, delay: 1.2 },
  { text: "01001101", top: "6%", left: "40%", size: 11, opacity: 0.3, duration: 10, delay: 0.3 },
  { text: "192.168.10.50", top: "38%", left: "90%", size: 13, opacity: 0.5, duration: 8, delay: 1.8 },
  { text: "445/TCP", top: "48%", left: "2%", size: 12, opacity: 0.4, duration: 9.5, delay: 0.9 },
  { text: "TTL=64", top: "60%", left: "92%", size: 11, opacity: 0.35, duration: 7.5, delay: 2.1 },
  { text: "0x4F3F", top: "18%", left: "68%", size: 12, opacity: 0.4, duration: 8.8, delay: 1.5 },
  { text: "10.0.2.254", top: "72%", left: "8%", size: 13, opacity: 0.45, duration: 9.2, delay: 0.4 },
  { text: "11010010", top: "80%", left: "82%", size: 11, opacity: 0.3, duration: 10.5, delay: 2.4 },
  { text: "RST", top: "32%", left: "20%", size: 10, opacity: 0.3, duration: 7, delay: 1.9 },
  { text: "22/SSH", top: "88%", left: "50%", size: 12, opacity: 0.35, duration: 8.2, delay: 0.7 },
  { text: "8.8.4.4", top: "4%", left: "76%", size: 12, opacity: 0.4, duration: 9.7, delay: 1.1 },
  { text: "3389/RDP", top: "56%", left: "94%", size: 11, opacity: 0.35, duration: 7.8, delay: 2.6 },
]

/**
 * The signature ambient element of the home page -- floating network
 * telemetry standing in for the reference design's hovering molecule
 * hexagons, on-theme for this project's actual subject matter. Purely
 * decorative (aria-hidden), absolutely positioned within the hero's
 * relative container, each item breathing independently.
 */
export function FloatingArtifacts() {
  return (
    <div aria-hidden className="pointer-events-none absolute inset-0 hidden overflow-hidden sm:block">
      {ARTIFACTS.map((a, i) => (
        <span
          key={i}
          className="animate-drift absolute font-mono text-secondary-fixed-dim"
          style={{
            top: a.top,
            left: a.left,
            fontSize: a.size,
            ["--drift-opacity" as string]: a.opacity,
            ["--drift-duration" as string]: `${a.duration}s`,
            ["--drift-delay" as string]: `${a.delay}s`,
          }}
        >
          {a.text}
        </span>
      ))}
    </div>
  )
}
