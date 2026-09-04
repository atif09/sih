import { useState } from "react"
import App from "./App"
import { Home } from "./pages/Home"

type Page = "home" | "console"

/**
 * Minimal client-side page switch -- two pages, no deep-linking
 * requirement, so a router dependency wasn't worth adding for this.
 */
export function Root() {
  const [page, setPage] = useState<Page>("home")

  return page === "home" ? <Home onLaunch={() => setPage("console")} /> : <App onHome={() => setPage("home")} />
}
