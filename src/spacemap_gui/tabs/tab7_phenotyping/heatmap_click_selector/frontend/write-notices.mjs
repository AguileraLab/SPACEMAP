// Preserve the licenses of dependencies shipped in the compiled widget.
import { readFileSync, readdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const lock = JSON.parse(readFileSync("package-lock.json", "utf8"));
const sections = ["Third-party notices for the SPACEMAP heatmap selector.\nThese dependencies retain their own licenses.\n"];
for (const [directory, info] of Object.entries(lock.packages).sort()) {
  if (!directory || info.dev) continue;
  const files = readdirSync(directory).filter(name => /^(licen[cs]e|copying|notice)(\.|$)/i.test(name)).sort();
  if (!files.length && directory === "node_modules/streamlit-component-lib") {
    sections.push(`\nstreamlit-component-lib (${info.version})\nSource: https://github.com/streamlit/streamlit/blob/1.40.0/LICENSE\n`);
    sections.push(readFileSync("vendor-licenses/streamlit-component-lib-LICENSE.txt", "utf8"));
    continue;
  }
  if (!files.length) throw new Error(`Missing license text: ${directory}`);
  sections.push(`\n${directory} (${info.version})\n${"=".repeat(72)}\n`);
  for (const file of files) sections.push(`${file}\n${readFileSync(join(directory, file), "utf8")}\n`);
}
// Normalize whitespace only; retain the complete upstream notice text.
const cleanText = text => text.replace(/\r\n/g, "\n").split("\n").map(line => line.trimEnd()).join("\n").trimEnd() + "\n";
writeFileSync("build/THIRD_PARTY_NOTICES.txt", cleanText(sections.join("\n")));
writeFileSync("build/index.html", cleanText(readFileSync("build/index.html", "utf8")));
