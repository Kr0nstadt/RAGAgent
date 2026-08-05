// Use the subagent extension shipped with the installed Herdr/Pi distribution.
// Keeping this loader tiny avoids maintaining a private fork of Pi's process,
// streaming, cancellation and project-agent confirmation logic.
export { default } from "../../../../../AppData/Roaming/npm/node_modules/@earendil-works/pi-coding-agent/examples/extensions/subagent/index.ts";
