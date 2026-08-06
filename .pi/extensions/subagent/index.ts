/** Portable project-local subagent tool for Herdr/Pi.
 *
 * Delegation is intentionally sequential: ERP agents share durable files and
 * this avoids parallel API storms and user-specific global extension paths.
 */
import { spawn } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { type ExtensionAPI, parseFrontmatter } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

interface AgentConfig {
	name: string;
	description: string;
	tools?: string[];
	model?: string;
	systemPrompt: string;
}

function discoverAgents(cwd: string): AgentConfig[] {
	const directory = path.join(cwd, ".pi", "agents");
	if (!fs.existsSync(directory)) return [];
	const result: AgentConfig[] = [];
	for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
		if (!entry.isFile() || !entry.name.endsWith(".md")) continue;
		const content = fs.readFileSync(path.join(directory, entry.name), "utf8");
		const { frontmatter, body } = parseFrontmatter<Record<string, string>>(content);
		if (!frontmatter.name || !frontmatter.description) continue;
		result.push({
			name: frontmatter.name,
			description: frontmatter.description,
			tools: frontmatter.tools?.split(",").map((item) => item.trim()).filter(Boolean),
			model: frontmatter.model,
			systemPrompt: body,
		});
	}
	return result;
}

function piInvocation(args: string[]): { command: string; args: string[] } {
	const currentScript = process.argv[1];
	if (currentScript && !currentScript.startsWith("/$bunfs/root/") && fs.existsSync(currentScript)) {
		return { command: process.execPath, args: [currentScript, ...args] };
	}
	const executable = path.basename(process.execPath).toLowerCase();
	if (!/^(node|bun)(\.exe)?$/.test(executable)) return { command: process.execPath, args };
	return { command: process.platform === "win32" ? "pi.cmd" : "pi", args };
}

function textFromMessage(message: any): string {
	if (message?.role !== "assistant" || !Array.isArray(message.content)) return "";
	return message.content
		.filter((part: any) => part?.type === "text")
		.map((part: any) => part.text)
		.join("\n");
}

export default function (pi: ExtensionAPI) {
	pi.registerTool({
		name: "subagent",
		label: "ERP Subagent",
		description: "Run one project-local specialist from .pi/agents in an isolated Pi process.",
		parameters: Type.Object({
			agent: Type.String({ description: "Agent name from .pi/agents" }),
			task: Type.String({ description: "Bounded task containing the exact Task ID" }),
			agentScope: Type.Optional(Type.String({ description: "Compatibility field; project scope is always used" })),
			confirmProjectAgents: Type.Optional(Type.Boolean({ description: "Compatibility field" })),
		}),
		async execute(_id, params, signal, onUpdate, ctx) {
			const agents = discoverAgents(ctx.cwd);
			const agent = agents.find((candidate) => candidate.name === params.agent);
			if (!agent) {
				return {
					content: [{ type: "text", text: `Unknown project agent: ${params.agent}. Available: ${agents.map((a) => a.name).join(", ")}` }],
					isError: true,
				};
			}

			const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "erp-pi-agent-"));
			const promptFile = path.join(tempDir, "system.md");
			fs.writeFileSync(promptFile, agent.systemPrompt, { encoding: "utf8", mode: 0o600 });
			const args = ["--mode", "json", "--print", "--no-session", "--offline", "--append-system-prompt", promptFile];
			if (agent.model) args.push("--model", agent.model);
			if (agent.tools?.length) args.push("--tools", agent.tools.join(","));
			args.push(`Task: ${params.task}`);

			let finalText = "";
			let stderr = "";
			try {
				const invocation = piInvocation(args);
				const exitCode = await new Promise<number>((resolve) => {
					const child = spawn(invocation.command, invocation.args, {
						cwd: ctx.cwd,
						shell: false,
						stdio: ["ignore", "pipe", "pipe"],
						env: { ...process.env, PI_OFFLINE: "1" },
					});
					let buffer = "";
					child.stdout.on("data", (chunk) => {
						buffer += chunk.toString();
						const lines = buffer.split("\n");
						buffer = lines.pop() ?? "";
						for (const line of lines) {
							try {
								const event = JSON.parse(line);
								const text = event.type === "message_end" ? textFromMessage(event.message) : "";
								if (text) {
									finalText = text;
									onUpdate?.({ content: [{ type: "text", text: finalText.slice(-8000) }] });
								}
							} catch { /* Ignore non-JSON diagnostics. */ }
						}
					});
					child.stderr.on("data", (chunk) => { stderr += chunk.toString(); });
					child.on("error", () => resolve(1));
					child.on("close", (code) => resolve(code ?? 1));
					const stop = () => child.kill();
					if (signal.aborted) stop(); else signal.addEventListener("abort", stop, { once: true });
				});
				if (exitCode !== 0) {
					return {
						content: [{ type: "text", text: stderr || finalText || `Subagent exited with code ${exitCode}` }],
						isError: true,
					};
				}
				return { content: [{ type: "text", text: finalText || "Subagent completed; inspect task artifacts." }] };
			} finally {
				fs.rmSync(tempDir, { recursive: true, force: true });
			}
		},
	});
}
