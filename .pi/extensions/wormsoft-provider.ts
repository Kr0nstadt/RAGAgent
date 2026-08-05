import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export default function (pi: ExtensionAPI) {
	pi.registerProvider("wormsoft-gateway", {
		baseUrl: "https://ai.wormsoft.ru/api/gpt",
		api: "openai-completions",
		apiKey: "$WORMSOFT_API_KEY",
		authHeader: true,
		compat: {
			supportsDeveloperRole: false,
			supportsReasoningEffort: false,
		},
		models: [
			{
				id: "wormsoft/agent/high",
				name: "Wormsoft Agent High",
				reasoning: true,
				input: ["text"],
				contextWindow: 128000,
				maxTokens: 16384,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			},
		],
	});
}
