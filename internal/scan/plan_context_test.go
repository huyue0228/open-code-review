// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 alibaba/open-code-review Contributors

package scan

import (
	"context"
	"strings"
	"testing"

	"github.com/alibaba/open-code-review/internal/config/template"
	"github.com/alibaba/open-code-review/internal/llm"
	"github.com/alibaba/open-code-review/internal/model"
)

type planContextClient struct{ requests []llm.ChatRequest }

func (c *planContextClient) CompletionsWithCtx(_ context.Context, req llm.ChatRequest) (*llm.ChatResponse, error) {
	c.requests = append(c.requests, req)
	return &llm.ChatResponse{Choices: []llm.Choice{{Message: llm.ResponseMessage{ToolCalls: []llm.ToolCall{{ID: "done", Type: "function", Function: llm.FunctionCall{Name: "task_done", Arguments: `{"state":"DONE"}`}}}}}}}, nil
}

func TestOversizedPlanContextFallsBackToMain(t *testing.T) {
	for _, field := range []string{"tools", "background"} {
		t.Run(field, func(t *testing.T) {
			tpl := makeTemplateWithFullScan()
			// Only the plan exceeds the ceiling; main must still review the file.
			tpl.PlanTask = &template.LlmConversation{Messages: []template.ChatMessage{{Role: "user", Content: "{{plan_tools}} {{requirement_background}}"}}}
			tpl.MainTask.Messages = []template.ChatMessage{{Role: "user", Content: "review {{current_file_path}} {{plan_guidance}}"}}
			a := newAgentForTest(t, tpl)
			client := &planContextClient{}
			// NewAgent wires the main runner at construction, so build with the same client.
			args := a.args
			args.LLMClient = client
			if field == "tools" {
				args.PlanTools = strings.Repeat("tool description ", 2000)
			} else {
				args.Background = strings.Repeat("business context ", 2000)
			}
			a = NewAgent(args)
			completed, reason, err := a.executeSubtask(t.Context(), model.ScanItem{Path: "order.go", Content: "package orders\n"})
			if err != nil || !completed {
				t.Fatalf("completed=%v reason=%q err=%v", completed, reason, err)
			}
			if len(client.requests) != 1 {
				t.Fatalf("requests=%d want only Main", len(client.requests))
			}
			found := false
			for _, w := range a.Warnings() {
				if w.Type == "plan_token_threshold_exceeded" {
					found = true
				}
			}
			if !found {
				t.Fatal("missing plan context warning")
			}
		})
	}
}
