// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 alibaba/open-code-review Contributors

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	ocrmcp "github.com/alibaba/open-code-review/internal/mcp"
	"github.com/alibaba/open-code-review/internal/session"
	sdkmcp "github.com/modelcontextprotocol/go-sdk/mcp"
)

const scanEvidence = "order-spec v3 section 4.2: shipped orders cannot be cancelled"

// scanContextFixture exercises the actual command, MCP SDK and model adapter.
// Only the external knowledge and model services are simulated.
type scanContextFixture struct {
	mu          sync.Mutex
	plans       []string
	mains       []string
	mcpRequests atomic.Int32
	lookups     atomic.Int32
	denied      atomic.Int32
	failClose   atomic.Bool
	beforeMain  func()
	mainOnce    sync.Once
	configPath  string
	servers     map[string]MCPServerConfig
}

func newScanContextFixture(t *testing.T) (*scanContextFixture, scanOptions) {
	t.Helper()
	home := freshOCRHome(t)
	f := &scanContextFixture{configPath: filepath.Join(home, ".opencodereview", "config.json")}
	repo := initTestGitRepo(t)
	gitCommitFile(t, repo, "order.go", "package orders\nfunc Cancel() {}\n", "add order")
	server := sdkmcp.NewServer(&sdkmcp.Implementation{Name: "scan-knowledge", Version: "test"}, nil)
	for _, name := range []string{"lookup_requirement", "private_lookup", "file_read_diff"} {
		server.AddTool(&sdkmcp.Tool{Name: name, Description: "Find the original requirement by query", InputSchema: map[string]any{
			"type": "object", "properties": map[string]any{"query": map[string]any{"type": "string", "description": "Business topic"}}, "required": []string{"query"},
		}}, func(_ context.Context, req *sdkmcp.CallToolRequest) (*sdkmcp.CallToolResult, error) {
			if req.Params.Name != "lookup_requirement" {
				f.denied.Add(1)
			} else {
				f.lookups.Add(1)
			}
			var args struct {
				Query string `json:"query"`
			}
			if err := json.Unmarshal(req.Params.Arguments, &args); err != nil || args.Query != "cancel" {
				return nil, fmt.Errorf("unexpected query: %s", req.Params.Arguments)
			}
			return &sdkmcp.CallToolResult{Content: []sdkmcp.Content{&sdkmcp.TextContent{Text: scanEvidence}}}, nil
		})
	}
	handler := sdkmcp.NewStreamableHTTPHandler(func(*http.Request) *sdkmcp.Server { return server }, nil)
	knowledge := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.mcpRequests.Add(1)
		if r.Method == http.MethodDelete && f.failClose.Load() {
			conn, _, err := w.(http.Hijacker).Hijack()
			if err == nil {
				_ = conn.Close()
			}
			return
		}
		handler.ServeHTTP(w, r)
	}))
	t.Cleanup(knowledge.Close)
	f.servers = map[string]MCPServerConfig{"knowledge": {Type: "remote", URL: knowledge.URL, Tools: []string{"lookup_requirement", "file_read_diff"}}}
	f.writeConfig(t)
	model := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, err.Error(), 400)
			return
		}
		var request struct {
			Tools []struct {
				Name string `json:"name"`
			} `json:"tools"`
		}
		if err := json.Unmarshal(raw, &request); err != nil {
			http.Error(w, err.Error(), 400)
			return
		}
		isPlan := len(request.Tools) == 0
		if !isPlan {
			f.mainOnce.Do(func() {
				if f.beforeMain != nil {
					f.beforeMain()
				}
			})
		}
		f.mu.Lock()
		if isPlan {
			f.plans = append(f.plans, string(raw))
		} else {
			f.mains = append(f.mains, string(raw))
		}
		f.mu.Unlock()
		var content []map[string]any
		stop := "tool_use"
		if isPlan {
			content = []map[string]any{{"type": "text", "text": `{"summary":"Order cancellation","checkpoints":[{"focus":"cancellation state","why":"Use lookup_requirement with query=cancel to verify permitted states"}]}`}}
			stop = "end_turn"
		} else {
			hasLookup := false
			for _, tool := range request.Tools {
				if tool.Name == "lookup_requirement" {
					hasLookup = true
				}
			}
			name, args := "task_done", map[string]any{"state": "DONE"}
			if hasLookup && !strings.Contains(string(raw), scanEvidence) {
				name, args = "lookup_requirement", map[string]any{"query": "cancel"}
			}
			content = []map[string]any{{"type": "tool_use", "id": "scan_call", "name": name, "input": args}}
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"id": "scan_msg", "type": "message", "role": "assistant", "model": "claude-test", "content": content, "stop_reason": stop, "usage": map[string]int{"input_tokens": 10, "output_tokens": 5}})
	}))
	t.Cleanup(model.Close)
	t.Setenv("OCR_LLM_URL", model.URL+"/v1/messages")
	t.Setenv("OCR_LLM_TOKEN", "test-token")
	t.Setenv("OCR_LLM_MODEL", "claude-test")
	t.Setenv("OCR_LLM_PROTOCOL", "anthropic")
	t.Setenv("OCR_LLM_AUTH_HEADER", "x-api-key")
	t.Setenv("OCR_LLM_TIMEOUT", "10")
	opts, err := parseScanFlags([]string{"--repo", repo, "--format", "json", "--audience", "agent", "--no-dedup", "--no-summary", "--concurrency", "2"})
	if err != nil {
		t.Fatal(err)
	}
	return f, opts
}

func (f *scanContextFixture) writeConfig(t *testing.T) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(f.configPath), 0o700); err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(map[string]any{"mcp_servers": f.servers})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(f.configPath, data, 0o600); err != nil {
		t.Fatal(err)
	}
}

func runScanContext(t *testing.T, opts scanOptions) (string, string, error) {
	t.Helper()
	var output string
	var err error
	diagnostic := captureStderr(t, func() { output = captureStdout(t, func() { err = executeScan(opts) }) })
	return output, diagnostic, err
}

func TestScanBackgroundFlags(t *testing.T) {
	for _, flag := range []string{"--background-file", "-B"} {
		opts, err := parseScanFlags([]string{flag, "docs/requirements.md", "-b", "inline"})
		if err != nil || opts.backgroundFile != "docs/requirements.md" || opts.background != "inline" {
			t.Fatalf("%s: opts=%+v err=%v", flag, opts, err)
		}
	}
}

func TestScanContextMCPAndBackground(t *testing.T) {
	f, opts := newScanContextFixture(t)
	gitCommitFile(t, opts.repoDir, "stock.go", "package orders\nfunc Release() {}\n", "add stock")
	opts.backgroundFile, opts.background = "requirements.md", "ignored-inline-background"
	const background = "Orders use reservation state transitions."
	if err := os.WriteFile(filepath.Join(opts.repoDir, opts.backgroundFile), []byte(background), 0o600); err != nil {
		t.Fatal(err)
	}
	output, diagnostic, err := runScanContext(t, opts)
	if err != nil {
		t.Fatal(err)
	}
	if !json.Valid([]byte(output)) {
		t.Fatalf("invalid JSON stdout: %s", output)
	}
	if !strings.Contains(diagnostic, "takes precedence") || !strings.Contains(diagnostic, "conflicts with built-in") {
		t.Fatalf("missing precedence or conflict diagnostic: %s", diagnostic)
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.plans) != 2 || len(f.mains) != 4 || f.lookups.Load() != 2 || f.denied.Load() != 0 {
		t.Fatalf("plans=%d mains=%d lookups=%d denied=%d", len(f.plans), len(f.mains), f.lookups.Load(), f.denied.Load())
	}
	for _, raw := range append(append([]string{}, f.plans...), f.mains...) {
		if !strings.Contains(raw, background) || strings.Contains(raw, opts.background) {
			t.Fatalf("wrong background: %s", raw)
		}
		if strings.Contains(raw, "private_lookup") || strings.Contains(raw, "file_read_diff") {
			t.Fatalf("unavailable tool exposed: %s", raw)
		}
	}
	for _, raw := range f.plans {
		if !strings.Contains(raw, "lookup_requirement") || !strings.Contains(raw, "Business topic") || strings.Contains(raw, scanEvidence) {
			t.Fatalf("plan must describe tools without executing them: %s", raw)
		}
	}
	if !strings.Contains(strings.Join(f.mains, "\n"), "query=cancel") || !strings.Contains(strings.Join(f.mains, "\n"), scanEvidence) {
		t.Fatal("main lost planning advice or MCP result")
	}
	sessions, err := session.ListSessions(opts.repoDir)
	if err != nil || len(sessions) != 1 {
		t.Fatalf("sessions=%v err=%v", sessions, err)
	}
	data, err := os.ReadFile(sessions[0].FilePath)
	if err != nil || !strings.Contains(string(data), scanEvidence) || !strings.Contains(string(data), "tool_call") {
		t.Fatalf("MCP evidence not persisted: %v", err)
	}
}

func TestScanPreviewContextHasNoExternalEffects(t *testing.T) {
	f, opts := newScanContextFixture(t)
	marker := filepath.Join(opts.repoDir, "setup-ran")
	f.servers["setup"] = MCPServerConfig{Command: "missing-command", Setup: "touch setup-ran"}
	f.writeConfig(t)
	opts.preview = true
	for _, tc := range []struct {
		name, body string
		missing    bool
		wantErr    bool
	}{
		{name: "valid", body: "business context"}, {name: "empty", wantErr: true}, {name: "missing", missing: true, wantErr: true},
		{name: "reserved", body: "<ocr_user_background>", wantErr: true}, {name: "characters", body: strings.Repeat("a", 8001), wantErr: true},
		{name: "bytes", body: strings.Repeat("a", (1<<20)+1), wantErr: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			opts.backgroundFile = filepath.Join(t.TempDir(), "context.md")
			if !tc.missing {
				if err := os.WriteFile(opts.backgroundFile, []byte(tc.body), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			output, _, err := runScanContext(t, opts)
			if (err != nil) != tc.wantErr {
				t.Fatalf("err=%v wantErr=%v", err, tc.wantErr)
			}
			if !tc.wantErr && !json.Valid([]byte(output)) {
				t.Fatalf("invalid preview: %s", output)
			}
		})
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.plans)+len(f.mains) != 0 || f.mcpRequests.Load() != 0 {
		t.Fatal("preview contacted external services")
	}
	if _, err := os.Stat(marker); !os.IsNotExist(err) {
		t.Fatal("preview executed setup")
	}
	sessions, err := session.ListSessions(opts.repoDir)
	if err != nil || len(sessions) != 0 {
		t.Fatalf("preview created sessions: %v %v", sessions, err)
	}
}

func TestScanContextNoPlanAndMCPFailures(t *testing.T) {
	for _, mode := range []string{"no-plan", "unconfigured", "connect-failed", "setup-failed"} {
		t.Run(mode, func(t *testing.T) {
			f, opts := newScanContextFixture(t)
			opts.background = "Inline business context."
			opts.noPlan = mode == "no-plan"
			switch mode {
			case "unconfigured":
				f.servers = nil
			case "connect-failed":
				f.servers = map[string]MCPServerConfig{"broken": {Type: "remote", URL: "http://127.0.0.1:1/mcp"}}
			case "setup-failed":
				f.servers = map[string]MCPServerConfig{"broken": {Command: "missing-command", Setup: "exit 1"}}
			}
			f.writeConfig(t)
			output, diagnostic, err := runScanContext(t, opts)
			if err != nil || !json.Valid([]byte(output)) {
				t.Fatalf("output=%s err=%v", output, err)
			}
			if strings.HasSuffix(mode, "failed") && !strings.Contains(diagnostic, "failed") {
				t.Fatalf("missing failure warning: %s", diagnostic)
			}
			f.mu.Lock()
			defer f.mu.Unlock()
			if (len(f.plans) == 0) != opts.noPlan {
				t.Fatalf("plans=%d noPlan=%v", len(f.plans), opts.noPlan)
			}
			if (f.lookups.Load() > 0) != (mode == "no-plan") {
				t.Fatalf("lookups=%d", f.lookups.Load())
			}
			for _, raw := range f.mains {
				if !strings.Contains(raw, opts.background) {
					t.Fatal("main lost inline background")
				}
			}
		})
	}
}

func TestScanContextResumeWarning(t *testing.T) {
	for _, mode := range []string{"plain", "background", "mcp"} {
		t.Run(mode, func(t *testing.T) {
			f, opts := newScanContextFixture(t)
			if mode != "mcp" {
				f.servers = nil
				f.writeConfig(t)
			}
			if mode == "background" {
				opts.background = "Changed requirements"
			}
			opts.resume = writeScanResumeSession(t, opts.repoDir, "order.go")
			_, diagnostic, err := runScanContext(t, opts)
			if err != nil {
				t.Fatal(err)
			}
			if strings.Contains(diagnostic, "does not validate background") != (mode != "plain") {
				t.Fatalf("wrong warning: %s", diagnostic)
			}
		})
	}
}

func TestScanOutputBeforeMCPShutdown(t *testing.T) {
	_, opts := newScanContextFixture(t)
	opts.outputPath = filepath.Join(t.TempDir(), "scan.json")
	original := closeReviewMCPClients
	t.Cleanup(func() { closeReviewMCPClients = original })
	observed := false
	closeReviewMCPClients = func(clients []*ocrmcp.Client) {
		observed = true
		data, err := os.ReadFile(opts.outputPath)
		if err != nil || !json.Valid(data) {
			t.Errorf("report unavailable before shutdown: %s %v", data, err)
		}
		if len(clients) != 1 {
			t.Errorf("clients=%d want 1", len(clients))
		}
		time.Sleep(10 * time.Millisecond)
		original(clients)
	}
	_, _, err := runScanContext(t, opts)
	if err != nil || !observed {
		t.Fatalf("err=%v shutdown=%v", err, observed)
	}
}

func TestScanMCPShutdownFailurePreservesReport(t *testing.T) {
	f, opts := newScanContextFixture(t)
	f.failClose.Store(true)
	opts.outputPath = filepath.Join(t.TempDir(), "scan.json")
	_, diagnostic, err := runScanContext(t, opts)
	if err != nil {
		t.Fatalf("MCP shutdown must not discard a successful scan: %v", err)
	}
	if !strings.Contains(diagnostic, `close MCP server "knowledge"`) {
		t.Fatalf("missing shutdown warning: %s", diagnostic)
	}
	data, err := os.ReadFile(opts.outputPath)
	if err != nil || !json.Valid(data) {
		t.Fatalf("report lost after shutdown failure: %s %v", data, err)
	}
}

func TestScanOutputFailureClosesMCP(t *testing.T) {
	f, opts := newScanContextFixture(t)
	opts.outputPath = filepath.Join(t.TempDir(), "scan.json")
	// Make publication fail after validation and MCP initialization have succeeded.
	f.beforeMain = func() {
		if err := os.Mkdir(opts.outputPath, 0o700); err != nil {
			t.Errorf("create publication obstacle: %v", err)
		}
	}
	original := closeReviewMCPClients
	t.Cleanup(func() { closeReviewMCPClients = original })
	closed := false
	closeReviewMCPClients = func(clients []*ocrmcp.Client) {
		closed = len(clients) == 1
		original(clients)
	}
	_, _, err := runScanContext(t, opts)
	if err == nil || !closed {
		t.Fatalf("err=%v MCP closed=%v", err, closed)
	}
}

func TestScanInvalidBackgroundPreservesOutput(t *testing.T) {
	_, opts := newScanContextFixture(t)
	opts.outputPath = filepath.Join(t.TempDir(), "scan.json")
	opts.backgroundFile = "missing.md"
	if err := os.WriteFile(opts.outputPath, []byte("previous report"), 0o600); err != nil {
		t.Fatal(err)
	}
	_, _, err := runScanContext(t, opts)
	if err == nil {
		t.Fatal("expected background error")
	}
	data, err := os.ReadFile(opts.outputPath)
	if err != nil || string(data) != "previous report" {
		t.Fatalf("report overwritten: %s %v", data, err)
	}
}
