const std = @import("std");
const Allocator = std.mem.Allocator;
const Provider = @import("../Provider.zig");
const types = @import("../types.zig");
const errors = @import("../errors.zig");
const ProviderError = errors.ProviderError;
const http = @import("../http.zig");
const sse = @import("../sse.zig");
const jh = @import("../json_helpers.zig");

const Anthropic = @This();

allocator: Allocator,
api_key: []const u8,
api_base: []const u8,
last_error: ?errors.ErrorDetails = null,

pub const Config = struct {
    api_key: []const u8,
    api_base: ?[]const u8 = null,
};

const default_api_base = "https://api.anthropic.com";
const api_version = "2023-06-01";
const messages_path = "/v1/messages";

pub fn init(allocator: Allocator, config: Config) ProviderError!*Anthropic {
    const self = allocator.create(Anthropic) catch return error.OutOfMemory;
    self.* = .{
        .allocator = allocator,
        .api_key = config.api_key,
        .api_base = config.api_base orelse default_api_base,
    };
    return self;
}

pub fn deinit(self: *Anthropic) void {
    self.clearLastError();
    self.allocator.destroy(self);
}

pub fn lastError(self: *const Anthropic) ?[]const u8 {
    if (self.last_error) |details| {
        return details.message;
    }
    return null;
}

pub fn lastErrorDetails(self: *const Anthropic) ?errors.ErrorDetails {
    return self.last_error;
}

fn clearLastError(self: *Anthropic) void {
    if (self.last_error) |*details| {
        details.deinit();
        self.last_error = null;
    }
}

pub fn provider(self: *Anthropic) Provider {
    return Provider.init(self);
}

// --- Provider interface implementation ---

pub fn complete(self: *Anthropic, request: Provider.CompletionRequest, allocator: Allocator) ProviderError!Provider.CompletionResponse {
    self.clearLastError();

    const body = buildRequestBody(request, false, allocator) catch return error.OutOfMemory;
    defer allocator.free(body);

    var resp = try doPost(self, body, allocator);
    defer resp.deinit();

    if (http.mapStatusError(resp.status)) |err| {
        return self.mapApiError(resp.body) orelse err;
    }

    return self.parseCompletionResponseWithErrorCapture(resp.body, allocator);
}

pub fn stream(self: *Anthropic, request: Provider.CompletionRequest, allocator: Allocator) ProviderError!Provider.StreamIterator {
    self.clearLastError();

    const body = buildRequestBody(request, true, allocator) catch return error.OutOfMemory;
    defer allocator.free(body);

    const url = std.fmt.allocPrint(allocator, "{s}{s}", .{ self.api_base, messages_path }) catch return error.OutOfMemory;
    defer allocator.free(url);

    const http_stream = try http.openStream(allocator, url, &.{
        .{ .name = "x-api-key", .value = self.api_key },
        .{ .name = "anthropic-version", .value = api_version },
    }, body);
    errdefer http_stream.deinit();

    const ctx = allocator.create(StreamContext) catch return error.OutOfMemory;
    ctx.* = .{
        .allocator = allocator,
        .http_stream = http_stream,
        .sse_reader = sse.SseLineReader.init(http_stream.reader(), allocator),
        .done = false,
        .current_block_type = null,
        .current_tool_id = null,
    };

    return Provider.StreamIterator.initFrom(ctx);
}

pub fn listModels(_: *Anthropic, allocator: Allocator) ProviderError![]const Provider.ModelInfo {
    const models = allocator.alloc(Provider.ModelInfo, known_models.len) catch return error.OutOfMemory;
    @memcpy(models, &known_models);
    return models;
}

// --- HTTP ---

fn doPost(self: *Anthropic, body: []const u8, allocator: Allocator) ProviderError!http.HttpResponse {
    const url = std.fmt.allocPrint(allocator, "{s}{s}", .{ self.api_base, messages_path }) catch return error.OutOfMemory;
    defer allocator.free(url);

    return http.post(allocator, url, &.{
        .{ .name = "x-api-key", .value = self.api_key },
        .{ .name = "anthropic-version", .value = api_version },
    }, body);
}

// --- Request building ---

fn buildRequestBody(request: Provider.CompletionRequest, do_stream: bool, allocator: Allocator) ![]u8 {
    var buf: std.ArrayList(u8) = .{};
    errdefer buf.deinit(allocator);
    var jw = jh.jsonWriter(buf.writer(allocator));

    try jw.beginObject();

    try jw.field("model");
    try jw.valueString(request.model);

    try jw.field("max_tokens");
    try jw.valueInt(@intCast(request.max_tokens));

    if (do_stream) {
        try jw.field("stream");
        try jw.valueBool(true);
    }

    if (request.temperature) |temp| {
        try jw.field("temperature");
        try jw.valueFloat(@floatCast(temp));
    }

    if (request.top_p) |top_p| {
        try jw.field("top_p");
        try jw.valueFloat(@floatCast(top_p));
    }

    if (request.top_k) |top_k| {
        try jw.field("top_k");
        try jw.valueInt(@intCast(top_k));
    }

    if (request.system_prompt) |sys| {
        try jw.field("system");
        try jw.valueString(sys);
    }

    if (request.stop_sequences.len > 0) {
        try jw.field("stop_sequences");
        try jw.beginArray();
        for (request.stop_sequences) |seq| {
            try jw.valueString(seq);
        }
        try jw.endArray();
    }

    if (request.tools.len > 0) {
        try jw.field("tools");
        try jw.beginArray();
        for (request.tools) |tool| {
            try jw.beginObject();
            try jw.field("name");
            try jw.valueString(tool.name);
            try jw.field("description");
            try jw.valueString(tool.description);
            try jw.field("input_schema");
            try jw.valueRaw(tool.input_schema);
            try jw.endObject();
        }
        try jw.endArray();
    }

    // Messages
    try jw.field("messages");
    try jw.beginArray();
    for (request.messages) |msg| {
        if (msg.role == .system) continue;
        try writeMessage(&jw, msg);
    }
    try jw.endArray();

    try jw.endObject();

    return buf.toOwnedSlice(allocator);
}

fn writeMessage(jw: anytype, msg: types.Message) !void {
    try jw.beginObject();

    try jw.field("role");
    // Anthropic uses "user" for tool results
    try jw.valueString(if (msg.role == .tool) "user" else msg.role.toString());

    try jw.field("content");
    if (msg.content.len == 1) {
        switch (msg.content[0]) {
            .text => |t| try jw.valueString(t.text),
            else => {
                try jw.beginArray();
                try writeContentBlock(jw, msg.content[0]);
                try jw.endArray();
            },
        }
    } else {
        try jw.beginArray();
        for (msg.content) |block| {
            try writeContentBlock(jw, block);
        }
        try jw.endArray();
    }

    try jw.endObject();
}

fn writeContentBlock(jw: anytype, block: types.ContentBlock) !void {
    switch (block) {
        .text => |t| {
            try jw.beginObject();
            try jw.field("type");
            try jw.valueString("text");
            try jw.field("text");
            try jw.valueString(t.text);
            try jw.endObject();
        },
        .image => |img| {
            try jw.beginObject();
            try jw.field("type");
            try jw.valueString("image");
            try jw.field("source");
            try jw.beginObject();
            try jw.field("type");
            try jw.valueString(switch (img.source_type) {
                .base64 => "base64",
                .url => "url",
            });
            try jw.field("data");
            try jw.valueString(img.data);
            if (img.media_type) |mt| {
                try jw.field("media_type");
                try jw.valueString(mt);
            }
            try jw.endObject();
            try jw.endObject();
        },
        .tool_use => |tu| {
            try jw.beginObject();
            try jw.field("type");
            try jw.valueString("tool_use");
            try jw.field("id");
            try jw.valueString(tu.id);
            try jw.field("name");
            try jw.valueString(tu.name);
            try jw.field("input");
            try jw.valueRaw(if (tu.input_json.len == 0) "{}" else tu.input_json);
            try jw.endObject();
        },
        .tool_result => |tr| {
            try jw.beginObject();
            try jw.field("type");
            try jw.valueString("tool_result");
            try jw.field("tool_use_id");
            try jw.valueString(tr.tool_use_id);
            try jw.field("content");
            try jw.valueString(tr.content);
            if (tr.is_error) {
                try jw.field("is_error");
                try jw.valueBool(true);
            }
            try jw.endObject();
        },
        .thinking => |th| {
            try jw.beginObject();
            try jw.field("type");
            try jw.valueString("thinking");
            try jw.field("thinking");
            try jw.valueString(th.text);
            try jw.endObject();
        },
    }
}

// --- Response parsing ---

fn parseCompletionResponse(body: []const u8, allocator: Allocator) ProviderError!Provider.CompletionResponse {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch return error.InvalidResponse;
    defer parsed.deinit();
    const root = parsed.value;

    // Check for API error in response body
    if (jh.getPath(root, "error.type")) |_| {
        return classifyApiError(root) orelse error.ProviderError;
    }

    return parseCompletionFromParsedJson(root, allocator);
}

fn parseCompletionFromParsedJson(root: std.json.Value, allocator: Allocator) ProviderError!Provider.CompletionResponse {
    const model_str = jh.getJsonString(jh.getPath(root, "model") orelse return error.InvalidResponse) orelse return error.InvalidResponse;
    const model = allocator.dupe(u8, model_str) catch return error.OutOfMemory;
    errdefer allocator.free(model);

    const stop_str = jh.getJsonString(jh.getPath(root, "stop_reason") orelse return error.InvalidResponse) orelse "unknown";
    const stop_reason = types.StopReason.fromString(stop_str);

    // Parse usage
    const usage = parseUsage(jh.getPath(root, "usage"));

    // Parse content blocks
    const content_val = jh.getPath(root, "content") orelse return error.InvalidResponse;
    const content_arr = jh.getJsonArray(content_val) orelse return error.InvalidResponse;

    var content_blocks: std.ArrayList(types.ContentBlock) = .{};
    defer content_blocks.deinit(allocator);

    for (content_arr.items) |item| {
        const block = parseContentBlock(item, allocator) catch continue;
        content_blocks.append(allocator, block) catch return error.OutOfMemory;
    }

    const content = content_blocks.toOwnedSlice(allocator) catch return error.OutOfMemory;

    return .{
        .message = .{
            .role = .assistant,
            .content = content,
            .allocator = allocator,
        },
        .usage = usage,
        .model = model,
        .stop_reason = stop_reason,
        .allocator = allocator,
    };
}

fn parseContentBlock(value: std.json.Value, allocator: Allocator) ProviderError!types.ContentBlock {
    const obj = jh.getJsonObject(value) orelse return error.InvalidResponse;
    const type_str = jh.getJsonString(obj.get("type") orelse return error.InvalidResponse) orelse return error.InvalidResponse;

    if (std.mem.eql(u8, type_str, "text")) {
        const text = jh.getJsonString(obj.get("text") orelse return error.InvalidResponse) orelse return error.InvalidResponse;
        return .{ .text = .{ .text = allocator.dupe(u8, text) catch return error.OutOfMemory } };
    } else if (std.mem.eql(u8, type_str, "tool_use")) {
        const id = jh.getJsonString(obj.get("id") orelse return error.InvalidResponse) orelse return error.InvalidResponse;
        const name = jh.getJsonString(obj.get("name") orelse return error.InvalidResponse) orelse return error.InvalidResponse;
        const input_val = obj.get("input") orelse return error.InvalidResponse;
        const input_json = jh.stringifyValue(allocator, input_val) catch return error.OutOfMemory;

        return .{ .tool_use = .{
            .id = allocator.dupe(u8, id) catch return error.OutOfMemory,
            .name = allocator.dupe(u8, name) catch return error.OutOfMemory,
            .input_json = input_json,
        } };
    } else if (std.mem.eql(u8, type_str, "thinking")) {
        const text = jh.getJsonString(obj.get("thinking") orelse return error.InvalidResponse) orelse return error.InvalidResponse;
        return .{ .thinking = .{ .text = allocator.dupe(u8, text) catch return error.OutOfMemory } };
    }

    return error.InvalidResponse;
}

fn parseUsage(value: ?std.json.Value) types.TokenUsage {
    const v = value orelse return .{};
    return .{
        .input_tokens = getOptionalInt(v, "input_tokens"),
        .output_tokens = getOptionalInt(v, "output_tokens"),
        .cache_read_tokens = getOptionalInt(v, "cache_read_input_tokens"),
        .cache_creation_tokens = getOptionalInt(v, "cache_creation_input_tokens"),
    };
}

fn getOptionalInt(value: std.json.Value, path: []const u8) u32 {
    const v = jh.getPath(value, path) orelse return 0;
    return intToU32(jh.getJsonInt(v));
}

fn intToU32(v: ?i64) u32 {
    const i = v orelse return 0;
    if (i < 0) return 0;
    return @intCast(@min(i, std.math.maxInt(u32)));
}

// --- Error mapping ---

fn mapApiError(self: *Anthropic, body: []const u8) ?ProviderError {
    const parsed = std.json.parseFromSlice(std.json.Value, std.heap.page_allocator, body, .{}) catch return null;
    defer parsed.deinit();
    return self.mapApiErrorFromJson(parsed.value);
}

fn mapApiErrorFromJson(self: *Anthropic, root: std.json.Value) ?ProviderError {
    const err_type = jh.getJsonString(jh.getPath(root, "error.type") orelse return null) orelse return null;
    const err_message = if (jh.getPath(root, "error.message")) |msg_val| (jh.getJsonString(msg_val) orelse "unknown error") else "unknown error";

    self.storeErrorDetails(err_type, err_message);

    return classifyErrorType(err_type);
}

fn classifyApiError(root: std.json.Value) ?ProviderError {
    const err_type = jh.getJsonString(jh.getPath(root, "error.type") orelse return null) orelse return null;
    return classifyErrorType(err_type);
}

fn classifyErrorType(err_type: []const u8) ProviderError {
    if (std.mem.eql(u8, err_type, "authentication_error")) return error.AuthenticationFailed;
    if (std.mem.eql(u8, err_type, "rate_limit_error")) return error.RateLimited;
    if (std.mem.eql(u8, err_type, "invalid_request_error")) return error.InvalidRequest;
    if (std.mem.eql(u8, err_type, "overloaded_error")) return error.Overloaded;
    if (std.mem.eql(u8, err_type, "not_found_error")) return error.ModelNotFound;
    if (std.mem.eql(u8, err_type, "permission_error")) return error.AuthenticationFailed;

    return error.ProviderError;
}

fn storeErrorDetails(self: *Anthropic, err_type: []const u8, err_message: []const u8) void {
    self.clearLastError();
    const duped_message = self.allocator.dupe(u8, err_message) catch return;
    const duped_type = self.allocator.dupe(u8, err_type) catch {
        self.allocator.free(duped_message);
        return;
    };
    self.last_error = .{
        .message = duped_message,
        .error_type = duped_type,
        .allocator = self.allocator,
    };
}

fn parseCompletionResponseWithErrorCapture(self: *Anthropic, body: []const u8, allocator: Allocator) ProviderError!Provider.CompletionResponse {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch return error.InvalidResponse;
    defer parsed.deinit();
    const root = parsed.value;

    // Check for API error in response body
    if (jh.getPath(root, "error.type")) |_| {
        return self.mapApiErrorFromJson(root) orelse error.ProviderError;
    }

    return parseCompletionFromParsedJson(root, allocator);
}

// --- Stream context ---

const BlockType = enum { text, thinking, tool_use };

const StreamContext = struct {
    allocator: Allocator,
    http_stream: *http.HttpStream,
    sse_reader: sse.SseLineReader,
    done: bool,
    current_block_type: ?BlockType,
    current_tool_id: ?[]const u8,

    pub fn next(self: *StreamContext) ProviderError!?types.StreamEvent {
        while (!self.done) {
            const event = (try self.sse_reader.nextEvent()) orelse return null;

            const event_type = event.event_type orelse continue;

            if (std.mem.eql(u8, event_type, "message_start")) {
                return try self.parseMessageStart(event.data);
            } else if (std.mem.eql(u8, event_type, "content_block_start")) {
                return try self.parseContentBlockStart(event.data);
            } else if (std.mem.eql(u8, event_type, "content_block_delta")) {
                return try self.parseContentBlockDelta(event.data);
            } else if (std.mem.eql(u8, event_type, "content_block_stop")) {
                self.current_block_type = null;
                return .content_block_stop;
            } else if (std.mem.eql(u8, event_type, "message_delta")) {
                return try self.parseMessageDelta(event.data);
            } else if (std.mem.eql(u8, event_type, "message_stop")) {
                self.done = true;
                return .message_stop;
            } else if (std.mem.eql(u8, event_type, "error")) {
                return error.ProviderError;
            }
            // Skip ping and unknown events
        }
        return null;
    }

    fn parseMessageStart(self: *StreamContext, data: []const u8) ProviderError!types.StreamEvent {
        const parsed = std.json.parseFromSlice(std.json.Value, self.allocator, data, .{}) catch return error.InvalidResponse;
        defer parsed.deinit();

        const msg_id = if (jh.getPath(parsed.value, "message.id")) |v|
            (if (jh.getJsonString(v)) |s| self.allocator.dupe(u8, s) catch return error.OutOfMemory else null)
        else
            null;

        const model = if (jh.getPath(parsed.value, "message.model")) |v|
            (if (jh.getJsonString(v)) |s| self.allocator.dupe(u8, s) catch return error.OutOfMemory else null)
        else
            null;

        return .{ .message_start = .{ .message_id = msg_id, .model = model } };
    }

    fn parseContentBlockStart(self: *StreamContext, data: []const u8) ProviderError!types.StreamEvent {
        const parsed = std.json.parseFromSlice(std.json.Value, self.allocator, data, .{}) catch return error.InvalidResponse;
        defer parsed.deinit();

        const block_type = jh.getJsonString(jh.getPath(parsed.value, "content_block.type") orelse return error.InvalidResponse) orelse return error.InvalidResponse;

        if (std.mem.eql(u8, block_type, "tool_use")) {
            self.current_block_type = .tool_use;
            const id = jh.getJsonString(jh.getPath(parsed.value, "content_block.id") orelse return error.InvalidResponse) orelse return error.InvalidResponse;
            const name = jh.getJsonString(jh.getPath(parsed.value, "content_block.name") orelse return error.InvalidResponse) orelse return error.InvalidResponse;

            const id_dup = self.allocator.dupe(u8, id) catch return error.OutOfMemory;
            errdefer self.allocator.free(id_dup);
            // Also store for tracking
            if (self.current_tool_id) |old| self.allocator.free(old);
            self.current_tool_id = self.allocator.dupe(u8, id) catch return error.OutOfMemory;

            return .{ .tool_use_start = .{
                .id = id_dup,
                .name = self.allocator.dupe(u8, name) catch return error.OutOfMemory,
            } };
        } else if (std.mem.eql(u8, block_type, "thinking")) {
            self.current_block_type = .thinking;
            // Return a thinking delta with empty text to signal start
            return .{ .thinking_delta = .{ .text = self.allocator.dupe(u8, "") catch return error.OutOfMemory } };
        } else {
            self.current_block_type = .text;
            // Return a text delta with empty text to signal start
            return .{ .text_delta = .{ .text = self.allocator.dupe(u8, "") catch return error.OutOfMemory } };
        }
    }

    fn parseContentBlockDelta(self: *StreamContext, data: []const u8) ProviderError!types.StreamEvent {
        const parsed = std.json.parseFromSlice(std.json.Value, self.allocator, data, .{}) catch return error.InvalidResponse;
        defer parsed.deinit();

        const delta_type = jh.getJsonString(jh.getPath(parsed.value, "delta.type") orelse return error.InvalidResponse) orelse return error.InvalidResponse;

        if (std.mem.eql(u8, delta_type, "text_delta")) {
            const text = jh.getJsonString(jh.getPath(parsed.value, "delta.text") orelse return error.InvalidResponse) orelse return error.InvalidResponse;
            return .{ .text_delta = .{ .text = self.allocator.dupe(u8, text) catch return error.OutOfMemory } };
        } else if (std.mem.eql(u8, delta_type, "thinking_delta")) {
            const text = jh.getJsonString(jh.getPath(parsed.value, "delta.thinking") orelse return error.InvalidResponse) orelse return error.InvalidResponse;
            return .{ .thinking_delta = .{ .text = self.allocator.dupe(u8, text) catch return error.OutOfMemory } };
        } else if (std.mem.eql(u8, delta_type, "input_json_delta")) {
            const json_str = jh.getJsonString(jh.getPath(parsed.value, "delta.partial_json") orelse return error.InvalidResponse) orelse return error.InvalidResponse;
            return .{ .tool_input_delta = .{ .json = self.allocator.dupe(u8, json_str) catch return error.OutOfMemory } };
        }

        return error.InvalidResponse;
    }

    fn parseMessageDelta(self: *StreamContext, data: []const u8) ProviderError!types.StreamEvent {
        const parsed = std.json.parseFromSlice(std.json.Value, self.allocator, data, .{}) catch return error.InvalidResponse;
        defer parsed.deinit();

        const stop_reason = if (jh.getPath(parsed.value, "delta.stop_reason")) |v|
            (if (jh.getJsonString(v)) |s| self.allocator.dupe(u8, s) catch return error.OutOfMemory else null)
        else
            null;

        const usage = parseUsage(jh.getPath(parsed.value, "usage"));

        return .{ .message_delta = .{ .stop_reason = stop_reason, .usage = usage } };
    }

    pub fn deinit(self: *StreamContext) void {
        if (self.current_tool_id) |id| self.allocator.free(id);
        self.sse_reader.deinit();
        self.http_stream.deinit();
        self.allocator.destroy(self);
    }
};

// --- Known models ---

const known_models = [_]Provider.ModelInfo{
    .{
        .id = "claude-opus-4-20250514",
        .display_name = "Claude Opus 4",
        .context_window = 200000,
        .max_output_tokens = 32000,
        .supports_tools = true,
        .supports_vision = true,
        .supports_streaming = true,
    },
    .{
        .id = "claude-sonnet-4-20250514",
        .display_name = "Claude Sonnet 4",
        .context_window = 200000,
        .max_output_tokens = 16000,
        .supports_tools = true,
        .supports_vision = true,
        .supports_streaming = true,
    },
    .{
        .id = "claude-3-5-sonnet-20241022",
        .display_name = "Claude 3.5 Sonnet",
        .context_window = 200000,
        .max_output_tokens = 8192,
        .supports_tools = true,
        .supports_vision = true,
        .supports_streaming = true,
    },
    .{
        .id = "claude-3-5-haiku-20241022",
        .display_name = "Claude 3.5 Haiku",
        .context_window = 200000,
        .max_output_tokens = 8192,
        .supports_tools = true,
        .supports_vision = false,
        .supports_streaming = true,
    },
    .{
        .id = "claude-3-haiku-20240307",
        .display_name = "Claude 3 Haiku",
        .context_window = 200000,
        .max_output_tokens = 4096,
        .supports_tools = true,
        .supports_vision = true,
        .supports_streaming = true,
    },
};

// --- Tests ---

test "buildRequestBody basic" {
    const allocator = std.testing.allocator;

    var content = try allocator.alloc(types.ContentBlock, 1);
    defer allocator.free(content);
    content[0] = .{ .text = .{ .text = "Hello" } };

    const msgs = try allocator.alloc(types.Message, 1);
    defer allocator.free(msgs);
    msgs[0] = .{
        .role = .user,
        .content = content,
        .allocator = allocator,
    };

    const body = try buildRequestBody(.{
        .model = "claude-sonnet-4-20250514",
        .messages = msgs,
        .system_prompt = "You are helpful.",
    }, false, allocator);
    defer allocator.free(body);

    // Verify it's valid JSON
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
    defer parsed.deinit();

    const model = jh.getJsonString(jh.getPath(parsed.value, "model").?).?;
    try std.testing.expectEqualStrings("claude-sonnet-4-20250514", model);

    const sys = jh.getJsonString(jh.getPath(parsed.value, "system").?).?;
    try std.testing.expectEqualStrings("You are helpful.", sys);
}

test "buildRequestBody with tools" {
    const allocator = std.testing.allocator;

    const msgs = try allocator.alloc(types.Message, 0);
    defer allocator.free(msgs);

    const tools = [_]types.ToolDefinition{
        .{
            .name = "get_weather",
            .description = "Get weather for a city",
            .input_schema = "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}}}",
        },
    };

    const body = try buildRequestBody(.{
        .model = "claude-sonnet-4-20250514",
        .messages = msgs,
        .tools = &tools,
    }, false, allocator);
    defer allocator.free(body);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
    defer parsed.deinit();

    const tool_name = jh.getJsonString(jh.getPath(parsed.value, "tools.0.name").?).?;
    try std.testing.expectEqualStrings("get_weather", tool_name);
}

test "parseCompletionResponse" {
    const allocator = std.testing.allocator;

    const response_json =
        \\{"id":"msg_123","type":"message","role":"assistant","model":"claude-sonnet-4-20250514","content":[{"type":"text","text":"Hello!"}],"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":5}}
    ;

    var resp = try parseCompletionResponse(response_json, allocator);
    defer resp.deinit();

    try std.testing.expectEqualStrings("Hello!", resp.text().?);
    try std.testing.expectEqualStrings("claude-sonnet-4-20250514", resp.model);
    try std.testing.expectEqual(types.StopReason.end_turn, resp.stop_reason);
    try std.testing.expectEqual(@as(u32, 10), resp.usage.input_tokens);
    try std.testing.expectEqual(@as(u32, 5), resp.usage.output_tokens);
}

test "parseCompletionResponse with tool_use" {
    const allocator = std.testing.allocator;

    const response_json =
        \\{"id":"msg_456","type":"message","role":"assistant","model":"claude-sonnet-4-20250514","content":[{"type":"text","text":"Let me check."},{"type":"tool_use","id":"tu_1","name":"get_weather","input":{"city":"London"}}],"stop_reason":"tool_use","usage":{"input_tokens":20,"output_tokens":15}}
    ;

    var resp = try parseCompletionResponse(response_json, allocator);
    defer resp.deinit();

    try std.testing.expectEqual(types.StopReason.tool_use, resp.stop_reason);
    try std.testing.expect(resp.message.hasToolUse());

    var iter = resp.message.toolUseBlocks();
    const tu = iter.next().?;
    try std.testing.expectEqualStrings("get_weather", tu.name);
    try std.testing.expectEqualStrings("tu_1", tu.id);
}

test "mapApiErrorFromJson" {
    const allocator = std.testing.allocator;

    const error_json = "{\"error\":{\"type\":\"authentication_error\",\"message\":\"invalid key\"}}";
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, error_json, .{});
    defer parsed.deinit();

    const err = classifyApiError(parsed.value);
    try std.testing.expectEqual(@as(?ProviderError, error.AuthenticationFailed), err);
}

test "mapApiErrorFromJson captures error details" {
    const allocator = std.testing.allocator;
    var anthropic = try Anthropic.init(allocator, .{ .api_key = "test" });
    defer anthropic.deinit();

    const error_json = "{\"error\":{\"type\":\"authentication_error\",\"message\":\"Your API key has been revoked\"}}";
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, error_json, .{});
    defer parsed.deinit();

    const err = anthropic.mapApiErrorFromJson(parsed.value);
    try std.testing.expectEqual(@as(?ProviderError, error.AuthenticationFailed), err);

    // Verify error details were captured
    try std.testing.expectEqualStrings("Your API key has been revoked", anthropic.lastError().?);
    const details = anthropic.lastErrorDetails().?;
    try std.testing.expectEqualStrings("authentication_error", details.error_type);
    try std.testing.expectEqualStrings("Your API key has been revoked", details.message);
}

test "mapApiErrorFromJson captures rate limit error details" {
    const allocator = std.testing.allocator;
    var anthropic = try Anthropic.init(allocator, .{ .api_key = "test" });
    defer anthropic.deinit();

    const error_json = "{\"error\":{\"type\":\"rate_limit_error\",\"message\":\"Rate limit exceeded, please slow down\"}}";
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, error_json, .{});
    defer parsed.deinit();

    const err = anthropic.mapApiErrorFromJson(parsed.value);
    try std.testing.expectEqual(@as(?ProviderError, error.RateLimited), err);

    try std.testing.expectEqualStrings("Rate limit exceeded, please slow down", anthropic.lastError().?);
}

test "clearLastError clears previous error" {
    const allocator = std.testing.allocator;
    var anthropic = try Anthropic.init(allocator, .{ .api_key = "test" });
    defer anthropic.deinit();

    const error_json = "{\"error\":{\"type\":\"overloaded_error\",\"message\":\"Server is overloaded\"}}";
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, error_json, .{});
    defer parsed.deinit();

    _ = anthropic.mapApiErrorFromJson(parsed.value);
    try std.testing.expect(anthropic.lastError() != null);

    anthropic.clearLastError();
    try std.testing.expect(anthropic.lastError() == null);
}

test "last_error is null when no error has occurred" {
    const allocator = std.testing.allocator;
    var anthropic = try Anthropic.init(allocator, .{ .api_key = "test" });
    defer anthropic.deinit();

    try std.testing.expect(anthropic.lastError() == null);
    try std.testing.expect(anthropic.lastErrorDetails() == null);
}

test "mapApiErrorFromJson captures unknown error type" {
    const allocator = std.testing.allocator;
    var anthropic = try Anthropic.init(allocator, .{ .api_key = "test" });
    defer anthropic.deinit();

    const error_json = "{\"error\":{\"type\":\"some_new_error\",\"message\":\"Something unexpected happened\"}}";
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, error_json, .{});
    defer parsed.deinit();

    const err = anthropic.mapApiErrorFromJson(parsed.value);
    try std.testing.expectEqual(@as(?ProviderError, error.ProviderError), err);

    try std.testing.expectEqualStrings("Something unexpected happened", anthropic.lastError().?);
    try std.testing.expectEqualStrings("some_new_error", anthropic.lastErrorDetails().?.error_type);
}

test "successive errors replace previous error details" {
    const allocator = std.testing.allocator;
    var anthropic = try Anthropic.init(allocator, .{ .api_key = "test" });
    defer anthropic.deinit();

    // First error
    const error_json1 = "{\"error\":{\"type\":\"authentication_error\",\"message\":\"Invalid key\"}}";
    const parsed1 = try std.json.parseFromSlice(std.json.Value, allocator, error_json1, .{});
    defer parsed1.deinit();
    _ = anthropic.mapApiErrorFromJson(parsed1.value);
    try std.testing.expectEqualStrings("Invalid key", anthropic.lastError().?);

    // Second error replaces first
    const error_json2 = "{\"error\":{\"type\":\"rate_limit_error\",\"message\":\"Too many requests\"}}";
    const parsed2 = try std.json.parseFromSlice(std.json.Value, allocator, error_json2, .{});
    defer parsed2.deinit();
    _ = anthropic.mapApiErrorFromJson(parsed2.value);
    try std.testing.expectEqualStrings("Too many requests", anthropic.lastError().?);
    try std.testing.expectEqualStrings("rate_limit_error", anthropic.lastErrorDetails().?.error_type);
}

test "known models list" {
    const allocator = std.testing.allocator;
    var anthropic = try Anthropic.init(allocator, .{ .api_key = "test" });
    defer anthropic.deinit();

    const models = try anthropic.listModels(allocator);
    defer allocator.free(models);

    try std.testing.expect(models.len > 0);
    try std.testing.expectEqualStrings("claude-opus-4-20250514", models[0].id);
}

test "buildRequestBody with sampling params" {
    const allocator = std.testing.allocator;
    const msgs = try allocator.alloc(types.Message, 0);
    defer allocator.free(msgs);

    const body = try buildRequestBody(.{
        .model = "claude-sonnet-4-20250514",
        .messages = msgs,
        .top_p = 0.9,
        .top_k = 40,
        .temperature = 0.7,
    }, false, allocator);
    defer allocator.free(body);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
    defer parsed.deinit();

    try std.testing.expect(jh.getPath(parsed.value, "top_p") != null);
    try std.testing.expect(jh.getPath(parsed.value, "top_k") != null);
    try std.testing.expect(jh.getPath(parsed.value, "temperature") != null);
}
