# zig-llm

Talk to LLMs from Zig. No wrappers around C libraries, no pulling in half the internet -- just `std.http.Client`, `std.json`, and a clean API that feels like Zig.

Inspired by [ruby_llm](https://github.com/crmne/ruby_llm)'s "make it dead simple" philosophy, but built for a language where you actually know who owns your memory.

Ships with **Anthropic Claude** support. Adding new providers? Implement three functions and you're done.

## Why This Exists

Every LLM library out there is either Python, JavaScript, or a Zig wrapper around a C/C++ binding. If you want to talk to Claude from Zig, you're hand-rolling HTTP requests and parsing JSON yourself. This library does that part so you can focus on the interesting stuff.

## Show Me the Code

```zig
const std = @import("std");
const llm = @import("zig-llm");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const api_key = std.posix.getenv("ANTHROPIC_API_KEY") orelse return;

    var anthropic = try llm.providers.Anthropic.init(allocator, .{
        .api_key = api_key,
    });
    defer anthropic.deinit();

    var chat = llm.Chat.init(allocator, anthropic.provider(), "claude-sonnet-4-20250514");
    defer chat.deinit();
    chat.system_prompt = "You are a helpful assistant.";

    var response = try chat.send("Hello! What is Zig?");
    defer response.deinit();

    if (response.text()) |text| {
        std.debug.print("{s}\n", .{text});
    }

    // History is maintained -- just keep talking
    var response2 = try chat.send("What are its main advantages?");
    defer response2.deinit();
}
```

That's it. No boilerplate, no ceremony.

## Streaming

True incremental streaming -- tokens arrive as the API generates them, not buffered. Under the hood, the library holds an open HTTP connection and parses SSE events off the wire one at a time. You get an iterator and control the loop.

```zig
var iter = try chat.sendStreaming("Tell me a joke.");
defer iter.deinit();

while (try iter.next()) |event| {
    switch (event) {
        .text_delta => |td| {
            std.debug.print("{s}", .{td.text});
            event.deinit(allocator);
        },
        .message_stop => break,
        else => event.deinit(allocator),
    }
}
```

## Tool Use

Give the model tools, write a handler, call `sendWithTools`. The library runs the tool loop for you -- dispatching calls, collecting results, and repeating until the model has what it needs.

Here's a complete, runnable example:

```zig
const std = @import("std");
const llm = @import("zig-llm");

// 1. Define your tools as JSON Schema
const tools = [_]llm.ToolDefinition{
    .{
        .name = "get_weather",
        .description = "Get the current weather for a given city.",
        .input_schema =
        \\{"type":"object","properties":{"city":{"type":"string","description":"The city name"}},"required":["city"]}
        ,
    },
};

// 2. Write a handler -- this is where your actual logic lives
fn handleToolCall(_: void, name: []const u8, input_json: []const u8) llm.Chat.ToolResult {
    std.debug.print("Tool called: {s}({s})\n", .{ name, input_json });

    if (std.mem.eql(u8, name, "get_weather")) {
        // In real code, you'd parse input_json, call an API, etc.
        return .{ .content = "{\"temp_c\": 15, \"condition\": \"partly cloudy\"}" };
    }
    return .{ .content = "Unknown tool", .is_error = true };
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const api_key = std.posix.getenv("ANTHROPIC_API_KEY") orelse return;

    var anthropic = try llm.providers.Anthropic.init(allocator, .{ .api_key = api_key });
    defer anthropic.deinit();

    var chat = llm.Chat.init(allocator, anthropic.provider(), "claude-sonnet-4-20250514");
    defer chat.deinit();
    chat.tools = &tools;

    // 3. sendWithTools handles the entire tool loop
    var response = try chat.sendWithTools("What's the weather like in London?", {}, handleToolCall);
    defer response.deinit();

    std.debug.print("Assistant: {s}\n", .{response.text().?});
}
```

The `context` parameter (the `{}` above) follows the same pattern as `std.sort` -- pass any value and it arrives as the first argument to your handler. Useful when your tools need access to a database, HTTP client, or other state:

```zig
var response = try chat.sendWithTools("query something", &my_app, MyApp.handleToolCall);
```

You can also drive the loop manually with `send()` + `sendToolResult()` if you need more control.

## Installation

Add to your `build.zig.zon`:

```zig
.dependencies = .{
    .zig_llm = .{
        .path = "../zig-llm", // or a URL
    },
},
```

Then in `build.zig`:

```zig
const zig_llm = b.dependency("zig_llm", .{
    .target = target,
    .optimize = optimize,
});
exe.root_module.addImport("zig-llm", zig_llm.module("zig-llm"));
```

## Try It

```sh
export ANTHROPIC_API_KEY="your-key-here"

zig build run-basic_chat    # Simple conversation
zig build run-streaming     # Watch tokens arrive in real time
zig build run-tool_use      # Tool calling loop
```

## How It Works

### The Provider Interface

The core of the library is `Provider` -- a VTable interface, the same pattern Zig's standard library uses for `std.mem.Allocator`, `std.io.Reader`, and friends. A fat pointer: `ptr: *anyopaque` + `vtable: *const VTable`.

```zig
const Provider = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    const VTable = struct {
        complete: *const fn (...) ProviderError!CompletionResponse,
        stream: *const fn (...) ProviderError!StreamIterator,
        listModels: *const fn (...) ProviderError![]const ModelInfo,
    };
};
```

Want to add your own provider? Just implement three methods and call `Provider.init(&your_thing)` -- the vtable is generated at comptime:

```zig
var my_provider = MyCustomProvider.init(allocator, config);
const provider = Provider.init(&my_provider);
```

### Memory Ownership

No surprises here:

- **`CompletionRequest`** borrows all strings -- caller owns them
- **`CompletionResponse`** and **`StreamEvent`** own their strings -- caller calls `deinit()`
- **`Chat`** owns its history -- freed on `chat.deinit()`
- Everything goes through an explicit `Allocator`. No globals, no hidden allocs.

## Testing

```sh
zig build test    # 30 tests, 0 leaks
```

## License

MIT
