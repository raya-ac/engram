package dev.engram.paper;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.Proxy;
import java.net.ProxySelector;
import java.net.SocketAddress;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.http.HttpTimeoutException;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Flow;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.regex.Pattern;

/** Bounded local HTTP transport. Call from a worker, never the game thread. */
final class BridgeClient implements AutoCloseable {
    static final int MAX_RESPONSE_BYTES = 32768;
    static final Pattern LABEL = Pattern.compile("[A-Za-z0-9][A-Za-z0-9_.-]{0,63}");
    static final Set<String> KINDS = Set.of("rule", "build", "handoff", "note");
    private final URI base;
    private final String token;
    private final Duration timeout;
    private final HttpClient http;

    BridgeClient(String endpoint, String token, Duration timeout) {
        URI candidate = URI.create(endpoint);
        String host = candidate.getHost();
        if (!"http".equals(candidate.getScheme()) || host == null
                || !Set.of("127.0.0.1", "[::1]", "::1").contains(host)
                || candidate.getPort() < 1 || candidate.getPort() > 65535
                || candidate.getRawUserInfo() != null || candidate.getRawQuery() != null
                || candidate.getRawFragment() != null
                || !(candidate.getRawPath().isEmpty() || candidate.getRawPath().equals("/"))) {
            throw new IllegalArgumentException("bridge-url must be an http loopback address with an explicit port and no path");
        }
        if (token == null || !token.matches("[A-Za-z0-9_-]{32,256}")) {
            throw new IllegalArgumentException("configure a 32–256 character token containing letters, digits, _ or -");
        }
        if (timeout == null || timeout.compareTo(Duration.ofSeconds(1)) < 0
                || timeout.compareTo(Duration.ofSeconds(30)) > 0) {
            throw new IllegalArgumentException("request-timeout-seconds must be from 1 to 30");
        }
        this.base = candidate.resolve("/");
        this.token = token;
        this.timeout = timeout;
        this.http = HttpClient.newBuilder().connectTimeout(timeout)
                .followRedirects(HttpClient.Redirect.NEVER)
                // Never forward the local bridge token to a configured system proxy.
                .proxy(new ProxySelector() {
                    @Override public List<Proxy> select(URI uri) { return List.of(Proxy.NO_PROXY); }
                    @Override public void connectFailed(URI uri, SocketAddress address, IOException failure) { }
                }).build();
    }

    static void validateKey(String world, String kind, String key) {
        if (world == null || key == null || !LABEL.matcher(world).matches() || !LABEL.matcher(key).matches()
                || kind == null || !KINDS.contains(kind)) {
            throw new IllegalArgumentException("use a valid kind and 1–64 character world/key labels (letters, digits, ._-)");
        }
    }

    JsonObject health() throws BridgeException {
        JsonObject result = request("GET", "health", null);
        if (!result.has("status") || !result.get("status").isJsonPrimitive()
                || !result.getAsJsonPrimitive("status").isString()
                || !"ok".equals(result.get("status").getAsString())) {
            throw new BridgeException("Bridge did not confirm initialized storage.");
        }
        return result;
    }

    JsonObject save(String world, String kind, String key, String summary) throws BridgeException {
        validateKey(world, kind, key);
        if (summary == null || summary.isBlank() || summary.length() > 4000) {
            throw new IllegalArgumentException("summary must contain 1–4000 characters");
        }
        JsonObject payload = new JsonObject();
        payload.addProperty("world", world);
        payload.addProperty("kind", kind);
        payload.addProperty("key", key);
        payload.addProperty("summary", summary);
        JsonObject result = request("POST", "v1/checkpoints", payload);
        if (!result.has("status") || !result.get("status").isJsonPrimitive()
                || !"saved".equals(result.get("status").getAsString())) {
            throw new BridgeException("Bridge did not confirm a saved checkpoint.");
        }
        return result;
    }

    JsonObject recall(String world, String kind, String key) throws BridgeException {
        validateKey(world, kind, key);
        // Validated labels contain only URI-safe ASCII; no user-supplied URL or scope.
        return request("GET", "v1/checkpoints?world=" + world + "&kind=" + kind + "&key=" + key, null);
    }

    private JsonObject request(String method, String path, JsonObject payload) throws BridgeException {
        HttpRequest.Builder builder = HttpRequest.newBuilder(base.resolve(path)).timeout(timeout)
                .header("Authorization", "Bearer " + token).header("Accept", "application/json");
        if (payload == null) {
            builder.GET();
        } else {
            byte[] body = payload.toString().getBytes(StandardCharsets.UTF_8);
            if (body.length > 16384) throw new BridgeException("Checkpoint exceeds the bridge request limit.");
            builder.header("Content-Type", "application/json").method(method, HttpRequest.BodyPublishers.ofByteArray(body));
        }
        var pending = http.sendAsync(builder.build(), info -> new LimitedBody());
        try {
            // Covers receipt of the complete bounded body, including a stalled peer.
            HttpResponse<byte[]> response = pending.get(timeout.toMillis(), TimeUnit.MILLISECONDS);
            if (response.statusCode() == 401 || response.statusCode() == 403) {
                throw new BridgeException("Bridge refused the token or world; check both configurations.");
            }
            if (response.statusCode() != 200) {
                throw new BridgeException("Bridge returned HTTP " + response.statusCode() + "; the operation was not confirmed.");
            }
            var parsed = JsonParser.parseString(new String(response.body(), StandardCharsets.UTF_8));
            if (!parsed.isJsonObject()) throw new BridgeException("Bridge returned an invalid response.");
            return parsed.getAsJsonObject();
        } catch (TimeoutException failure) {
            throw new BridgeException("Bridge timed out. A save may have completed; recall before retrying.");
        } catch (InterruptedException failure) {
            Thread.currentThread().interrupt();
            throw new BridgeException("Bridge request interrupted; outcome is unknown.");
        } catch (ExecutionException failure) {
            if (failure.getCause() instanceof HttpTimeoutException) {
                throw new BridgeException("Bridge timed out. A save may have completed; recall before retrying.");
            }
            throw new BridgeException("Bridge connection or response failed; no result was confirmed.");
        } catch (RuntimeException failure) {
            throw new BridgeException("Bridge connection or response failed; no result was confirmed.");
        } finally {
            if (!pending.isDone()) pending.cancel(true);
        }
    }

    @Override public void close() { http.shutdownNow(); }

    static final class BridgeException extends Exception {
        BridgeException(String message) { super(message); }
    }

    static final class LimitedBody implements HttpResponse.BodySubscriber<byte[]> {
        private final CompletableFuture<byte[]> result = new CompletableFuture<>();
        private final ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        private Flow.Subscription subscription;
        @Override public CompletionStage<byte[]> getBody() { return result; }
        @Override public void onSubscribe(Flow.Subscription value) {
            if (subscription != null) { value.cancel(); return; }
            subscription = value;
            subscription.request(1);
        }
        @Override public void onNext(List<ByteBuffer> chunks) {
            for (ByteBuffer chunk : chunks) {
                if (chunk.remaining() > MAX_RESPONSE_BYTES - bytes.size()) {
                    subscription.cancel();
                    result.completeExceptionally(new IOException("response exceeds limit"));
                    return;
                }
                byte[] part = new byte[chunk.remaining()];
                chunk.get(part);
                bytes.writeBytes(part);
            }
            subscription.request(1);
        }
        @Override public void onError(Throwable failure) { result.completeExceptionally(failure); }
        @Override public void onComplete() { result.complete(bytes.toByteArray()); }
    }
}
