package dev.engram.paper;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import static org.junit.jupiter.api.Assertions.*;

class BridgeClientTest {
    private static final String TOKEN = "synthetic-test-token-0123456789abcdef";
    private HttpServer server;
    private String endpoint;

    @BeforeEach void start() throws IOException {
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        endpoint = "http://127.0.0.1:" + server.getAddress().getPort();
        server.start();
    }

    @AfterEach void stop() { server.stop(0); }

    private static void reply(HttpExchange exchange, int status, String value) throws IOException {
        byte[] body = value.getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().add("Content-Type", "application/json");
        exchange.sendResponseHeaders(status, body.length);
        try (var output = exchange.getResponseBody()) { output.write(body); }
    }

    @Test void explicitWorldNoteAndRecallUseAuthenticatedContract() throws Exception {
        AtomicReference<JsonObject> saved = new AtomicReference<>();
        AtomicReference<String> query = new AtomicReference<>();
        server.createContext("/v1/checkpoints", exchange -> {
            if (!("Bearer " + TOKEN).equals(exchange.getRequestHeaders().getFirst("Authorization"))) {
                reply(exchange, 401, "{}"); return;
            }
            if (exchange.getRequestMethod().equals("POST")) {
                saved.set(JsonParser.parseString(new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8)).getAsJsonObject());
                reply(exchange, 200, "{\"status\":\"saved\"}");
            } else {
                query.set(exchange.getRequestURI().getRawQuery());
                reply(exchange, 200, "{\"found\":false,\"checkpoints\":[]}");
            }
        });
        try (BridgeClient client = new BridgeClient(endpoint, TOKEN, Duration.ofSeconds(3))) {
            client.save("world_nether", "build", "station", "x=2 y=70 z=-8\nKeep the west path open.");
            assertFalse(client.recall("world_nether", "build", "station").get("found").getAsBoolean());
        }
        assertEquals("world_nether", saved.get().get("world").getAsString());
        assertEquals("x=2 y=70 z=-8\nKeep the west path open.", saved.get().get("summary").getAsString());
        assertFalse(saved.get().has("project_id"));
        assertEquals("world=world_nether&kind=build&key=station", query.get());
    }

    @Test void healthRequiresActualSuccessShape() throws Exception {
        server.createContext("/health", exchange -> reply(exchange, 200, "{\"status\":\"ok\",\"storage\":{}}"));
        try (BridgeClient client = new BridgeClient(endpoint, TOKEN, Duration.ofSeconds(2))) {
            assertEquals("ok", client.health().get("status").getAsString());
        }
    }

    @Test void unauthorizedResponsesDoNotEchoPrivateServerDetails() {
        server.createContext("/health", exchange -> reply(exchange, 401, "{\"error\":\"SECRET database details\"}"));
        try (BridgeClient client = new BridgeClient(endpoint, TOKEN, Duration.ofSeconds(2))) {
            var error = assertThrows(BridgeClient.BridgeException.class, client::health);
            assertTrue(error.getMessage().contains("refused"));
            assertFalse(error.getMessage().contains("SECRET"));
        }
    }

    @Test void redirectsAreNotFollowed() {
        server.createContext("/health", exchange -> {
            exchange.getResponseHeaders().add("Location", endpoint + "/should-not-run");
            reply(exchange, 302, "{}");
        });
        AtomicReference<Boolean> followed = new AtomicReference<>(false);
        server.createContext("/should-not-run", exchange -> { followed.set(true); reply(exchange, 200, "{}"); });
        try (BridgeClient client = new BridgeClient(endpoint, TOKEN, Duration.ofSeconds(2))) {
            assertThrows(BridgeClient.BridgeException.class, client::health);
            assertFalse(followed.get());
        }
    }

    @Test void oversizedResponseIsRejected() {
        server.createContext("/health", exchange -> reply(exchange, 200, "x".repeat(BridgeClient.MAX_RESPONSE_BYTES + 1)));
        try (BridgeClient client = new BridgeClient(endpoint, TOKEN, Duration.ofSeconds(2))) {
            assertThrows(BridgeClient.BridgeException.class, client::health);
        }
    }

    @Test void timeoutCoversStalledBodyAfterHeaders() {
        server.createContext("/health", exchange -> {
            exchange.sendResponseHeaders(200, 100);
            try (var output = exchange.getResponseBody()) {
                output.write('{'); output.flush();
                try { Thread.sleep(2000); } catch (InterruptedException failure) { Thread.currentThread().interrupt(); }
            } catch (IOException ignored) { }
        });
        try (BridgeClient client = new BridgeClient(endpoint, TOKEN, Duration.ofSeconds(1))) {
            assertTimeoutPreemptively(Duration.ofSeconds(3), () -> {
                var error = assertThrows(BridgeClient.BridgeException.class, client::health);
                assertTrue(error.getMessage().contains("timed out"));
            });
        }
    }

    @ParameterizedTest
    @ValueSource(strings = {"https://127.0.0.1:8422", "http://example.com:8422", "http://localhost:8422",
            "http://127.0.0.1:8422/path", "http://user@127.0.0.1:8422", "http://127.0.0.1:8422?token=value"})
    void rejectsExternalOrAmbiguousEndpoints(String value) {
        assertThrows(IllegalArgumentException.class, () -> new BridgeClient(value, TOKEN, Duration.ofSeconds(2)));
    }

    @Test void rejectsMalformedKeysBeforeSending() {
        assertThrows(IllegalArgumentException.class, () -> BridgeClient.validateKey("world&other=x", "note", "key"));
        assertThrows(IllegalArgumentException.class, () -> BridgeClient.validateKey("world", "execute", "key"));
        assertThrows(IllegalArgumentException.class, () -> BridgeClient.validateKey("world", "note", "../path"));
        assertThrows(IllegalArgumentException.class, () -> BridgeClient.validateKey("world", null, "key"));
    }
}
