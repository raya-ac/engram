package dev.engram.paper;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import dev.engram.paper.api.EngramMemoryService.*;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class QueuedMemoryServiceTest {
    private HttpServer server;
    private QueuedMemoryService service;
    private final Key key = new Key("world", "note", "npc-summary");

    @BeforeEach void start() throws IOException {
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.start();
        service = new QueuedMemoryService(new BridgeClient("http://127.0.0.1:" + server.getAddress().getPort(),
                "synthetic-service-token-0123456789abcdef", Duration.ofSeconds(3)), Set.of("world"));
    }

    @AfterEach void stop() { service.close(); server.stop(0); }

    private static void reply(HttpExchange exchange, String value) throws IOException {
        byte[] body = value.getBytes(StandardCharsets.UTF_8);
        exchange.sendResponseHeaders(200, body.length);
        try (var output = exchange.getResponseBody()) { output.write(body); }
    }

    private static Failure failure(CompletableFuture<?> result) {
        var error = assertThrows(ExecutionException.class, () -> result.get(2, TimeUnit.SECONDS));
        return assertInstanceOf(Failure.class, error.getCause());
    }

    @Test void typedSaveAndRecallPreserveStructuredSnapshotAndIdentity() throws Exception {
        AtomicReference<JsonObject> saved = new AtomicReference<>();
        server.createContext("/v1/checkpoints", exchange -> {
            if (exchange.getRequestMethod().equals("POST")) {
                saved.set(JsonParser.parseString(new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8)).getAsJsonObject());
                reply(exchange, "{\"status\":\"saved\"}");
            } else {
                reply(exchange, "{\"found\":true,\"checkpoints\":[{\"task\":\"game:world:note:npc-summary\",\"summary\":\"bridge repaired\",\"decisions\":[\"keep north gate open\"],\"next_steps\":[\"inspect lamps\"],\"blockers\":[],\"updated_at\":1700000000.25}]}");
            }
        });
        var decisions = new ArrayList<>(List.of("keep north gate open"));
        Note note = new Note(key, "bridge repaired", decisions, List.of("inspect lamps"), List.of());
        decisions.set(0, "caller mutation must not replace saved text");
        assertEquals(key, service.save(note).get(2, TimeUnit.SECONDS).key());
        assertEquals("keep north gate open", saved.get().getAsJsonArray("decisions").get(0).getAsString());
        assertFalse(saved.get().has("project_id"));
        Checkpoint checkpoint = service.recall(key).get(2, TimeUnit.SECONDS).orElseThrow();
        assertEquals(key, checkpoint.note().key());
        assertEquals(List.of("inspect lamps"), checkpoint.note().nextSteps());
        assertEquals(250_000_000, checkpoint.updatedAt().getNano());
        assertThrows(UnsupportedOperationException.class, () -> checkpoint.note().decisions().add("bad"));
    }

    @Test void missingNotesAndFailuresAreDifferentResults() throws Exception {
        server.createContext("/v1/checkpoints", exchange -> reply(exchange, "{\"found\":false,\"checkpoints\":[]}"));
        assertTrue(service.recall(key).get(2, TimeUnit.SECONDS).isEmpty());
        assertEquals(FailureReason.WORLD_NOT_ALLOWED,
                failure(service.recall(new Key("other_world", "note", "npc-summary"))).reason());
    }

    @Test void rejectsUnexpectedTaskRatherThanReturningAnotherNpcsNote() {
        server.createContext("/v1/checkpoints", exchange -> reply(exchange,
                "{\"found\":true,\"checkpoints\":[{\"task\":\"game:world:note:other-npc\",\"summary\":\"private\",\"updated_at\":1700000000}]}"));
        var error = failure(service.recall(key));
        assertEquals(FailureReason.INVALID_RESPONSE, error.reason());
        assertFalse(error.getMessage().contains("private"));
    }

    @Test void callersShareBoundedQueueAndShutdownCompletesWaitingFutures() throws Exception {
        CountDownLatch entered = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        AtomicInteger requests = new AtomicInteger();
        server.createContext("/v1/checkpoints", exchange -> {
            requests.incrementAndGet(); entered.countDown();
            try { release.await(5, TimeUnit.SECONDS); }
            catch (InterruptedException failure) { Thread.currentThread().interrupt(); }
            try { reply(exchange, "{\"found\":false,\"checkpoints\":[]}"); } catch (IOException ignored) { }
        });
        List<CompletableFuture<?>> accepted = new ArrayList<>();
        try {
            accepted.add(service.recall(key));
            assertTrue(entered.await(2, TimeUnit.SECONDS));
            // Mix a command-style task with public service calls in the same queue.
            accepted.add(service.submit(() -> "command result"));
            for (int i = 0; i < 7; i++) accepted.add(service.recall(key));
            assertEquals(FailureReason.BUSY, failure(service.recall(key)).reason());
            service.close();
            for (var result : accepted) assertTrue(result.isCompletedExceptionally());
            assertEquals(FailureReason.STOPPED, failure(service.recall(key)).reason());
            assertEquals(1, requests.get());
        } finally { release.countDown(); }
    }
}
