package dev.engram.paper;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import dev.engram.paper.api.EngramMemoryService;
import java.time.Instant;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/** One queue shared by the public plugin service and staff commands. */
final class QueuedMemoryService implements EngramMemoryService, AutoCloseable {
    private final BridgeClient bridge;
    private final Set<String> worlds;
    private final ThreadPoolExecutor worker;
    private final Set<CompletableFuture<?>> pending = ConcurrentHashMap.newKeySet();
    private final Object admission = new Object();
    private boolean closed;

    QueuedMemoryService(BridgeClient bridge, Set<String> worlds) {
        this.bridge = bridge;
        this.worlds = Set.copyOf(worlds);
        worker = new ThreadPoolExecutor(1, 1, 0, TimeUnit.MILLISECONDS, new ArrayBlockingQueue<>(8), runnable -> {
            Thread thread = new Thread(runnable, "engram-bridge");
            thread.setDaemon(true);
            return thread;
        }, new ThreadPoolExecutor.AbortPolicy());
    }

    @Override public CompletableFuture<SaveResult> save(Note note) {
        if (note == null) throw new IllegalArgumentException("note is required");
        if (!worlds.contains(note.key().world())) return deniedWorld();
        return submit(() -> { bridge.save(note); return new SaveResult(note.key()); });
    }

    @Override public CompletableFuture<Optional<Checkpoint>> recall(Key key) {
        if (key == null) throw new IllegalArgumentException("key is required");
        if (!worlds.contains(key.world())) return deniedWorld();
        return submit(() -> parse(key, bridge.recall(key.world(), key.kind(), key.key())));
    }

    private static <T> CompletableFuture<T> deniedWorld() {
        return CompletableFuture.failedFuture(new Failure(FailureReason.WORLD_NOT_ALLOWED,
                "This world is not enabled in the plugin configuration."));
    }

    <T> CompletableFuture<T> submit(Callable<T> operation) {
        CompletableFuture<T> result = new CompletableFuture<>();
        synchronized (admission) {
            if (closed) return CompletableFuture.failedFuture(stopped());
            pending.add(result);
            try {
                worker.execute(() -> {
                    try {
                        if (!result.isDone()) result.complete(operation.call());
                    } catch (BridgeClient.BridgeException failure) {
                        result.completeExceptionally(new Failure(FailureReason.BRIDGE_FAILURE, failure.getMessage()));
                    } catch (Failure failure) {
                        result.completeExceptionally(failure);
                    } catch (Exception failure) {
                        result.completeExceptionally(new Failure(FailureReason.INVALID_RESPONSE,
                                "Bridge returned an unexpected result; no operation was confirmed."));
                    } finally {
                        pending.remove(result);
                    }
                });
            } catch (RejectedExecutionException failure) {
                pending.remove(result);
                result.completeExceptionally(new Failure(FailureReason.BUSY,
                        "Engram is busy; try again after the pending requests finish."));
            }
        }
        return result;
    }

    static Optional<Checkpoint> parse(Key key, JsonObject result) {
        // Reuse the command response validator for the shared string/list limits.
        ContextView.render(result);
        if (!result.get("found").getAsBoolean()) return Optional.empty();
        JsonObject item = result.getAsJsonArray("checkpoints").get(0).getAsJsonObject();
        String expectedTask = "game:" + key.world() + ":" + key.kind() + ":" + key.key();
        if (!item.has("task") || !item.get("task").isJsonPrimitive()
                || !item.getAsJsonPrimitive("task").isString()
                || !expectedTask.equals(item.get("task").getAsString())
                || !item.has("updated_at") || !item.get("updated_at").isJsonPrimitive()
                || !item.getAsJsonPrimitive("updated_at").isNumber()) {
            throw new IllegalArgumentException("checkpoint identity or timestamp is invalid");
        }
        double timestamp = item.get("updated_at").getAsDouble();
        if (!Double.isFinite(timestamp) || timestamp < 0 || timestamp > 253402300799d) {
            throw new IllegalArgumentException("checkpoint timestamp is invalid");
        }
        long seconds = (long) timestamp;
        Instant updatedAt = Instant.ofEpochSecond(seconds, (long) ((timestamp - seconds) * 1_000_000_000));
        return Optional.of(new Checkpoint(new Note(key, item.get("summary").getAsString(),
                strings(item, "decisions"), strings(item, "next_steps"), strings(item, "blockers")), updatedAt));
    }

    private static List<String> strings(JsonObject item, String field) {
        if (!item.has(field)) return List.of();
        return item.getAsJsonArray(field).asList().stream().map(JsonElement::getAsString).toList();
    }

    private static Failure stopped() { return new Failure(FailureReason.STOPPED, "Engram service has stopped; the outcome of an in-flight write may be unknown."); }

    @Override public void close() {
        synchronized (admission) {
            if (closed) return;
            closed = true;
            worker.shutdownNow();
        }
        // Queued tasks removed by shutdownNow must still complete their futures.
        pending.forEach(result -> result.completeExceptionally(stopped()));
        pending.clear();
        bridge.close();
    }
}
