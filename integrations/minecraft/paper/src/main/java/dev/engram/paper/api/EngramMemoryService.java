package dev.engram.paper.api;

import java.time.Instant;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.regex.Pattern;

/**
 * Obtain through Bukkit's ServicesManager after declaring depend: [EngramMemory].
 * Calls enqueue bounded I/O and may be made from any thread. Completion threads
 * are unspecified: schedule all subsequent Bukkit access on the server thread.
 * Never join/get a pending future on that thread. The caller authorizes players
 * and snapshots game state; this server-side service is not a per-player ACL.
 * Cancelling a future cannot undo a write already sent to the bridge.
 */
public interface EngramMemoryService {
    CompletableFuture<SaveResult> save(Note note);
    CompletableFuture<Optional<Checkpoint>> recall(Key key);

    /** Exact checkpoint identity inside the bridge's fixed server project. */
    record Key(String world, String kind, String key) {
        private static final Pattern LABEL = Pattern.compile("[A-Za-z0-9][A-Za-z0-9_.-]{0,63}");
        private static final Set<String> KINDS = Set.of("rule", "build", "handoff", "note");
        public Key {
            if (world == null || key == null || kind == null || !LABEL.matcher(world).matches()
                    || !LABEL.matcher(key).matches() || !KINDS.contains(kind)) {
                throw new IllegalArgumentException("invalid world/kind/key checkpoint identity");
            }
        }
    }

    /** Saving replaces this key's complete note, including all three lists. */
    record Note(Key key, String summary, List<String> decisions, List<String> nextSteps, List<String> blockers) {
        public Note {
            Objects.requireNonNull(key, "key");
            summary = text(summary, 4000);
            decisions = items(decisions);
            nextSteps = items(nextSteps);
            blockers = items(blockers);
        }
        public Note(Key key, String summary) { this(key, summary, List.of(), List.of(), List.of()); }
        private static String text(String value, int maximum) {
            if (value == null || value.isBlank() || value.length() > maximum) {
                throw new IllegalArgumentException("note text is missing or exceeds its limit");
            }
            return value.strip();
        }
        private static List<String> items(List<String> values) {
            if (values == null || values.size() > 8) throw new IllegalArgumentException("note lists allow at most eight entries");
            return values.stream().map(value -> text(value, 500)).toList();
        }
    }

    /** Timestamp is when Engram saved the checkpoint, not verification of game state. */
    record Checkpoint(Note note, Instant updatedAt) {
        public Checkpoint { Objects.requireNonNull(note, "note"); Objects.requireNonNull(updatedAt, "updatedAt"); }
    }

    /** Constructed only after the bridge confirms status=saved. */
    record SaveResult(Key key) { public SaveResult { Objects.requireNonNull(key, "key"); } }

    enum FailureReason { WORLD_NOT_ALLOWED, BUSY, STOPPED, BRIDGE_FAILURE, INVALID_RESPONSE }

    /** Bounded, sanitized failure; no raw response, credential or connection exception. */
    final class Failure extends RuntimeException {
        private final FailureReason reason;
        public Failure(FailureReason reason, String message) {
            super(message);
            this.reason = Objects.requireNonNull(reason);
        }
        public FailureReason reason() { return reason; }
    }
}
