package dev.engram.paper;

import dev.engram.paper.api.EngramMemoryService;
import dev.engram.paper.recipes.NpcMemoryHooks;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class NpcMemoryHooksTest {
    private static final UUID NPC = UUID.fromString("00000000-0000-0000-0000-000000000001");
    private static final UUID PLAYER = UUID.fromString("00000000-0000-0000-0000-000000000002");

    @Test void stableNpcAndPlayerIdsSeparateKeysAndKeepWorldScope() {
        var same = NpcMemoryHooks.key("world", NPC, PLAYER);
        assertEquals(same, NpcMemoryHooks.key("world", NPC, PLAYER));
        assertEquals(64, same.key().length());
        assertNotEquals(same, NpcMemoryHooks.key("world", PLAYER, NPC));
        assertNotEquals(same, NpcMemoryHooks.key("world", NPC, UUID.randomUUID()));
        assertNotEquals(same, NpcMemoryHooks.key("other_world", NPC, PLAYER));
    }

    @Test void hookUsesOnlyExplicitSaveAndExactRecallWithoutSimulatingAnNpc() {
        class RecordingService implements EngramMemoryService {
            Note saved;
            Key recalled;
            @Override public CompletableFuture<SaveResult> save(Note note) {
                saved = note; return CompletableFuture.completedFuture(new SaveResult(note.key()));
            }
            @Override public CompletableFuture<Optional<Checkpoint>> recall(Key key) {
                recalled = key; return CompletableFuture.completedFuture(Optional.empty());
            }
        }
        var service = new RecordingService();
        var hooks = new NpcMemoryHooks(service);
        assertNull(service.saved);
        assertTrue(hooks.recall("world", NPC, PLAYER).join().isEmpty());
        assertNull(service.saved);
        hooks.saveReviewedState("world", NPC, PLAYER, "Confirmed bridge delivery; town invitation remains pending.").join();
        assertEquals(service.recalled, service.saved.key());
        assertTrue(service.saved.summary().contains("invitation remains pending"));
        assertTrue(service.saved.decisions().isEmpty());
    }
}
