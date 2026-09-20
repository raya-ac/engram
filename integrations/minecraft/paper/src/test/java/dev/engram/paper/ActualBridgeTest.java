package dev.engram.paper;

import java.time.Duration;
import java.util.List;
import java.util.Set;
import dev.engram.paper.api.EngramMemoryService.Key;
import dev.engram.paper.api.EngramMemoryService.Note;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/** Hosted CI supplies a real Python bridge backed by a disposable SQLite store. */
class ActualBridgeTest {
    @Test void publicPluginServicePersistsTypedSnapshotThroughActualBridge() throws Exception {
        String endpoint = System.getenv("ENGRAM_TEST_BRIDGE_URL");
        String token = System.getenv("ENGRAM_TEST_BRIDGE_TOKEN");
        assumeTrue(endpoint != null && token != null, "set bridge test environment to run the cross-language check");
        var key = new Key("world", "note", "ci-service-checkpoint");
        try (var service = new QueuedMemoryService(new BridgeClient(endpoint, token, Duration.ofSeconds(10)), Set.of("world"))) {
            var receipt = service.save(new Note(key, "Synthetic NPC state: bridge delivery accepted.",
                    List.of("reward already recorded by quest plugin"), List.of("offer town tour"), List.of()))
                    .get(15, TimeUnit.SECONDS);
            assertEquals(key, receipt.key());
        }
        try (var reopened = new QueuedMemoryService(new BridgeClient(endpoint, token, Duration.ofSeconds(10)), Set.of("world"))) {
            var checkpoint = reopened.recall(key).get(15, TimeUnit.SECONDS).orElseThrow();
            assertEquals(List.of("offer town tour"), checkpoint.note().nextSteps());
            assertEquals("Synthetic NPC state: bridge delivery accepted.", checkpoint.note().summary());
        }
    }

    @Test void javaToPythonToSQLiteCheckpointRoundTrip() throws Exception {
        String endpoint = System.getenv("ENGRAM_TEST_BRIDGE_URL");
        String token = System.getenv("ENGRAM_TEST_BRIDGE_TOKEN");
        assumeTrue(endpoint != null && token != null, "set bridge test environment to run the cross-language check");
        try (BridgeClient client = new BridgeClient(endpoint, token, Duration.ofSeconds(10))) {
            assertEquals("ok", client.health().get("status").getAsString());
            client.save("world", "build", "ci-checkpoint", "location: minecraft:overworld x=248 y=72 z=-316\nSynthetic lighthouse shell complete.");
            var first = client.recall("world", "build", "ci-checkpoint");
            assertTrue(first.get("found").getAsBoolean());
            assertEquals(1, first.getAsJsonArray("checkpoints").size());
            assertTrue(String.join(" ", ContextView.render(first)).contains("x=248 y=72 z=-316"));
            client.save("world", "build", "ci-checkpoint", "Synthetic lighthouse stairs complete; glass remains.");
            var replaced = client.recall("world", "build", "ci-checkpoint");
            assertEquals("Synthetic lighthouse stairs complete; glass remains.",
                    replaced.getAsJsonArray("checkpoints").get(0).getAsJsonObject().get("summary").getAsString());
            assertFalse(client.recall("world", "rule", "ci-checkpoint").get("found").getAsBoolean());
        }
    }
}
