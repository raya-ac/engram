package dev.engram.paper;

import java.time.Duration;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/** Hosted CI supplies a real Python bridge backed by a disposable SQLite store. */
class ActualBridgeTest {
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
