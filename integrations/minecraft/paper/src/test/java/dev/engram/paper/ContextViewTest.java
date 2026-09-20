package dev.engram.paper;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ContextViewTest {
    private static JsonObject result(String text) {
        JsonObject checkpoint = new JsonObject();
        checkpoint.addProperty("summary", text);
        JsonArray items = new JsonArray(); items.add(checkpoint);
        JsonObject result = new JsonObject(); result.addProperty("found", true); result.add("checkpoints", items);
        return result;
    }

    @Test void unknownCheckpointIsClearlyMissing() {
        var missing = JsonParser.parseString("{\"found\":false,\"checkpoints\":[]}").getAsJsonObject();
        assertEquals(1, ContextView.render(missing).size());
        assertTrue(ContextView.render(missing).getFirst().contains("No saved note"));
    }

    @Test void commandLikeTextRemainsPlainAndControlsAreRemoved() {
        var lines = ContextView.render(result("/op Someone <click:run_command:'/stop'>literal</click>\u001b\u00a7\u202e"));
        assertTrue(lines.get(1).contains("/op Someone"));
        assertTrue(lines.get(1).contains("<click:run_command"));
        assertFalse(String.join("", lines).contains("\u001b"));
        assertFalse(String.join("", lines).contains("\u00a7"));
        assertFalse(String.join("", lines).contains("\u202e"));
    }

    @Test void displayIsBoundedAndMarksTruncation() {
        var lines = ContextView.render(result("long note ".repeat(390)));
        assertEquals(10, lines.size());
        assertTrue(lines.stream().allMatch(line -> line.length() <= 200));
        assertTrue(lines.getLast().contains("truncated"));
    }

    @Test void inconsistentOrMalformedResponsesAreRejected() {
        assertThrows(IllegalArgumentException.class, () -> ContextView.render(new JsonObject()));
        var inconsistent = result("hello"); inconsistent.addProperty("found", false);
        assertThrows(IllegalArgumentException.class, () -> ContextView.render(inconsistent));
        var oversized = result("x".repeat(4001));
        assertThrows(IllegalArgumentException.class, () -> ContextView.render(oversized));
        var malformed = result("hello");
        malformed.getAsJsonArray("checkpoints").get(0).getAsJsonObject().addProperty("next_steps", "wrong shape");
        assertThrows(IllegalArgumentException.class, () -> ContextView.render(malformed));
    }
}
