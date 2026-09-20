package dev.engram.paper;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import java.util.ArrayList;
import java.util.List;

/** Treat retrieved content as bounded plain text, never commands or MiniMessage. */
final class ContextView {
    private ContextView() { }

    static List<String> render(JsonObject result) {
        if (!result.has("found") || !result.get("found").isJsonPrimitive()
                || !result.getAsJsonPrimitive("found").isBoolean()
                || !result.has("checkpoints") || !result.get("checkpoints").isJsonArray()) {
            throw new IllegalArgumentException("invalid checkpoint response");
        }
        JsonArray checkpoints = result.getAsJsonArray("checkpoints");
        boolean found = result.get("found").getAsBoolean();
        if (!found && checkpoints.isEmpty()) return List.of("No saved note for this world, kind and key.");
        if (!found || checkpoints.size() != 1 || !checkpoints.get(0).isJsonObject()) {
            throw new IllegalArgumentException("invalid checkpoint response");
        }
        JsonObject item = checkpoints.get(0).getAsJsonObject();
        List<String> text = new ArrayList<>();
        text.add("Saved reference note; check current world state before acting.");
        text.add(string(item.get("summary"), 4000));
        for (String field : List.of("decisions", "next_steps", "blockers")) {
            if (!item.has(field)) continue;
            if (!item.get(field).isJsonArray() || item.getAsJsonArray(field).size() > 8) {
                throw new IllegalArgumentException("invalid checkpoint response");
            }
            for (JsonElement value : item.getAsJsonArray(field)) text.add(field + ": " + string(value, 500));
        }
        List<String> lines = new ArrayList<>();
        for (String value : text) {
            String plain = plain(value);
            for (int offset = 0; offset < plain.length(); offset += 200) {
                if (lines.size() == 9) {
                    lines.add("[display truncated; the full checkpoint remains in Engram]");
                    return List.copyOf(lines);
                }
                lines.add(plain.substring(offset, Math.min(offset + 200, plain.length())));
            }
        }
        return List.copyOf(lines);
    }

    private static String string(JsonElement value, int maximum) {
        if (value == null || !value.isJsonPrimitive() || !value.getAsJsonPrimitive().isString()
                || value.getAsString().length() > maximum) {
            throw new IllegalArgumentException("invalid checkpoint response");
        }
        return value.getAsString();
    }

    static String plain(String value) {
        StringBuilder result = new StringBuilder();
        value.codePoints().forEach(cp -> {
            int type = Character.getType(cp);
            if (cp == '\n' || cp == '\r' || cp == '\t') result.append(' ');
            else if (type != Character.CONTROL && type != Character.FORMAT && cp != '\u00a7') result.appendCodePoint(cp);
        });
        return result.toString();
    }
}
