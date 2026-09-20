package dev.engram.paper.recipes;

import dev.engram.paper.api.EngramMemoryService;
import dev.engram.paper.api.EngramMemoryService.Checkpoint;
import dev.engram.paper.api.EngramMemoryService.Key;
import dev.engram.paper.api.EngramMemoryService.Note;
import dev.engram.paper.api.EngramMemoryService.SaveResult;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.Objects;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;

/**
 * A source hook for an existing NPC/quest plugin, with no Citizens dependency.
 * The caller supplies a persistent NPC UUID and server-authenticated player UUID.
 * It owns permission checks, interaction events and verification of game outcomes.
 * Nothing here spawns an NPC, runs dialogue, rewards a player or listens to chat.
 */
public final class NpcMemoryHooks {
    private final EngramMemoryService memory;

    public NpcMemoryHooks(EngramMemoryService memory) { this.memory = Objects.requireNonNull(memory); }

    /** Read the last explicit snapshot for this NPC/player pair in this world. */
    public CompletableFuture<Optional<Checkpoint>> recall(String world, UUID npcId, UUID playerId) {
        return memory.recall(key(world, npcId, playerId));
    }

    /** Replace the complete snapshot after the caller has reviewed/verified it. */
    public CompletableFuture<SaveResult> saveReviewedState(String world, UUID npcId, UUID playerId, String summary) {
        return memory.save(new Note(key(world, npcId, playerId), summary));
    }

    /** Stable opaque 64-character key. Hashing is namespacing, not authorization. */
    public static Key key(String world, UUID npcId, UUID playerId) {
        Objects.requireNonNull(npcId, "npcId");
        Objects.requireNonNull(playerId, "playerId");
        try {
            byte[] bytes = ("npc-player-v1:" + npcId + ":" + playerId).getBytes(StandardCharsets.UTF_8);
            String digest = HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes));
            return new Key(world, "note", digest);
        } catch (NoSuchAlgorithmException impossible) {
            throw new IllegalStateException("SHA-256 unavailable");
        }
    }
}
