package dev.engram.paper;

import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import dev.engram.paper.api.EngramMemoryService;
import net.kyori.adventure.text.Component;
import org.bukkit.Location;
import org.bukkit.command.Command;
import org.bukkit.command.CommandExecutor;
import org.bukkit.command.CommandSender;
import org.bukkit.command.ConsoleCommandSender;
import org.bukkit.command.TabCompleter;
import org.bukkit.entity.Player;
import org.bukkit.plugin.IllegalPluginAccessException;
import org.bukkit.plugin.ServicePriority;
import org.bukkit.plugin.java.JavaPlugin;

/** Staff commands and a bounded Java service; no automatic capture or world mutations. */
public final class EngramPlugin extends JavaPlugin implements CommandExecutor, TabCompleter {
    private BridgeClient bridge;
    private QueuedMemoryService service;
    private Set<String> worlds;
    private String consoleWorld;
    private volatile boolean stopping;
    // Accessed on the main thread only.
    private final Set<String> pendingSenders = new HashSet<>();

    @Override public void onEnable() {
        stopping = false;
        saveDefaultConfig();
        try {
            worlds = Set.copyOf(getConfig().getStringList("allowed-worlds"));
            if (worlds.isEmpty() || worlds.stream().anyMatch(world -> !BridgeClient.LABEL.matcher(world).matches())) {
                throw new IllegalArgumentException("allowed-worlds must contain valid world labels");
            }
            consoleWorld = getConfig().getString("console-world", "world");
            if (!worlds.contains(consoleWorld)) throw new IllegalArgumentException("console-world must be in allowed-worlds");
            String environmentName = getConfig().getString("token-env", "GAME_MEMORY_TOKEN");
            String token = environmentName.isBlank() ? null : System.getenv(environmentName);
            if (token == null || token.isBlank()) token = getConfig().getString("token", "");
            bridge = new BridgeClient(getConfig().getString("bridge-url", "http://127.0.0.1:8422"), token,
                    Duration.ofSeconds(getConfig().getInt("request-timeout-seconds", 5)));
            service = new QueuedMemoryService(bridge, worlds);
            var command = getCommand("engram");
            if (command == null) throw new IllegalArgumentException("engram command is missing from plugin.yml");
            command.setExecutor(this);
            command.setTabCompleter(this);
            getServer().getServicesManager().register(EngramMemoryService.class, service, this, ServicePriority.Normal);
            getLogger().info("Engram commands ready. Use /engram status to check the local bridge.");
        } catch (IllegalArgumentException failure) {
            // Configuration errors never include token values or response bodies.
            getLogger().severe("Engram configuration is invalid; check bridge-url, token, timeout and allowed-worlds.");
            getServer().getPluginManager().disablePlugin(this);
        }
    }

    @Override public void onDisable() {
        stopping = true;
        getServer().getServicesManager().unregisterAll(this);
        if (service != null) service.close();
        else if (bridge != null) bridge.close();
        pendingSenders.clear();
    }

    @Override public boolean onCommand(CommandSender sender, Command command, String label, String[] args) {
        if (!(sender instanceof Player) && !(sender instanceof ConsoleCommandSender)) {
            send(sender, "Use Engram from a player account or the server console.");
            return true;
        }
        if (args.length == 0 || args[0].equalsIgnoreCase("help")) {
            send(sender, "/engram status | save <rule|build|handoff|note> <key> <text...> | recall <kind> <key>");
            return true;
        }
        String action = args[0].toLowerCase(Locale.ROOT);
        if (!List.of("status", "save", "recall").contains(action)) {
            send(sender, "Unknown action. Use /engram help.");
            return true;
        }
        if (!sender.hasPermission("engram." + action)) {
            send(sender, "You do not have permission to use this Engram command.");
            return true;
        }
        if (action.equals("status")) {
            if (args.length != 1) { send(sender, "Use /engram status."); return true; }
            submit(sender, () -> { bridge.health(); return List.of("Engram bridge and initialized storage are available."); });
            return true;
        }
        if ((action.equals("save") && args.length < 4) || (action.equals("recall") && args.length != 3)) {
            send(sender, "Use /engram save <kind> <key> <text...> or /engram recall <kind> <key>.");
            return true;
        }
        String kind = args[1].toLowerCase(Locale.ROOT);
        String key = args[2];
        // Snapshot game state on the main thread. Worker code never touches a world.
        String world = sender instanceof Player player ? player.getWorld().getName() : consoleWorld;
        try {
            BridgeClient.validateKey(world, kind, key);
            if (!worlds.contains(world)) throw new IllegalArgumentException("This world is not enabled in the plugin configuration.");
            if (action.equals("recall")) {
                submit(sender, () -> ContextView.render(bridge.recall(world, kind, key)));
            } else {
                String summary = String.join(" ", Arrays.copyOfRange(args, 3, args.length));
                if (summary.isBlank() || summary.length() > 3000) throw new IllegalArgumentException("Use 1–3000 characters of note text.");
                if (kind.equals("build")) {
                    if (!(sender instanceof Player player)) throw new IllegalArgumentException("Save a build note in-game so its coordinates come from the server.");
                    Location location = player.getLocation();
                    summary = "location: " + player.getWorld().getKey() + " x=" + location.getBlockX()
                            + " y=" + location.getBlockY() + " z=" + location.getBlockZ() + "\n" + summary;
                }
                String capturedSummary = summary;
                submit(sender, () -> {
                    bridge.save(world, kind, key, capturedSummary);
                    return List.of("Saved " + kind + "/" + key + " for " + world + ". Saving the same key replaces its previous note.");
                });
            }
        } catch (IllegalArgumentException failure) {
            send(sender, failure.getMessage());
        }
        return true;
    }

    private void submit(CommandSender sender, BridgeWork work) {
        String senderId = sender instanceof Player player ? player.getUniqueId().toString() : "console";
        if (stopping || !pendingSenders.add(senderId)) {
            send(sender, "An Engram request is already pending; wait for its result.");
            return;
        }
        var request = service.submit(work::run);
        request.whenComplete((response, failure) -> {
                List<String> capturedResponse = failure == null ? response : List.of(
                        failure instanceof EngramMemoryService.Failure ? failure.getMessage() : "Engram request failed; no result was confirmed.");
                if (stopping) return;
                try {
                    getServer().getScheduler().runTask(this, () -> {
                        pendingSenders.remove(senderId);
                        if (sender instanceof Player player && !player.isOnline()) return;
                        for (String line : capturedResponse) send(sender, line);
                    });
                } catch (IllegalPluginAccessException ignored) {
                    // Plugin stopped between completion and scheduling the reply.
                }
        });
        if (!request.isCompletedExceptionally()) send(sender, "Engram request queued.");
    }

    private static void send(CommandSender sender, String text) {
        sender.sendMessage(Component.text("[engram] " + ContextView.plain(text)));
    }

    @Override public List<String> onTabComplete(CommandSender sender, Command command, String alias, String[] args) {
        List<String> options = new ArrayList<>();
        if (args.length == 1) {
            for (String action : List.of("status", "save", "recall")) if (sender.hasPermission("engram." + action)) options.add(action);
            options.add("help");
        } else if (args.length == 2 && Set.of("save", "recall").contains(args[0].toLowerCase(Locale.ROOT))
                && sender.hasPermission("engram." + args[0].toLowerCase(Locale.ROOT))) {
            options.addAll(List.of("rule", "build", "handoff", "note"));
        }
        String prefix = args.length == 0 ? "" : args[args.length - 1].toLowerCase(Locale.ROOT);
        return options.stream().filter(value -> value.startsWith(prefix)).toList();
    }

    @FunctionalInterface private interface BridgeWork {
        List<String> run() throws BridgeClient.BridgeException;
    }
}
