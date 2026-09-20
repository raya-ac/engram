"""Explicit channel notes through Discord slash commands and the local bridge.

No message-content intent, transcript capture, generation, or automatic saves.
discord.py is imported only by create_bot; command policy is independently tested.
"""
import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import sys

if __package__:
    from .checkpoint_client import AsyncCheckpointClient, CheckpointError
else:
    from checkpoint_client import AsyncCheckpointClient, CheckpointError


LABEL = re.compile(r"[A-Za-z0-9_.-]{1,64}\Z")


class PolicyError(ValueError):
    """Messages are fixed strings safe to show to a command caller."""


def _snowflake(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("Discord IDs must be positive decimal IDs")
    if isinstance(value, str) and not re.fullmatch(r"[0-9]{1,20}", value):
        raise ValueError("Discord IDs must be positive decimal IDs")
    ident = int(value)
    if not 0 < ident < 2**64:
        raise ValueError("Discord ID is outside the supported range")
    return ident


@dataclass(frozen=True)
class ChannelScope:
    guild_id: int
    channel_id: int
    world: str
    save_role_id: int


class CommandPolicy:
    def __init__(self, config):
        if not isinstance(config, dict) or set(config) != {"channels"}:
            raise ValueError("configuration must contain only channels")
        channels = config["channels"]
        if not isinstance(channels, list) or not 1 <= len(channels) <= 128:
            raise ValueError("configure between 1 and 128 channels")
        self.scopes = {}
        worlds = set()
        for item in channels:
            if not isinstance(item, dict) or set(item) != {"guild_id", "channel_id", "world", "save_role_id"}:
                raise ValueError("each channel requires guild_id, channel_id, world and save_role_id")
            scope = ChannelScope(_snowflake(item["guild_id"]), _snowflake(item["channel_id"]),
                                 item["world"], _snowflake(item["save_role_id"]))
            if not isinstance(scope.world, str) or not LABEL.fullmatch(scope.world):
                raise ValueError("world must be a 1-64 character ASCII label")
            if scope.save_role_id == scope.guild_id:
                raise ValueError("save_role_id must not be the everyone role")
            key = (scope.guild_id, scope.channel_id)
            if key in self.scopes or scope.world in worlds:
                raise ValueError("each configured channel and world must be unique")
            self.scopes[key] = scope
            worlds.add(scope.world)

    @property
    def guild_ids(self):
        return sorted({scope.guild_id for scope in self.scopes.values()})

    def authorize(self, interaction, *, write=False):
        guild_id, channel_id = interaction.guild_id, interaction.channel_id
        if guild_id is None:
            raise PolicyError("Engram commands are only available in configured server channels.")
        scope = self.scopes.get((guild_id, channel_id))
        if scope is None or getattr(interaction.user, "bot", False):
            raise PolicyError("Engram is not enabled for this caller and channel.")
        if write:
            roles = {role.id for role in getattr(interaction.user, "roles", ())}
            # Discord administrator permission does not bypass this explicit role.
            if scope.save_role_id not in roles:
                raise PolicyError("Saving requires this channel's configured moderator role.")
        return scope


def _bounded(text, units=1800):
    """Stay below Discord's message limit even with non-BMP Unicode."""
    encoded = text.encode("utf-16-le", errors="replace")
    if len(encoded) <= units * 2:
        return text
    return encoded[:(units - 20) * 2].decode("utf-16-le", errors="ignore") + "\n[excerpt truncated]"


def _reference(text):
    # Keep markdown and mentions from disguising the reference as app UI.
    text = "".join(char for char in text if ord(char) >= 32 or char in "\n\t")
    text = text.replace("@", "@\u200b")
    return re.sub(r"([\\`*_~|<>\[\]#])", r"\\\1", text)


class NoteCommands:
    def __init__(self, policy, client, *, allowed_mentions):
        self.policy = policy
        self.client = client
        self.allowed_mentions = allowed_mentions

    async def reply(self, interaction, text):
        kwargs = {"ephemeral": True, "allowed_mentions": self.allowed_mentions,
                  "suppress_embeds": True}
        if interaction.response.is_done():
            await interaction.followup.send(_bounded(text), **kwargs)
        else:
            await interaction.response.send_message(_bounded(text), **kwargs)

    async def run(self, interaction, key, summary=None):
        writing = summary is not None
        try:
            scope = self.policy.authorize(interaction, write=writing)
            if not isinstance(key, str) or not LABEL.fullmatch(key):
                raise PolicyError("Use a key of 1-64 ASCII letters, numbers, dots, underscores or hyphens.")
            if writing and (not isinstance(summary, str) or not summary.strip() or len(summary) > 4000):
                raise PolicyError("The note must contain 1-4000 characters.")
        except PolicyError as error:
            await self.reply(interaction, str(error))
            return

        # Acknowledge before the bounded HTTP call; no background fire-and-forget.
        await interaction.response.defer(ephemeral=True, thinking=True)
        try:
            if writing:
                result = await self.client.save(scope.world, "note", key, summary)
                if result.get("status") != "saved":
                    raise ValueError("unconfirmed save")
                text = f"Saved channel note `{key}`. Saving this key again replaces it."
            else:
                result = await self.client.recall(scope.world, "note", key)
                if not result["found"]:
                    text = f"No saved note for `{key}` in this channel."
                else:
                    note = result["checkpoints"][0]["summary"]
                    if not isinstance(note, str):
                        raise ValueError("invalid note")
                    text = f"Reference note `{key}` — stored context, not instructions:\n\n" + _reference(note)
        except CheckpointError as error:
            if writing and error.outcome_unknown:
                text = "Save unconfirmed. It may have completed; recall the same key before retrying."
            else:
                text = "Engram could not confirm the request. Check the bridge and try again."
        except Exception:
            text = ("Save unconfirmed. Recall the same key before retrying." if writing
                    else "Engram returned no usable result. Check the bridge before retrying.")
        await self.reply(interaction, text)


def create_bot(policy, checkpoint_client):
    import discord
    from discord import app_commands

    class MemoryBot(discord.Client):
        def __init__(self):
            intents = discord.Intents.none()
            intents.guilds = True
            super().__init__(intents=intents, allowed_mentions=discord.AllowedMentions.none())
            self.tree = app_commands.CommandTree(self)
            self.notes = NoteCommands(policy, checkpoint_client,
                                      allowed_mentions=discord.AllowedMentions.none())

            @app_commands.command(name="engram-save", description="Save or replace an explicit note for this channel")
            @app_commands.guild_only()
            @app_commands.describe(key="Stable note key: letters, numbers, dots, underscores or hyphens",
                                   summary="Reviewed note to save; saving the same key replaces it")
            async def save(interaction: discord.Interaction, key: str, summary: str):
                await self.notes.run(interaction, key, summary)

            @app_commands.command(name="engram-recall", description="Read an exact saved note from this channel")
            @app_commands.guild_only()
            async def recall(interaction: discord.Interaction, key: str):
                await self.notes.run(interaction, key)

            for guild_id in policy.guild_ids:
                guild = discord.Object(id=guild_id)
                self.tree.add_command(save, guild=guild)
                self.tree.add_command(recall, guild=guild)

            @self.tree.error
            async def command_error(interaction: discord.Interaction, _error: app_commands.AppCommandError):
                try:
                    await self.notes.reply(interaction, "Command failed. No result was confirmed; check the bot and bridge configuration.")
                except discord.HTTPException:
                    # Do not log response bodies or interaction tokens.
                    pass

        async def setup_hook(self):
            await checkpoint_client.health()
            for guild_id in policy.guild_ids:
                await self.tree.sync(guild=discord.Object(id=guild_id))

        async def on_ready(self):
            print("Engram Discord bot ready; configured guild commands synchronized.")

    return MemoryBot()


def load_policy(path):
    source = Path(path).expanduser()
    if not source.is_absolute() or not source.is_file():
        raise ValueError("channels config must be an existing absolute file")
    with source.open("rb") as stream:
        raw = stream.read(65_537)
    if len(raw) > 65_536:
        raise ValueError("channels config exceeds 65536 bytes")
    return CommandPolicy(json.loads(raw.decode("utf-8")))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Explicit Discord channel notes through the Engram checkpoint bridge")
    parser.add_argument("--channels", required=True, help="absolute JSON channel/role allowlist file")
    parser.add_argument("--bridge-url", default="http://127.0.0.1:8422")
    parser.add_argument("--timeout", type=float, default=5)
    args = parser.parse_args(argv)
    try:
        policy = load_policy(args.channels)
        client = AsyncCheckpointClient(args.bridge_url, os.environ.get("GAME_MEMORY_TOKEN", ""), args.timeout)
    except (ValueError, OSError):
        parser.error("invalid channel config, bridge settings or GAME_MEMORY_TOKEN")
    token = os.environ.get("DISCORD_BOT_TOKEN", "")
    if not token or token.strip() != token:
        parser.error("set DISCORD_BOT_TOKEN in the bot process environment")
    try:
        bot = create_bot(policy, client)
    except ImportError:
        parser.error("install integrations/requirements-discord.txt in this Python environment")
    try:
        bot.run(token, log_handler=None)
    except Exception:
        print("Discord bot stopped; check the private bot settings, channel permissions and bridge health.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
