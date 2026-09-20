# A memory filter for Open WebUI

use this source integration to give one Open WebUI user a dedicated Engram
store. ordinary messages recall relevant notes before the model answers.
`/remember <note>` saves a note you explicitly selected. replies and whole
transcripts are never saved automatically.

the complete, editable source is
[`integrations/open_webui/engram_filter.py`](https://github.com/raya-ac/engram/blob/main/integrations/open_webui/engram_filter.py).
it is a Python `Filter` with `inlet` and `outlet` methods, plus configurable
Valves. those extension points and the conversation toggle come from
[Open WebUI's filter interface](https://docs.openwebui.com/features/extensibility/plugin/functions/filter/).

## 1. create a dedicated memory store

install Engram in an isolated environment using the
[installation guide](../getting-started/installation.md). choose new paths:

```sh
engram --config /absolute/chat-memory/config.yaml init --yes --preset portable \
  --db-path /absolute/chat-memory/memory.db
engram --config /absolute/chat-memory/config.yaml doctor --full
```

the paths are placeholders; create their parent directory first. for an existing
dedicated store, keep its config and skip init. review the `llm` settings:
the REST save operation may attempt hypothetical-question generation through
that backend, and hosted providers receive the text they process.

generate a private token:

```sh
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

set it as `web.auth_token` in your private Engram config, keeping the existing
settings. for a same-machine connection, use:

```yaml
web:
  host: 127.0.0.1
  port: 8420
  auth_token: "replace-with-your-generated-token"
```

start that store's API:

```sh
engram --config /absolute/chat-memory/config.yaml serve --web
```

the filter uses the same authenticated API as the workspace. use the token in
its Valve below; do not put it into a public source file or URL.

## 2. add the source to Open WebUI

as an administrator, open the Functions editor, create a function and paste the
complete filter source. save it and attach it to the model you want to use.
Open WebUI documents [function management](https://docs.openwebui.com/features/extensibility/plugin/functions/)
and [Valves](https://docs.openwebui.com/features/extensibility/plugin/development/valves/).

configure these Valves:

| Valve | Value |
|---|---|
| `enabled` | `true` after reviewing the code |
| `base_url` | `http://127.0.0.1:8420` when both processes share the host |
| `api_token` | the same private token as `web.auth_token` |
| `allowed_user_id` | your Open WebUI user ID |
| `top_k` | `3` initially |
| `timeout_seconds` | `30`; allow more time for first model loading if needed |

to find your ID, temporarily leave `allowed_user_id` blank, select the filter
in your conversation and send a message. the filter reports your own ID in a
status event and makes no Engram request. copy it into the Valve and save.

the filter is selectable per conversation. enable its toggle in the chat you
want to use. enabling the function in administration and selecting it in a
conversation are separate steps.

### if Open WebUI runs in a container

`127.0.0.1` means the container itself. use a reachable service address and an
appropriate private network or HTTPS proxy. Engram must listen on an interface
reachable from that network. `allow_private_http` is an explicit opt-in for
unencrypted HTTP away from loopback; it does **not** create a private network
or configure a firewall. leave it false for HTTPS or same-host use.

## 3. verify saving and recall

with the filter selected, send:

```text
/remember The Lantern project uses SQLite for local development.
```

look for the filter's **Engram saved the explicit note** status. then open a
new chat with the filter selected and ask:

```text
Which database does the Lantern project use locally?
```

the filter reports how many reference notes it recalled. inspect the Engram
workspace to confirm the saved note. a model repeating a plausible answer is
not sufficient evidence that the filter ran.

turn the filter off and confirm no Engram status appears. a different user ID
also makes no request. this verifies the two intended boundaries.

## how the source works

the inlet finds the last text user message. it makes an authenticated
`GET /api/search/explain` request, takes up to five bounded memory excerpts,
and adds a separate user-level reference message before the question. it leaves
the system prompt intact and marks recalled text as reference data. the debug
search path does not reinforce memory access counts.

an explicit `/remember` message instead calls `POST /api/remember`. the outlet
returns the response unchanged. HTTP calls have timeouts, response-size limits,
and no redirect following. failures produce a short status without credentials
or raw query text. a timed-out write is **unconfirmed**: check the store before
retrying because the server may already have saved it.

## adapt it to another chat app

keep the same separation: authenticate the app user, choose that user's store,
retrieve before generation, and save only on an explicit action. the HTTP calls
can live in middleware, a chat callback or a Save to memory button. preserve
message attachments; this source deliberately skips multipart messages rather
than flattening images and files into text.

Engram's REST search is store-wide. the user-ID check here restricts who this
filter will serve; it does not turn one Engram database into a multi-tenant
service. use separate stores and tokens for separate users, or implement and
test your own server-side routing. never let the model or message text choose
a token, database path or account.

## verification scope

the repository tests cover explicit saves, reference insertion, repeated-filter
behavior, user separation, multipart preservation, timeouts and HTTP contracts.
they use isolated storage and stubbed model work. this source has not been
accepted inside a running Open WebUI instance; follow the checks above with
your installed version. it is editable integration code, not a hosted service.
