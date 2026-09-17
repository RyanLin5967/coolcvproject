> For the complete documentation index, see [llms.txt](https://docs.roboflow.com/llms.txt). Markdown versions of documentation pages are available by appending `.md` to page URLs; this page is available as [Markdown](https://docs.roboflow.com/deployment/self-hosted/inference-server/configuration/security-migration.md).

# Security Configuration Migration

Check release scope and update self-hosted Inference settings for localhost binding, model authorization, video limits, and runtime hardening.

Use this guide when upgrading a self-hosted Inference Server, CLI, or image. It separates changes to network exposure from changes that require an explicit feature opt-in.

## Release status

This documentation was checked against source on September 11, 2026. A merge into source does not establish which published package or image contains the change.

<table data-search="false"><thead><tr><th>Change</th><th>Source status</th><th>Release scope</th></tr></thead><tbody><tr><td>Localhost publishing and desktop binding, PR #2943</td><td>Merged</td><td>Check the installed CLI and desktop bundle; first released versions have not been verified here.</td></tr><tr><td>Managed pipeline process cap and RAM guard, PR #2944</td><td>Merged</td><td>Requires a server image containing the change; first released version has not been verified here.</td></tr><tr><td>Local model authorization guard, PR #2956</td><td>Merged</td><td>Requires a server build containing the change; first released version has not been verified here.</td></tr><tr><td>Model, media, TLS, gateway, offline execution, and OPC UA hardening, PR #2952</td><td>Pending, reviewed at commit 54a663f1d</td><td>Not a claim about released images. The sections below describe the proposed runtime-hardening build.</td></tr></tbody></table>

Pin the package and image versions you deploy, then confirm that they contain the relevant changes. Existing containers do not acquire these changes merely because source or image defaults change.

## Localhost publishing

The CLI publishes on `127.0.0.1` instead of every host interface for ordinary CPU and GPU launches. Remote clients must use an explicitly exposed address, such as `inference server start --bind-address 0.0.0.0`, with access restricted to trusted clients.

Jetson CLI launches keep `0.0.0.0`, and `--tunnel` also publishes on `0.0.0.0` so the tunnel can connect. Desktop bundles default to `127.0.0.1` and honor `HOST`; manual Docker mappings remain the operator's choice.

### Feature and observability defaults

Custom Python remains enabled with local execution by default; the earlier announcement of a June 19, 2026 default flip no longer applies. Stream API enablement and Workspace authentication retain their existing behavior, with no new stream administration token.

Health, documentation, and metrics endpoints retain their existing API-key exemptions. See [server security](/deployment/self-hosted/inference-server/configuration/security.md) for the network boundary and the endpoint list.

## Pipeline capacity

New process allocation stops at `STREAM_MANAGER_MAX_ACTIVE_PIPELINES`, default `8`, with the effective cap raised to the preload count if higher. Existing idle workers can still be assigned at the cap, and warm-pool replenishment cannot exceed it.

Deployments that need more workers must choose a higher cap after checking capacity. The RAM guard also handles workers without samples instead of failing during initialization; see [Video Configuration](/deployment/self-hosted/inference-server/configuration/video-configuration.md#managed-pipeline-limits).

## Local model authorization

Online servers reject `MODELS_CACHE_AUTH_ENABLED=True` combined with `ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES=True`. Keep local loading disabled when you need per-model API-key checks, or disable model authorization only when you deliberately trust local package callers.

Offline deployments retain the explicit `ALLOW_OFFLINE_MODEL_CACHE_AUTH_BYPASS` requirement. See [Model Package Security](/deployment/self-hosted/inference-server/configuration/model-security.md).

## Video source validation

In the pending runtime-hardening build, raw GStreamer descriptions are rejected by default, and bare filenames such as `video.mp4` must become `./video.mp4`. Use `ALLOW_UNSAFE_GSTREAMER_PIPELINES=True` only for workloads that need and trust raw pipeline interpretation.

The [supported reference formats](/deployment/self-hosted/inference-server/configuration/video-configuration.md#server-side-video-references) include ordinary RTSP and RTMP URLs without that opt-in. Direct local Python media use and files opened on the SDK client are unchanged.

## Model package trust

In the pending runtime-hardening build, JetPack 5.1.1, 6.2.0, and 7.2.0 images switch `ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES` from `True` to `False`. The `inference-models-speed` benchmark also switches its CLI, adapter, and Python helper defaults to reject untrusted packages.

Explicitly opt in with the image setting or benchmark's `--allow-untrusted-packages` flag only when you trust the executable package. See [Model Package Security](/deployment/self-hosted/inference-server/configuration/model-security.md#package-trust-defaults); direct local package access is a separate setting.

Previously published JetPack 6.0 images are not updated by this change, and the reviewed source no longer has that Dockerfile. Inventory those deployments separately.

## TLS client certificates

In the pending runtime-hardening build, the Python Uvicorn, shell Uvicorn, and parallel Gunicorn launchers require a valid client certificate when both `ENABLE_HTTPS=True` and `SSL_CA_CERTS` are configured. Provision client certificates for applications and health or metrics checks before upgrading.

HTTPS without a client CA retains server-only TLS, and HTTP remains the default when HTTPS is disabled. See [Serving Inference over HTTPS](/deployment/self-hosted/inference-server/configuration/https.md#mutual-tls).

### Encrypted keys

The parallel Gunicorn launcher rejects `SSL_KEYFILE_PASSWORD` because it does not support that option. Use a Uvicorn launcher for an encrypted key; both Python and shell Uvicorn support the password.

## Gateway transport

In the pending runtime-hardening build, a bare `SECURE_GATEWAY=host:port` switches from HTTP to HTTPS and emits a migration warning. Use an explicit `https://` URL and configure the gateway certificate trust before upgrading.

Plain HTTP is accepted only for an explicitly configured loopback IP or `localhost`, such as `http://127.0.0.1:8080`; remote and LAN gateways must serve HTTPS. Unsupported schemes, embedded credentials, queries, fragments, and whitespace fail configuration import in both Inference and the separately installed `inference-models` package.

A loopback tunnel must be reachable from the server's own network namespace. Follow [Connecting Inference Servers](/deployment/self-hosted/enterprise/secure-gateway.md#connecting-inference-servers) for gateway setup, including the legacy `LICENSE_SERVER` alias.

## Offline Custom Python

In the pending runtime-hardening build, `OFFLINE_MODE=True` with `WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE=modal` fails startup instead of silently running code locally. Explicitly choose `local` with local execution permitted for trusted code, or retain online connectivity for Modal.

The execution mode accepts `local` and `modal`, ignoring case and surrounding whitespace; unknown values fail startup. This does not disable ordinary local Custom Python.

## OPC UA credentials

In the pending runtime-hardening build, pooled OPC UA sessions are isolated by endpoint, username, and password. A request with an incorrect, empty, or omitted password cannot reuse a session authenticated with another password; credential rotation creates a separate connection.

Workflow authors keep using the same credential fields. Integrations that call the connection manager directly must pass the original password to `release_connection` and `invalidate_connection`, including explicit `None` for anonymous sessions.

## Other fixes

The pending runtime-hardening build fixes a usage-collector stall when the in-memory queue is full; it does not disable usage collection or change monitoring opt-ins. CI shell-input validation and review-tool installation changes do not require deployment configuration changes.
