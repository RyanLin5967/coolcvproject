# Public demo on Cloudflare Pages

The public site is a static, recorded-results presentation. It exposes no training jobs, write API or credentials. Local training remains available through `coveragecv serve`.

1. In Cloudflare, open **Workers & Pages → Create application → Pages → Import an existing Git repository** (some accounts show **Connect to Git**).
2. Connect GitHub and choose **RyanLin5967/coolcvproject**. Set the production branch to **main**.
3. Set the framework to **None**, root directory to **public-demo**, build command to **exit 0**, and build output directory to **.**. Leave environment variables empty.
4. Deploy and check the generated `*.pages.dev` address. The merge interaction, all three benchmark tabs, prediction images and walkthrough should work.
5. Open the Pages project's **Custom domains → Set up a domain**. Enter **coverage.ryanlin.dev** and continue. If the zone is already in your Cloudflare account, accept its proposed DNS record. If DNS is managed elsewhere, add a CNAME named `coverage` targeting the project's actual `*.pages.dev` hostname.
6. Wait for Cloudflare's domain verification and HTTPS certificate. Add the resulting HTTPS address to the GitHub repository's website field.

Register the hostname through Pages before adding a manual CNAME. The `public-demo/_headers` file supplies the static site's security headers. The largest asset is below Pages' 25 MiB per-file limit; no paid compute or video service is needed for this demo.

Official instructions: [Git integration](https://developers.cloudflare.com/pages/get-started/git-integration/), [plain static sites](https://developers.cloudflare.com/pages/framework-guides/deploy-anything/), [custom domains](https://developers.cloudflare.com/pages/configuration/custom-domains/).

## Updating the demo

UI changes are made in `src/coveragecv/workbench/static/`, then copied with:

```sh
python3 scripts/build_public_demo.py
```

To publish newly collected measurements, run the local workbench, then:

```sh
uv run --no-sync python scripts/export_public_demo.py
python3 scripts/build_public_demo.py
```

The exporter checks checkpoint/data/evaluation identities and metric contents, strips machine paths and private provider details, and copies a fixed, evenly spaced set of validation images. It does not train or run inference. Commit the rebuilt `public-demo/` alongside the source changes. Git pushes to `main` will trigger Pages deployments after connecting the repository.

## Demo video

The public merge page has a **Watch the walkthrough** button. It plays `public-demo/media/walkthrough.mp4` with `walkthrough.vtt` captions. It is an actual recorded interface, with no audio or synthetic training progress.

For a personal application video, record your own voice over the same path: missing-label failure → coverage rule → matched benchmark → real predictions. Describe CoverageCV as a coverage-aware dataset compiler and RF-DETR training adapter. Explain the infrastructure behind it: immutable artifacts, provenance, loss/gradient parity, durable workers and verified Roboflow integration. Roboflow is an integration and the source of RF-DETR, not the product being cloned.
