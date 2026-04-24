#!/usr/bin/env python3
"""Find all registry tags pointing to the same digest as a local Docker image.

Usage:
    python3 find_docker_tags.py <image>

Example:
    python3 find_docker_tags.py ghcr.io/tenstorrent/tt-xla/tt-xla-ird-ubuntu-24-04:latest
"""

import sys
import json
import subprocess
import urllib.request
import urllib.error


def get_local_digest(image):
    result = subprocess.run(
        ["docker", "inspect", image, "--format", "{{json .RepoDigests}}"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Error: could not inspect image '{image}'", file=sys.stderr)
        sys.exit(1)
    digests = json.loads(result.stdout.strip())
    if not digests:
        print("Error: image has no RepoDigests (locally built image?)", file=sys.stderr)
        sys.exit(1)
    return digests[0]  # e.g. ghcr.io/org/repo@sha256:abc123...


def parse_repo_digest(repo_digest):
    """Return (registry, repo_path, digest) from 'registry/org/repo@sha256:...'"""
    at = repo_digest.rfind("@")
    digest = repo_digest[at + 1:]
    full_name = repo_digest[:at]
    parts = full_name.split("/")
    if "." in parts[0] or ":" in parts[0]:
        registry = parts[0]
        repo_path = "/".join(parts[1:])
    else:
        registry = "registry-1.docker.io"
        repo_path = full_name
    return registry, repo_path, digest


def get_token(registry, repo_path):
    url = f"https://{registry}/token?scope=repository:{repo_path}:pull&service={registry}"
    try:
        resp = urllib.request.urlopen(url)
        return json.loads(resp.read()).get("token", "")
    except urllib.error.HTTPError:
        return ""


def registry_get(registry, path, token, accept=None):
    url = f"https://{registry}/v2/{path}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    if accept:
        headers["Accept"] = accept
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers))


def get_tags(registry, repo_path, token):
    resp = registry_get(registry, f"{repo_path}/tags/list", token)
    return json.loads(resp.read()).get("tags", [])


def get_manifest_digest(registry, repo_path, token, tag):
    resp = registry_get(registry, f"{repo_path}/manifests/{tag}", token,
                        accept="application/vnd.docker.distribution.manifest.v2+json")
    return resp.headers.get("Docker-Content-Digest", "")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <image>", file=sys.stderr)
        sys.exit(1)

    image = sys.argv[1]
    print(f"Inspecting local image: {image}")

    repo_digest = get_local_digest(image)
    registry, repo_path, target_digest = parse_repo_digest(repo_digest)

    print(f"Registry:     {registry}")
    print(f"Repository:   {repo_path}")
    print(f"Digest:       {target_digest}")
    print(f"Querying registry for all tags ({registry})...")

    token = get_token(registry, repo_path)
    tags = get_tags(registry, repo_path, token)
    print(f"Checking {len(tags)} tags...\n")

    matches = []
    for tag in tags:
        try:
            if get_manifest_digest(registry, repo_path, token, tag) == target_digest:
                matches.append(tag)
        except urllib.error.HTTPError as e:
            print(f"  Warning: could not check tag '{tag}': {e}", file=sys.stderr)

    if matches:
        print(f"Tags pointing to {target_digest}:")
        for tag in matches:
            print(f"  {tag}")
    else:
        print(f"No tags found matching {target_digest}")


if __name__ == "__main__":
    main()
