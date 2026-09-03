#!/usr/bin/env python3
"""Report what review approvals a GitHub PR still needs, and who can give them.

Answers the practical question "am I done, and who do I ping?" by combining four
sources GitHub never shows together:

  1. the branch's ``pull_request`` rules (how many approvals, is code-owner
     review required) -- the union of every ruleset that applies, since GitHub
     enforces the strictest,
  2. CODEOWNERS resolved against the PR's changed files, grouped into ownership
     groups (files that share an identical owner set),
  3. the reviews already submitted, expanded through team membership, to mark
     each group satisfied or pending,
  4. recent merged-PR approval history, to rank who actually signs off for each
     pending group.

With ``require_code_owner_review`` the headline "N approvals required" is
misleading: the real requirement is >=N approvals AND every owned path approved
by one of its owners. So the number of people you need is however many it takes
to cover the pending groups -- computed here as a minimum cover, which also
reveals when one person can clear two groups at once.

Usage:
    pr_review_status.py <pr-number-or-url> [--repo owner/name] [options]

Requires the `gh` CLI, authenticated with `read:org` (for team membership).
"""

import argparse
import base64
import json
import re
import subprocess
import sys
from collections import defaultdict

BOT_SUFFIX = "[bot]"
# Review states that do not carry a verdict and so never replace a user's
# effective review state (GitHub applies the same rule).
NON_VERDICT_STATES = {"COMMENTED", "PENDING"}


# --------------------------------------------------------------------------- #
# gh plumbing
# --------------------------------------------------------------------------- #

def gh(args, parse_json=True, check=True):
    """Run `gh`. Returns parsed JSON, or None when check=False and it failed."""
    out = subprocess.run(["gh"] + args, capture_output=True, text=True)
    if out.returncode != 0:
        if not check:
            return None
        sys.exit(f"gh {' '.join(args)} failed:\n{out.stderr}")
    if not parse_json:
        return out.stdout
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError:
        if not check:
            return None
        sys.exit(f"gh {' '.join(args)} returned non-JSON output:\n{out.stdout[:400]}")


def gh_graphql(query, variables):
    args = ["api", "graphql", "-f", f"query={query}"]
    for k, v in variables.items():
        args += ["-F", f"{k}={v}"]
    return gh(args)


def parse_pr(arg, repo_flag):
    """Return (owner, repo, number) from a URL, or a number plus --repo/cwd."""
    m = re.search(r"github\.com/([^/]+)/([^/]+)/pull/(\d+)", arg)
    if m:
        return m.group(1), m.group(2), int(m.group(3))
    if not arg.isdigit():
        sys.exit(f"Cannot parse '{arg}' as a PR number or URL.")
    if repo_flag:
        if "/" not in repo_flag:
            sys.exit("--repo must be owner/name")
        owner, repo = repo_flag.split("/", 1)
        return owner, repo, int(arg)
    info = gh(["repo", "view", "--json", "owner,name"], check=False)
    if not info:
        sys.exit("Not in a GitHub repo -- pass a full PR URL or --repo owner/name.")
    return info["owner"]["login"], info["name"], int(arg)


# --------------------------------------------------------------------------- #
# Branch rules: what does merging actually require?
# --------------------------------------------------------------------------- #

def branch_review_rules(owner, repo, branch):
    """Union of every pull_request rule applying to `branch`.

    GitHub evaluates all applicable rulesets and enforces the strictest, so the
    union (max of counts, OR of flags) is what a merge is held to.
    """
    result = {
        "source": None,
        "rule_count": 0,
        "required_approving_review_count": 0,
        "require_code_owner_review": False,
        "dismiss_stale_reviews_on_push": False,
        "require_last_push_approval": False,
        "required_review_thread_resolution": False,
        "require_extra_approval_for_unattributed_changes": False,
    }
    flags = [k for k in result if isinstance(result[k], bool)]

    rules = gh(["api", f"repos/{owner}/{repo}/rules/branches/{branch}"], check=False)
    pr_rules = [r for r in (rules or []) if r.get("type") == "pull_request"]
    if pr_rules:
        result["source"] = "rulesets"
        result["rule_count"] = len(pr_rules)
        for r in pr_rules:
            p = r.get("parameters") or {}
            result["required_approving_review_count"] = max(
                result["required_approving_review_count"],
                p.get("required_approving_review_count", 0),
            )
            for f in flags:
                result[f] = result[f] or bool(p.get(f, False))
        return result

    # Classic branch protection (needs admin; tolerate a 403/404).
    prot = gh(["api", f"repos/{owner}/{repo}/branches/{branch}/protection"], check=False)
    rev = (prot or {}).get("required_pull_request_reviews")
    if rev:
        result["source"] = "branch protection"
        result["rule_count"] = 1
        result["required_approving_review_count"] = rev.get("required_approving_review_count", 0)
        result["require_code_owner_review"] = bool(rev.get("require_code_owner_reviews"))
        result["dismiss_stale_reviews_on_push"] = bool(rev.get("dismiss_stale_reviews"))
        result["require_last_push_approval"] = bool(rev.get("require_last_push_approval"))
    return result


# --------------------------------------------------------------------------- #
# CODEOWNERS
# --------------------------------------------------------------------------- #

CODEOWNERS_PATHS = (".github/CODEOWNERS", "CODEOWNERS", "docs/CODEOWNERS")


def fetch_codeowners(owner, repo, ref=None):
    """Return (text, path) for the first CODEOWNERS GitHub would honor."""
    for path in CODEOWNERS_PATHS:
        endpoint = f"repos/{owner}/{repo}/contents/{path}"
        if ref:
            endpoint += f"?ref={ref}"
        content = gh(["api", endpoint, "--jq", ".content"], parse_json=False, check=False)
        if content and content.strip():
            return base64.b64decode(content).decode("utf-8", "replace"), path
    return "", None


def _pattern_to_regex(pattern):
    """Compile a CODEOWNERS pattern for matching repo-relative paths.

    Follows the rules GitHub documents, which are gitignore-like with one
    notable difference: a wildcard in the final segment does not recurse, so
    `docs/*` owns `docs/a.md` but not `docs/sub/b.md`, while the literal
    `docs/sub` owns that directory and everything under it.
    """
    dir_only = pattern.endswith("/")
    core = pattern[:-1] if dir_only else pattern
    # A slash at the start or middle anchors the pattern to the repo root; a
    # trailing slash does not, so `apps/` matches an apps dir at any depth.
    anchored = core.startswith("/") or "/" in core.strip("/")
    core = core.lstrip("/")

    body, i, n = "", 0, len(core)
    while i < n:
        if core.startswith("/**", i) and i + 3 == n:
            body += "/.*"                  # trailing "/**": everything below
            i += 3
        elif core.startswith("**/", i):
            body += "(?:.*/)?"             # "**/" spans zero or more dirs
            i += 3
        elif core.startswith("**", i):
            body += ".*"
            i += 2
        elif core[i] == "*":
            body += "[^/]*"                # a single "*" never crosses "/"
            i += 1
        elif core[i] == "?":
            body += "[^/]"
            i += 1
        else:
            body += re.escape(core[i])
            i += 1

    if dir_only:
        body += "/.*"                      # "dir/" owns the contents
    elif not any(c in core.rsplit("/", 1)[-1] for c in "*?"):
        body += "(/.*)?"                   # literal name: the file, or a dir's contents
    return re.compile(("^" if anchored else "^(?:.*/)?") + body + "$")


def parse_codeowners(text):
    """Return ordered [(pattern, regex, owners)]; the LAST match wins."""
    rules = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("["):   # skip gitlab-style section headers
            continue
        parts = line.split()
        rules.append((parts[0], _pattern_to_regex(parts[0]), parts[1:]))
    return rules


def match_owners(path, rules):
    """Last matching rule wins (GitHub semantics). Returns (owners, pattern)."""
    owners, pattern = [], None
    for pat, regex, owns in rules:
        if regex.match(path):
            owners, pattern = owns, pat
    return owners, pattern


# --------------------------------------------------------------------------- #
# PR facts
# --------------------------------------------------------------------------- #

def pr_overview(owner, repo, number):
    return gh([
        "pr", "view", str(number), "--repo", f"{owner}/{repo}", "--json",
        "number,title,state,isDraft,author,baseRefName,headRefOid,url,"
        "reviewDecision,mergeable,mergeStateStatus,changedFiles",
    ])


def changed_files(owner, repo, number):
    files = gh(["api", f"repos/{owner}/{repo}/pulls/{number}/files", "--paginate"])
    return [f["filename"] for f in files]


def effective_reviews(owner, repo, number):
    """login -> {state, commit, submitted_at} using each user's latest verdict.

    Plain comments never overwrite a verdict, matching how GitHub decides
    whether a user's approval still stands.
    """
    revs = gh(["api", f"repos/{owner}/{repo}/pulls/{number}/reviews", "--paginate"])
    latest = {}
    for r in revs:
        login = (r.get("user") or {}).get("login")
        state = r.get("state")
        if not login or state in NON_VERDICT_STATES:
            continue
        latest[login] = {
            "state": state,
            "commit": r.get("commit_id") or "",
            "submitted_at": r.get("submitted_at") or "",
        }
    return latest


def requested_reviewers(owner, repo, number):
    data = gh(["api", f"repos/{owner}/{repo}/pulls/{number}/requested_reviewers"])
    return (
        [u["login"] for u in data.get("users", [])],
        [t["slug"] for t in data.get("teams", [])],
    )


class TeamCache:
    """Team slug -> member logins. GitHub includes child-team members."""

    def __init__(self, org):
        self.org = org
        self._cache = {}
        self.errors = {}

    def members(self, slug):
        if slug not in self._cache:
            data = gh(
                ["api", f"orgs/{self.org}/teams/{slug}/members", "--paginate"],
                check=False,
            )
            if data is None:
                self.errors[slug] = "not readable (needs read:org, or team not in this org)"
                self._cache[slug] = []
            else:
                self._cache[slug] = [m["login"] for m in data]
        return self._cache[slug]


# --------------------------------------------------------------------------- #
# Ownership groups
# --------------------------------------------------------------------------- #

def ownership_groups(files, rules):
    """Group changed files by identical owner set.

    Each group is one code-owner requirement: any single owner of the group can
    satisfy every file in it.
    """
    groups = defaultdict(lambda: {"files": [], "patterns": set()})
    unowned = []
    for path in files:
        owners, pattern = match_owners(path, rules)
        if not owners:
            unowned.append(path)
            continue
        key = tuple(sorted(owners, key=str.lower))
        groups[key]["files"].append(path)
        groups[key]["patterns"].add(pattern)

    out = []
    for owners, info in groups.items():
        out.append({
            "owners": list(owners),
            "files": sorted(info["files"]),
            "patterns": sorted(info["patterns"]),
        })
    out.sort(key=lambda g: (-len(g["files"]), g["owners"]))
    return out, sorted(unowned)


def expand_group(group, org, teams):
    """Split a group's owners into teams/users and resolve eligible approvers."""
    prefix = f"@{org}/"
    group["teams"] = [o[len(prefix):] for o in group["owners"] if o.lower().startswith(prefix.lower())]
    group["users"] = [
        o[1:] for o in group["owners"]
        if o.startswith("@") and not o.lower().startswith(prefix.lower())
    ]
    # A team owner from another org cannot be resolved; keep it visible.
    group["foreign"] = [o for o in group["owners"] if not o.startswith("@")]
    eligible = set(group["users"])
    for slug in group["teams"]:
        eligible.update(teams.members(slug))
    group["eligible"] = eligible
    return group


def classify_groups(groups, reviews, author):
    """Mark each group satisfied / changes-requested / pending.

    The PR author cannot approve their own PR, so they never satisfy a group.
    """
    approvers = {u for u, r in reviews.items() if r["state"] == "APPROVED" and u != author}
    blockers = {u for u, r in reviews.items() if r["state"] == "CHANGES_REQUESTED"}
    for g in groups:
        g["approved_by"] = sorted(g["eligible"] & approvers, key=str.lower)
        g["changes_requested_by"] = sorted(g["eligible"] & blockers, key=str.lower)
        if g["changes_requested_by"]:
            g["status"] = "changes_requested"
        elif g["approved_by"]:
            g["status"] = "satisfied"
        else:
            g["status"] = "pending"
    return groups


def min_approver_cover(pending, score):
    """Smallest set of people whose reviews would clear every pending group.

    This is a minimum set cover, solved by iterative deepening with the
    most-constrained-group-first branching rule, so the answer is provably
    minimal rather than merely sufficient -- which is the whole point: it is
    what tells you two groups can collapse onto one reviewer, or cannot.
    Falls back to greedy (flagged inexact) if the search blows its budget.

    Returns (cover, exact).
    """
    if not pending:
        return [], True

    masks = {}
    for i, g in enumerate(pending):
        for login in g["candidates"]:
            masks[login] = masks.get(login, 0) | (1 << i)
    if not masks:
        return [], True

    full = (1 << len(pending)) - 1

    # One representative per distinct coverage, preferring whoever approves most.
    by_mask = {}
    for login, mask in masks.items():
        if mask not in by_mask or score(login) > score(by_mask[mask]):
            by_mask[mask] = login
    # A coverage that is a subset of another can never be needed for a minimum
    # cover, so dropping it shrinks the search without changing the answer.
    ordered = sorted(by_mask.items(), key=lambda kv: -bin(kv[0]).count("1"))
    kept = []
    for mask, login in ordered:
        if not any(mask & k == mask for k, _ in kept):
            kept.append((mask, login))

    covering = [[c for c in kept if c[0] >> i & 1] for i in range(len(pending))]
    budget = [100_000]
    best = None

    def rec(covered, chosen, depth):
        nonlocal best
        if covered == full:
            total = sum(score(l) for _, l in chosen)
            if best is None or total > best[0]:
                best = (total, [l for _, l in chosen])
            return True
        if depth == 0:
            return False
        budget[0] -= 1
        if budget[0] <= 0:
            return False
        # Branch on the hardest-to-cover group; any cover must include one of
        # its owners, so this enumerates without duplicating permutations.
        idx = min((i for i in range(len(pending)) if not covered >> i & 1),
                  key=lambda i: len(covering[i]))
        found = False
        for cand in covering[idx]:
            chosen.append(cand)
            found |= rec(covered | cand[0], chosen, depth - 1)
            chosen.pop()
        return found

    for size in range(1, len(pending) + 1):
        best = None
        if rec(0, [], size) and best:
            return best[1], True
        if budget[0] <= 0:
            break

    cover, covered = [], 0
    for mask, login in sorted(kept, key=lambda p: (-bin(p[0]).count("1"), -score(p[1]))):
        if mask & ~covered:
            cover.append(login)
            covered |= mask
        if covered == full:
            break
    return cover, False


# --------------------------------------------------------------------------- #
# Approval history -- who actually signs off
# --------------------------------------------------------------------------- #

HISTORY_QUERY = """
query($owner:String!, $repo:String!, $cursor:String) {
  repository(owner:$owner, name:$repo) {
    pullRequests(states:MERGED, first:50,
                 orderBy:{field:UPDATED_AT, direction:DESC}, after:$cursor) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number
        reviews(first:100) {
          nodes { state submittedAt author { login } }
        }
      }
    }
  }
}
"""


def approval_history(owner, repo, scan):
    """Up to `scan` recently-updated merged PRs, newest first, with approvers."""
    prs, cursor = [], None
    while len(prs) < scan:
        variables = {"owner": owner, "repo": repo}
        if cursor:
            variables["cursor"] = cursor
        data = gh_graphql(HISTORY_QUERY, variables)
        conn = data["data"]["repository"]["pullRequests"]
        for node in conn["nodes"]:
            approvers = {}
            for rev in node["reviews"]["nodes"]:
                if rev["state"] == "APPROVED" and rev.get("author"):
                    login = rev["author"]["login"]
                    when = rev.get("submittedAt") or ""
                    approvers[login] = max(approvers.get(login, ""), when)
            prs.append({"number": node["number"], "approvers": approvers})
        if not conn["pageInfo"]["hasNextPage"]:
            break
        cursor = conn["pageInfo"]["endCursor"]
    return prs[:scan]


def user_stats(prs):
    """login -> {approvals, last} across the scanned window."""
    stats = defaultdict(lambda: {"approvals": 0, "last": ""})
    for pr in prs:
        for login, when in pr["approvers"].items():
            s = stats[login]
            s["approvals"] += 1
            s["last"] = max(s["last"], when)
    return stats


def rank_approvers(candidates, prs, last_n):
    """Rank `candidates` by approvals over the most recent `last_n` PRs any of
    them approved -- i.e. the PRs that this ownership group actually reviewed.

    Scoping to the group's own PRs (rather than the whole window) keeps a
    low-traffic team's ranking meaningful instead of drowning it in repo noise.
    """
    cand = set(candidates)
    counts, lasts, used = defaultdict(int), {}, []
    for pr in prs:                                    # newest first
        hits = {l: w for l, w in pr["approvers"].items() if l in cand}
        if not hits:
            continue
        used.append(pr["number"])
        for login, when in hits.items():
            counts[login] += 1
            lasts[login] = max(lasts.get(login, ""), when)
        if len(used) >= last_n:
            break
    ranked = [
        {"login": l, "approvals": c, "last": lasts.get(l, "")}
        for l, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0].lower()))
    ]
    return ranked, used


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

MARK = {"satisfied": "OK", "pending": "PENDING", "changes_requested": "CHANGES REQ"}
DAY = slice(0, 10)


def plural(n, word):
    return f"{word}s" if n != 1 else word


def fmt_owners(group, org):
    """Shorten @org/team to @team so the owner column stays readable."""
    return " ".join(
        o.replace(f"@{org}/", "@") if o.lower().startswith(f"@{org}/".lower()) else o
        for o in group["owners"]
    )


def render_text(rep, args):
    pr, rules = rep["pr"], rep["rules"]
    org = rep["org"]
    out = []
    w = out.append

    w(f"PR {rep['repo']}#{pr['number']} - {pr['title']}")
    w(f"  {pr['url']}")
    extra = " (DRAFT)" if pr["isDraft"] else ""
    w(f"  author @{pr['author']}   base {pr['baseRefName']}   state {pr['state']}{extra}")
    w(f"  GitHub review decision: {pr.get('reviewDecision') or 'none'}"
      f"   mergeable: {pr.get('mergeable')}   merge state: {pr.get('mergeStateStatus')}")
    w("")

    if rules["source"]:
        w(f"Review rules on '{pr['baseRefName']}' (from {rules['source']}"
          + (f", union of {rules['rule_count']} pull_request rules -- GitHub enforces"
             " the strictest" if rules["rule_count"] > 1 else "")
          + "):")
    else:
        w(f"Review rules on '{pr['baseRefName']}': not readable"
          " (classic branch protection needs admin) -- assuming none:")
    w(f"  {'approvals required':<31} : {rules['required_approving_review_count']}")
    w(f"  {'code owner review required':<31} : "
      f"{'yes' if rules['require_code_owner_review'] else 'no'}")
    for key, label in (
        ("dismiss_stale_reviews_on_push", "dismiss stale reviews on push"),
        ("require_last_push_approval", "require last push approval"),
        ("required_review_thread_resolution", "review threads must resolve"),
        ("require_extra_approval_for_unattributed_changes", "extra approval for unattributed"),
    ):
        if rules[key]:
            w(f"  {label:<31} : yes")
    if not rules["source"]:
        w("  Trust GitHub's review decision above over this section.")
    w("")

    counted = rep["counted_approvals"]
    need = rules["required_approving_review_count"]
    if counted:
        w(f"Approvals so far ({len(counted)}/{need} count requirement): "
          + ", ".join("@" + u for u in counted))
    else:
        w(f"Approvals so far (0/{need} count requirement): none")
    for u, r in sorted(rep["reviews"].items()):
        if r["state"] == "CHANGES_REQUESTED":
            w(f"  !! changes requested by @{u}")
        if r.get("stale"):
            w(f"  !! @{u}'s {r['state'].lower()} is on an older commit"
              f" ({r['commit'][:8]}) and is dismissed on push")
    w("")

    if not rep["codeowners_path"]:
        w("No CODEOWNERS file found -- ownership groups unavailable.")
    else:
        groups = rep["groups"]
        w(f"Code ownership of {len(rep['files'])} changed file(s) via {rep['codeowners_path']}"
          f" -> {len(groups)} group(s):")
        w("")
        wid = max([len(fmt_owners(g, org)) for g in groups] + [20])
        w(f"  {'':<13}  {'files':>5}  {'owners':<{wid}}  detail")
        for g in groups:
            detail = ""
            if g["status"] == "satisfied":
                detail = "approved by " + ", ".join("@" + u for u in g["approved_by"])
            elif g["status"] == "changes_requested":
                detail = "changes requested by " + ", ".join("@" + u for u in g["changes_requested_by"])
            elif not g["eligible"]:
                detail = "no resolvable owners (team membership unreadable?)"
            elif g.get("ranked"):
                detail = "ping " + ", ".join("@" + r["login"] for r in g["ranked"][:2])
            w(f"  [{MARK[g['status']]:<11}]  {len(g['files']):>5}  "
              f"{fmt_owners(g, org):<{wid}}  {detail}")
        if rep["unowned"]:
            w(f"  [{'unowned':<11}]  {len(rep['unowned']):>5}  "
              f"{'-':<{wid}}  no CODEOWNERS entry")
        w("")

        for g in groups:
            if g["status"] == "satisfied" and not args.all_groups:
                continue
            w(f"-- {MARK[g['status']]}: {fmt_owners(g, org)} "
              f"({len(g['files'])} file(s)) --")
            shown = g["files"] if args.files else g["files"][:args.max_files]
            for f in shown:
                w(f"     {f}")
            if len(g["files"]) > len(shown):
                w(f"     ... and {len(g['files']) - len(shown)} more (--files to list all)")
            if g["patterns"]:
                w(f"     matched CODEOWNERS pattern(s): {', '.join(g['patterns'])}")
            for slug in g["teams"]:
                err = rep["team_errors"].get(slug)
                w(f"     team @{org}/{slug}: "
                  + (err if err else f"{len(rep['team_members'].get(slug, []))} members"))
            if g["status"] != "satisfied":
                if g.get("author_is_owner"):
                    w(f"     note: PR author @{pr['author']} owns this path but cannot self-approve")
                ranked = g.get("ranked", [])
                if ranked:
                    w(f"     who signs off here (last {len(g.get('history_prs', []))}"
                      f" PR(s) this group approved):")
                    for r in ranked[:args.top]:
                        w(f"       {r['approvals']:>2} "
                          f"{plural(r['approvals'], 'approval'):<9}  "
                          f"@{r['login']:<22} last {r['last'][DAY] or '-'}")
                else:
                    if rep["history_scanned"]:
                        w(f"     no approvals from these owners in the last"
                          f" {rep['history_scanned']} merged PRs (try --scan)")
                    pool = g.get("owner_pool", [])
                    w(f"     {len(pool)} eligible approver(s)"
                      + (f": {', '.join('@' + c for c in pool[:12])}"
                         + (" ..." if len(pool) > 12 else "") if pool else ""))
            w("")

    pending = [g for g in rep["groups"] if g["status"] != "satisfied"]
    w("Bottom line:")
    if pr["state"] != "OPEN":
        w(f"  This PR is {pr['state']} -- the review state below is historical.")
    if not rep["codeowners_path"]:
        w(f"  {len(counted)}/{need} approvals; no CODEOWNERS to check.")
    elif not pending:
        if len(counted) >= need:
            w("  All code-owner groups satisfied and the approval count is met."
              + ("" if pr["state"] != "OPEN" else " Ready to merge."))
        else:
            w(f"  All code-owner groups satisfied, but only {len(counted)}/{need}"
              f" approvals -- {need - len(counted)} more from anyone with write access.")
    else:
        blocked = [g for g in pending if g["status"] == "changes_requested"]
        cover = rep["cover"]
        w(f"  {len(pending)} of {len(rep['groups'])} ownership"
          f" {plural(len(rep['groups']), 'group')} still outstanding.")
        if pr["isDraft"]:
            w("  This PR is a draft, so GitHub has not sent review requests yet --"
              " these are the groups that will be required.")
        for owners in rep["unresolvable"]:
            w(f"  No eligible approver could be resolved for {' '.join(owners)}"
              " (team membership unreadable? needs `read:org`)")
        if blocked:
            w(f"  {len(blocked)} group(s) have CHANGES REQUESTED -- those must be resolved"
              " and re-approved, not just approved by someone else.")
        if cover and not rep["history_scanned"]:
            noun = "person" if len(cover) == 1 else "people"
            w(f"  Minimum {len(cover)} more {noun} needed"
              f"{'' if rep['cover_exact'] else ' (approx)'}"
              f" to cover the {len(pending)} pending"
              f" {plural(len(pending), 'group')}:")
            for g in pending:
                w(f"      {fmt_owners(g, org)}")
            w("  (run without --no-history to name who usually approves for each)")
        elif cover:
            noun = "person" if len(cover) == 1 else "people"
            blockers = {u for g in pending if g["status"] == "changes_requested"
                        for u in g["changes_requested_by"]}
            names = ", ".join(
                "@" + u + (" (must re-review)" if u in blockers else "") for u in cover
            )
            w(f"  Minimum {len(cover)} more {noun} needed"
              f"{'' if rep['cover_exact'] else ' (approx)'}: {names}")
            if len(cover) < len(pending) and rep["history_scanned"]:
                for login in cover:
                    hit = [g for g in pending if login in g.get("candidates", [])]
                    if len(hit) > 1:
                        w(f"    @{login} alone clears {len(hit)} of them: "
                          + "; ".join(fmt_owners(g, org) for g in hit))
            elif len(pending) > 1:
                w("  (the pending groups share no eligible owner, so it cannot collapse further)")
            if len(counted) + len(cover) < need:
                w(f"  That still leaves the count short: {need} approvals required,"
                  f" {len(counted)} so far.")
        else:
            w("  Could not resolve any eligible approver -- check `read:org` access"
              " or the owners listed above.")
        still = rep["requested"]
        if still[0] or still[1]:
            bits = ["@" + u for u in still[0]] + [f"@{org}/{t}" for t in still[1]]
            w(f"  GitHub still lists review requests for: {', '.join(bits)}")
            w("  (individual requests linger even after a co-owner approves the same path)")
    if pr.get("mergeable") == "CONFLICTING":
        w("  Heads up: the branch conflicts with its base, so GitHub cannot build the"
          " merge ref -- pull_request CI workflows will silently never run. Rebase first.")
    return "\n".join(out)


def render_markdown(rep, args):
    pr, org = rep["pr"], rep["org"]
    sym = {"satisfied": "✅ satisfied", "pending": "⏳ pending",
           "changes_requested": "❌ changes requested"}
    out = [
        f"**PR [{rep['repo']}#{pr['number']}]({pr['url']})** - {pr['title']}",
        "",
        f"Rules on `{pr['baseRefName']}`: {rep['rules']['required_approving_review_count']}"
        f" approval(s) required, code-owner review"
        f" {'**required**' if rep['rules']['require_code_owner_review'] else 'not required'}.",
        "",
        "| paths | owners | status | who to ping |",
        "|---|---|---|---|",
    ]
    for g in rep["groups"]:
        head = g["files"][0] if len(g["files"]) == 1 else \
            f"{g['patterns'][0] if g['patterns'] else '?'} ({len(g['files'])} files)"
        if g["status"] == "satisfied":
            detail = "approved by " + ", ".join("@" + u for u in g["approved_by"])
        else:
            detail = ", ".join(f"@{r['login']} ({r['approvals']})"
                               for r in g.get("ranked", [])[:args.top]) or "-"
        out.append(f"| `{head}` | {fmt_owners(g, org)} | {sym[g['status']]} | {detail} |")
    out.append("")
    cover = rep["cover"]
    if cover:
        out.append(f"**Need {len(cover)} more approval(s):** "
                   + ", ".join("@" + u for u in cover))
    else:
        out.append("**No outstanding code-owner groups.**")
    return "\n".join(out)


def report_teams(args, org, teams, history):
    """Rank a team's habitual approvers, PR aside -- "who signs off for X?"."""
    slugs = [t.strip().lstrip("@").split("/")[-1] for t in args.teams.split(",") if t.strip()]
    for slug in slugs:
        members = teams.members(slug)
        err = teams.errors.get(slug)
        print(f"-- team @{org}/{slug}"
              + (f": {err} --" if err else f" ({len(members)} members) --"))
        if err:
            continue
        ranked, used = rank_approvers(members, history, args.prs)
        if not ranked:
            print(f"     no approvals from this team in the last {len(history)} merged PRs"
                  " (try --scan)")
            continue
        print(f"     top approvers (last {len(used)} PR(s) this team approved):")
        for r in ranked[:args.top]:
            print(f"       {r['approvals']:>2} {plural(r['approvals'], 'approval'):<9}  "
                  f"@{r['login']:<22} last {r['last'][DAY] or '-'}")
        silent = [mm for mm in members if mm not in {r['login'] for r in ranked}]
        if silent and args.top >= len(ranked):
            print(f"     no recent approvals from: {', '.join('@' + mm for mm in silent)}")


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description="Report outstanding reviews on a GitHub PR and who can clear them.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  pr_review_status.py https://github.com/tenstorrent/tt-metal/pull/55279
  pr_review_status.py 55279 --repo tenstorrent/tt-metal --json
  pr_review_status.py 8777 --repo tenstorrent/tt-mlir --markdown
  pr_review_status.py --repo tenstorrent/tt-metal \\
      --teams metalium-developers-infra        # no PR: who approves for a team
""")
    ap.add_argument("pr", nargs="?", help="PR number or URL")
    ap.add_argument("--repo", help="owner/name, when the PR is a bare number")
    ap.add_argument("--prs", type=int, default=10, metavar="N",
                    help="rank approvers over the last N PRs each group reviewed (default 10)")
    ap.add_argument("--scan", type=int, default=200, metavar="N",
                    help="recent merged PRs to scan for approval history (default 200)")
    ap.add_argument("--top", type=int, default=5, metavar="N",
                    help="how many approvers to list per group (default 5)")
    ap.add_argument("--teams", metavar="a,b",
                    help="also report top approvers for these team slugs, PR aside")
    ap.add_argument("--files", action="store_true", help="list every file in a group")
    ap.add_argument("--max-files", type=int, default=8, metavar="N",
                    help="files shown per group without --files (default 8)")
    ap.add_argument("--all-groups", action="store_true",
                    help="show detail for satisfied groups too")
    ap.add_argument("--no-history", action="store_true",
                    help="skip the approval-history scan (faster; no approver ranking)")
    ap.add_argument("--json", action="store_true", help="emit the full report as JSON")
    ap.add_argument("--markdown", action="store_true",
                    help="emit a markdown table for pasting into Slack/GitHub")
    args = ap.parse_args()

    if not args.pr and not args.teams:
        ap.error("a PR number or URL is required (or --teams with --repo)")

    if args.pr:
        owner, repo, number = parse_pr(args.pr, args.repo)
    else:
        # --teams on its own: no PR to analyze, just rank a team's approvers.
        owner, repo, number = parse_pr("0", args.repo)
    org = owner
    teams = TeamCache(org)

    if not args.pr:
        history = [] if args.no_history else approval_history(owner, repo, args.scan)
        report_teams(args, org, teams, history)
        return

    pr = pr_overview(owner, repo, number)
    pr["author"] = (pr.get("author") or {}).get("login", "")
    rules = branch_review_rules(owner, repo, pr["baseRefName"])
    files = changed_files(owner, repo, number)
    reviews = effective_reviews(owner, repo, number)
    requested = requested_reviewers(owner, repo, number)

    head = pr.get("headRefOid") or ""
    for r in reviews.values():
        r["stale"] = bool(
            rules["dismiss_stale_reviews_on_push"] and head and r["commit"]
            and r["commit"] != head
        )

    # Bots and the author never count toward the approval requirement.
    counted = sorted(
        u for u, r in reviews.items()
        if r["state"] == "APPROVED" and u != pr["author"]
        and not u.endswith(BOT_SUFFIX) and not r["stale"]
    )

    text, co_path = fetch_codeowners(owner, repo, pr["baseRefName"])
    co_rules = parse_codeowners(text)
    groups, unowned = ownership_groups(files, co_rules) if co_rules else ([], list(files))
    for g in groups:
        expand_group(g, org, teams)
        g["author_is_owner"] = pr["author"] in g["eligible"]
    classify_groups(groups, reviews, pr["author"])

    history = [] if args.no_history else approval_history(owner, repo, args.scan)
    stats = user_stats(history)

    pending = [g for g in groups if g["status"] != "satisfied"]
    for g in pending:
        # Owners who could approve: everyone eligible bar the author and bots.
        g["owner_pool"] = sorted(
            u for u in g["eligible"]
            if u != pr["author"] and not u.endswith(BOT_SUFFIX)
        )
        # What it takes to clear the group. A CHANGES_REQUESTED review is only
        # cleared by that same reviewer (or a dismissal), so they are the sole
        # candidate -- another owner approving leaves the block in place.
        g["candidates"] = (
            list(g["changes_requested_by"]) if g["status"] == "changes_requested"
            else g["owner_pool"]
        )
        g["ranked"], g["history_prs"] = rank_approvers(g["owner_pool"], history, args.prs)
    for g in groups:
        if g["status"] == "satisfied" and args.all_groups:
            g["owner_pool"] = g["candidates"] = sorted(g["eligible"])
            g["ranked"], g["history_prs"] = rank_approvers(g["owner_pool"], history, args.prs)

    solvable = [g for g in pending if g["candidates"]]
    unresolvable = [g for g in pending if not g["candidates"]]
    cover, cover_exact = min_approver_cover(
        solvable, lambda login: stats.get(login, {}).get("approvals", 0)
    )
    # Present the cover most-active-first so the first name is the best ping.
    cover.sort(key=lambda l: (-stats.get(l, {}).get("approvals", 0), l.lower()))

    rep = {
        "repo": f"{owner}/{repo}",
        "org": org,
        "pr": pr,
        "rules": rules,
        "files": files,
        "codeowners_path": co_path,
        "reviews": reviews,
        "counted_approvals": counted,
        "requested": requested,
        "groups": groups,
        "unowned": unowned,
        "cover": cover,
        "cover_exact": cover_exact,
        "unresolvable": [g["owners"] for g in unresolvable],
        "history_scanned": len(history),
        "team_errors": teams.errors,
        "team_members": {s: teams.members(s) for g in groups for s in g["teams"]},
    }

    if args.json:
        clean = json.loads(json.dumps(rep, default=lambda o: sorted(o) if isinstance(o, set) else str(o)))
        print(json.dumps(clean, indent=2))
    elif args.markdown:
        print(render_markdown(rep, args))
    else:
        print(render_text(rep, args))

    if args.teams:
        print()
        report_teams(args, org, teams, history)


if __name__ == "__main__":
    main()
