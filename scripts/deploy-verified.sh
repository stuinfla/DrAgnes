#!/bin/bash
# Mela Verified Deployment Script (Standalone Repo)
#
# 7-step pipeline:
# 1. Pre-flight checks (gh CLI, vercel CLI, agent-browser, git clean)
# 2. Bumps version in all 3 places (package.json, +page.svelte, README.md)
# 3. Build check (vite build)
# 4. Commit + push to GitHub (stuinfla/Mela)
# 5. GitHub verification (commit visible, no CI failures)
# 6. Vercel verification (deployment ready, no build errors)
# 7. Agent-browser verification (load live site, confirm version, screenshot)
#
# Usage: bash scripts/deploy-verified.sh [major|minor|patch]
# Default: patch

set -euo pipefail

BUMP_TYPE="${1:-patch}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
GITHUB_REPO="stuinfla/Mela"
VERCEL_URL="https://mela-app.vercel.app"
VERCEL_PROJECT="mela"
REPORT_FILE="/tmp/mela-deploy-report.txt"

cd "$REPO_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "   ${GREEN}PASS${NC} $1"; }
fail() { echo -e "   ${RED}FAIL${NC} $1"; }
warn() { echo -e "   ${YELLOW}WARN${NC} $1"; }
step() { echo -e "\n${YELLOW}[$1/7]${NC} $2"; }

# Initialize report
echo "Mela Deploy Report — $(date)" > "$REPORT_FILE"
echo "========================================" >> "$REPORT_FILE"
ERRORS=0

# ============================================================
# Step 1: Pre-flight checks
# ============================================================
step 1 "Pre-flight checks"

# Check gh CLI
if command -v gh &> /dev/null; then
  GH_AUTH=$(gh auth status 2>&1 || true)
  if echo "$GH_AUTH" | grep -q "Logged in"; then
    pass "gh CLI authenticated"
  else
    fail "gh CLI not authenticated — run: gh auth login"
    ERRORS=$((ERRORS + 1))
  fi
else
  fail "gh CLI not found — install: brew install gh"
  ERRORS=$((ERRORS + 1))
fi

# Check vercel CLI
if command -v vercel &> /dev/null; then
  pass "vercel CLI found"
else
  fail "vercel CLI not found — install: npm i -g vercel"
  ERRORS=$((ERRORS + 1))
fi

# Check agent-browser
if command -v agent-browser &> /dev/null; then
  pass "agent-browser found"
else
  warn "agent-browser not found — browser verification will use curl fallback"
fi

# Check git is clean (except untracked)
if [ -n "$(git diff --cached --name-only)" ]; then
  fail "Staged changes exist — commit or unstage first"
  ERRORS=$((ERRORS + 1))
else
  pass "Git staging area clean"
fi

# Check we're on main
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" = "main" ]; then
  pass "On branch main"
else
  fail "On branch '$CURRENT_BRANCH' — switch to main first"
  ERRORS=$((ERRORS + 1))
fi

# Check remote is reachable
if gh repo view "$GITHUB_REPO" --json name -q '.name' &> /dev/null; then
  pass "GitHub repo $GITHUB_REPO reachable"
else
  fail "Cannot reach GitHub repo $GITHUB_REPO"
  ERRORS=$((ERRORS + 1))
fi

# Check Vercel project is linked
VERCEL_INSPECT=$(vercel inspect "$VERCEL_URL" 2>&1 || true)
if echo "$VERCEL_INSPECT" | grep -q "name"; then
  pass "Vercel project '$VERCEL_PROJECT' linked"
else
  warn "Could not inspect Vercel project — deploy may still work"
fi

if [ $ERRORS -gt 0 ]; then
  echo ""
  fail "Pre-flight failed with $ERRORS error(s). Fix above issues and retry."
  exit 1
fi

# ============================================================
# Step 2: Bump version
# ============================================================
step 2 "Version bump ($BUMP_TYPE)"

OLD_VERSION=$(node -p "require('./package.json').version")
IFS='.' read -r MAJOR MINOR PATCH <<< "$OLD_VERSION"

case "$BUMP_TYPE" in
  major) MAJOR=$((MAJOR + 1)); MINOR=0; PATCH=0 ;;
  minor) MINOR=$((MINOR + 1)); PATCH=0 ;;
  patch) PATCH=$((PATCH + 1)) ;;
  *) fail "Invalid bump type: $BUMP_TYPE (use major|minor|patch)"; exit 1 ;;
esac

NEW_VERSION="$MAJOR.$MINOR.$PATCH"
echo "   $OLD_VERSION -> $NEW_VERSION"
echo "Version: $OLD_VERSION -> $NEW_VERSION ($BUMP_TYPE)" >> "$REPORT_FILE"

# Update package.json
node -e "
const fs = require('fs');
const pkg = JSON.parse(fs.readFileSync('package.json', 'utf-8'));
pkg.version = '$NEW_VERSION';
fs.writeFileSync('package.json', JSON.stringify(pkg, null, 2) + '\n');
"
pass "package.json"

# Update +page.svelte header
sed -i '' "s/v[0-9]*\.[0-9]*\.[0-9]*<\/span>/v${NEW_VERSION}<\/span>/" src/routes/+page.svelte
pass "+page.svelte"

# Update README headline
sed -i '' "s/Version [0-9]*\.[0-9]*\.[0-9]*/Version ${NEW_VERSION}/" README.md
pass "README.md"

# Verify all 3 match
PKG_V=$(node -p "require('./package.json').version")
SVELTE_V=$(grep -o 'v[0-9]*\.[0-9]*\.[0-9]*' src/routes/+page.svelte | head -1)
README_V=$(grep -o 'Version [0-9]*\.[0-9]*\.[0-9]*' README.md | head -1 | sed 's/Version //')

if [ "$PKG_V" = "$NEW_VERSION" ] && [ "$SVELTE_V" = "v$NEW_VERSION" ] && [ "$README_V" = "$NEW_VERSION" ]; then
  pass "All 3 version locations in sync: $NEW_VERSION"
else
  fail "Version mismatch — pkg:$PKG_V svelte:$SVELTE_V readme:$README_V"
  exit 1
fi

# ============================================================
# Step 3: Build check
# ============================================================
step 3 "Build check"

BUILD_OUTPUT=$(npm run build 2>&1)
BUILD_EXIT=$?

if [ $BUILD_EXIT -eq 0 ]; then
  pass "vite build succeeded"
else
  fail "Build failed:"
  echo "$BUILD_OUTPUT" | tail -20
  echo "BUILD FAILED" >> "$REPORT_FILE"
  echo "$BUILD_OUTPUT" | tail -20 >> "$REPORT_FILE"
  exit 1
fi

# ============================================================
# Step 4: Commit + push to GitHub
# ============================================================
step 4 "Commit + push"

git add package.json src/routes/+page.svelte README.md
git commit -m "chore(mela): bump version to $NEW_VERSION

Co-Authored-By: claude-flow <ruv@ruv.net>"
pass "Committed v$NEW_VERSION"

git push origin main 2>&1
PUSH_EXIT=$?

if [ $PUSH_EXIT -eq 0 ]; then
  pass "Pushed to origin/main"
else
  fail "Push failed"
  exit 1
fi

# ============================================================
# Step 5: GitHub verification
# ============================================================
step 5 "GitHub verification"

sleep 3

# Verify commit is visible on GitHub
LATEST_SHA=$(gh api "repos/$GITHUB_REPO/commits/main" --jq '.sha' 2>/dev/null || echo "FETCH_FAILED")
LOCAL_SHA=$(git rev-parse HEAD)

if [ "${LATEST_SHA:0:8}" = "${LOCAL_SHA:0:8}" ]; then
  pass "Commit ${LOCAL_SHA:0:8} visible on GitHub"
else
  warn "GitHub shows ${LATEST_SHA:0:8}, local is ${LOCAL_SHA:0:8} — may need a moment"
fi

# Check for CI status (GitHub Actions)
CI_STATUS=$(gh api "repos/$GITHUB_REPO/commits/$LOCAL_SHA/check-runs" --jq '.total_count' 2>/dev/null || echo "0")
if [ "$CI_STATUS" = "0" ]; then
  warn "No CI checks configured (consider adding GitHub Actions)"
else
  CI_CONCLUSION=$(gh api "repos/$GITHUB_REPO/commits/$LOCAL_SHA/check-runs" --jq '.check_runs[0].conclusion // "pending"' 2>/dev/null || echo "unknown")
  if [ "$CI_CONCLUSION" = "success" ]; then
    pass "CI checks passed"
  elif [ "$CI_CONCLUSION" = "pending" ] || [ "$CI_CONCLUSION" = "null" ]; then
    warn "CI checks still running"
  else
    warn "CI status: $CI_CONCLUSION"
  fi
fi

echo "GitHub: commit ${LOCAL_SHA:0:8} pushed, CI: ${CI_STATUS} checks" >> "$REPORT_FILE"

# ============================================================
# Step 6: Vercel deployment verification
# ============================================================
step 6 "Vercel deployment verification"

# vercel ls outputs bare URLs (no table) when piped, so we compare URL lists.
# The first URL is always the most recent deployment.
PRE_DEPLOY_URL=$(vercel ls --scope stuart-kerrs-projects 2>/dev/null | head -1 || echo "")
echo "   Pre-push latest: ${PRE_DEPLOY_URL:-none}"
echo "   Waiting for Vercel to pick up the push..."

MAX_WAIT=240
WAITED=0
DEPLOY_READY="false"
LATEST_DEPLOY_URL=""

while [ $WAITED -lt $MAX_WAIT ]; do
  CANDIDATE_URL=$(vercel ls --scope stuart-kerrs-projects 2>/dev/null | head -1 || echo "")

  # A new URL appeared at the top of the list (new deployment triggered)
  if [ -n "$CANDIDATE_URL" ] && [ "$CANDIDATE_URL" != "$PRE_DEPLOY_URL" ]; then
    echo "   New deployment detected: $CANDIDATE_URL"
    # Now wait for it to be Ready (vercel inspect shows status)
    INSPECT_STATUS=$(vercel inspect "$CANDIDATE_URL" --scope stuart-kerrs-projects 2>&1 | grep -oE "Ready|Building|Error|Canceled" | head -1 || echo "unknown")

    if [ "$INSPECT_STATUS" = "Ready" ]; then
      LATEST_DEPLOY_URL="$CANDIDATE_URL"
      DEPLOY_READY="true"
      pass "Deployment ready: $LATEST_DEPLOY_URL"
      break
    elif [ "$INSPECT_STATUS" = "Error" ] || [ "$INSPECT_STATUS" = "Canceled" ]; then
      fail "Deployment failed with status: $INSPECT_STATUS"
      echo "   Run: vercel inspect $CANDIDATE_URL --scope stuart-kerrs-projects --logs"
      break
    else
      echo "   Status: $INSPECT_STATUS — still building..."
    fi
  fi

  sleep 10
  WAITED=$((WAITED + 10))
  echo "   Waiting... ${WAITED}s / ${MAX_WAIT}s"
done

# Timeout fallback: use whatever is at the top of the list
if [ "$DEPLOY_READY" != "true" ] && [ $WAITED -ge $MAX_WAIT ]; then
  warn "Timed out (${MAX_WAIT}s) — using latest deployment"
  LATEST_DEPLOY_URL=$(vercel ls --scope stuart-kerrs-projects 2>/dev/null | head -1 || echo "")
  if [ -n "$LATEST_DEPLOY_URL" ]; then
    INSPECT_STATUS=$(vercel inspect "$LATEST_DEPLOY_URL" --scope stuart-kerrs-projects 2>&1 | grep -oE "Ready|Building|Error" | head -1 || echo "unknown")
    if [ "$INSPECT_STATUS" = "Ready" ]; then
      DEPLOY_READY="true"
      warn "Latest deployment is Ready: $LATEST_DEPLOY_URL"
    else
      fail "Latest deployment status: $INSPECT_STATUS"
    fi
  else
    fail "No deployments found"
  fi
fi

echo "Vercel: ${LATEST_DEPLOY_URL:-none} (${DEPLOY_READY})" >> "$REPORT_FILE"

# Always update the production alias
if [ -n "$LATEST_DEPLOY_URL" ]; then
  ALIAS_OUT=$(vercel alias "$LATEST_DEPLOY_URL" "$VERCEL_URL" --scope stuart-kerrs-projects 2>&1 || true)
  if echo "$ALIAS_OUT" | grep -qiE "success|already"; then
    pass "Alias: $VERCEL_URL -> $LATEST_DEPLOY_URL"
  else
    warn "Alias may have failed: $ALIAS_OUT"
  fi
fi

# Check build logs for real errors (externalized dep warnings are expected)
if [ -n "$LATEST_DEPLOY_URL" ]; then
  VERCEL_LOGS=$(vercel inspect "$LATEST_DEPLOY_URL" --scope stuart-kerrs-projects --logs 2>&1 | tail -30 || true)
  REAL_ERRORS=$(echo "$VERCEL_LOGS" | grep -i "error\|failed\|ERR_" | grep -vi "could not be resolved.*external\|treating it as an external" || true)
  if [ -n "$REAL_ERRORS" ]; then
    fail "Build errors detected:"
    echo "$REAL_ERRORS" | head -5
    echo "VERCEL BUILD ERRORS: $REAL_ERRORS" >> "$REPORT_FILE"
  else
    pass "No build errors"
  fi
fi

echo "   Waiting 10s for CDN propagation..."
sleep 10

# ============================================================
# Step 7: Live site verification (JS bundle check)
# ============================================================
step 7 "Live site verification"

SITE_OK="false"
VERSION_OK="false"
HEADER_OK="false"
SCREENSHOT=""

# Primary method: check the JS bundle directly.
# SvelteKit is a client-rendered SPA — the version badge is in the JS, not raw HTML.
# This works regardless of agent-browser or service workers.
echo "   Checking JS bundle for version..."

# Find the page chunk URL from the app entry point
APP_JS_URL=$(curl -s "$VERCEL_URL/" 2>/dev/null | grep -oE '/_app/immutable/entry/app\.[^"]+\.js' | head -1 || echo "")
if [ -n "$APP_JS_URL" ]; then
  # Get the page node chunk (node 2 = main page in SvelteKit)
  PAGE_CHUNK=$(curl -s "${VERCEL_URL}${APP_JS_URL}" 2>/dev/null | grep -oE '"../nodes/2\.[^"]+\.js"' | tr -d '"' || echo "")
  if [ -n "$PAGE_CHUNK" ]; then
    # Convert relative path to absolute
    CHUNK_URL="${VERCEL_URL}/_app/immutable/${PAGE_CHUNK#../}"
    CHUNK_CONTENT=$(curl -s "$CHUNK_URL" 2>/dev/null || echo "")

    if echo "$CHUNK_CONTENT" | grep -q "v${NEW_VERSION}"; then
      VERSION_OK="true"
      pass "Version v${NEW_VERSION} confirmed in JS bundle"
    else
      # Show what version IS in the bundle
      FOUND_VERSION=$(echo "$CHUNK_CONTENT" | grep -oE 'font-semibold">[^<]+</span>' | head -1 || echo "none found")
      fail "Version v${NEW_VERSION} NOT in JS bundle. Found: $FOUND_VERSION"
    fi

    if echo "$CHUNK_CONTENT" | grep -q "Mela"; then
      HEADER_OK="true"
      pass "Mela branding confirmed in JS bundle"
    fi
  else
    warn "Could not find page chunk in app.js"
  fi
else
  warn "Could not find app.js entry point"
fi

# Secondary: agent-browser screenshot (visual record, not for pass/fail)
if command -v agent-browser &> /dev/null; then
  echo "   Taking screenshot for visual record..."
  SCREENSHOT="/tmp/mela-deploy-v${NEW_VERSION}.png"
  (
    agent-browser close 2>/dev/null || true
    sleep 1
    agent-browser open "$VERCEL_URL" 2>/dev/null || true
    sleep 3
    # Dismiss disclaimer if present
    SNAP=$(agent-browser snapshot -i 2>/dev/null || echo "")
    if echo "$SNAP" | grep -q "I Understand"; then
      DISMISS_REF=$(echo "$SNAP" | grep "I Understand" | grep -oE 'ref=e[0-9]+' | head -1 | sed 's/ref=//')
      if [ -n "$DISMISS_REF" ]; then
        agent-browser click "$DISMISS_REF" 2>/dev/null || true
        sleep 2
      fi
    fi
    agent-browser screenshot "$SCREENSHOT" 2>/dev/null || true
    agent-browser close 2>/dev/null || true
  ) 2>/dev/null
  if [ -f "$SCREENSHOT" ]; then
    pass "Screenshot: $SCREENSHOT"
  fi
fi

# Curl fallback for header check if JS bundle method didn't work
if [ "$HEADER_OK" != "true" ]; then
  SITE_HTML=$(curl -s --max-time 10 "$VERCEL_URL" 2>/dev/null || echo "")
  if echo "$SITE_HTML" | grep -q "Mela"; then
    HEADER_OK="true"
    pass "Mela branding confirmed in HTML"
  fi
fi

if [ "$VERSION_OK" = "true" ] && [ "$HEADER_OK" = "true" ]; then
  SITE_OK="true"
fi

# ============================================================
# Extract changelog (commits since last tag/version)
# ============================================================
CHANGELOG=""
PREV_TAG=$(git tag --sort=-version:refname | head -1 2>/dev/null || echo "")
if [ -n "$PREV_TAG" ]; then
  CHANGELOG=$(git log "$PREV_TAG"..HEAD --pretty=format:"  - %s" --no-merges 2>/dev/null | head -20)
else
  # No tags — show commits since last version bump
  LAST_BUMP=$(git log --oneline --grep="bump version" -1 --format="%H" 2>/dev/null || echo "")
  if [ -n "$LAST_BUMP" ]; then
    CHANGELOG=$(git log "$LAST_BUMP"..HEAD --pretty=format:"  - %s" --no-merges 2>/dev/null | head -20)
  else
    CHANGELOG=$(git log -5 --pretty=format:"  - %s" --no-merges 2>/dev/null)
  fi
fi

# ============================================================
# Final Report
# ============================================================
echo ""
echo "============================================"
echo "  DEPLOYMENT REPORT — v$NEW_VERSION"
echo "============================================"
echo ""

if [ "$SITE_OK" = "true" ]; then
  echo -e "  ${GREEN}DEPLOYMENT SUCCESSFUL${NC}"
  echo ""
else
  echo -e "  ${RED}DEPLOYMENT VERIFICATION FAILED${NC}"
  echo ""
fi

echo "  Version:     $OLD_VERSION -> $NEW_VERSION"
echo "  GitHub:      https://github.com/$GITHUB_REPO"
echo "  Vercel:      $VERCEL_URL"
echo "  Commit:      ${LOCAL_SHA:0:12}"
echo ""
echo "  Verification:"
echo "    Build:           $([ $BUILD_EXIT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "    Git push:        $([ $PUSH_EXIT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "    GitHub visible:  $([ "${LATEST_SHA:0:8}" = "${LOCAL_SHA:0:8}" ] && echo 'PASS' || echo 'PENDING')"
echo "    Vercel ready:    $([ "$DEPLOY_READY" = "true" ] && echo 'PASS' || echo 'PENDING')"
echo "    Version on site: $([ "$VERSION_OK" = "true" ] && echo 'PASS' || echo 'FAIL')"
echo "    Header on site:  $([ "$HEADER_OK" = "true" ] && echo 'PASS' || echo 'FAIL')"
if [ -n "$SCREENSHOT" ] && [ -f "$SCREENSHOT" ]; then
  echo "    Screenshot:      $SCREENSHOT"
fi
echo ""

if [ -n "$CHANGELOG" ]; then
  echo "  What's new in v$NEW_VERSION:"
  echo "$CHANGELOG"
  echo ""
fi

# Write full report to file
{
  echo ""
  echo "Mela Deployment Report"
  echo "========================"
  echo "Date: $(date)"
  echo "Version: $OLD_VERSION -> $NEW_VERSION ($BUMP_TYPE)"
  echo "Commit: ${LOCAL_SHA:0:12}"
  echo "GitHub: https://github.com/$GITHUB_REPO"
  echo "Vercel: $VERCEL_URL"
  echo ""
  echo "Verification Results:"
  echo "  Build: $([ $BUILD_EXIT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
  echo "  Push: $([ $PUSH_EXIT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
  echo "  GitHub: $([ "${LATEST_SHA:0:8}" = "${LOCAL_SHA:0:8}" ] && echo 'PASS' || echo 'PENDING')"
  echo "  Vercel: $([ "$DEPLOY_READY" = "true" ] && echo 'PASS' || echo 'PENDING')"
  echo "  Version visible: $VERSION_OK"
  echo "  Header visible: $HEADER_OK"
  echo "  Screenshot: ${SCREENSHOT:-none}"
  echo ""
  if [ -n "$CHANGELOG" ]; then
    echo "What's new in v$NEW_VERSION:"
    echo "$CHANGELOG"
    echo ""
  fi
} > "$REPORT_FILE"

if [ "$SITE_OK" = "true" ]; then
  echo "RESULT: SUCCESS" >> "$REPORT_FILE"
  echo "  Full report: $REPORT_FILE"
  echo "============================================"
  exit 0
else
  echo "RESULT: FAILED" >> "$REPORT_FILE"
  echo ""
  echo "  Troubleshoot:"
  echo "    gh api repos/$GITHUB_REPO/commits/main --jq '.sha'"
  echo "    vercel ls"
  echo "    vercel inspect $VERCEL_URL --logs"
  echo "    curl -s $VERCEL_URL | grep 'v$NEW_VERSION'"
  echo ""
  echo "  Full report: $REPORT_FILE"
  echo "============================================"
  exit 1
fi
