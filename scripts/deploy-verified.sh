#!/bin/bash
# Dr. Agnes Verified Deployment Script
# 1. Bumps version (patch)
# 2. Commits + pushes to GitHub (stuinfla/DrAgnes)
# 3. Waits for Vercel deployment
# 4. Verifies live site shows new version via Playwright
# 5. Reports success/failure
#
# Usage: bash scripts/deploy-verified.sh [major|minor|patch]
# Default: patch

set -euo pipefail

BUMP_TYPE="${1:-patch}"
MONOREPO_ROOT="/Users/stuartkerr/RuVector_New/RuVector"
DRAGNES_DIR="$MONOREPO_ROOT/examples/dragnes"
GITHUB_REPO="stuinfla/DrAgnes"
VERCEL_URL="https://dragnes.vercel.app"
PLAYWRIGHT_DIR="$MONOREPO_ROOT/ui/ruvocal"  # has playwright installed

cd "$DRAGNES_DIR"

echo "============================================"
echo "Dr. Agnes Verified Deployment"
echo "============================================"
echo ""

# Step 1: Bump version
OLD_VERSION=$(node -p "require('./package.json').version")
IFS='.' read -r MAJOR MINOR PATCH <<< "$OLD_VERSION"

case "$BUMP_TYPE" in
  major) MAJOR=$((MAJOR + 1)); MINOR=0; PATCH=0 ;;
  minor) MINOR=$((MINOR + 1)); PATCH=0 ;;
  patch) PATCH=$((PATCH + 1)) ;;
  *) echo "ERROR: Invalid bump type: $BUMP_TYPE (use major|minor|patch)"; exit 1 ;;
esac

NEW_VERSION="$MAJOR.$MINOR.$PATCH"
echo "📦 Version: $OLD_VERSION → $NEW_VERSION ($BUMP_TYPE)"

# Update package.json
cd "$DRAGNES_DIR"
node -e "
const fs = require('fs');
const pkg = JSON.parse(fs.readFileSync('package.json', 'utf-8'));
pkg.version = '$NEW_VERSION';
fs.writeFileSync('package.json', JSON.stringify(pkg, null, 2) + '\n');
"
echo "   ✅ package.json updated"

# Update version in +page.svelte header (single source of truth: package.json → here)
cd "$DRAGNES_DIR"
sed -i '' "s/v[0-9]*\.[0-9]*\.[0-9]*<\/span>/v${NEW_VERSION}<\/span>/" src/routes/+page.svelte
echo "   ✅ +page.svelte header updated"

# Update version in README headline
sed -i '' "s/Version [0-9]*\.[0-9]*\.[0-9]*/Version ${NEW_VERSION}/" README.md
echo "   ✅ README.md version updated"

# Step 2: Commit
cd "$MONOREPO_ROOT"
git add examples/dragnes/package.json examples/dragnes/src/routes/+page.svelte examples/dragnes/README.md
git commit -m "chore(dragnes): bump version to $NEW_VERSION

Co-Authored-By: claude-flow <ruv@ruv.net>"
echo "   ✅ Committed"

# Step 3: Subtree split + push to GitHub
echo ""
echo "🚀 Pushing to GitHub ($GITHUB_REPO)..."
SPLIT_SHA=$(git subtree split --prefix=examples/dragnes -b dragnes-deploy-$NEW_VERSION 2>/dev/null | tail -1)
git push fork dragnes-deploy-$NEW_VERSION:main --force 2>&1
echo "   ✅ Pushed to $GITHUB_REPO (SHA: ${SPLIT_SHA:0:8})"

# Step 4: Wait for Vercel deployment
echo ""
echo "⏳ Waiting for Vercel deployment..."
MAX_WAIT=120
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
  DEPLOY_STATE=$(gh api "repos/$GITHUB_REPO/commits/main/status" --jq '.state' 2>/dev/null || echo "unknown")
  if [ "$DEPLOY_STATE" = "success" ] || [ "$DEPLOY_STATE" = "pending" ]; then
    break
  fi
  sleep 10
  WAITED=$((WAITED + 10))
  echo "   Waiting... ${WAITED}s"
done

# Give Vercel extra time to build
echo "   Waiting 30s for Vercel build..."
sleep 30

# Step 5: Verify with Playwright
echo ""
echo "🔍 Verifying live deployment..."
cd "$PLAYWRIGHT_DIR"

VERIFY_RESULT=$(node -e "
const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  await page.goto('$VERCEL_URL', { waitUntil: 'networkidle', timeout: 30000 });

  const html = await page.content();
  const hasVersion = html.includes('v$NEW_VERSION');
  const hasHeader = html.includes('Dr. Agnes');
  const hasButtons = html.includes('Take Photo') || html.includes('Upload Photo');

  await page.screenshot({ path: '/tmp/deploy-verify-$NEW_VERSION.png', fullPage: true });

  console.log(JSON.stringify({
    version_visible: hasVersion,
    header_present: hasHeader,
    buttons_present: hasButtons,
    url: '$VERCEL_URL'
  }));

  await browser.close();
})().catch(e => { console.log(JSON.stringify({ error: e.message })); process.exit(1); });
" 2>&1)

echo "   Verification: $VERIFY_RESULT"

# Parse result
VERSION_OK=$(echo "$VERIFY_RESULT" | node -p "JSON.parse(require('fs').readFileSync('/dev/stdin','utf8')).version_visible" 2>/dev/null || echo "false")
HEADER_OK=$(echo "$VERIFY_RESULT" | node -p "JSON.parse(require('fs').readFileSync('/dev/stdin','utf8')).header_present" 2>/dev/null || echo "false")

echo ""
echo "============================================"
if [ "$VERSION_OK" = "true" ] && [ "$HEADER_OK" = "true" ]; then
  echo "✅ DEPLOYMENT VERIFIED: v$NEW_VERSION live at $VERCEL_URL"
  echo "   Screenshot: /tmp/deploy-verify-$NEW_VERSION.png"
  echo "============================================"
  exit 0
else
  echo "❌ DEPLOYMENT VERIFICATION FAILED"
  echo "   Version visible: $VERSION_OK"
  echo "   Header present: $HEADER_OK"
  echo "   Screenshot: /tmp/deploy-verify-$NEW_VERSION.png"
  echo "   Check $VERCEL_URL manually"
  echo "============================================"
  exit 1
fi
