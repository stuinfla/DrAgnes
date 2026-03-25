#!/bin/bash
# Dr. Agnes Verified Deployment Script (Standalone Repo)
# 1. Bumps version (patch/minor/major)
# 2. Updates version in all 3 places (package.json, +page.svelte, README.md)
# 3. Commits + pushes to GitHub (stuinfla/DrAgnes)
# 4. Waits for Vercel deployment
# 5. Verifies live site shows new version
#
# Usage: bash scripts/deploy-verified.sh [major|minor|patch]
# Default: patch

set -euo pipefail

BUMP_TYPE="${1:-patch}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
GITHUB_REPO="stuinfla/DrAgnes"
VERCEL_URL="https://dragnes.vercel.app"

cd "$REPO_DIR"

echo "============================================"
echo "Dr. Agnes Verified Deployment (Standalone)"
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
echo "Version: $OLD_VERSION -> $NEW_VERSION ($BUMP_TYPE)"

# Update package.json
node -e "
const fs = require('fs');
const pkg = JSON.parse(fs.readFileSync('package.json', 'utf-8'));
pkg.version = '$NEW_VERSION';
fs.writeFileSync('package.json', JSON.stringify(pkg, null, 2) + '\n');
"
echo "   package.json updated"

# Update version in +page.svelte header
sed -i '' "s/v[0-9]*\.[0-9]*\.[0-9]*<\/span>/v${NEW_VERSION}<\/span>/" src/routes/+page.svelte
echo "   +page.svelte header updated"

# Update version in README headline
sed -i '' "s/Version [0-9]*\.[0-9]*\.[0-9]*/Version ${NEW_VERSION}/" README.md
echo "   README.md version updated"

# Step 2: Build check
echo ""
echo "Running build check..."
npm run build > /dev/null 2>&1
echo "   Build passed"

# Step 3: Commit + push
echo ""
echo "Committing and pushing..."
git add package.json src/routes/+page.svelte README.md
git commit -m "chore(dragnes): bump version to $NEW_VERSION

Co-Authored-By: claude-flow <ruv@ruv.net>"
echo "   Committed"

git push origin main
echo "   Pushed to $GITHUB_REPO"

# Step 4: Wait for Vercel deployment
echo ""
echo "Waiting for Vercel deployment..."
MAX_WAIT=180
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
  # Check if Vercel has a new deployment
  DEPLOY_STATUS=$(vercel ls 2>/dev/null | head -5 | grep -c "Ready" || true)
  if [ "$DEPLOY_STATUS" -gt 0 ]; then
    # Check if the deployment is recent (within last 3 minutes)
    break
  fi
  sleep 10
  WAITED=$((WAITED + 10))
  echo "   Waiting... ${WAITED}s"
done

# Give Vercel extra time to propagate
echo "   Waiting 20s for CDN propagation..."
sleep 20

# Step 5: Verify deployment
echo ""
echo "Verifying live deployment..."

# Use curl to check version string on live site
SITE_HTML=$(curl -s --max-time 15 "$VERCEL_URL" 2>/dev/null || echo "FETCH_FAILED")

VERSION_OK="false"
HEADER_OK="false"

if echo "$SITE_HTML" | grep -q "v${NEW_VERSION}"; then
  VERSION_OK="true"
fi

if echo "$SITE_HTML" | grep -q "Dr. Agnes"; then
  HEADER_OK="true"
fi

echo ""
echo "============================================"
if [ "$VERSION_OK" = "true" ] && [ "$HEADER_OK" = "true" ]; then
  echo "DEPLOYMENT VERIFIED: v$NEW_VERSION live at $VERCEL_URL"
  echo "   Version visible: $VERSION_OK"
  echo "   Header present: $HEADER_OK"
  echo "============================================"
  exit 0
else
  echo "DEPLOYMENT VERIFICATION FAILED"
  echo "   Version visible: $VERSION_OK"
  echo "   Header present: $HEADER_OK"
  echo "   Check $VERCEL_URL manually"
  echo "   Run: vercel ls && vercel inspect $VERCEL_URL --logs"
  echo "============================================"
  exit 1
fi
