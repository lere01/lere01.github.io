SHELL := /bin/sh

.DEFAULT_GOAL := help

HUGO ?= hugo
GIT ?= git
SITE_URL ?= http://localhost:1313
CHECK_DIR ?= /private/tmp/lere01.github.io-check

.PHONY: help doctor deps serve serve-production new list-drafts check build build-drafts clean prepublish publish

help: ## Show the available tasks.
	@awk 'BEGIN { FS = ":.*## "; printf "Usage: make <target> [VARIABLE=value]\n\nTargets:\n" } /^[a-zA-Z0-9_-]+:.*## / { printf "  %-20s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@printf "\nExamples:\n"
	@printf "  make serve\n"
	@printf "  make new SLUG=my-article-title\n"
	@printf "  make prepublish\n"
	@printf "  make publish PUBLISH=1\n"

doctor: ## Check the local Hugo, Go, and Git installations.
	@command -v "$(HUGO)" >/dev/null || { printf "Hugo is not installed or not on PATH.\n"; exit 1; }
	@command -v go >/dev/null || { printf "Go is not installed or not on PATH.\n"; exit 1; }
	@command -v "$(GIT)" >/dev/null || { printf "Git is not installed or not on PATH.\n"; exit 1; }
	@"$(HUGO)" version
	@go version
	@"$(GIT)" --version

deps: doctor ## Download the Hugo module dependencies.
	@"$(HUGO)" mod get

serve: doctor ## Serve locally with drafts and live reload at SITE_URL.
	@printf "Serving drafts at %s\n" "$(SITE_URL)"
	@"$(HUGO)" server --buildDrafts

serve-production: doctor ## Preview only production-eligible content locally.
	@printf "Serving the production view at %s\n" "$(SITE_URL)"
	@"$(HUGO)" server --environment production

new: doctor ## Create a draft post bundle. Usage: make new SLUG=my-post.
	@test -n "$(SLUG)" || { printf "SLUG is required. Example: make new SLUG=my-post\n"; exit 2; }
	@case "$(SLUG)" in *[!a-z0-9-]*|'') printf "SLUG must contain only lowercase letters, numbers, and hyphens.\n"; exit 2;; esac
	@test ! -e "content/posts/$(SLUG)" || { printf "content/posts/%s already exists.\n" "$(SLUG)"; exit 2; }
	@"$(HUGO)" new content "posts/$(SLUG)/index.md"
	@printf "Created content/posts/%s/index.md\n" "$(SLUG)"

list-drafts: ## List all Markdown files currently marked as drafts.
	@rg -l '^draft:[[:space:]]*true[[:space:]]*$$' content --glob '*.md' | sort || true

check: doctor ## Validate all content, including drafts, with a clean private build.
	@"$(HUGO)" --buildDrafts --cleanDestinationDir --destination "$(CHECK_DIR)" --panicOnWarning
	@find content/posts -name '*.svg' -type f -exec xmllint --noout {} + 2>/dev/null || \
		{ command -v xmllint >/dev/null && exit 1 || printf "xmllint not installed, skipping SVG validation.\n"; }
	@printf "Validation build written to %s\n" "$(CHECK_DIR)"

build: doctor ## Build the minified production site into public/.
	@"$(HUGO)" --environment production --gc --minify --cleanDestinationDir

build-drafts: doctor ## Build a minified preview including drafts into public/.
	@"$(HUGO)" --buildDrafts --gc --minify --cleanDestinationDir

clean: ## Remove generated Hugo output and resource caches.
	@rm -rf ./public ./resources/_gen
	@printf "Removed public/ and resources/_gen/.\n"

prepublish: check ## Verify branch, worktree, and the production build before publishing.
	@test "$$("$(GIT)" branch --show-current)" = "main" || { printf "Publishing is only allowed from the main branch.\n"; exit 1; }
	@test -z "$$("$(GIT)" status --porcelain)" || { printf "The worktree is not clean. Commit or stash changes before publishing.\n"; exit 1; }
	@draft_count="$$(rg -l '^draft:[[:space:]]*true[[:space:]]*$$' content/posts --glob '*.md' | wc -l | tr -d ' ')"; \
		printf "%s draft post(s) will remain excluded from production.\n" "$$draft_count"
	@$(MAKE) --no-print-directory build
	@printf "Pre-publish checks passed. GitHub Pages will deploy when main is pushed.\n"

publish: ## Push clean, validated main to origin. Requires PUBLISH=1.
	@test "$(PUBLISH)" = "1" || { printf "Refusing to publish without confirmation. Run: make publish PUBLISH=1\n"; exit 2; }
	@$(MAKE) --no-print-directory prepublish
	@"$(GIT)" push origin main
