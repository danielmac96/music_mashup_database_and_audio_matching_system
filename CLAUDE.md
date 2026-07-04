# CLAUDE.md — AI Assistant Guide

current goal: The onboarding flow is built and working. Importing a SoundCloud link in the
web app now auto-processes every track through download → stems → analyze → structure via a
bounded, resumable job queue (Import + Library tabs; see readme "First run"). Current focus:
robustness (error surfacing + retry, stale-preview re-verify, dependency health check) and
improving **pairwise** mashup quality (matching + the Audition Studio playground). Staying
local (FastAPI + Vite + SQLite); no cloud and no multi-song "Big Bootie" set builder yet.

---

## Project Purpose

Take soundcloud link. Get all info on songs as possible from soundcloud. Download using the current download script.
Improve ingest and download folders where possible.

We want very simple user friendly steps. Ultimately the web app will be used to interact with playlist links to download 
to a specificed local location. 