# llm-architectures-refresher

Runnable receipts for Han's LLM Architecture Refresh blog series (posts in
`~/Documents/GitHub/bearbearyu1223.github.io`).

- Read `~/.claude/projects/-Users-hanyu/memory/MEMORY.md` first; all series memories live there.
- The review and publish workflow is the blog's post-desk skill
  (`~/Documents/GitHub/bearbearyu1223.github.io/.claude/skills/post-desk/SKILL.md`). Its gate
  (`blog_gate.py`, wired in `.claude/settings.json` here) turns every push of this repo into a permission prompt.
  This repo is pushed before the blog.
- Long runs: launch detached (`nohup caffeinate -i uv run demoNN ... &`), watch with Monitor, and send Han a
  push notification when a run finishes or fails.
- Capture stdout only (`> outputs/demoNN.txt`, stderr elsewhere); recapture before checking receipts.
