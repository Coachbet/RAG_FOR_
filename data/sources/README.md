# Data sources (links only)

The FIA documents are public but **not redistributed** in this repository: only the links are here, and
`make data` downloads them into `data/raw/fia/` (git-ignored).

- `rulebooks.csv`: the rule documents indexed by the RAG. One row per PDF. To move to a newer issue of the
  Sporting Regulations, replace the URL and the date (only the latest issue should be indexed).
- `seasons.csv`: the FIA "Decision documents" season pages. The download script walks every event of each
  `enabled=yes` season and keeps only the stewards' decisions about one car ("Offence", "Infringement",
  "Decision" documents). Set 2024 to `yes` if we have fewer than 300 usable incidents.
- `extra_decisions.txt` (optional): one decision PDF URL per line, added on top of the season pages.

Rules followed: public pages only, `Crawl-delay: 10` from fia.com/robots.txt (one request every 10 s, so a full
season takes about 1 to 2 hours the first time; everything is cached afterwards), drivers anonymised before any
file leaves `data/raw/`, short extracts only in the app and the report.
