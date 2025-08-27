Files for two-agent knowledge-exchange RL demo
---------------------------------------------
obstacles.json : Grid size, goal, and obstacle coordinates (0-based).
knownA.csv     : 10x10 binary matrix; A's initial obstacle knowledge (north half).
knownB.csv     : 10x10 binary matrix; B's initial obstacle knowledge (east half).

Place all files in your Windows folder:
C:\backupcgi\final_bak\two_agents_knowlodge

Your script should load:
- obstacles.json (JSON)
- knownA.csv, knownB.csv (CSV without header)

These files are deterministic and match the example used in the paper.
