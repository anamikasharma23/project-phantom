# Judge demo guide

## Before you present

1. `python -m phantom serve` and open http://127.0.0.1:8765 (live console, decoy mission loaded, status READY).
2. Close the guide. Set the speed to Normal.
3. Backup if anything goes wrong: double-click `ui/index.html` for the saved replay, or play `docs/mission-replay.mp4`.

## Two-minute demo script

**0:00 The problem (15 s).**
"WALL-E has to find EVE on an unknown terrain map using signals that may be noisy or fake, with limited battery and time. My agent, PHANTOM, keeps a probability map of where EVE could be, decides what to do by scoring its options, and logs every decision so you can replay it."

**0:15 The map (20 s).** Point at the map.
"The glow is WALL-E's belief: brighter means more likely. This is his guess, not the answer. The lines are signal bearings. There are three sources here, and one of them, ghost_echo, is secretly a decoy."

**0:35 Start, then pause around step 10 to 13 (25 s).** Press the round Start button; this is the real agent running live.
"Look: the hottest cells are now in the top-left, where the decoy points. Early on, WALL-E has no independent evidence yet, so even a doubtful source still pulls the belief. Watch what happens next." Step forward to 14. "At step 13 Signal DNA rated the decoy 'conflicting', and at step 14 the belief moved back toward the real signals."

**1:00 Investigation screen (30 s).** Click the Investigation tab.
"These are all the options WALL-E considered at this step, with real numbers: expected information in bits, battery, time, risk, and the score. The rejected ones are the ghosts. Some are vetoed because they would leave too little battery to get home." Click Signal DNA. "Each source has a trust estimate with a range. Sources are only judged against scans and physical clues, never against each other, so the decoy can't make an honest source look bad."

**1:30 The result (20 s).** Press Run to end, then open the Report tab.
"WALL-E confirmed EVE at (19, 3) with his own scan, inferred she is active, reports confidence and resources, and recommends the next move. This report is generated only from the trace. The ground truth is hidden behind this switch, and the agent never had access to it."

**1:50 Evidence it works (10 s).**
"68 automated tests pass, including a test that the agent code never imports the environment. On 100 random missions, PHANTOM found EVE far more often than a sweep baseline, and it was never stranded. The README lists the exact numbers and limitations."

## Likely questions

**Is this machine learning?**
No. It's probabilistic reasoning plus a hand-designed scoring rule. The "learning" in Signal DNA is counting agreements and disagreements (a Beta distribution), which is statistics, not a trained model.

**How do you know the agent isn't cheating?**
EVE's position only exists inside `world.py`. The agent only receives a public briefing and observation objects. A test checks that agent files never import the environment, and another changes only the hidden truth and checks that the agent's first decision is identical.

**What is a Bayes filter, simply?**
Start with "EVE could be anywhere". Each clue says how likely it would be if EVE were in each cell. Multiply those in and rescale so everything adds to 100%. Cells that fit all the evidence end up bright.

**What does "information gain in bits" mean?**
Entropy measures how spread out the belief is. One bit is like halving the number of equally likely places. A scan's information gain is how much it is expected to shrink that spread.

**Why not just drive to the brightest cell?**
Because driving costs battery and time. The score divides expected information by cost, so a cheap scan nearby can beat a long drive. A safety veto stops any plan that would leave WALL-E unable to get home.

**How does Signal DNA avoid circular reasoning?**
A source is never judged by a belief that includes its own readings, or another signal source's readings. It's judged only against empty scans, physical clues and the old initial signal. Once EVE is actually seen, every past reading is re-scored against her real position.

**Did Signal DNA actually help?**
In our benchmark it mattered in the decoy scenario; in the other scenarios the results with and without it are within noise. We report both.

**What are the limitations?**
Greedy one-step planning, EVE doesn't move, terrain is known in advance, the agent models fake readings as random rather than as a consistent decoy, and the benchmark covers only our simulator.

**What happens with a bad config or too little battery?**
Bad configs stop with a specific message. With too little battery the agent stays safe, explains that the likely area is out of round-trip range, and recommends a bigger battery or a forward charging point.

## Files to understand before presenting

1. `phantom/world.py`: what the world hides and what an observation contains.
2. `phantom/belief.py`: the probability grid and how each evidence type updates it.
3. `phantom/planner.py`: candidate scoring, the veto, and the safety rules.
4. `phantom/signal_dna.py`: the reliability counters and the independence rule.
5. `phantom/mission.py`: the main loop in `MissionRunner.step()`: decide, act, observe, log.
6. `configs/decoy_mission.yaml`: what a mission config looks like.
7. `tests/test_agent.py`: the no-cheating and DNA tests (good to show).
