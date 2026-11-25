# The verdict: GUILTY, 11 to 1

**The case:** Record about 20 minutes of real video at a Nairobi junction, then have the project's model run the traffic and signal control on that same traffic, and see the difference against how the junction actually performed.

**The charge:** The case fails as stated: it should not go ahead the way it is described.

## How the jury voted

| Juror | Vote | Why |
|---|---|---|
| The Accountant | GUILTY | The plan is costed only in words like "cheap" and has no sourced figure for time or effort, yet it spends recording, extraction and replay work, before 24 April 2027, on a comparison whose result can't be trusted as described: the "agent" is a stand-in rule in a stand-in microsimulation, and both sides of the comparison come from a detector that has never run, judged against a "roughly" with no number attached. |
| The First-Time Customer | GUILTY | If someone showed me this "difference" for the first time, I'd assume I was watching the project's trained controller beat the real junction, and I'd feel misled to learn it was a hand-written rule in a toy simulator replaying arrivals that today's lights had already shaped. |
| The Competitor | GUILTY | If I were the rival vendor, I would let them show their "difference" and then point out that it is a hand-written pressure rule replaying traffic already shaped by today's lights inside a toy simulator, judged against an undefined "roughly", and that one rebuttal would wipe out their headline; as described, this demo gives me ammunition and gives them no edge. |
| The Retired Judge | GUILTY | The defense's own concessions show the comparison as described would measure a stand-in rule in a stand-in simulator against an unvalidated detector, judged by an undefined "roughly", so its case for going ahead this way rests on hope, not on evidence in the record. |
| The Teenager | NOT GUILTY | Honestly, the multi-seed confidence intervals put me to sleep, and real Nairobi traffic going through the model next to what actually happened on the video is the only part of this anyone would watch, talk about or remember, so do it, as long as the check against today's plan passes first. |
| The Engineer | GUILTY | As a builder I can see the recording and the See run working, but the comparison as described feeds arrivals from a detector that has never run into a stand-in microsimulation driven by a stand-in rule, and checks it against an undefined "roughly" with no hand counts, so a mismatch or a "difference" could come from any of those parts and nobody could tell which. |
| The Small-Business Owner | GUILTY | One person is doing all the work here, and the main product isn't on the shelf yet because SUMO isn't installed and nothing is trained, so spending the next free Monday on a side demo that pits a stand-in rule against an unchecked count is stocking the window before the shop has anything to sell. |
| The Statistician | GUILTY | A single 20-minute clip gives one difference with no error bar, and the only yardstick for separating real controller effect from replay and simulator error is a "roughly" match that nobody has defined, so whatever gap appears cannot be told apart from noise, and once shown at a viva it will be read as a result. |
| The Parent | GUILTY | I'd back a cheap recording trip, but not one where the "model" being shown off is really a hand-written stand-in rule judged against detector numbers nobody has checked, because a demo you can't trust is worse than no demo when the time could go to the experiments the grade depends on. |
| The Investor | GUILTY | I'd pay for the recording and the first run through See, because that buys real information cheaply, but as described the headline step pits a hand-written stand-in rule against today's plan in a toy simulator with no defined pass mark, so the main thing it can produce is a misleading number shown at a viva, and the plan carries that risk without the matching upside. |
| The Designer | GUILTY | Anyone watching a "model vs. the real junction" comparison will take away "the ML controller beat today's traffic," but as described the thing doing the controlling is a hand-written stand-in rule in a toy simulator, so the headline would tell them something that isn't true. |
| The Lawyer | GUILTY | As described, the plan goes straight from "record about 20 minutes" at a public Nairobi junction to processing every car, boda boda and pedestrian in the clip, with no step for filming permission, university ethics clearance or handling of identifiable people and number plates, and it would present a hand-written stand-in rule as "this model", which is a misrepresentation risk at the viva; both are fixable, but the plan as stated leaves them out. |

## Charges that stuck

1. **There is no model to test yet.** The "agent" in the console is a hand-written stand-in rule, so a "model vs. real junction" difference would misstate what was tested (10 jurors).
2. **"Roughly" has no number.** The step-4 check that simulated "today's plan" matches the video has no tolerance, so no gap can be told apart from replay error (6 jurors in their reasons; 10 asked for a fixed tolerance).
3. **The detector has never run and is unchecked.** The same See output supplies both the demand and the "actual" performance, with no hand count to catch its errors (6 jurors).

## What the defense got right

1. It would be the pipeline's first real end-to-end run. Even the prosecution called this the defense's best point.
2. The record already treats the clip as a demonstration, not the main evaluation, so a failed check costs a demo and not the project.

## Conditions for acquittal

1. Rewrite the plan so the agent-versus-today comparison runs only with the trained agent in the calibrated SUMO model, and label any earlier run as "stand-in rule". This week, install SUMO, Gymnasium and Stable-Baselines3 so that path exists.
2. Write down a numeric match tolerance for "today's plan" now, before any footage is recorded, and report a gap only if it is larger than that tolerance.
3. Add a hand-count check of See against a few minutes of the clip, log arrivals upstream of the queue, and confirm filming permission and the junction's real control mode.

## The sentence

Record the footage and run it through See. That part is sound. Do not show a "difference" until the real agent exists and the yardstick is fixed in advance. Unblock the core experiments first.

_The full trial is in C:\makutano\.court\20261025-150210: the opening statements, the rebuttals and every juror's vote._

_Made with [/court](https://github.com/alexyc9381/court-skill), a free Claude Code skill by Alex Chen ([@nocodealex](https://instagram.com/nocodealex))._
