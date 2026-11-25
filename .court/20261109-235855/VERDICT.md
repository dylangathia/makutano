# The verdict: GUILTY, 12 to 0

**The case:** Keep the Makutano final-year project (video → forecast → reinforcement learning signal control in a calibrated SUMO simulation) and add an evidence-grounded "explain why" layer: every signal decision is logged with its evidence, an LLM agent answers operator questions only from those records, and a verifier rejects any answer whose numbers or claims don't match the logs.

**The charge:** The case fails as stated: it should not go ahead the way it is described.

## How the jury voted

| Juror | Vote | Why |
|---|---|---|
| The Accountant | GUILTY | Nobody can tell me what this layer costs: there's no LLM model, no price per query, no offline plan and no labelling effort. It wants to spend effort on a fourth module while the three it's supposed to explain haven't run a single time, and I won't sign a budget line with no figure on it that also comes before the work it depends on. |
| The First-Time Customer | GUILTY | Hearing this for the first time, I'm told it "can prove why it made every decision," and then I find out there is no trained agent, no footage and no SUMO run, and even the defense admits "prove" overstates what it does, so I wouldn't believe the pitch as it's written. |
| The Competitor | GUILTY | If I were the rival pitching against this tomorrow, I would not need to copy it: I would ask the examiners to see the trained RL agent whose decisions it "can prove", and because that agent has never run and the logs hold only inputs and outputs, not the policy's reasons, the headline claim falls apart before I have to say anything else. |
| The Retired Judge | GUILTY | The charge is about the case as stated, and as stated its central claim, that the system "can prove why it made every decision", goes beyond what the record supports; the defense admits this, and its fixes (rewording, building the layer only after the core runs, a held-out error set) are promises, not anything in the record. |
| The Teenager | GUILTY | A chatbot explaining an RL agent that has never been trained is just a chatbot explaining a pressure rule, and nobody's coming back next week to ask why a fake agent did a fake thing, plus even the defense admits the "prove why" pitch is overselling it. |
| The Engineer | GUILTY | As an engineer I won't sign off on a fourth layer stacked on a base that has never run, because right now the explanation layer would only explain a stand-in pressure rule, and the plan as described doesn't say which LLM it uses, how it runs offline, or how a non-numeric "why" claim gets verified. |
| The Small-Business Owner | GUILTY | You don't open a second counter when the first one has never sold anything: the core product hasn't run yet, one person does all the work, and as written this plan puts a new line on the shelf with no order of work, no cost for the LLM, and a sign out front ("can prove why") that even the defense says overpromises. |
| The Statistician | GUILTY | Every number here would come from a sample of zero: no footage, no trained agent and no SUMO run exist yet. The verifier's headline catch rate would also be measured on errors its own author injected, so the record gives me no base rate that supports adding a fourth module or a claim to "prove why." |
| The Parent | GUILTY | If my child came home with this plan I'd say no: it spends scarce final-year time on a new LLM layer that explains a stand-in rule, while the core work it depends on hasn't run, and it promises to "prove why", which even its own defender admits it can't do. |
| The Investor | GUILTY | I would not put money into this as described, because it adds a fourth module on a core that has never produced a single result, and it is sold on a headline ("can prove why it made every decision") that even the defense admits the design cannot deliver. |
| The Designer | GUILTY | The one-line pitch is the first thing a stranger reads, and it promises "prove why" when both sides agree the design only shows the evidence around a decision, so the story is clear but wrong, and the project also goes from three parts to four before any of the first three has run. |
| The Lawyer | GUILTY | As written, the pitch makes two claims the record shows are untrue, that Makutano "runs a Nairobi junction's signals" when it only controls a simulation that has not run yet, and that it "can prove why it made every decision" when the verifier only checks answers against logs the system writes about itself; I would not sign off on a headline like that. |

## Charges that stuck

1. **It explains a base that has never run.** No footage, no trained RL agent and no SUMO run exist, so today the layer would explain a stand-in pressure rule. All 12 jurors leaned on this.
2. **The pitch overclaims.** "Can prove why it made every decision" goes beyond what logs of inputs, outputs and shield firings can show, and the defense conceded this. 10 jurors cited it in their reasons.
3. **The LLM side has no plan.** The case names no model, cost, offline fallback or labelling effort. 5 jurors raised it.

## What the defense got right

1. The evidence fields (measured queues, forecast, choice, shield rule fired) already exist in the pipeline and console, so the layer reuses data instead of starting a new field.
2. The evaluation can be tested: ground truth comes from the logs, and the verifier's catch rate can be measured. The case is still silent on how non-numeric claims get checked.

## Conditions for acquittal

1. **Write the order of work into the proposal:** the explain layer starts only after the trained RL agent has produced results in calibrated SUMO (all 12 jurors).
2. **Reword the pitch** to something like "shows the logged evidence behind every decision" in a simulated junction, dropping "prove why" and "runs a Nairobi junction" (all 12).
3. **Name the LLM, its cost or offline fallback, and a verifier test set written by someone other than you** (6 jurors across these asks).

## The sentence

The court finds the idea sound but the case overstated and out of order. This week, install SUMO, Gymnasium and Stable-Baselines3 and get the core running. Revise the proposal so the explain layer comes last, under an honest pitch, with its model and test set named. Then bring it back.

_The full trial is in C:\makutano\.court\20261109-235855: the opening statements, the rebuttals and every juror's vote._

_Made with [/court](https://github.com/alexyc9381/court-skill), a free Claude Code skill by Alex Chen ([@nocodealex](https://instagram.com/nocodealex))._
