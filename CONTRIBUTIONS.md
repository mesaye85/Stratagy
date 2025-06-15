Civic Horizon Project: Contribution Guide

Purpose

This guide outlines how to contribute to the Structural Democracy Project — a community-driven initiative to build a transparent, behaviorally grounded, non-ideological dataset for evaluating the democratic legitimacy of nation-states.

Unlike conventional indices, we do not measure democracy by alignment with Western liberal norms, international institutions, or procedural formalism. Instead, we measure it by the **structural empowerment of people to shape their future**, using behavior-based indicators across five domains.

This guide explains the structure, intent, and required documentation for each contribution.

---

 Dataset Structure

Each country is scored across **20 fields**, grouped into **5 structural domains**, using a 0–10 scale. Contributors must:

* Provide a numeric score (0 to 10)
* Include a 1–2 sentence justification
* Cite the source (e.g., law, news, official record)

> ⚠️ **Do not rely on perception-based indices or NGO opinions.**
> Use verifiable public data, legal facts, or documented behavior.

---

## 🧱 Domains & Fields

### 🟦 1. Power Responsiveness Domain

Measures the structural capacity for leadership change and policy responsiveness.

| Field                         | Description                                                                                 | Examples                                      |
| ----------------------------- | ------------------------------------------------------------------------------------------- | --------------------------------------------- |
| `Leadership_Turnover_Score`   | Can leadership realistically change hands through opposition or nonviolent public pressure? | Peaceful regime changes, opposition victories |
| `Local_Governance_Score`      | Do municipalities have budget and policy control over key sectors?                          | Independent school boards, regional taxation  |
| `Policy_Responsiveness_Score` | Do protests or public pressure lead to actual reforms?                                      | Legislation passed after demonstrations       |

---

### 🟧 2. Sociopolitical Coherence Domain

Assesses whether the system reflects its population, not in kindness or rhetoric, but in functional inclusion.

| Field                                      | Description                                                                | Examples                                           |
| ------------------------------------------ | -------------------------------------------------------------------------- | -------------------------------------------------- |
| `Language_Alignment_Score`                 | Does the state operate in a language understood by the majority?           | Census vs. administrative language                 |
| `Participation_Accessibility_Score`        | Can people from poor/rural/minority backgrounds enter governance?          | Parliamentary bios, electoral access laws          |
| `Civilian_Control_Score`                   | Are elected officials supreme over military decisions and budget?          | Civilian-led MoD, coup absence                     |
| `Structural_Advocacy_and_Prosperity_Score` | Can minority groups organize, compete, and rise without legal obstruction? | Minority-owned businesses, protest protection laws |

---

### 🟨 3. Material Well-being & Agency Domain

Focuses on whether citizens have the minimum conditions to meaningfully act.

| Field                        | Description                                                  | Examples                                        |
| ---------------------------- | ------------------------------------------------------------ | ----------------------------------------------- |
| `Healthcare_Access_Score`    | Can poor or rural people access essential healthcare?        | Public clinic maps, coverage laws               |
| `Education_Completion_Score` | Do low-income or rural students complete secondary school?   | Graduation stats by income or region            |
| `Food_Sovereignty_Score`     | Can the country feed itself without total import dependency? | FAO production ratios, caloric self-sufficiency |
| `Migration_Freedom_Score`    | Can people move freely for work or housing?                  | Internal passport laws, housing permit systems  |

---

### 🟩 4. Future Perception Domain

Assesses whether people believe they have a future within the system.

| Field                                | Description                                                | Examples                                    |
| ------------------------------------ | ---------------------------------------------------------- | ------------------------------------------- |
| `Agency_Projection_Score`            | Do people believe they can impact the system?              | Grassroots orgs, non-violent change history |
| `Pathway_to_Influence_Score`         | Can regular people rise into public influence?             | Mayors from unions or local businesses      |
| `Collective_Future_Confidence_Score` | Do people believe the country has a viable future?         | Emigration rates, birthrate rebound         |
| `System_Adaptability_Score`          | Has the system reformed in response to legitimate demands? | Legal amendments, new inclusive structures  |

---

### 🟨 5. Behavioral Future Engagement Domain

Measures whether people are acting *now* as if they believe in the system’s future.

| Field                              | Description                                                 | Examples                                              |
| ---------------------------------- | ----------------------------------------------------------- | ----------------------------------------------------- |
| `Collective_Momentum_Score`        | Are new civic movements, unions, or parties forming?        | Registration logs, public organizing                  |
| `Life_Investment_Score`            | Are young people settling, starting families, buying homes? | Fertility rate, mortgage growth, education investment |
| `Exit_vs_Voice_Behavior_Score`     | Are people more likely to emigrate or protest?              | Emigration vs. civic participation data               |
| `Narrative_Action_Alignment_Score` | Does national rhetoric match what people are *doing*?       | "Reform" narrative vs. grassroots shifts              |

---

## 🧮 Scoring Guidelines

Use the 0–10 scale based on factual performance:

| Score | Meaning                                        |
| ----- | ---------------------------------------------- |
| 10    | Fully realized access, structure is empowering |
| 7–9   | Strong but imperfect                           |
| 4–6   | Accessible in theory, spotty in practice       |
| 1–3   | Symbolic or highly obstructed                  |
| 0     | Effectively prohibited or absent               |

Each score must be **justified** with a short note and source.

---

## 🧾 Example Submission

```csv
Country,Leadership_Turnover_Score,...,Democracy_Status
India,7,...,Democratic
```

Example justification:

> `Leadership_Turnover_Score: 7` — Power changed hands in 2014 and 2019 through competitive elections. Opposition had access to media and voting mechanisms. Source: \[Election Commission of India, Reuters]

---

## 🚀 How to Contribute

1. Clone or fork the repository.
2. Open `data/structural_democracy_template.csv`.
3. Add your country and scores with justifications.
4. Submit a Pull Request (PR) with your additions.
5. If unsure, submit via our [Google Form](#).

We review all entries manually for:

* Source credibility
* Reasonable justification
* Alignment with the structural philosophy of the model

---

## 🛑 What Not to Submit

* Opinion pieces, editorial ratings, or NGO perception indices
* Scores without justification or sources
* Scores derived from aggregated liberal/conservative ideology indexes

---

## 📬 Questions or Feedback?

Join the discussion on [GitHub Issues](#) or email us at `contact@civichorizon.org`

---

## 🌐 License

This project is open-source under the [MIT License](LICENSE). All contributions must include open-access sources or citation permissions.
