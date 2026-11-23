# Architecture

## 1. Agent workflow

```mermaid
flowchart LR
    U([User]) -->|"file (.csv / .xlsx / .json)<br/>or sample name + goal"| V

    subgraph Team["AI data team (Google ADK + Gemini)"]
        V["<b>Verify Agent</b><br/>load + EDA<br/>agree on goal"]
        D["<b>Data Science Agent</b><br/>clean, train 3 models,<br/>pick winner, charts"]
        R["<b>Report Agent</b><br/>write 5 sections,<br/>build PDF"]
        V -- "confirm_goal()<br/>hands off" --> D
        D -- "bundle_report_package()<br/>hands off" --> R
    end

    V -. "can't support goal<br/>→ ask again" .-> U
    R -->|report.pdf| U

    P1[["Payload 1<br/>user_responses.txt<br/>dataVerified.txt<br/>usergoal.txt<br/>dataset.csv"]]
    P2[["Payload 2<br/>+ dataset_changes.txt<br/>+ model_selection_why.txt<br/>+ metrics.json<br/>+ charts/*.png<br/>+ report_agent_package.txt"]]
    V --> P1 --> D
    D --> P2 --> R
```

## 2. Verify Agent conversation loop

```mermaid
flowchart TD
    S([Start]) --> W[Welcome + explain accepted formats]
    W --> F{File or sample<br/>in message?}
    F -- no --> NF["'I didn't see a dataset…'<br/>(file_not_supported loop)"] --> F
    F -- yes --> L[load_dataset → dataVerified.txt]
    L --> OK{Supported and<br/>≥ 20 rows?}
    OK -- no --> NF
    OK -- yes --> Q["'What do you want to analyze?'<br/>+ 2 examples"]
    Q --> G{Goal given?}
    G -- "'I don't know'" --> DEF[check_goal('') → default target,<br/>'Explain the whole dataset']
    G -- "needs a worked-out column<br/>(pass/fail, change, total)" --> CT["create_target(name, formula)<br/>e.g. passed = G3 >= 10"] --> C
    G -- yes --> C[check_goal(target)]
    C --> SUP{Data supports it?}
    SUP -- no --> NO["'Your dataset doesn't support that'"] --> Q
    SUP -- yes --> CG[confirm_goal → usergoal.txt]
    DEF --> CG
    CG --> DS([Data Science Agent])
```

## 3. Data Science Agent steps (all fixed, tested tools)

```mermaid
flowchart TD
    A[read_verify_payload] --> B[clean_dataset<br/>→ dataset_changes.txt<br/>+ leakage check]
    B --> C["train_models('all_features')<br/>60/20/20 split · 3 candidates"]
    C --> T{Validation scores<br/>within 0.01?}
    T -- yes --> SIMP[Keep simpler model]
    T -- no --> BEST[Keep best model]
    SIMP & BEST --> FIT[Refit on train+val,<br/>score on test, permutation importance]
    FIT --> LK{Leaky features flagged?<br/>e.g. G1, G2}
    LK -- yes --> C2["train_models('without_G1_G2')"] --> CH
    LK -- no --> CH[make_charts]
    CH --> PK[bundle_report_package → Report Agent]
```

## 4. Deployment

```mermaid
flowchart LR
    Browser -->|HTTPS| CR["Cloud Run<br/>adk web (container)"]
    CR -->|API key| GEM[Gemini API<br/>AI Studio]
    CR -.->|"GOOGLE_GENAI_USE_VERTEXAI=TRUE"| VX[Vertex AI Gemini]
    CR -->|run folders + PDF| GCS[(Cloud Storage<br/>OUTPUT_BUCKET)]
```
