```mermaid
flowchart TD
    %% צבעים
    classDef target fill:#b6e3b6,stroke:#2e7d32,color:#000;
    classDef nontarget fill:#f4b6b6,stroke:#b71c1c,color:#000;
    classDef unknown fill:#e0e0e0,stroke:#424242,color:#000;
    classDef question fill:#e3f2fd,stroke:#1565c0,color:#000;

    A([Start]) --> B[Is a location specified<br/>in the profile?]
    class B question

    B -->|Yes| C[Does the location explicitly<br/>indicate Iran?]
    class C question

    B -->|No| D[Does the profile bio indicate<br/>Iranian identity?]
    class D question

    C -->|Yes| T[TARGET]
    class T target

    C -->|No| D

    D -->|Yes| T
    D -->|No| E[Is there linguistic or semantic<br/>evidence in tweets?]
    class E question

    E -->|Yes| T
    E -->|No| F[Is the information insufficient<br/>or contradictory?]
    class F question

    F -->|Yes| U[UNKNOWN]
    class U unknown

    F -->|No| N[NON_TARGET]
    class N nontarget
