```mermaid
flowchart TD
    %% צבעים – אותו עיצוב
    classDef local fill:#b6e3b6,stroke:#2e7d32,color:#000;
    classDef diaspora fill:#f4b6b6,stroke:#b71c1c,color:#000;
    classDef unknown fill:#e0e0e0,stroke:#424242,color:#000;
    classDef skip fill:#eeeeee,stroke:#616161,color:#000;
    classDef question fill:#e3f2fd,stroke:#1565c0,color:#000;

    A([Start]) --> B[Is the user labeled as TARGET<br/>from Flow 1]
    class B question

    B -->|No| X[Column not relevant<br/>Skip classification]
    class X skip

    B -->|Yes| C[Is a profile location specified?]
    class C question

    C -->|Yes| D[Does the location indicate<br/>residence in Iran?]
    class D question

    D -->|Yes| L[LOCAL]
    class L local

    D -->|Other city or country| R[DIASPORA]
    class R diaspora

    C -->|No| E[Does the profile bio suggest<br/>living in Iran?]
    class E question

    E -->|Yes| L

    E -->|No| F[Does tweet content suggest<br/>day-to-day life in Iran?]
    class F question

    F -->|Yes| L

    F -->|No| U[UNKNOWN]
    class U unknown
