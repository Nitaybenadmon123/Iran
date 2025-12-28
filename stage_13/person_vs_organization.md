```mermaid
flowchart TD
    %% צבעים – אותו עיצוב כמו תרשימים 1–2
    classDef person fill:#b6e3b6,stroke:#2e7d32,color:#000;
    classDef org fill:#f4b6b6,stroke:#b71c1c,color:#000;
    classDef unknown fill:#e0e0e0,stroke:#424242,color:#000;
    classDef question fill:#e3f2fd,stroke:#1565c0,color:#000;

    A([Start]) --> B[Does the username or display name<br/>contain organizational keywords?]
    class B question

    B -->|Yes| O[ORGANIZATION]
    class O org

    B -->|No| C[Does the profile image appear<br/>to be a personal photo?]
    class C question

    C -->|Yes| P[PERSON]
    class P person

    C -->|No| D[Does the profile image appear<br/>to be a logo or brand symbol?]
    class D question

    D -->|Yes| O

    D -->|No| E[Does the bio describe a person<br/>and personal roles or professions?]
    class E question

    E -->|Yes| P

    E -->|No| F[Does the bio describe an institution,<br/>media outlet, or organization?]
    class F question

    F -->|Yes| O

    F -->|No| G[Does the tweet content reflect<br/>personal opinions or daily activity?]
    class G question

    G -->|Yes| P

    G -->|No| U[UNKNOWN]
    class U unknown
