# Data

Do not commit downloaded datasets, caches, or generated training corpora here.

Record each dataset used in an experiment in the corresponding config and result metadata, including:

- canonical dataset name and source
- exact split / subset
- preprocessing version or script commit
- number of examples
- contamination / overlap checks if applicable
- role: ID train, ID validation, OOD evaluation, or anchor set

The ID/OOD definition should follow an existing protocol when possible rather than being chosen after seeing results.
