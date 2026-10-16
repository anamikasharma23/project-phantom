# Benchmark (20 seeded missions per scenario)

| Scenario | Agent | Success | 95% CI | Steps when found | Battery spent | Stranded | P(true cell) | Status right |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | phantom | 100% | 84%–100% | 34.1 | 55.4 | 0 | 1.00 | 95% |
| clean | phantom_no_dna | 100% | 84%–100% | 37.0 | 62.1 | 0 | 1.00 | 95% |
| clean | sweep | 50% | 30%–70% | 75.2 | 186.0 | 0 | 0.68 | 90% |
| clean | random | 5% | 1%–24% | 211 | 340.3 | 0 | 0.06 | 50% |
| noisy | phantom | 100% | 84%–100% | 42.9 | 63.0 | 0 | 1.00 | 90% |
| noisy | phantom_no_dna | 100% | 84%–100% | 38.5 | 62.3 | 0 | 1.00 | 100% |
| noisy | sweep | 50% | 30%–70% | 75.2 | 186.0 | 0 | 0.68 | 90% |
| noisy | random | 5% | 1%–24% | 211 | 340.3 | 0 | 0.06 | 50% |
| unreliable | phantom | 100% | 84%–100% | 50.9 | 81.3 | 0 | 1.00 | 100% |
| unreliable | phantom_no_dna | 100% | 84%–100% | 52.8 | 90.5 | 0 | 1.00 | 100% |
| unreliable | sweep | 40% | 22%–61% | 75.9 | 210.0 | 0 | 0.64 | 90% |
| unreliable | random | 0% | 0%–16% | – | 341.8 | 0 | 0.01 | 55% |
| decoy | phantom | 95% | 76%–99% | 63.4 | 116.5 | 0 | 0.95 | 95% |
| decoy | phantom_no_dna | 35% | 18%–57% | 88.6 | 193.9 | 0 | 0.45 | 95% |
| decoy | sweep | 50% | 30%–70% | 75.2 | 186.0 | 0 | 0.68 | 90% |
| decoy | random | 5% | 1%–24% | 211 | 340.3 | 0 | 0.06 | 50% |
| tight_battery | phantom | 100% | 84%–100% | 67.0 | 94.0 | 0 | 1.00 | 100% |
| tight_battery | phantom_no_dna | 90% | 70%–97% | 72.3 | 104.7 | 0 | 1.00 | 100% |
| tight_battery | sweep | 45% | 26%–66% | 102.6 | 199.4 | 0 | 0.58 | 95% |
| tight_battery | random | 0% | 0%–16% | – | 339.7 | 0 | 0.01 | 60% |
