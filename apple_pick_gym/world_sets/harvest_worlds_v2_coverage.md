# harvest_worlds_v2 coverage

Candidates: v1 (seeds 0-9 x 64) + v2 (seeds 10-23 x 128) = 2432 from `fruiting_system_ranges_rl_harvest_real_g05_m1.json`; 2212 passed both screening builds; `harvest_worlds_v2.jsonl` is a seeded random 2000 of those.

2212/2432 worlds accepted

| knob | candidate range | accepted range | reject rate low / mid / high tercile |
| --- | --- | --- | --- |
| primary.density | 325.5 .. 1218 | 325.5 .. 1218 | 10% / 10% / 7% |
| spur.length | 0.0892 .. 0.1056 | 0.0892 .. 0.1056 | 9% / 8% / 9% |
| spur.radius | 0.004268 .. 0.005836 | 0.004268 .. 0.005836 | 10% / 7% / 11% |
| spur.density | 2880 .. 8193 | 2880 .. 8193 | 8% / 11% / 9% |
| spur.youngs_modulus_pa | 4.967e+07 .. 1.068e+09 | 4.967e+07 .. 1.068e+09 | 10% / 7% / 10% |
| spur.flexural_modulus_pa | 1.854e+06 .. 4.509e+07 | 1.854e+06 .. 4.509e+07 | 8% / 11% / 8% |
| spur.damping_ratio | 0.1971 .. 0.7227 | 0.1971 .. 0.7227 | 8% / 7% / 12% |
| stem.length | 0.01374 .. 0.01477 | 0.01374 .. 0.01477 | 9% / 10% / 8% |
| stem.youngs_modulus_pa | 3.024e+07 .. 5.635e+09 | 3.024e+07 .. 5.635e+09 | 9% / 7% / 11% |
| stem.flexural_modulus_pa | 2.968e+07 .. 1.998e+08 | 2.968e+07 .. 1.998e+08 | 14% / 8% / 6% |
| stem.damping_ratio | 0.2524 .. 0.7647 | 0.2524 .. 0.7647 | 8% / 9% / 10% |
| stem.elevation_deg | -89.5 .. -30.25 | -89.5 .. -30.25 | 3% / 7% / 17% |
| apple_radius | 0.03009 .. 0.03355 | 0.03009 .. 0.03355 | 8% / 8% / 11% |
| apple_density | 1617 .. 2238 | 1617 .. 2238 | 9% / 9% / 9% |
| support_kp | 186.2 .. 301.5 | 186.2 .. 301.5 | 8% / 10% / 9% |
| support_roll_kp | 0.4841 .. 1.013 | 0.4841 .. 1.013 | 9% / 9% / 10% |
| support_zeta | 0.2513 .. 0.7818 | 0.2513 .. 0.7818 | 8% / 10% / 9% |
| arm_link_mass_scale | 0.9 .. 1.1 | 0.9 .. 1.1 | 9% / 9% / 10% |
| arm_link_inertia_scale | 0.9002 .. 1.1 | 0.9002 .. 1.1 | 9% / 8% / 10% |
| arm_ee_mass_scale | 0.95 .. 1.05 | 0.95 .. 1.05 | 9% / 9% / 10% |
| arm_ee_inertia_scale | 0.95 .. 1.05 | 0.95 .. 1.05 | 9% / 10% / 9% |
| weld_polar_deg | 1.854 .. 29.94 | 1.854 .. 29.94 | 8% / 10% / 10% |
