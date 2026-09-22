# harvest_worlds_v1 coverage

Screened with `example_screen_harvest_worlds.py` (default ScreeningConfig) on the default ranges fixture `fruiting_system_ranges_rl_harvest_real_g05_m1.json`: pass 1 = seeds 0-9 x 64 envs (603/640), pass 2 = survivors rebuilt shuffled in chunks of 128 (587/603). Accepted worlds passed both builds.

587/640 worlds accepted

| knob | candidate range | accepted range | reject rate low / mid / high tercile |
| --- | --- | --- | --- |
| primary.density | 326.7 .. 1218 | 326.7 .. 1218 | 13% / 8% / 3% |
| spur.length | 0.0892 .. 0.1056 | 0.0892 .. 0.1056 | 8% / 8% / 9% |
| spur.radius | 0.004282 .. 0.005811 | 0.004282 .. 0.005811 | 8% / 5% / 11% |
| spur.density | 2898 .. 8183 | 2898 .. 8183 | 8% / 10% / 7% |
| spur.youngs_modulus_pa | 5.038e+07 .. 1.052e+09 | 5.038e+07 .. 1.052e+09 | 11% / 6% / 8% |
| spur.flexural_modulus_pa | 3.144e+06 .. 4.509e+07 | 3.144e+06 .. 4.509e+07 | 6% / 11% / 8% |
| spur.damping_ratio | 0.1975 .. 0.7227 | 0.1975 .. 0.7227 | 8% / 6% / 11% |
| stem.length | 0.01374 .. 0.01475 | 0.01374 .. 0.01475 | 11% / 8% / 6% |
| stem.youngs_modulus_pa | 3.024e+07 .. 5.635e+09 | 3.024e+07 .. 5.635e+09 | 10% / 4% / 10% |
| stem.flexural_modulus_pa | 2.968e+07 .. 1.998e+08 | 2.968e+07 .. 1.998e+08 | 14% / 6% / 5% |
| stem.damping_ratio | 0.2524 .. 0.7647 | 0.2524 .. 0.7647 | 8% / 6% / 11% |
| stem.elevation_deg | -88.87 .. -30.27 | -88.87 .. -30.27 | 4% / 6% / 15% |
| apple_radius | 0.03009 .. 0.03355 | 0.03009 .. 0.03355 | 8% / 8% / 9% |
| apple_density | 1618 .. 2238 | 1618 .. 2238 | 5% / 11% / 9% |
| support_kp | 186.3 .. 301.5 | 186.3 .. 301.5 | 7% / 7% / 11% |
| support_roll_kp | 0.4843 .. 1.01 | 0.4843 .. 1.01 | 10% / 6% / 9% |
| support_zeta | 0.2519 .. 0.7818 | 0.2519 .. 0.7818 | 6% / 9% / 10% |
| arm_link_mass_scale | 0.901 .. 1.1 | 0.9013 .. 1.1 | 9% / 9% / 7% |
| arm_link_inertia_scale | 0.9004 .. 1.1 | 0.9004 .. 1.1 | 8% / 6% / 11% |
| arm_ee_mass_scale | 0.9501 .. 1.05 | 0.9501 .. 1.05 | 9% / 7% / 9% |
| arm_ee_inertia_scale | 0.9501 .. 1.049 | 0.9501 .. 1.049 | 6% / 11% / 8% |
| weld_polar_deg | 2.622 .. 29.88 | 2.622 .. 29.88 | 7% / 10% / 8% |
