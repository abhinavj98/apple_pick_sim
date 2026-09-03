# Stem harvest without penalty-weld damping

| Field | Value |
| ----- | ----- |
| **Status** | Planned |
| **Date** | 2026-09-03 |
| **Canonical living docs** | H1 `docs/handbook-coupled-simulation.md` §6; `docs/WRENCH_READOUT.md`; `docs/explicit-apple-load-tcp-harvest.md`; `docs/damping-tuning.md` |
| **Depends on** | Stem–apple FIXED gather (`gather_joint_wrench_child_com_device`); lagged TCP `body_f` apply |
| **Out of scope** | First-pull leftover-\(C\) slam; fixture `stem_apple` \(k_d\); Newton submodule; plot LPF |

## Purpose

Real-replay `ft_wrist` and the lagged plant wrench on the FR3 TCP include the
AVBD **penalty-weld damper** \(k_d\dot C\) on the stem–apple `FIXED` joint.
That term is sized as critical damping of a \(10^5\,\mathrm{N/m}\) numerical
spring on the apple mass (\(\zeta=0.3\) ⇒ \(k_d\sim 90\,\mathrm{N{\cdot}s/m}\)),
not as pedicel viscosity. Because the apple is prescribed (co-teleported with
TCP), \(\dot C\approx v_{\mathrm{tcp}}\) at pull start, so harvest jumps
\(\sim 2\,\mathrm{N}\) while real wrist force at the same speed jumps
\(\sim 0.01\,\mathrm{N}\).

This slice **stops sending \(k_d\dot C\) to the arm and to `ft_wrist`**. VBD
still solves with \(k_d\) (and \(k_d/\Delta t\) in the Hessian). Wood
stretch/bend \(\zeta\) is unchanged.

## Physics (why this is not “delete real damping”)

- Real viscosity lives in the **deforming stem** (already VBD cable
  stretch/bend damping). A true rigid grasp has \(C=\dot C=0\); the wrist load
  is the reaction \(\lambda\) that transmits that wood constitutive force.
- Sim \(k_d\dot C\) is the residual rate of a **soft penalty glue** on a
  prescribed apple. Harvesting it double-counts a numerical residual as a
  physical wrist load.
- Quasi-static bench pulls confirm \(|F|\) tracks stretch and stays up on
  hold, not \(v_{\mathrm{tcp}}\).

## Decision (locked)

| Topic | Choice |
| --- | --- |
| Where to strip \(k_d\) | **Gather inputs**, not fixture / not VBD solve |
| Mechanism | `gather_joint_wrench_child_com_device(..., include_penalty_damping=False)` passes a **cached zeros** array in place of `solver.joint_penalty_kd`. Do not mutate `solver.joint_penalty_kd`. Do not patch Newton. |
| Who passes `False` | `harvest_stem_tension_for_tcp`, `_harvest_stem_tension_for_tcp_cpu`, `harvest_batched_stem_tension` |
| Who keeps default `True` | Woody / debug gather (`batched_obs` junction wrenches, `fixed_joint_wrenches_child_com_vbd` default, equilibrium tests) |
| Angular + linear | Zeros **all** penalty-\(k_d\) slots for that gather (linear and angular). Translation-only \(\dot C\) tests still isolate the linear slot. |
| Caps / \(mg\) / inertia | Unchanged; applied after gather |
| First-episode \(C\) spike | **Out of scope** (leftover constraint error, not \(k_d v\)) |

## API

```python
def gather_joint_wrench_child_com_device(
    model,
    solver,
    *,
    body_q,
    body_q_prev,
    joint_indices,
    dt,
    control=None,
    out_f=None,
    out_t=None,
    include_penalty_damping: bool = True,
) -> tuple[wp.array, wp.array]:
```

When `include_penalty_damping` is false, launch
`gather_joint_wrench_child_at_com_kernel` with a device array of zeros the
same shape/dtype/device as `solver.joint_penalty_kd`. Cache that zeros buffer
on the solver (e.g. `_joint_penalty_kd_harvest_zeros`) so coupled substeps do
not allocate every tick.

`fixed_joint_wrenches_child_com_vbd` gains the same flag (default `True`) and
forwards it through the device gather, then copies to NumPy. CPU harvest must
use `False` so CPU/GPU harvest parity stays on the same law.

## Call graph after the change

```text
VBD step  (joint_penalty_kd unchanged)
  → harvest_stem_tension_* (include_penalty_damping=False)
      → gather: F = -(kC + λ)     # no kd Ċ
      → transport to TCP + mg + caps
      → proxy_forces[tcp]
  → next MuJoCo: body_f[tcp] ← coupling_forces_cache
  → gym ft_wrist ← coupling_forces_cache
```

## Tests (contract)

1. **Identity (no VBD between gathers).** Same `body_q`, `body_q_prev` with
   apple translation shifted by \(-\delta\) on \(x\), \(\Delta t=\) `SUB_DT`.
   \(\dot C=\delta/\Delta t\,\mathbf{e}_x\). Read linear \(k_d\) from
   `joint_penalty_kd[joint_constraint_start[stem]+0]`.
   `F_on - F_off ≈ -k_d \dot C` on the child (atol ~0.05 N).
2. Existing sign / \(mg\) / gain / cap harvest tests still pass once their
   reference gather uses the **same** `include_penalty_damping=False` as
   harvest (`_stem_apple_wrench_from_scene`).
3. Batched harvest mock may assert the new kwarg is `False`.

## Docs to update when implementing

- H1 §6: harvested stem wrench is \(kC+\lambda\) (penalty \(k_d\dot C\)
  omitted); VBD still uses \(k_d\).
- `docs/explicit-apple-load-tcp-harvest.md` and `docs/WRENCH_READOUT.md`:
  harvest vs debug gather flag.
- Optional one-liner in `docs/damping-tuning.md`: weld \(k_d\) remains in
  VBD; it is not applied to TCP/`ft_wrist`.
